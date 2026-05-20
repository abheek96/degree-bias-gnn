"""
Tests for graph symmetry and consistency between inspect_node_aggregation
and _khop_neighbors after the incoming-edge fix.

Uses the same data pipeline as main.py (CC filtering + random split seeded
at 42) so train_mask matches what the influence analysis sees.
"""

import pytest

from influence import _khop_neighbors
from models.gcn import inspect_node_aggregation
from utils import get_khop_cardinality


# ── helpers ───────────────────────────────────────────────────────────────────

def _incoming_neighbors(edge_index, node_idx):
    """Nodes that send an edge TO node_idx (col == node_idx)."""
    col = edge_index[1]
    mask = col == node_idx
    return set(edge_index[0][mask].tolist())


# ── tests ─────────────────────────────────────────────────────────────────────

@pytest.mark.slow
def test_graph_symmetry(cora_data):
    """Every edge (u, v) has a reverse (v, u) — graph is undirected."""
    ei = cora_data.edge_index
    edges = set(zip(ei[0].tolist(), ei[1].tolist()))
    asymmetric = [(u, v) for u, v in edges if (v, u) not in edges]
    assert asymmetric == [], f"Found {len(asymmetric)} asymmetric edges: {asymmetric[:5]}"


@pytest.mark.slow
def test_khop_matches_incoming(cora_data, node_idx=1894):
    """_khop_neighbors(k=1) equals the incoming-neighbor set for a given node."""
    N = cora_data.num_nodes
    khop = _khop_neighbors(cora_data.edge_index, node_idx, k=1, num_nodes=N)
    incoming = _incoming_neighbors(cora_data.edge_index, node_idx)
    incoming.discard(node_idx)
    assert khop == incoming, (
        f"Mismatch — only in khop: {sorted(khop - incoming)}, "
        f"only in incoming: {sorted(incoming - khop)}"
    )


@pytest.mark.slow
def test_node_1894_edges(cora_data):
    """Check presence of specific edges around node 1894."""
    edges = set(zip(cora_data.edge_index[0].tolist(), cora_data.edge_index[1].tolist()))
    expected_present = [(184, 1894), (1828, 1894), (271, 1894)]
    for u, v in expected_present:
        assert (u, v) in edges, f"Expected edge ({u}, {v}) not found"


@pytest.mark.slow
def test_inspect_matches_khop(cora_data, node_idx=1894):
    """inspect_node_aggregation neighbor set equals _khop_neighbors(k=1)."""
    N = cora_data.num_nodes
    df = inspect_node_aggregation(
        node_idx=node_idx,
        edge_index=cora_data.edge_index,
        train_mask=cora_data.train_mask,
        y=cora_data.y,
    )
    inspect_neighbors = set(df["neighbor"].tolist()) - {node_idx}
    khop = _khop_neighbors(cora_data.edge_index, node_idx, k=1, num_nodes=N)
    assert inspect_neighbors == khop, (
        f"only in inspect: {sorted(inspect_neighbors - khop)}, "
        f"only in khop: {sorted(khop - inspect_neighbors)}"
    )


@pytest.mark.slow
def test_train_nodes_consistent(cora_data, node_idx=1894):
    """Training neighbors agree between _khop_neighbors + train_mask and inspect_node_aggregation."""
    N = cora_data.num_nodes
    train_set = set(cora_data.train_mask.cpu().nonzero(as_tuple=False).view(-1).tolist())

    khop = _khop_neighbors(cora_data.edge_index, node_idx, k=1, num_nodes=N)
    khop_train = {n for n in khop if n in train_set}

    df = inspect_node_aggregation(
        node_idx=node_idx,
        edge_index=cora_data.edge_index,
        train_mask=cora_data.train_mask,
        y=cora_data.y,
    )
    inspect_train = set(df.loc[df["in_train_set"], "neighbor"].tolist())
    inspect_train.discard(node_idx)

    assert khop_train == inspect_train, (
        f"only in khop: {sorted(khop_train - inspect_train)}, "
        f"only in inspect: {sorted(inspect_train - khop_train)}"
    )


@pytest.mark.slow
@pytest.mark.parametrize("node", [387, 1894, 1362])
@pytest.mark.parametrize("k", [1, 2])
def test_khop_cardinality_matches_bfs(cora_data, node, k):
    """get_khop_cardinality matches len(_khop_neighbors) for every (node, k)."""
    N = cora_data.num_nodes
    bfs_count = len(_khop_neighbors(cora_data.edge_index, node, k, N))
    tensor_count = get_khop_cardinality(cora_data, k)[node].item()
    assert bfs_count == tensor_count, (
        f"node={node} k={k}: BFS={bfs_count} tensor={tensor_count}"
    )


@pytest.mark.slow
def test_influence_nodes_in_receptive_field(cora_data):
    """Nodes found by influence analysis (absent from 1-hop) are present at k=2."""
    from torch_geometric.datasets import Planetoid
    from torch_geometric.transforms import LargestConnectedComponents
    from dataset_utils import apply_split
    import random, numpy as np, torch

    # Load public split for the node_387 case
    data_pub = Planetoid(root="./data", name="Cora", split="public")[0]
    data_pub = LargestConnectedComponents()(data_pub)

    for data, node, influence_nodes, k in [
        (data_pub,  387,  {2221, 456, 2248}, 2),
        (cora_data, 1894, {184, 271},        2),
    ]:
        N = data.num_nodes
        nb_k1 = _khop_neighbors(data.edge_index, node, k=1, num_nodes=N)
        nb_k  = _khop_neighbors(data.edge_index, node, k=k, num_nodes=N)
        assert influence_nodes <= nb_k, (
            f"node={node}: influence nodes {influence_nodes - nb_k} not in {k}-hop"
        )
        assert influence_nodes.isdisjoint(nb_k1), (
            f"node={node}: influence nodes {influence_nodes & nb_k1} unexpectedly in 1-hop"
        )
