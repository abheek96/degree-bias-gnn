"""
Tests for graph symmetry and consistency between inspect_node_aggregation
and _khop_neighbors after the incoming-edge fix.

Run with:
    python tests_graph_symmetry.py
"""

import torch
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import LargestConnectedComponents

from influence import _khop_neighbors
from models.gcn import inspect_node_aggregation


# ── fixtures ──────────────────────────────────────────────────────────────────

def load_cora():
    data = Planetoid(root="./data", name="Cora")[0]
    data = LargestConnectedComponents()(data)
    return data


# ── helpers ───────────────────────────────────────────────────────────────────

def incoming_neighbors(edge_index, node_idx):
    """Nodes that send an edge TO node_idx (col == node_idx)."""
    col = edge_index[1]
    mask = col == node_idx
    return set(edge_index[0][mask].tolist())


def outgoing_neighbors(edge_index, node_idx):
    """Nodes that node_idx sends an edge TO (row == node_idx)."""
    row = edge_index[0]
    mask = row == node_idx
    return set(edge_index[1][mask].tolist())


# ── test 1: graph symmetry ─────────────────────────────────────────────────────

def test_graph_symmetry():
    """Every edge (u, v) should have a reverse (v, u) for an undirected graph."""
    data = load_cora()
    ei = data.edge_index
    edges = set(zip(ei[0].tolist(), ei[1].tolist()))
    asymmetric = [(u, v) for u, v in edges if (v, u) not in edges]

    print(f"[test_graph_symmetry]")
    print(f"  Total edges : {ei.size(1)}")
    print(f"  Asymmetric  : {len(asymmetric)}")
    if asymmetric:
        print(f"  Examples    : {asymmetric[:10]}")
        print("  FAIL — graph has asymmetric edges")
    else:
        print("  PASS — graph is fully symmetric")
    return len(asymmetric) == 0


# ── test 2: _khop_neighbors matches incoming neighbors for k=1 ────────────────

def test_khop_matches_incoming(node_idx=1894):
    """After the fix, _khop_neighbors(k=1) should equal the incoming neighbor
    set (i.e. what inspect_node_aggregation aggregates from), excluding
    node_idx itself."""
    data = load_cora()
    N = data.num_nodes

    khop = _khop_neighbors(data.edge_index, node_idx, k=1, num_nodes=N)
    incoming = incoming_neighbors(data.edge_index, node_idx)
    incoming.discard(node_idx)  # exclude self-loop if present in raw edge_index

    only_in_khop     = khop - incoming
    only_in_incoming = incoming - khop

    print(f"\n[test_khop_matches_incoming]  node={node_idx}")
    print(f"  _khop_neighbors : {sorted(khop)}")
    print(f"  incoming (col)  : {sorted(incoming)}")
    if only_in_khop or only_in_incoming:
        print(f"  only in _khop     : {sorted(only_in_khop)}")
        print(f"  only in incoming  : {sorted(only_in_incoming)}")
        print("  FAIL — mismatch")
    else:
        print("  PASS — sets are identical")
    return khop == incoming


# ── test 3: specific edges for node 1894 ──────────────────────────────────────

def test_node_1894_edges():
    """Check whether the edges (1894->184), (184->1894), (1894->271),
    (271->1894) exist in edge_index, explaining why 184 and 271 did or
    did not appear in the aggregation table."""
    data = load_cora()
    edges = set(zip(data.edge_index[0].tolist(), data.edge_index[1].tolist()))
    pairs = [(1894, 184), (184, 1894), (1894, 271), (271, 1894), (1894, 1828), (1828, 1894)]

    print(f"\n[test_node_1894_edges]")
    for u, v in pairs:
        exists = (u, v) in edges
        print(f"  edge ({u:>4}, {v:>4}) : {'EXISTS' if exists else 'absent'}")


# ── test 4: inspect_node_aggregation neighbors match _khop_neighbors ──────────

def test_inspect_matches_khop(node_idx=1894):
    """The neighbor column of inspect_node_aggregation (excluding the self-loop
    row) should equal _khop_neighbors(k=1)."""
    data = load_cora()
    N = data.num_nodes

    df = inspect_node_aggregation(
        node_idx=node_idx,
        edge_index=data.edge_index,
        train_mask=data.train_mask,
        y=data.y,
    )

    inspect_neighbors = set(df["neighbor"].tolist()) - {node_idx}  # remove self-loop
    khop = _khop_neighbors(data.edge_index, node_idx, k=1, num_nodes=N)

    only_in_inspect = inspect_neighbors - khop
    only_in_khop    = khop - inspect_neighbors

    print(f"\n[test_inspect_matches_khop]  node={node_idx}")
    if only_in_inspect or only_in_khop:
        print(f"  only in inspect_node_aggregation : {sorted(only_in_inspect)}")
        print(f"  only in _khop_neighbors          : {sorted(only_in_khop)}")
        print("  FAIL — mismatch")
    else:
        print("  PASS — neighbor sets are identical")
    return inspect_neighbors == khop


# ── run all ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    results = {}
    results["graph_symmetry"]        = test_graph_symmetry()
    results["khop_matches_incoming"] = test_khop_matches_incoming(node_idx=1894)
    test_node_1894_edges()
    results["inspect_matches_khop"]  = test_inspect_matches_khop(node_idx=1894)

    print("\n── Summary ──")
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {status}  {name}")
