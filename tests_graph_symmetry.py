"""
Tests for graph symmetry and consistency between inspect_node_aggregation
and _khop_neighbors after the incoming-edge fix.

IMPORTANT: uses the same data pipeline as main.py — CC filtering + random
split seeded at 42 — so train_mask matches what the influence analysis sees.

Run with:
    python tests_graph_symmetry.py
"""

import random
import numpy as np
import torch
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import LargestConnectedComponents

from dataset_utils import apply_split
from influence import _khop_neighbors
from models.gcn import inspect_node_aggregation
from utils import get_khop_cardinality


# ── fixtures ──────────────────────────────────────────────────────────────────

def _set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_cora():
    """Load Cora with CC filtering and random split (seed=42), matching main.py."""
    data = Planetoid(root="./data", name="Cora")[0]
    data = LargestConnectedComponents()(data)

    _set_seed(42)
    dataset_cfg = {
        "name": "Cora",
        "use_cc": True,
        "num_train_per_class": 20,
        "num_val": 500,
    }
    data = apply_split(data, split="random", dataset_cfg=dataset_cfg)
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


# ── test 5: training nodes in inspect_node_aggregation match influence analysis ──

def test_train_nodes_consistent(node_idx=1894):
    """The training nodes flagged in inspect_node_aggregation must be exactly
    the nodes that _khop_neighbors returns which are also in train_mask.
    This catches any split mismatch between the two code paths."""
    data = load_cora()
    N = data.num_nodes

    train_set = set(data.train_mask.cpu().nonzero(as_tuple=False).view(-1).tolist())

    # training neighbors according to _khop_neighbors + train_mask
    khop = _khop_neighbors(data.edge_index, node_idx, k=1, num_nodes=N)
    khop_train = {n for n in khop if n in train_set}

    # training neighbors according to inspect_node_aggregation
    df = inspect_node_aggregation(
        node_idx=node_idx,
        edge_index=data.edge_index,
        train_mask=data.train_mask,
        y=data.y,
    )
    inspect_train = set(df.loc[df["in_train_set"], "neighbor"].tolist())
    inspect_train.discard(node_idx)  # self-loop

    only_in_khop    = khop_train - inspect_train
    only_in_inspect = inspect_train - khop_train

    print(f"\n[test_train_nodes_consistent]  node={node_idx}")
    print(f"  train neighbors via _khop_neighbors        : {sorted(khop_train)}")
    print(f"  train neighbors via inspect_node_aggregation: {sorted(inspect_train)}")
    if only_in_khop or only_in_inspect:
        print(f"  only in _khop     : {sorted(only_in_khop)}")
        print(f"  only in inspect   : {sorted(only_in_inspect)}")
        print("  FAIL — mismatch between the two code paths")
    else:
        print("  PASS — training neighbor sets are identical")
    return khop_train == inspect_train


# ── test 6: get_khop_cardinality matches _khop_neighbors count ───────────────

def test_khop_cardinality_matches_neighbors(nodes=(387, 1894, 1362), ks=(1, 2)):
    """get_khop_cardinality(data, k)[node] must equal len(_khop_neighbors(..., k))
    for every (node, k) pair.  Validates that the dense-adjacency-power
    implementation in utils.py and the BFS in influence.py agree on the count
    of distinct nodes within each k-hop neighbourhood."""
    data = load_cora()
    N = data.num_nodes

    print(f"\n[test_khop_cardinality_matches_neighbors]")
    passed = True
    for k in ks:
        card = get_khop_cardinality(data, k)
        for node in nodes:
            bfs_count    = len(_khop_neighbors(data.edge_index, node, k, N))
            tensor_count = card[node].item()
            match = bfs_count == tensor_count
            if not match:
                passed = False
            status = "ok" if match else "MISMATCH"
            print(f"  node={node:>4}  k={k}  bfs={bfs_count:>4}  tensor={tensor_count:>4}  {status}")
    if passed:
        print("  PASS — all counts agree")
    else:
        print("  FAIL — count mismatch(es) found")
    return passed


# ── test 7: node 387 — receptive field depth vs aggregation table ─────────────

def load_cora_public():
    """Load Cora with CC filtering and the fixed public split."""
    data = Planetoid(root="./data", name="Cora", split="public")[0]
    data = LargestConnectedComponents()(data)
    return data


def test_node_387_receptive_field():
    """For node 387 (degree=16, public split):

    - k=1 neighbourhood has 0 training nodes — consistent with the aggregation
      table showing all 17 rows as in_train_set=False.
    - k=2 neighbourhood surfaces the 3 training nodes logged by the influence
      analysis (nodes 2221, 456, 2248) when num_layers=3 (k_hops=2).
    - get_khop_cardinality gives the correct count at both depths and agrees
      with the raw _khop_neighbors BFS.

    This confirms: inspect_node_aggregation is hardcoded to 1-hop; the
    influence analysis searches k_hops = num_layers - 1 hops.  The two
    analyses diverge only when num_layers > 2, which is exactly the condition
    under which the influence log found training nodes absent from the table.
    """
    node = 387
    data = load_cora_public()
    N    = data.num_nodes

    train_set = set(data.train_mask.cpu().nonzero(as_tuple=False).view(-1).tolist())

    nb_k1 = _khop_neighbors(data.edge_index, node, k=1, num_nodes=N)
    nb_k2 = _khop_neighbors(data.edge_index, node, k=2, num_nodes=N)

    train_k1   = {n for n in nb_k1 if n in train_set}
    train_k2   = {n for n in nb_k2 if n in train_set}
    only_at_k2 = train_k2 - train_k1   # nodes reachable only via 2 hops

    card_k1 = get_khop_cardinality(data, k=1)[node].item()
    card_k2 = get_khop_cardinality(data, k=2)[node].item()

    # Ground truth from the influence log (num_layers=3, public split)
    expected_train_k2 = {2221, 456, 2248}

    print(f"\n[test_node_387_receptive_field]  node={node}")
    print(f"  k=1 : {len(nb_k1):>3} distinct neighbours  |  {len(train_k1)} training nodes"
          f"  (cardinality={card_k1})")
    print(f"  k=2 : {len(nb_k2):>3} distinct neighbours  |  {len(train_k2)} training nodes"
          f"  (cardinality={card_k2})")
    print(f"  training at k=1          : {sorted(train_k1)}")
    print(f"  training at k=2          : {sorted(train_k2)}")
    print(f"  new training nodes at k=2: {sorted(only_at_k2)}")

    checks = {
        "k=1 has 0 training nodes (matches aggregation table)": len(train_k1) == 0,
        "k=2 includes all training nodes from influence log"  : expected_train_k2 <= train_k2,
        "influence-log nodes absent from k=1 neighbourhood"  : expected_train_k2.isdisjoint(train_k1),
        "cardinality k=1 == degree 16"                       : card_k1 == 16,
        "cardinality k=2 > k=1 (receptive field grows)"      : card_k2 > card_k1,
        "cardinality k=1 agrees with BFS count"              : card_k1 == len(nb_k1),
        "cardinality k=2 agrees with BFS count"              : card_k2 == len(nb_k2),
    }

    passed = True
    for desc, ok in checks.items():
        status = "ok" if ok else "FAIL"
        print(f"  [{status}] {desc}")
        if not ok:
            passed = False

    print("  PASS" if passed else "  FAIL")
    return passed


# ── run all ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    results = {}
    results["graph_symmetry"]                   = test_graph_symmetry()
    results["khop_matches_incoming"]            = test_khop_matches_incoming(node_idx=1894)
    test_node_1894_edges()
    results["inspect_matches_khop"]             = test_inspect_matches_khop(node_idx=1894)
    results["train_nodes_consistent"]           = test_train_nodes_consistent(node_idx=1894)
    results["khop_cardinality_matches_neighbors"] = test_khop_cardinality_matches_neighbors()
    results["node_387_receptive_field"]         = test_node_387_receptive_field()

    print("\n── Summary ──")
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {status}  {name}")
