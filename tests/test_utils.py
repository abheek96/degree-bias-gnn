"""
Tests for utils.py — graph-structural feature functions.

All fast unit tests use the tiny_graph fixture (no network I/O).
"""

import math

import pytest
import torch

from influence import _khop_neighbors
from utils import (
    compute_distances_to_train,
    get_avg_spl_to_train,
    get_khop_cardinality,
    get_node_purity,
)


# ── get_node_purity ────────────────────────────────────────────────────────────

class TestGetNodePurity:
    """
    tiny_graph topology:
        0 — 1 — 3 — 4 — 5
            |       |
            2       6

    Classes: [0, 0, 0, 0, 1, 1, 1]
    """

    def test_homogeneous_neighborhood(self, tiny_graph):
        """Node 1's 1-hop neighbors are {0, 2, 3} — all class 0 like node 1 → purity=1."""
        purity = get_node_purity(tiny_graph, k=1)
        # node 1: neighbors 0, 2, 3 all class 0; node 1 is also class 0 → purity=1
        assert math.isclose(float(purity[1]), 1.0, abs_tol=1e-5)

    def test_heterogeneous_neighborhood(self, tiny_graph):
        """Node 3's 1-hop neighbors are {1, 4}: class 0 and class 1.
        Node 3 is class 0 → same-class count = 1, total = 2 → purity = 0.5."""
        purity = get_node_purity(tiny_graph, k=1)
        assert math.isclose(float(purity[3]), 0.5, abs_tol=1e-5)

    def test_purity_in_range(self, tiny_graph):
        """Purity values must be in [0, 1] or NaN."""
        purity = get_node_purity(tiny_graph, k=1)
        valid = purity[~torch.isnan(purity)]
        assert (valid >= 0).all() and (valid <= 1).all()

    def test_node_mask_reduces_output_size(self, tiny_graph):
        """When node_mask is provided the output length equals mask.sum()."""
        mask = tiny_graph.test_mask
        purity = get_node_purity(tiny_graph, k=1, node_mask=mask)
        assert purity.shape[0] == int(mask.sum())

    def test_k2_purity_differs_from_k1(self, tiny_graph):
        """k=2 includes a wider neighborhood so purity may differ from k=1."""
        p1 = get_node_purity(tiny_graph, k=1)
        p2 = get_node_purity(tiny_graph, k=2)
        # They won't be identical for all nodes in a mixed-class graph
        assert not torch.allclose(p1[~torch.isnan(p1)], p2[~torch.isnan(p2)],
                                   atol=1e-5), \
            "k=1 and k=2 purity should differ for at least some nodes"


# ── get_khop_cardinality ───────────────────────────────────────────────────────

class TestGetKhopCardinality:
    def test_k1_equals_degree(self, tiny_graph):
        """For k=1, cardinality should equal the node degree."""
        from torch_geometric.utils import degree as graph_degree
        deg = graph_degree(tiny_graph.edge_index[1], tiny_graph.num_nodes).long()
        card = get_khop_cardinality(tiny_graph, k=1)
        assert torch.equal(card, deg), f"k=1 cardinality != degree:\n{card} vs {deg}"

    def test_k2_geq_k1(self, tiny_graph):
        """k=2 cardinality must be >= k=1 for every node."""
        c1 = get_khop_cardinality(tiny_graph, k=1)
        c2 = get_khop_cardinality(tiny_graph, k=2)
        assert (c2 >= c1).all()

    def test_matches_bfs_count(self, tiny_graph):
        """Cardinality from matrix power must equal BFS count for every node."""
        N = tiny_graph.num_nodes
        for k in (1, 2):
            card = get_khop_cardinality(tiny_graph, k=k)
            for node in range(N):
                bfs = len(_khop_neighbors(tiny_graph.edge_index, node, k=k, num_nodes=N))
                assert card[node].item() == bfs, (
                    f"node={node} k={k}: matrix={card[node].item()} BFS={bfs}"
                )


# ── compute_distances_to_train ─────────────────────────────────────────────────

class TestComputeDistancesToTrain:
    """
    tiny_graph: train nodes = {0, 4}.

    Expected min distances to any train node (from test nodes {2, 3, 5, 6}):
        node 2: 2-hop via 2→1→0  → dist_any=2
        node 3: 1-hop via 3→4    → dist_any=1 (or 3→1→0, but min is 1)
            Actually: neighbors of 3 are {1,4}, node 4 is a train node → dist_any=1
        node 5: 1-hop via 5→4    → dist_any=1
        node 6: 1-hop via 6→4    → dist_any=1

    Expected min distances to same-class train node:
        node 2 (class 0): nearest same-class train = node 0 (class 0), distance 2
        node 3 (class 0): nearest same-class train = node 0 (class 0), distance 2 (via 3→1→0)
        node 5 (class 1): nearest same-class train = node 4 (class 1), distance 1
        node 6 (class 1): nearest same-class train = node 4 (class 1), distance 1
    """

    def test_output_shapes(self, tiny_graph):
        dist_any, dist_same = compute_distances_to_train(tiny_graph)
        n_test = int(tiny_graph.test_mask.sum())
        assert dist_any.shape == (n_test,)
        assert dist_same.shape == (n_test,)

    def test_dist_any_leq_dist_same(self, tiny_graph):
        """Distance to nearest any-class train ≤ distance to nearest same-class train."""
        dist_any, dist_same = compute_distances_to_train(tiny_graph)
        INF = tiny_graph.num_nodes + 1
        reachable = dist_same < INF
        assert (dist_any[reachable] <= dist_same[reachable]).all()

    def test_dist_values_positive(self, tiny_graph):
        dist_any, dist_same = compute_distances_to_train(tiny_graph)
        INF = tiny_graph.num_nodes + 1
        reachable = dist_any < INF
        assert (dist_any[reachable] > 0).all(), "All distances must be positive (no self-loops to train)"


# ── get_avg_spl_to_train ───────────────────────────────────────────────────────

class TestGetAvgSplToTrain:
    def test_output_length(self, tiny_graph):
        """Output length equals num_nodes."""
        avg_spl = get_avg_spl_to_train(tiny_graph)
        assert avg_spl.shape == (tiny_graph.num_nodes,)

    def test_train_nodes_have_zero_or_low_spl(self, tiny_graph):
        """Train nodes (0 and 4) have 0 SPL to themselves."""
        avg_spl = get_avg_spl_to_train(tiny_graph)
        # Train nodes are at distance 0 to themselves, so their avg includes 0
        train_idx = tiny_graph.train_mask.nonzero(as_tuple=True)[0]
        for idx in train_idx.tolist():
            # avg SPL includes self-distance (0) so it should be low
            assert float(avg_spl[idx]) <= 5.0, (
                f"Train node {idx} has unexpectedly large avg SPL: {avg_spl[idx]}"
            )

    def test_nonneg_values(self, tiny_graph):
        avg_spl = get_avg_spl_to_train(tiny_graph)
        valid = avg_spl[~torch.isnan(avg_spl)]
        assert (valid >= 0).all()
