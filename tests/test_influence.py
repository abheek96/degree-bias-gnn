"""
Tests for influence.py — BFS helpers and Jacobian-based influence computation.

Fast unit tests use the tiny_graph fixture (no network I/O).
Slow integration tests use cora_data + trained_gcn (marked with @pytest.mark.slow).
"""

import pytest
import torch

from influence import (
    _khop_bfs,
    _khop_distances,
    _khop_neighbors,
    compute_influence_analysis,
    influence_distribution,
    k_hop_subsets_exact,
)


# ── _khop_neighbors / _khop_bfs ───────────────────────────────────────────────

class TestKhopNeighbors:
    """
    Graph topology (from tiny_graph fixture):

        0 — 1 — 3 — 4 — 5
            |       |
            2       6

    Seed = node 3.
    k=1 neighbors of 3: {1, 4}
    k=2 neighbors of 3: {0, 2, 5, 6}
    """

    def test_k1_returns_direct_neighbors(self, tiny_graph):
        N = tiny_graph.num_nodes
        result = _khop_neighbors(tiny_graph.edge_index, node_x=3, k=1, num_nodes=N)
        assert result == {1, 4}

    def test_k2_is_superset_of_k1(self, tiny_graph):
        N = tiny_graph.num_nodes
        k1 = _khop_neighbors(tiny_graph.edge_index, node_x=3, k=1, num_nodes=N)
        k2 = _khop_neighbors(tiny_graph.edge_index, node_x=3, k=2, num_nodes=N)
        assert k1 < k2, "k=2 must be a strict superset of k=1"

    def test_k2_correct_new_nodes(self, tiny_graph):
        N = tiny_graph.num_nodes
        k2 = _khop_neighbors(tiny_graph.edge_index, node_x=3, k=2, num_nodes=N)
        assert k2 == {0, 1, 2, 4, 5, 6}

    def test_seed_excluded(self, tiny_graph):
        N = tiny_graph.num_nodes
        for k in (1, 2):
            result = _khop_neighbors(tiny_graph.edge_index, node_x=3, k=k, num_nodes=N)
            assert 3 not in result, f"Seed node 3 should not appear in k={k} result"

    def test_isolated_node_has_no_neighbors(self, tiny_graph):
        """A node with no edges returns an empty set."""
        import torch
        from torch_geometric.data import Data

        ei = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
        result = _khop_neighbors(ei, node_x=2, k=1, num_nodes=3)
        assert result == set()

    def test_bfs_and_khop_neighbors_agree(self, tiny_graph):
        """_khop_bfs(return_distances=False) == _khop_neighbors."""
        N = tiny_graph.num_nodes
        for node in range(N):
            for k in (1, 2):
                via_bfs = _khop_bfs(tiny_graph.edge_index, node, k, N)
                via_fn  = _khop_neighbors(tiny_graph.edge_index, node, k, N)
                assert via_bfs == via_fn, f"Mismatch at node={node} k={k}"


# ── _khop_distances ────────────────────────────────────────────────────────────

class TestKhopDistances:
    def test_k1_distances_are_one(self, tiny_graph):
        N = tiny_graph.num_nodes
        dist = _khop_distances(tiny_graph.edge_index, node_x=3, k=2, num_nodes=N)
        assert dist[1] == 1
        assert dist[4] == 1

    def test_k2_distances_are_two(self, tiny_graph):
        N = tiny_graph.num_nodes
        dist = _khop_distances(tiny_graph.edge_index, node_x=3, k=2, num_nodes=N)
        for node in (0, 2, 5, 6):
            assert dist[node] == 2, f"Expected dist[{node}]==2, got {dist.get(node)}"

    def test_keys_equal_khop_neighbors(self, tiny_graph):
        """Keys of _khop_distances == _khop_neighbors for every (node, k)."""
        N = tiny_graph.num_nodes
        for node in range(N):
            for k in (1, 2):
                dist_keys = set(_khop_distances(tiny_graph.edge_index, node, k, N).keys())
                neighbors = _khop_neighbors(tiny_graph.edge_index, node, k, N)
                assert dist_keys == neighbors, f"Mismatch at node={node} k={k}"

    def test_seed_not_in_distances(self, tiny_graph):
        N = tiny_graph.num_nodes
        dist = _khop_distances(tiny_graph.edge_index, node_x=3, k=2, num_nodes=N)
        assert 3 not in dist


# ── k_hop_subsets_exact ────────────────────────────────────────────────────────

class TestKhopSubsetsExact:
    def test_subsets_are_disjoint(self, tiny_graph):
        N = tiny_graph.num_nodes
        subsets = k_hop_subsets_exact(
            node_idx=3, num_hops=2,
            edge_index=tiny_graph.edge_index, num_nodes=N, device="cpu",
        )
        seen = set()
        for s in subsets:
            s_set = set(s.tolist())
            assert s_set.isdisjoint(seen), f"Overlapping nodes across subsets: {s_set & seen}"
            seen |= s_set

    def test_union_equals_khop_plus_seed(self, tiny_graph):
        """Union of all subsets = k-hop neighbors ∪ {seed}."""
        N = tiny_graph.num_nodes
        subsets = k_hop_subsets_exact(
            node_idx=3, num_hops=2,
            edge_index=tiny_graph.edge_index, num_nodes=N, device="cpu",
        )
        union = set()
        for s in subsets:
            union |= set(s.tolist())

        expected = _khop_neighbors(tiny_graph.edge_index, 3, k=2, num_nodes=N) | {3}
        assert union == expected

    def test_subset_0_is_seed(self, tiny_graph):
        """S_0 contains only the seed node."""
        N = tiny_graph.num_nodes
        subsets = k_hop_subsets_exact(
            node_idx=3, num_hops=2,
            edge_index=tiny_graph.edge_index, num_nodes=N, device="cpu",
        )
        assert subsets[0].tolist() == [3]

    def test_subset_1_matches_k1_neighbors(self, tiny_graph):
        N = tiny_graph.num_nodes
        subsets = k_hop_subsets_exact(
            node_idx=3, num_hops=2,
            edge_index=tiny_graph.edge_index, num_nodes=N, device="cpu",
        )
        s1 = set(subsets[1].tolist())
        k1 = _khop_neighbors(tiny_graph.edge_index, 3, k=1, num_nodes=N)
        assert s1 == k1

    def test_num_subsets_is_hops_plus_one(self, tiny_graph):
        N = tiny_graph.num_nodes
        for hops in (1, 2):
            subsets = k_hop_subsets_exact(
                node_idx=0, num_hops=hops,
                edge_index=tiny_graph.edge_index, num_nodes=N, device="cpu",
            )
            assert len(subsets) == hops + 1


# ── influence_distribution ─────────────────────────────────────────────────────

class TestInfluenceDistribution:
    """Uses tiny_graph with a randomly-initialized 2-layer GCN."""

    @pytest.fixture(scope="class")
    def gcn_tiny(self, tiny_graph):
        from models import get_model
        import torch
        torch.manual_seed(0)
        model = get_model(
            "GCN",
            in_dim=tiny_graph.num_node_features,
            hidden_dim=8,
            out_dim=2,
            num_layers=3,
            dropout=0.0,
        )
        model.eval()
        return model

    def test_output_length_equals_num_nodes(self, tiny_graph, gcn_tiny):
        N = tiny_graph.num_nodes
        inf = influence_distribution(gcn_tiny, tiny_graph, node_x=3, k_hops=2)
        assert inf.shape == (N,), f"Expected shape ({N},), got {inf.shape}"

    def test_all_values_nonneg(self, tiny_graph, gcn_tiny):
        inf = influence_distribution(gcn_tiny, tiny_graph, node_x=3, k_hops=2)
        assert (inf >= 0).all(), "Influence scores must be non-negative (L1 norm)"

    def test_zero_outside_khop(self, tiny_graph, gcn_tiny):
        """Nodes outside the k-hop receptive field have influence 0."""
        N = tiny_graph.num_nodes
        k = 2
        inf = influence_distribution(gcn_tiny, tiny_graph, node_x=3, k_hops=k)
        reachable = _khop_neighbors(tiny_graph.edge_index, 3, k=k, num_nodes=N) | {3}
        outside = set(range(N)) - reachable
        for node in outside:
            assert inf[node].item() == 0.0, f"Node {node} outside k-hop should have 0 influence"

    def test_self_influence_nonzero(self, tiny_graph, gcn_tiny):
        """Node 3's self-influence (hop 0) should be > 0."""
        inf = influence_distribution(gcn_tiny, tiny_graph, node_x=3, k_hops=2)
        assert inf[3].item() > 0.0, "Self-influence should be positive"


# ── compute_influence_analysis (slow integration test) ────────────────────────

@pytest.mark.slow
def test_compute_influence_analysis_structure(cora_data, trained_gcn):
    """compute_influence_analysis returns dicts with expected keys."""
    from torch_geometric.utils import degree as graph_degree

    model, pred = trained_gcn
    k_hops = 2
    results = compute_influence_analysis(
        model, cora_data, pred, k_hops=k_hops, target_nodes=[387, 1362]
    )
    assert len(results) > 0, "Expected at least one result"

    expected_keys = {"node_idx", "degree", "true_label", "pred_label",
                     "same_class_influence", "diff_class_influence",
                     "same_class_influence_norm", "diff_class_influence_norm",
                     "total_train_influence", "n_same_train", "n_diff_train",
                     "neighbors"}
    for r in results:
        missing = expected_keys - set(r.keys())
        assert not missing, f"Result dict missing keys: {missing}"
        assert isinstance(r["neighbors"], list)
