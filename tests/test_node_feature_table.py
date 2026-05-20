"""
Tests for analysis/node_feature_table.py — feature table builder and
logistic regression pipeline.

Fast unit tests use tiny_graph and synthetic DataFrames; no Cora download.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "analysis"))

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn.functional as F

from node_feature_table import _FEATURE_COLS, _build_rows, _run_logistic_regression
from torch_geometric.utils import degree as graph_degree


# ── minimal GCN-compatible model ──────────────────────────────────────────────

class _DummyModel(torch.nn.Module):
    """Minimal model with GCN-compatible forward signature."""

    def __init__(self, in_dim: int, out_dim: int = 2):
        super().__init__()
        self.fc = torch.nn.Linear(in_dim, out_dim)

    def forward(self, x, edge_index):
        return self.fc(x)


# ── shared helpers ─────────────────────────────────────────────────────────────

def _make_build_rows_inputs(tiny_graph):
    """Prepare all inputs needed to call _build_rows on tiny_graph."""
    from utils import compute_distances_to_train, get_avg_spl_to_train, get_avg_spl_to_same_class_train, get_node_purity
    from influence import k_hop_subsets_exact

    data = tiny_graph
    N = data.num_nodes
    device = torch.device("cpu")

    model = _DummyModel(in_dim=data.num_node_features, out_dim=2)
    model.eval()
    with torch.no_grad():
        logits = model(data.x, data.edge_index)
    pred = logits.argmax(dim=1)

    all_deg = graph_degree(data.edge_index[1], N).long()
    train_set = set(data.train_mask.nonzero(as_tuple=True)[0].tolist())
    y = data.y
    test_idx = data.test_mask.nonzero(as_tuple=True)[0].tolist()
    n_test = len(test_idx)

    purity_1 = get_node_purity(data, k=1, node_mask=data.test_mask)
    purity_2 = get_node_purity(data, k=2, node_mask=data.test_mask)

    dist_any, dist_same = compute_distances_to_train(data)

    avg_spl = get_avg_spl_to_train(data)
    avg_spl_same = get_avg_spl_to_same_class_train(data)

    return dict(
        data=data,
        model=model,
        pred=pred,
        k_hops=1,
        device=device,
        train_set=train_set,
        y=y,
        all_deg=all_deg,
        test_idx=test_idx,
        purity_1=purity_1,
        purity_2=purity_2,
        dist_any=dist_any,
        dist_same=dist_same,
        avg_spl=avg_spl,
        avg_spl_same=avg_spl_same,
        embeddings=None,
        skip_influence=True,
    )


def _make_synthetic_df(n: int = 100, seed: int = 42) -> pd.DataFrame:
    """Create a synthetic DataFrame matching the _FEATURE_COLS schema."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({col: rng.standard_normal(n) for col in _FEATURE_COLS})
    df["node_idx"] = np.arange(n)
    df["correct"] = rng.integers(0, 2, size=n)
    return df


# ── _FEATURE_COLS schema consistency ──────────────────────────────────────────

class TestFeatureColsSchema:
    def test_feature_cols_matches_build_rows_keys(self, tiny_graph):
        """_FEATURE_COLS must match the keys returned by _build_rows minus id/target."""
        inputs = _make_build_rows_inputs(tiny_graph)
        rows = _build_rows(**inputs)
        assert len(rows) > 0, "_build_rows returned no rows"
        row_keys = set(rows[0].keys()) - {"node_idx", "correct"}
        feat_set = set(_FEATURE_COLS)
        missing_from_rows = feat_set - row_keys
        extra_in_rows = row_keys - feat_set
        assert not missing_from_rows, f"_FEATURE_COLS has columns absent from row dict: {missing_from_rows}"
        assert not extra_in_rows, f"row dict has columns absent from _FEATURE_COLS: {extra_in_rows}"

    def test_feature_cols_no_duplicates(self):
        assert len(_FEATURE_COLS) == len(set(_FEATURE_COLS)), "Duplicate column names in _FEATURE_COLS"


# ── _build_rows output ─────────────────────────────────────────────────────────

class TestBuildRows:
    def test_one_row_per_test_node(self, tiny_graph):
        inputs = _make_build_rows_inputs(tiny_graph)
        rows = _build_rows(**inputs)
        n_test = int(tiny_graph.test_mask.sum())
        assert len(rows) == n_test

    def test_all_feature_cols_present(self, tiny_graph):
        inputs = _make_build_rows_inputs(tiny_graph)
        rows = _build_rows(**inputs)
        for row in rows:
            for col in _FEATURE_COLS:
                assert col in row, f"Feature column '{col}' missing from row"

    def test_correct_column_is_binary(self, tiny_graph):
        inputs = _make_build_rows_inputs(tiny_graph)
        rows = _build_rows(**inputs)
        for row in rows:
            assert row["correct"] in (0, 1), f"'correct' must be 0 or 1, got {row['correct']}"

    def test_degree_matches_graph(self, tiny_graph):
        inputs = _make_build_rows_inputs(tiny_graph)
        rows = _build_rows(**inputs)
        all_deg = inputs["all_deg"]
        for row in rows:
            expected = int(all_deg[row["node_idx"]].item())
            assert row["degree"] == expected

    def test_influence_cols_nan_when_skipped(self, tiny_graph):
        """All influence columns are NaN when skip_influence=True."""
        inputs = _make_build_rows_inputs(tiny_graph)
        rows = _build_rows(**inputs)
        infl_cols = [c for c in _FEATURE_COLS if "infl" in c]
        for row in rows:
            for col in infl_cols:
                assert np.isnan(row[col]), f"Expected NaN for '{col}' with skip_influence=True"

    def test_emb_cols_nan_without_embeddings(self, tiny_graph):
        """Embedding columns are NaN when embeddings=None."""
        inputs = _make_build_rows_inputs(tiny_graph)
        rows = _build_rows(**inputs)
        emb_cols = [c for c in _FEATURE_COLS if c.startswith("emb_")]
        for row in rows:
            for col in emb_cols:
                assert np.isnan(row[col]), f"Expected NaN for '{col}' with embeddings=None"


# ── _run_logistic_regression ───────────────────────────────────────────────────

class TestRunLogisticRegression:
    def test_return_structure(self):
        df = _make_synthetic_df(n=100)
        result = _run_logistic_regression(df)
        assert len(result) == 6, "Expected 6-tuple (auroc, acc, pr_auc, coef_df, p_misc, y_misc)"
        auroc, acc, pr_auc, coef_df, p_misc, y_misc = result
        assert hasattr(auroc, "__len__"), "auroc should be an array"
        assert hasattr(acc, "__len__"), "acc should be an array"
        assert isinstance(pr_auc, float)
        assert isinstance(coef_df, pd.DataFrame)

    def test_auroc_in_range(self):
        df = _make_synthetic_df(n=100)
        auroc, *_ = _run_logistic_regression(df)
        assert all(0.0 <= v <= 1.0 for v in auroc), f"AUROC out of [0, 1]: {auroc}"

    def test_pr_auc_in_range(self):
        df = _make_synthetic_df(n=100)
        _, _, pr_auc, *_ = _run_logistic_regression(df)
        assert 0.0 <= pr_auc <= 1.0, f"PR-AUC out of [0, 1]: {pr_auc}"

    def test_coef_df_has_expected_columns(self):
        df = _make_synthetic_df(n=100)
        _, _, _, coef_df, _, _ = _run_logistic_regression(df)
        assert "feature" in coef_df.columns
        assert "coefficient" in coef_df.columns

    def test_small_dataset(self):
        """Should not crash with n=20 (5-fold CV becomes challenging but valid)."""
        df = _make_synthetic_df(n=20, seed=7)
        # Ensure at least one of each class per fold
        df["correct"] = [0, 1] * 10
        # Should not raise
        _run_logistic_regression(df)

    def test_feature_subset(self):
        """Passing a subset of feature_cols should work."""
        df = _make_synthetic_df(n=80)
        subset = _FEATURE_COLS[:4]
        auroc, acc, pr_auc, coef_df, p_misc, y_misc = _run_logistic_regression(df, feature_cols=subset)
        assert len(coef_df) <= len(subset)
