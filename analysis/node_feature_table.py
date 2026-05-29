"""
node_feature_table.py — Build a per-test-node feature table and train a
logistic regression classifier to predict GCN misclassification.

Each row is one test node.  Features cover structural graph factors (degree,
SPL, purity, neighbourhood training-node composition) and model-specific
Jacobian-L1 influence factors.  The binary target ``correct`` is 1 when the GCN
correctly classifies the node and 0 otherwise.

Columns
-------
  node_idx                      — global node index (identifier, not a feature)
  degree                        — in-degree
  min_dist_to_train             — min shortest-path to any training node
  min_dist_to_same_class_train  — min shortest-path to a same-class training node
  avg_spl_to_train              — average shortest-path to all training nodes
  avg_spl_to_same_class_train   — average shortest-path to same-class training nodes
  purity_1hop                   — fraction of cumulative 1-hop neighbourhood that shares focal class
  purity_2hop                   — fraction of cumulative 2-hop neighbourhood that shares focal class
  n_same_train_1hop             — # same-class training nodes in the 1-hop ring
  n_diff_train_1hop             — # diff-class training nodes in the 1-hop ring
  n_same_train_2hop             — # same-class training nodes in the 2-hop ring
  n_diff_train_2hop             — # diff-class training nodes in the 2-hop ring
  same_train_ratio_1hop         — n_same_train_1hop / |1-hop ring|
  diff_train_ratio_1hop         — n_diff_train_1hop / |1-hop ring|
  same_train_ratio_2hop         — n_same_train_2hop / |2-hop ring|
  diff_train_ratio_2hop         — n_diff_train_2hop / |2-hop ring|
  mean_cosine_sim_1hop          — mean cosine similarity of focal node and 1-hop neighbours (raw features)
  total_infl_same_1hop          — Jacobian-L1 influence from all same-class 1-hop nodes
  total_infl_diff_1hop          — Jacobian-L1 influence from all diff-class 1-hop nodes
  total_infl_same_2hop          — Jacobian-L1 influence from all same-class 2-hop nodes
  total_infl_diff_2hop          — Jacobian-L1 influence from all diff-class 2-hop nodes
  same_train_infl_frac_1hop     — fraction of total influence from same-class training nodes at hop 1
  diff_train_infl_frac_1hop     — fraction of total influence from diff-class training nodes at hop 1
  same_train_infl_frac_2hop     — fraction of total influence from same-class training nodes at hop 2
  diff_train_infl_frac_2hop     — fraction of total influence from diff-class training nodes at hop 2
  closeness_centrality          — reciprocal of sum of shortest-path distances to all reachable nodes
  eigenvector_centrality        — principal eigenvector component (NaN if ARPACK fails)
  correct                       — 1 = correctly classified, 0 = misclassified (target)

The influence columns are omitted (set to NaN) when ``--no-influence`` is passed.
Fraction features (same/diff_train_infl_frac_*hop) use total I_x.sum() as the denominator.

Model source (mutually exclusive)
----------------------------------
  --checkpoint PATH   load a saved state_dict
  --run N             resolve results/{exec}/checkpoints/run{N:02d}_*.pt
  --multi-run         load all num_runs checkpoints, average influence features
                      across runs, and predict misclassification frequency
                      (Ridge regression instead of logistic regression)
  --subset-across-runs
                      load all num_runs checkpoints, compute the feature-subset
                      PR-AUC ablation per run, and report each subset's PR-AUC as
                      mean ± std across runs (CSV + bar chart with error bars)
  (neither)           retrain from scratch with config seed

Usage
-----
  uv run analysis/node_feature_table.py --run 1
  uv run analysis/node_feature_table.py --run 1 --no-influence
  uv run analysis/node_feature_table.py \\
      --checkpoint results/.../checkpoints/run01_seed42.pt --save-dir ./output
  uv run analysis/node_feature_table.py --multi-run --shap
  uv run analysis/node_feature_table.py --subset-across-runs
"""

import argparse
import logging
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from torch_geometric.utils import degree as graph_degree

from checkpoint_utils import (
    _build_model,       # noqa: F401 — imported for completeness; used via load_from_checkpoint
    _deep_merge,
    _resolve_run_checkpoint,
    load_cfg,
    load_from_checkpoint,
    set_seed,
    train_model,
)
from dataset_utils import load_or_create_split
from influence import influence_distribution, k_hop_subsets_exact
from utils import (
    compute_distances_to_train,
    get_avg_spl_to_same_class_train,
    get_avg_spl_to_train,
    get_closeness_centrality,
    get_eigenvector_centrality,
    get_node_purity,
)

log = logging.getLogger(__name__)

# Feature columns used for logistic regression (everything except node_idx and correct).
_FEATURE_COLS = [
    "degree",
    "min_dist_to_train",
    "min_dist_to_same_class_train",
    "avg_spl_to_train",
    "avg_spl_to_same_class_train",
    "purity_1hop",
    "purity_2hop",
    "n_same_train_1hop",
    "n_diff_train_1hop",
    "n_same_train_2hop",
    "n_diff_train_2hop",
    "same_train_ratio_1hop",
    "diff_train_ratio_1hop",
    "same_train_ratio_2hop",
    "diff_train_ratio_2hop",
    "mean_cosine_sim_1hop",
    "total_infl_same_1hop",
    "total_infl_diff_1hop",
    "total_infl_same_2hop",
    "total_infl_diff_2hop",
    "same_train_infl_frac_1hop",
    "diff_train_infl_frac_1hop",
    "same_train_infl_frac_2hop",
    "diff_train_infl_frac_2hop",
    "emb_sim_same_1hop",
    "emb_sim_diff_1hop",
    "emb_purity_delta",
    "closeness_centrality",
    "eigenvector_centrality",
]

# Human-readable labels for SHAP plots. Falls back to the raw column name for
# any key not listed here.
_FEATURE_DISPLAY_NAMES: dict[str, str] = {
    "degree":                       "Degree",
    "min_dist_to_train":            "Min dist → any train node",
    "min_dist_to_same_class_train": "Min dist → same-class train node",
    "avg_spl_to_train":             "Avg SPL → any train node",
    "avg_spl_to_same_class_train":  "Avg SPL → same-class train node",
    "purity_1hop":                  "Purity (1-hop)",
    "purity_2hop":                  "Purity (2-hop)",
    "n_same_train_1hop":            "# same-class train (1-hop ring)",
    "n_diff_train_1hop":            "# diff-class train (1-hop ring)",
    "n_same_train_2hop":            "# same-class train (2-hop ring)",
    "n_diff_train_2hop":            "# diff-class train (2-hop ring)",
    "same_train_ratio_1hop":        "Same-class train ratio (1-hop)",
    "diff_train_ratio_1hop":        "Diff-class train ratio (1-hop)",
    "same_train_ratio_2hop":        "Same-class train ratio (2-hop)",
    "diff_train_ratio_2hop":        "Diff-class train ratio (2-hop)",
    "mean_cosine_sim_1hop":         "Mean cosine sim (raw features, 1-hop)",
    "total_infl_same_1hop":         "Influence: same-class nodes (1-hop)",
    "total_infl_diff_1hop":         "Influence: diff-class nodes (1-hop)",
    "total_infl_same_2hop":         "Influence: same-class nodes (2-hop)",
    "total_infl_diff_2hop":         "Influence: diff-class nodes (2-hop)",
    "same_train_infl_frac_1hop":    "Influence frac: same-class train (1-hop)",
    "diff_train_infl_frac_1hop":    "Influence frac: diff-class train (1-hop)",
    "same_train_infl_frac_2hop":    "Influence frac: same-class train (2-hop)",
    "diff_train_infl_frac_2hop":    "Influence frac: diff-class train (2-hop)",
    "emb_sim_same_1hop":            "Embedding sim: same-class (1-hop)",
    "emb_sim_diff_1hop":            "Embedding sim: diff-class (1-hop)",
    "emb_purity_delta":             "Embedding purity delta (same − diff)",
    "closeness_centrality":         "Closeness centrality",
    "eigenvector_centrality":       "Eigenvector centrality",
    # multi-run averaged columns
    "avg_total_infl_same_1hop":      "Avg influence: same-class (1-hop)",
    "avg_total_infl_diff_1hop":      "Avg influence: diff-class (1-hop)",
    "avg_total_infl_same_2hop":      "Avg influence: same-class (2-hop)",
    "avg_total_infl_diff_2hop":      "Avg influence: diff-class (2-hop)",
    "avg_same_train_infl_frac_1hop": "Avg infl frac: same-class train (1-hop)",
    "avg_diff_train_infl_frac_1hop": "Avg infl frac: diff-class train (1-hop)",
    "avg_same_train_infl_frac_2hop": "Avg infl frac: same-class train (2-hop)",
    "avg_diff_train_infl_frac_2hop": "Avg infl frac: diff-class train (2-hop)",
    "avg_emb_sim_same_1hop":         "Avg emb sim: same-class (1-hop)",
    "avg_emb_sim_diff_1hop":         "Avg emb sim: diff-class (1-hop)",
    "avg_emb_purity_delta":          "Avg embedding purity delta",
    "std_misc_freq":                 "Std dev of misclassification across runs",
}

# Topology-fixed features shared by both single-run and multi-run modes.
_TOPO_COLS = [
    "degree",
    "min_dist_to_train",
    "min_dist_to_same_class_train",
    "avg_spl_to_train",
    "avg_spl_to_same_class_train",
    "purity_1hop",
    "purity_2hop",
    "n_same_train_1hop",
    "n_diff_train_1hop",
    "n_same_train_2hop",
    "n_diff_train_2hop",
    "same_train_ratio_1hop",
    "diff_train_ratio_1hop",
    "same_train_ratio_2hop",
    "diff_train_ratio_2hop",
    "mean_cosine_sim_1hop",
    "closeness_centrality",
    "eigenvector_centrality",
]

_MULTI_INFL_COLS = [
    "avg_total_infl_same_1hop",
    "avg_total_infl_diff_1hop",
    "avg_total_infl_same_2hop",
    "avg_total_infl_diff_2hop",
    "avg_same_train_infl_frac_1hop",
    "avg_diff_train_infl_frac_1hop",
    "avg_same_train_infl_frac_2hop",
    "avg_diff_train_infl_frac_2hop",
]

_MULTI_EMB_COLS = [
    "avg_emb_sim_same_1hop",
    "avg_emb_sim_diff_1hop",
    "avg_emb_purity_delta",
]

_FEATURE_COLS_MULTI = _TOPO_COLS + _MULTI_INFL_COLS + _MULTI_EMB_COLS

# Feature groups for subset-comparison plot (6 logical categories).
# Each value lists the column names present in single-run mode; multi-run mode
# uses the same groups but with avg_* equivalents for influence/embedding cols.
_FEATURE_GROUP_MAP: dict[str, list[str]] = {
    "degree": ["degree"],
    "purity": ["purity_1hop", "purity_2hop", "mean_cosine_sim_1hop"],
    "training_proximity": [
        "min_dist_to_train", "min_dist_to_same_class_train",
        "avg_spl_to_train", "avg_spl_to_same_class_train",
        "n_same_train_1hop", "n_diff_train_1hop",
        "n_same_train_2hop", "n_diff_train_2hop",
        "same_train_ratio_1hop", "diff_train_ratio_1hop",
        "same_train_ratio_2hop", "diff_train_ratio_2hop",
    ],
    "centrality": ["closeness_centrality", "eigenvector_centrality"],
    "influence": [
        "total_infl_same_1hop", "total_infl_diff_1hop",
        "total_infl_same_2hop", "total_infl_diff_2hop",
        "same_train_infl_frac_1hop", "diff_train_infl_frac_1hop",
        "same_train_infl_frac_2hop", "diff_train_infl_frac_2hop",
    ],
    "embedding": ["emb_sim_same_1hop", "emb_sim_diff_1hop", "emb_purity_delta"],
}

_FEATURE_GROUP_MAP_MULTI: dict[str, list[str]] = {
    **{k: v for k, v in _FEATURE_GROUP_MAP.items()
       if k not in ("influence", "embedding")},
    "influence": _MULTI_INFL_COLS,
    "embedding": _MULTI_EMB_COLS,
}

_GROUP_ORDER = ["degree", "purity", "training_proximity", "centrality", "influence", "embedding"]

_GROUP_DISPLAY_NAMES = {
    "degree":             "Degree",
    "purity":             "Purity",
    "training_proximity": "Training proximity",
    "centrality":         "Centrality",
    "influence":          "Influence (Jacobian)",
    "embedding":          "Embedding",
}

# Hardcoded subset specs for the feature subset comparison plot.
# Each entry is (display_label, list_of_group_keys).  Groups whose columns are
# absent at runtime (e.g. --no-influence) are silently skipped; specs that
# expand to an empty column set are dropped.  "Full model" is added dynamically
# as the union of all available groups so it always reflects the actual run.
_SUBSET_SPECS: list[tuple[str, list[str]]] = [
    # ── single groups ─────────────────────────────────────────────────────────
    ("Degree",                         ["degree"]),
    ("Purity",                         ["purity"]),
    ("Training proximity",             ["training_proximity"]),
    ("Centrality",                     ["centrality"]),
    ("Influence",                      ["influence"]),
    ("Embedding",                      ["embedding"]),
    # ── combinations ─────────────────────────────────────────────────────────
    ("Degree + Purity",                ["degree", "purity"]),
    ("Degree + Training proximity",    ["degree", "training_proximity"]),
    ("All structural",                 ["degree", "purity", "training_proximity", "centrality"]),
]


# ── helpers ────────────────────────────────────────────────────────────────────

def _build_adj_list(edge_index_cpu, N: int) -> dict:
    adj = defaultdict(set)
    for src, dst in zip(edge_index_cpu[0].tolist(), edge_index_cpu[1].tolist()):
        adj[src].add(dst)
    return adj


def _hop_ring_stats(hop_nodes: list, train_set: set, y, true_lbl: int) -> dict:
    """Count same/diff-class training nodes in a hop ring and compute ratios.

    Parameters
    ----------
    hop_nodes : node indices in this exact hop ring
    train_set : set of training node indices
    y         : global label tensor
    true_lbl  : true class of the focal node

    Returns keys: n_same, n_diff, n_total, ratio_same, ratio_diff
    """
    n = len(hop_nodes)
    n_same = sum(1 for v in hop_nodes if v in train_set and int(y[v].item()) == true_lbl)
    n_diff = sum(1 for v in hop_nodes if v in train_set and int(y[v].item()) != true_lbl)
    return {
        "n_same":    n_same,
        "n_diff":    n_diff,
        "n_total":   n,
        "ratio_same": n_same / n if n > 0 else float("nan"),
        "ratio_diff": n_diff / n if n > 0 else float("nan"),
    }


def _embedding_sim_features(
    node_x: int, neighbors_1hop: list, embeddings, y, true_lbl: int
) -> dict:
    """Penultimate-layer cosine similarity between focal node and 1-hop neighbours.

    Returns keys: emb_sim_same_1hop, emb_sim_diff_1hop, emb_purity_delta
    All values are NaN when embeddings is None or there are no neighbors.
    """
    nan = float("nan")
    if embeddings is None or not neighbors_1hop:
        return {"emb_sim_same_1hop": nan, "emb_sim_diff_1hop": nan, "emb_purity_delta": nan}

    focal_emb = F.normalize(embeddings[node_x].unsqueeze(0), dim=1)
    same_nbrs = [n for n in neighbors_1hop if int(y[n].item()) == true_lbl]
    diff_nbrs = [n for n in neighbors_1hop if int(y[n].item()) != true_lbl]

    emb_sim_same = float(
        (focal_emb * F.normalize(embeddings[same_nbrs], dim=1)).sum(dim=1).mean()
    ) if same_nbrs else nan

    emb_sim_diff = float(
        (focal_emb * F.normalize(embeddings[diff_nbrs], dim=1)).sum(dim=1).mean()
    ) if diff_nbrs else nan

    emb_purity_delta = (
        emb_sim_same - emb_sim_diff
        if not (np.isnan(emb_sim_same) or np.isnan(emb_sim_diff))
        else nan
    )
    return {
        "emb_sim_same_1hop":  emb_sim_same,
        "emb_sim_diff_1hop":  emb_sim_diff,
        "emb_purity_delta":   emb_purity_delta,
    }


def _influence_features(
    node_x: int, data, model, k_hops: int, subsets2, y, train_set: set, true_lbl: int
) -> dict:
    """Jacobian-L1 influence features for a single node.

    Returns a dict with the eight infl_* columns.  Called only when
    skip_influence=False; the caller is responsible for passing subsets2
    (already computed for 2-hop ring counts so they can be reused).
    """
    edge_index = data.edge_index
    N = data.num_nodes

    I_x        = influence_distribution(model, data, node_x, k_hops)
    total_infl = float(I_x.sum().item())
    hop_s      = k_hop_subsets_exact(node_x, k_hops, edge_index, N, I_x.device)

    def _infl_sum(nodes):
        return float(I_x[nodes].sum().item()) if nodes else 0.0

    def _train_frac(nodes):
        tr = [n for n in nodes if n in train_set]
        return float(I_x[tr].sum().item()) / total_infl if tr and total_infl > 0 else 0.0

    if len(hop_s) > 1:
        S1_i  = hop_s[1].tolist()
        same1 = [n for n in S1_i if int(y[n].item()) == true_lbl]
        diff1 = [n for n in S1_i if int(y[n].item()) != true_lbl]
        total_same_infl   = _infl_sum(same1)
        total_diff_infl   = _infl_sum(diff1)
        same_train_frac_1 = _train_frac(same1)
        diff_train_frac_1 = _train_frac(diff1)
    else:
        total_same_infl = total_diff_infl = 0.0
        same_train_frac_1 = diff_train_frac_1 = 0.0

    if len(subsets2) > 2 and len(subsets2[2]) > 0:
        S2_i  = subsets2[2].tolist()
        same2 = [n for n in S2_i if int(y[n].item()) == true_lbl]
        diff2 = [n for n in S2_i if int(y[n].item()) != true_lbl]
        total_same_infl_2  = _infl_sum(same2)
        total_diff_infl_2  = _infl_sum(diff2)
        same_train_frac_2  = _train_frac(same2)
        diff_train_frac_2  = _train_frac(diff2)
    else:
        total_same_infl_2 = total_diff_infl_2 = 0.0
        same_train_frac_2 = diff_train_frac_2 = 0.0

    return {
        "total_infl_same_1hop":      total_same_infl,
        "total_infl_diff_1hop":      total_diff_infl,
        "total_infl_same_2hop":      total_same_infl_2,
        "total_infl_diff_2hop":      total_diff_infl_2,
        "same_train_infl_frac_1hop": same_train_frac_1,
        "diff_train_infl_frac_1hop": diff_train_frac_1,
        "same_train_infl_frac_2hop": same_train_frac_2,
        "diff_train_infl_frac_2hop": diff_train_frac_2,
    }


_NAN_INFL = {
    "total_infl_same_1hop":      float("nan"),
    "total_infl_diff_1hop":      float("nan"),
    "total_infl_same_2hop":      float("nan"),
    "total_infl_diff_2hop":      float("nan"),
    "same_train_infl_frac_1hop": float("nan"),
    "diff_train_infl_frac_1hop": float("nan"),
    "same_train_infl_frac_2hop": float("nan"),
    "diff_train_infl_frac_2hop": float("nan"),
}

_NAN_EMB = {
    "emb_sim_same_1hop": float("nan"),
    "emb_sim_diff_1hop": float("nan"),
    "emb_purity_delta":  float("nan"),
}


# ── per-node feature computation ───────────────────────────────────────────────

def _build_rows(
    data,
    model,
    pred,
    k_hops: int,
    device,
    train_set: set,
    y,
    all_deg,
    test_idx: list,
    purity_1,               # FloatTensor [num_test]  — cumulative 1-hop, test_mask order
    purity_2,               # FloatTensor [num_test]  — cumulative 2-hop, test_mask order
    dist_any,               # LongTensor  [num_test]  — min SPL to any train node
    dist_same,              # LongTensor  [num_test]  — min SPL to same-class train node
    avg_spl,                # FloatTensor [num_nodes] — avg SPL to all train nodes
    avg_spl_same,           # FloatTensor [num_nodes] — avg SPL to same-class train nodes
    closeness,              # FloatTensor [num_nodes] — closeness centrality
    eigenvec,               # FloatTensor [num_nodes] — eigenvector centrality
    embeddings,             # FloatTensor [num_nodes, D] — penultimate GCN embeddings, CPU
    skip_influence: bool,
) -> list[dict]:
    N          = data.num_nodes
    INF        = N + 1      # sentinel used by compute_distances_to_train
    edge_index = data.edge_index
    x_feat     = data.x.cpu()    # raw input features for cosine sim
    adj        = _build_adj_list(edge_index.cpu(), N)

    rows = []
    for pos, node_x in enumerate(test_idx):
        true_lbl = int(y[node_x].item())

        # ── hop ring counts ───────────────────────────────────────────────────
        S1       = list(adj[node_x])
        subsets2 = k_hop_subsets_exact(node_x, 2, edge_index, N, device)
        S2       = subsets2[2].tolist() if len(subsets2) > 2 else []

        st1 = _hop_ring_stats(S1, train_set, y, true_lbl)
        st2 = _hop_ring_stats(S2, train_set, y, true_lbl)

        # ── cosine similarity (raw features, 1-hop) ───────────────────────────
        if S1:
            focal   = F.normalize(x_feat[node_x].unsqueeze(0), dim=1)
            nbr_mat = F.normalize(x_feat[S1], dim=1)
            cos_sim = float((focal * nbr_mat).sum(dim=1).mean().item())
        else:
            cos_sim = float("nan")

        # ── embedding-space similarity ────────────────────────────────────────
        emb = _embedding_sim_features(node_x, S1, embeddings, y, true_lbl)

        # ── Jacobian-L1 influence (expensive) ────────────────────────────────
        infl = (
            _influence_features(node_x, data, model, k_hops, subsets2, y, train_set, true_lbl)
            if not skip_influence
            else _NAN_INFL
        )

        # ── SPL / distance ────────────────────────────────────────────────────
        min_d  = int(dist_any[pos].item())
        min_ds = int(dist_same[pos].item())

        rows.append({
            "node_idx":                     node_x,
            "degree":                       int(all_deg[node_x].item()),
            "min_dist_to_train":            min_d  if min_d  < INF else float("nan"),
            "min_dist_to_same_class_train": min_ds if min_ds < INF else float("nan"),
            "avg_spl_to_train":             float(avg_spl[node_x].item()),
            "avg_spl_to_same_class_train":  float(avg_spl_same[node_x].item()),
            "purity_1hop":                  float(purity_1[pos].item()),
            "purity_2hop":                  float(purity_2[pos].item()),
            "n_same_train_1hop":            st1["n_same"],
            "n_diff_train_1hop":            st1["n_diff"],
            "n_same_train_2hop":            st2["n_same"],
            "n_diff_train_2hop":            st2["n_diff"],
            "same_train_ratio_1hop":        st1["ratio_same"],
            "diff_train_ratio_1hop":        st1["ratio_diff"],
            "same_train_ratio_2hop":        st2["ratio_same"],
            "diff_train_ratio_2hop":        st2["ratio_diff"],
            "mean_cosine_sim_1hop":         cos_sim,
            **infl,
            **emb,
            "closeness_centrality":         float(closeness[node_x].item()),
            "eigenvector_centrality":       float(eigenvec[node_x].item()),
            "correct":                      int(int(pred[node_x].item()) == true_lbl),
        })

        if (pos + 1) % 100 == 0 or (pos + 1) == len(test_idx):
            log.info("  processed %d / %d test nodes", pos + 1, len(test_idx))

    return rows


# ── multi-run helpers ─────────────────────────────────────────────────────────

def _compute_run_dependent_features(
    data, model, pred, k_hops: int, device,
    train_set: set, y, test_idx: list,
    skip_influence: bool, skip_embeddings: bool,
) -> list[dict]:
    """Compute per-node influence, embedding, and correctness for one run."""
    N          = data.num_nodes
    edge_index = data.edge_index
    adj        = _build_adj_list(edge_index.cpu(), N)

    if skip_embeddings:
        embeddings = None
    elif hasattr(model, "get_intermediate"):
        embeddings = model.get_intermediate(data.x, data.edge_index, layer=k_hops).cpu()
    else:
        with torch.no_grad():
            embeddings = model(data.x, data.edge_index).cpu()

    rows = []
    for node_x in test_idx:
        true_lbl = int(y[node_x].item())
        S1       = list(adj[node_x])
        subsets2 = k_hop_subsets_exact(node_x, 2, edge_index, N, device)

        infl = (
            _influence_features(node_x, data, model, k_hops, subsets2, y, train_set, true_lbl)
            if not skip_influence
            else _NAN_INFL
        )
        emb = (
            _embedding_sim_features(node_x, S1, embeddings, y, true_lbl)
            if not skip_embeddings
            else _NAN_EMB
        )

        rows.append({
            "node_idx": node_x,
            "correct":  int(int(pred[node_x].item()) == true_lbl),
            **infl,
            **emb,
        })

    return rows


def _load_all_runs(cfg, data_on_device, n_runs: int, device) -> list[tuple]:
    """Load all N run checkpoints; return list of (pred_cpu, model) tuples."""
    runs = []
    for run_id in range(1, n_runs + 1):
        ckpt_path = _resolve_run_checkpoint(cfg, run_id)
        if ckpt_path is None:
            raise RuntimeError(
                f"Could not resolve checkpoint for run {run_id}. "
                f"Ensure {n_runs} runs have been trained."
            )
        pred, model = load_from_checkpoint(cfg, data_on_device, device, ckpt_path)
        runs.append((pred.cpu(), model))
        log.info("Loaded run %d/%d: %s", run_id, n_runs, ckpt_path)
    return runs


def _aggregate_multi_run(
    topo_df: pd.DataFrame,
    run_features_list: list[list[dict]],
) -> pd.DataFrame:
    """Combine topology features with run-averaged influence/embedding features.

    run_features_list : one list[dict] per run; each dict has node_idx, correct,
                        8 infl cols, 3 emb cols.
    Returns a DataFrame with topology cols + avg_* cols + std_misc_freq + misc_freq.
    """
    run_dfs = [
        pd.DataFrame(feats).set_index("node_idx")
        for feats in run_features_list
    ]

    infl_emb_cols = [c for c in run_dfs[0].columns if c != "correct"]
    correct_arr   = np.stack([df["correct"].values for df in run_dfs], axis=1)  # [n_test, n_runs]
    misc_freq     = 1.0 - correct_arr.mean(axis=1)
    std_misc      = correct_arr.std(axis=1)

    avg_dict: dict[str, np.ndarray] = {}
    for col in infl_emb_cols:
        stacked = np.stack([df[col].values.astype(float) for df in run_dfs], axis=1)
        avg_dict[f"avg_{col}"] = np.nanmean(stacked, axis=1)

    avg_df = pd.DataFrame(avg_dict, index=run_dfs[0].index)
    avg_df["std_misc_freq"] = std_misc
    avg_df["misc_freq"]     = misc_freq
    avg_df = avg_df.reset_index()

    return topo_df.merge(avg_df, on="node_idx")


# ── univariate AUROC ──────────────────────────────────────────────────────────

def _univariate_auroc(df: pd.DataFrame):
    from sklearn.metrics import roc_auc_score

    results = []
    for col in _FEATURE_COLS:
        if col not in df.columns:
            continue
        sub = df[[col, "correct"]].dropna()
        if len(sub) < 10 or sub[col].nunique() < 2:
            continue
        y_m    = (sub["correct"] == 0).values.astype(int)
        scores = sub[col].values.astype(float)
        auroc  = roc_auc_score(y_m, scores)
        auroc  = max(auroc, 1.0 - auroc)
        results.append((col, auroc, len(sub)))

    results.sort(key=lambda x: x[1], reverse=True)

    log.info("── Univariate AUROC (feature used directly as ranking score) ───────")
    log.info("  %6s  %-42s  %s", "n", "feature", "AUROC")
    for col, auroc, n in results:
        log.info("  %6d  %-42s  %.4f", n, col, auroc)
    log.info("────────────────────────────────────────────────────────────────────")


# ── logistic regression ────────────────────────────────────────────────────────

def _run_logistic_regression(df: pd.DataFrame, feature_cols=None):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score
    from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    cols      = feature_cols if feature_cols is not None else _FEATURE_COLS
    available = [c for c in cols if c in df.columns]
    skipped_nan = [c for c in available if not df[c].notna().any()]
    available   = [c for c in available if df[c].notna().any()]
    if skipped_nan:
        log.info("Skipping all-NaN feature columns: %s", sorted(skipped_nan))
    df_clean  = df[available + ["correct"]].dropna()
    n_dropped = len(df) - len(df_clean)
    if n_dropped:
        log.warning("Dropped %d rows with NaN features before LR fitting", n_dropped)

    X = df_clean[available].values.astype(float)
    y = df_clean["correct"].values
    y_misc = 1 - y
    baseline = y_misc.mean()

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=42,
        )),
    ])

    cv    = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    auroc = cross_val_score(pipe, X, y, cv=cv, scoring="roc_auc")
    acc   = cross_val_score(pipe, X, y, cv=cv, scoring="accuracy")

    oof_proba   = cross_val_predict(pipe, X, y, cv=cv, method="predict_proba")
    p_misc      = oof_proba[:, 0]
    pr_auc      = average_precision_score(y_misc, p_misc)

    n           = len(y_misc)
    sorted_idx  = np.argsort(p_misc)[::-1]
    k_values    = [k for k in (50, 100, 200) if k <= n]
    k_values   += [int(round(n * pct)) for pct in (0.10, 0.20) if int(round(n * pct)) not in k_values]
    k_values    = sorted(set(k_values))

    pipe.fit(X, y)
    lr = pipe.named_steps["lr"]
    coef_df = (
        pd.DataFrame({"feature": available, "coefficient": lr.coef_[0]})
        .assign(abs_coef=lambda d: d["coefficient"].abs())
        .sort_values("abs_coef", ascending=False)
        .drop(columns="abs_coef")
        .reset_index(drop=True)
    )

    log.info("── Logistic Regression results (5-fold stratified CV) ──────────────")
    log.info("  n=%d  misclassified=%d (%.1f%%)  baseline_rate=%.3f",
             len(df_clean), int(y_misc.sum()), 100.0 * baseline, baseline)
    log.info("  AUROC:    %.4f ± %.4f", auroc.mean(), auroc.std())
    log.info("  Accuracy: %.4f ± %.4f", acc.mean(),   acc.std())
    log.info("  PR-AUC (avg precision, positive=misclassified):  %.4f", pr_auc)
    log.info("  Lift@k (ranked by P(misclassified), OOF scores):")
    log.info("    %6s  %10s  %10s  %6s", "k", "precision@k", "baseline", "lift")
    for k in k_values:
        top_k     = sorted_idx[:k]
        prec_at_k = y_misc[top_k].mean()
        lift      = prec_at_k / baseline if baseline > 0 else float("nan")
        log.info("    %6d  %10.3f  %10.3f  %6.2f×", k, prec_at_k, baseline, lift)
    log.info("  Coefficients (|coef| descending):")
    for _, row in coef_df.iterrows():
        log.info("    %-42s  %+.4f", row["feature"], row["coefficient"])
    log.info("────────────────────────────────────────────────────────────────────")

    return auroc, acc, pr_auc, coef_df, p_misc, y_misc


# ── Ridge regression (multi-run) ──────────────────────────────────────────────

def _run_ridge_regression(df: pd.DataFrame, feature_cols=None):
    from scipy.stats import spearmanr
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import KFold
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    cols      = feature_cols if feature_cols is not None else _FEATURE_COLS_MULTI
    available = [c for c in cols if c in df.columns]
    available = [c for c in available if df[c].notna().any()]
    df_clean  = df[available + ["misc_freq"]].dropna()
    n_dropped = len(df) - len(df_clean)
    if n_dropped:
        log.warning("Dropped %d rows with NaN before Ridge fitting", n_dropped)

    X      = df_clean[available].values.astype(float)
    y_freq = df_clean["misc_freq"].values

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("ridge",  Ridge(alpha=1.0)),
    ])

    cv        = KFold(n_splits=5, shuffle=True, random_state=42)
    oof_preds = np.zeros(len(y_freq))
    r2_scores: list[float] = []

    for train_idx, test_idx in cv.split(X):
        pipe.fit(X[train_idx], y_freq[train_idx])
        oof_preds[test_idx] = pipe.predict(X[test_idx])
        ss_res = np.sum((y_freq[test_idx] - oof_preds[test_idx]) ** 2)
        ss_tot = np.sum((y_freq[test_idx] - y_freq[test_idx].mean()) ** 2)
        r2_scores.append(1 - ss_res / ss_tot if ss_tot > 0 else float("nan"))

    spear_r, spear_p = spearmanr(y_freq, oof_preds)
    rmse = float(np.sqrt(np.mean((y_freq - oof_preds) ** 2)))
    r2   = float(np.nanmean(r2_scores))

    pipe.fit(X, y_freq)
    coef_df = (
        pd.DataFrame({"feature": available, "coefficient": pipe.named_steps["ridge"].coef_})
        .assign(abs_coef=lambda d: d["coefficient"].abs())
        .sort_values("abs_coef", ascending=False)
        .drop(columns="abs_coef")
        .reset_index(drop=True)
    )

    log.info("── Ridge Regression results (5-fold CV, target=misc_freq) ─────────")
    log.info("  n=%d  mean misc_freq=%.3f  std=%.3f", len(df_clean), y_freq.mean(), y_freq.std())
    log.info("  Spearman r=%.4f (p=%.4e)", spear_r, spear_p)
    log.info("  OOF R²=%.4f  RMSE=%.4f", r2, rmse)
    log.info("  Coefficients (|coef| descending):")
    for _, row in coef_df.iterrows():
        log.info("    %-42s  %+.4f", row["feature"], row["coefficient"])
    log.info("────────────────────────────────────────────────────────────────────")

    return spear_r, r2, rmse, coef_df, oof_preds, y_freq


# ── feature subset comparison ─────────────────────────────────────────────────

def _expand_groups(group_names, group_map, df):
    """Expand a list of group names to column names present in df."""
    cols = []
    for g in group_names:
        cols.extend(c for c in group_map.get(g, []) if c in df.columns)
    return cols


def _eval_subsets(df, group_map, metric_fn):
    """Evaluate each `_SUBSET_SPECS` entry plus a dynamic "Full model" on `df`.

    Parameters
    ----------
    df         : feature table (one row per test node).
    group_map  : feature-group → column-list mapping (single- or multi-run variant).
    metric_fn  : callable taking a list of column names and returning a float score.

    Returns a list of dicts with keys ``label``, ``score``, ``n_groups``. Subsets
    whose columns are all absent (or all-NaN) in `df` are skipped. "Full model" is
    the union of all available groups, so it reflects the columns actually present.
    """
    def _eval(label, groups):
        cols = [c for c in _expand_groups(groups, group_map, df) if df[c].notna().any()]
        if not cols:
            log.info("Skipping subset '%s' (no available columns)", label)
            return None
        return {"label": label, "score": float(metric_fn(cols)), "n_groups": len(groups)}

    results = []
    for label, groups in _SUBSET_SPECS:
        row = _eval(label, groups)
        if row is not None:
            results.append(row)

    # Full model = union of all available groups (added dynamically)
    avail_groups = [
        g for g in _GROUP_ORDER
        if any(c in df.columns and df[c].notna().any() for c in group_map.get(g, []))
    ]
    row = _eval("Full model", avail_groups)
    if row is not None:
        results.append(row)

    return results


def _run_subset_comparison(df, is_multi_run, save_dir, show, dataset, model_name):
    """Evaluate each entry in _SUBSET_SPECS plus a dynamic Full model bar.

    Saves one CSV and one horizontal bar chart PNG.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    metric_label = "Spearman r" if is_multi_run else "PR-AUC"
    suffix       = "multi" if is_multi_run else "single"
    group_map    = _FEATURE_GROUP_MAP_MULTI if is_multi_run else _FEATURE_GROUP_MAP

    if is_multi_run:
        def metric_fn(cols):
            spear_r, *_ = _run_ridge_regression(df, feature_cols=cols)
            return float(spear_r)
    else:
        def metric_fn(cols):
            _, _, pr_auc, *_ = _run_logistic_regression(df, feature_cols=cols)
            return float(pr_auc)

    results = _eval_subsets(df, group_map, metric_fn)
    for r in results:
        log.info("  %-35s  %s=%.4f", r["label"], metric_label, r["score"])

    if not results:
        log.warning("No subsets could be evaluated.")
        return pd.DataFrame()

    df_res = (
        pd.DataFrame([
            {"label": r["label"], metric_label: r["score"], "n_groups": r["n_groups"]}
            for r in results
        ])
        .sort_values(metric_label, ascending=True)
        .reset_index(drop=True)
    )

    # ── save CSV ──────────────────────────────────────────────────────────────
    if save_dir:
        sub = os.path.join(save_dir, "node_feature_table")
        os.makedirs(sub, exist_ok=True)
        path = os.path.join(sub, f"subset_comparison_{suffix}.csv")
        df_res.to_csv(path, index=False)
        log.info("Saved subset comparison CSV → %s", path)

    # ── plot ──────────────────────────────────────────────────────────────────
    _C_SINGLE = "#4878CF"   # individual feature group
    _C_COMBO  = "#EE854A"   # multi-group combination
    _C_FULL   = "#333333"   # full model

    def _bar_color(row):
        if row["label"] == "Full model":
            return _C_FULL
        return _C_SINGLE if row["n_groups"] == 1 else _C_COMBO

    colors = [_bar_color(row) for _, row in df_res.iterrows()]

    fig, ax = plt.subplots(figsize=(7, 0.55 * len(df_res) + 1.0))
    bars = ax.barh(df_res["label"], df_res[metric_label], color=colors, height=0.6, zorder=2)

    # Score labels at the end of each bar
    x_max = df_res[metric_label].max()
    x_min = df_res[metric_label].min()
    for bar, score in zip(bars, df_res[metric_label]):
        w      = bar.get_width()
        offset = (x_max - x_min) * 0.015
        ax.text(
            w + offset if score >= 0 else w - offset,
            bar.get_y() + bar.get_height() / 2,
            f"{score:.3f}", va="center",
            ha="left" if score >= 0 else "right", fontsize=7.5,
        )

    # Degree reference line
    degree_rows = df_res[df_res["label"] == "Degree"]
    if not degree_rows.empty:
        deg_score = float(degree_rows[metric_label].iloc[0])
        ax.axvline(deg_score, color="#D65F5F", ls="--", lw=1.2, zorder=3)
        label_ha = "left" if deg_score >= 0 else "right"
        ax.text(
            deg_score, len(df_res) - 0.1,
            f" degree\n ({deg_score:.3f})",
            color="#D65F5F", fontsize=7.5, va="top", ha=label_ha,
        )

    # Legend
    legend_handles = [
        Patch(facecolor=_C_SINGLE, label="Single group"),
        Patch(facecolor=_C_COMBO,  label="Combination"),
        Patch(facecolor=_C_FULL,   label="Full model"),
    ]
    ax.legend(handles=legend_handles, fontsize=8, loc="lower right")

    ax.set_xlabel(metric_label)
    ax.set_xlim(x_min - (x_max - x_min) * 0.05, x_max + (x_max - x_min) * 0.18)
    ax.set_title(f"{dataset} · {model_name} — feature subset comparison ({suffix})", fontsize=10)
    ax.grid(axis="x", lw=0.5, alpha=0.4, zorder=1)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()

    if save_dir:
        sub  = os.path.join(save_dir, "node_feature_table")
        path = os.path.join(sub, f"subset_comparison_{suffix}.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        log.info("Saved → %s", path)

    if show:
        plt.show()
    else:
        plt.close(fig)

    return df_res


# ── SHAP values ───────────────────────────────────────────────────────────────

def _compute_shap_values(df: pd.DataFrame, feature_cols=None):
    import shap
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    cols      = feature_cols if feature_cols is not None else _FEATURE_COLS
    available = [c for c in cols if c in df.columns]
    available = [c for c in available if df[c].notna().any()]

    keep = available + ["correct"]
    if "node_idx" in df.columns:
        keep = ["node_idx"] + keep
    df_clean = df[keep].dropna(subset=available + ["correct"])

    X          = df_clean[available].values.astype(float)
    y          = df_clean["correct"].values
    node_idxs  = df_clean["node_idx"].values.astype(int) if "node_idx" in df_clean.columns else np.arange(len(y))

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("lr",     LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)),
    ])
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    shap_vals   = np.zeros((len(y), len(available)))
    base_vals   = np.zeros(len(y))

    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X, y)):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr       = y[train_idx]

        pipe.fit(X_tr, y_tr)
        scaler   = pipe.named_steps["scaler"]
        lr_model = pipe.named_steps["lr"]

        X_tr_sc = scaler.transform(X_tr)
        X_te_sc = scaler.transform(X_te)

        explainer = shap.LinearExplainer(
            lr_model,
            masker=shap.maskers.Independent(X_tr_sc),
        )
        sv = explainer.shap_values(X_te_sc)
        shap_vals[test_idx] = sv
        base_vals[test_idx] = explainer.expected_value
        log.info("  SHAP fold %d/%d done", fold_idx + 1, cv.n_splits)

    display_names = [_FEATURE_DISPLAY_NAMES.get(c, c) for c in available]
    return -shap_vals, X, display_names, -base_vals, node_idxs


def _compute_shap_values_multi(df: pd.DataFrame, feature_cols=None):
    """OOF SHAP values for Ridge regression on misc_freq (multi-run mode)."""
    import shap
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import KFold
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    cols      = feature_cols if feature_cols is not None else _FEATURE_COLS_MULTI
    available = [c for c in cols if c in df.columns]
    available = [c for c in available if df[c].notna().any()]

    keep = available + ["misc_freq"]
    if "node_idx" in df.columns:
        keep = ["node_idx"] + keep
    df_clean  = df[keep].dropna(subset=available + ["misc_freq"])

    X         = df_clean[available].values.astype(float)
    y_freq    = df_clean["misc_freq"].values
    node_idxs = (
        df_clean["node_idx"].values.astype(int)
        if "node_idx" in df_clean.columns
        else np.arange(len(y_freq))
    )

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("ridge",  Ridge(alpha=1.0)),
    ])
    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    shap_vals = np.zeros((len(y_freq), len(available)))
    base_vals = np.zeros(len(y_freq))

    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X)):
        X_tr, X_te = X[train_idx], X[test_idx]
        pipe.fit(X_tr, y_freq[train_idx])
        scaler      = pipe.named_steps["scaler"]
        ridge_model = pipe.named_steps["ridge"]

        X_tr_sc = scaler.transform(X_tr)
        X_te_sc = scaler.transform(X_te)

        explainer = shap.LinearExplainer(
            ridge_model,
            masker=shap.maskers.Independent(X_tr_sc),
        )
        sv = explainer.shap_values(X_te_sc)
        shap_vals[test_idx] = sv
        base_vals[test_idx] = explainer.expected_value
        log.info("  SHAP fold %d/%d done", fold_idx + 1, cv.n_splits)

    display_names = [_FEATURE_DISPLAY_NAMES.get(c, c) for c in available]
    # No negation: positive SHAP → higher misc_freq → more misclassification
    return shap_vals, X, display_names, base_vals, node_idxs


def _plot_shap_beeswarm(shap_values, X_raw, feature_names, dataset, model_name, save_dir, show,
                        target_label="log-odds of misclassification", suffix=""):
    import shap
    import matplotlib.pyplot as plt

    n_features = len(feature_names)
    fig_h = max(5.0, n_features * 0.38 + 1.5)

    shap.summary_plot(
        shap_values,
        X_raw,
        feature_names=feature_names,
        plot_type="dot",
        show=False,
        plot_size=(10, fig_h),
    )
    fig = plt.gcf()
    fig.axes[0].set_xlabel(f"SHAP value  (impact on {target_label})")
    fig.suptitle(
        f"{dataset} · {model_name} — SHAP feature importance\n"
        "(positive = increases P(misclassified), OOF across 5 folds)",
        fontsize=9,
        y=1.01,
    )
    fig.tight_layout()

    if save_dir:
        sub  = os.path.join(save_dir, "shap")
        os.makedirs(sub, exist_ok=True)
        path = os.path.join(sub, f"{dataset}_{model_name}_shap_beeswarm{suffix}.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        log.info("Saved → %s", path)

    if show:
        plt.show()
    else:
        plt.close(fig)


def _plot_shap_bar(shap_values, feature_names, dataset, model_name, save_dir, show,
                   target_label="log-odds of misclassification", suffix=""):
    import shap
    import matplotlib.pyplot as plt

    n_features = len(feature_names)
    fig_h = max(5.0, n_features * 0.38 + 1.5)

    shap.summary_plot(
        shap_values,
        feature_names=feature_names,
        plot_type="bar",
        show=False,
        plot_size=(8, fig_h),
    )
    fig = plt.gcf()
    fig.axes[0].set_xlabel(f"mean |SHAP value|  (mean absolute impact on {target_label})")
    fig.suptitle(
        f"{dataset} · {model_name} — SHAP global feature importance (mean |SHAP|)\n"
        "OOF across 5 folds",
        fontsize=9,
        y=1.01,
    )
    fig.tight_layout()

    if save_dir:
        sub  = os.path.join(save_dir, "shap")
        os.makedirs(sub, exist_ok=True)
        path = os.path.join(sub, f"{dataset}_{model_name}_shap_bar{suffix}.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        log.info("Saved → %s", path)

    if show:
        plt.show()
    else:
        plt.close(fig)


def _plot_shap_heatmap(shap_values, X_raw, base_values, feature_names,
                       dataset, model_name, save_dir, show):
    import shap
    import matplotlib.pyplot as plt

    exp = shap.Explanation(
        values        = shap_values,
        base_values   = base_values,
        data          = X_raw,
        feature_names = feature_names,
    )
    fig_h = max(6.0, len(feature_names) * 0.38 + 2.0)
    plt.figure(figsize=(12, fig_h))
    shap.plots.heatmap(exp, show=False)
    fig = plt.gcf()
    fig.suptitle(
        f"{dataset} · {model_name} — SHAP heatmap (all test nodes)\n"
        "rows = nodes sorted by predicted P(misclassified); "
        "positive SHAP = increases P(misclassified)",
        fontsize=9,
        y=1.01,
    )
    fig.tight_layout()

    if save_dir:
        sub  = os.path.join(save_dir, "shap")
        os.makedirs(sub, exist_ok=True)
        path = os.path.join(sub, f"{dataset}_{model_name}_shap_heatmap.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        log.info("Saved → %s", path)

    if show:
        plt.show()
    else:
        plt.close(fig)


def _plot_shap_decision(shap_values, base_values, feature_names,
                        dataset, model_name, save_dir, show):
    import shap
    import matplotlib.pyplot as plt

    n_features = len(feature_names)
    fig_h = max(5.0, n_features * 0.38 + 1.5)
    mean_base = float(np.mean(base_values))
    plt.figure(figsize=(9, fig_h))
    shap.decision_plot(
        mean_base,
        shap_values,
        feature_names=feature_names,
        show=False,
    )
    fig = plt.gcf()
    fig.suptitle(
        f"{dataset} · {model_name} — SHAP decision plot (all test nodes)\n"
        "each line = one node; positive = increases P(misclassified)",
        fontsize=9,
        y=1.01,
    )
    fig.tight_layout()

    if save_dir:
        sub  = os.path.join(save_dir, "shap")
        os.makedirs(sub, exist_ok=True)
        path = os.path.join(sub, f"{dataset}_{model_name}_shap_decision.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        log.info("Saved → %s", path)

    if show:
        plt.show()
    else:
        plt.close(fig)


def _plot_shap_waterfall(shap_values, X_raw, base_values, node_idxs,
                         feature_names, target_node_idxs, df,
                         dataset, model_name, save_dir, show,
                         target_col="correct", suffix=""):
    import shap
    import matplotlib.pyplot as plt

    node_to_row = {int(nid): i for i, nid in enumerate(node_idxs)}

    for node_idx in target_node_idxs:
        if node_idx not in node_to_row:
            log.warning("Node %d not found in SHAP rows (not a test node or dropped due to NaN) — skipping", node_idx)
            continue

        row = node_to_row[node_idx]
        exp = shap.Explanation(
            values      = shap_values[row],
            base_values = base_values[row],
            data        = X_raw[row],
            feature_names = feature_names,
        )

        meta_row = df[df["node_idx"] == node_idx]
        if not meta_row.empty:
            deg = int(meta_row["degree"].iloc[0])
            pur = meta_row["purity_1hop"].iloc[0]
            if target_col == "misc_freq":
                mf       = float(meta_row["misc_freq"].iloc[0])
                subtitle = f"degree={deg}  purity_1hop={pur:.2f}  misc_freq={mf:.2f}"
            else:
                correct  = int(meta_row["correct"].iloc[0])
                status   = "correct" if correct else "misclassified"
                subtitle = f"degree={deg}  purity_1hop={pur:.2f}  {status}"
        else:
            subtitle = ""

        plt.figure(figsize=(9, max(4.0, len(feature_names) * 0.38 + 1.5)))
        shap.plots.waterfall(exp, show=False)
        fig = plt.gcf()
        fig.axes[0].set_title(
            f"{dataset} · {model_name} — node {node_idx}\n"
            f"{subtitle}\n"
            "(positive SHAP = increases P(misclassified))",
            fontsize=9,
        )
        fig.tight_layout()

        if save_dir:
            sub  = os.path.join(save_dir, "shap")
            os.makedirs(sub, exist_ok=True)
            path = os.path.join(sub, f"{dataset}_{model_name}_shap_waterfall_node{node_idx}{suffix}.png")
            fig.savefig(path, dpi=150, bbox_inches="tight")
            log.info("Saved → %s", path)

        if show:
            plt.show()
        else:
            plt.close(fig)


# ── ROC / PR curve plot ────────────────────────────────────────────────────────

def _plot_roc_pr(curves, baseline_rate, dataset, model_name, save_dir, show):
    import matplotlib.pyplot as plt
    from sklearn.metrics import (
        average_precision_score,
        precision_recall_curve,
        roc_auc_score,
        roc_curve,
    )

    _COLORS = {
        "Degree only":   "#9E9E9E",
        "Purity only":   "#F57C00",
        "Full features": "#1565C0",
    }

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    for label, (p_misc, y_misc) in curves.items():
        color = _COLORS.get(label, "black")
        auroc  = roc_auc_score(y_misc, p_misc)
        pr_auc = average_precision_score(y_misc, p_misc)

        fpr, tpr, _   = roc_curve(y_misc, p_misc)
        prec, rec, _  = precision_recall_curve(y_misc, p_misc)

        if label == "Purity only":
            print(f"\n[{label}] PR curve — first 5 points (rec, prec):")
            for i in range(min(5, len(rec))):
                print(f"  [{i}]  rec={rec[i]:.4f}  prec={prec[i]:.4f}")
            print(f"  Last : rec={rec[-1]:.4f}  prec={prec[-1]:.4f}")
            print(f"  Total points (incl. sklearn anchor): {len(rec)}")

        axes[0].plot(fpr, tpr, color=color, lw=1.8,
                     label=f"{label}  (AUROC {auroc:.3f})")
        axes[1].plot(rec, prec, color=color, lw=1.8,
                     label=f"{label}  (PR-AUC {pr_auc:.3f})")

    axes[0].plot([0, 1], [0, 1], "k--", lw=0.8, label="Random (0.500)")
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].set_title("ROC curve")
    axes[0].legend(loc="lower right", fontsize=8)
    axes[0].set_xlim(0, 1); axes[0].set_ylim(0, 1)

    axes[1].axhline(baseline_rate, color="k", ls="--", lw=0.8,
                    label=f"Random ({baseline_rate:.3f})")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title("Precision-Recall curve")
    axes[1].legend(loc="upper right", fontsize=8)
    axes[1].set_xlim(0, 1); axes[1].set_ylim(0, 1)

    fig.suptitle(f"{dataset} · {model_name} — baseline comparison", fontsize=10)
    fig.tight_layout()

    if save_dir:
        sub  = os.path.join(save_dir, "roc_pr_curves")
        os.makedirs(sub, exist_ok=True)
        path = os.path.join(sub, f"{dataset}_{model_name}_roc_pr.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        log.info("Saved → %s", path)

    if show:
        plt.show()
    else:
        plt.close(fig)


# ── main orchestration ─────────────────────────────────────────────────────────

def run(cfg, checkpoint_path, device, save_dir, skip_influence, skip_embeddings=False,
        feature_cols=None, univariate_auroc=False, plot_roc=False, show=False,
        compute_shap=False, shap_nodes=None, feature_selection=False):
    cache_dir = cfg.get("dataset_cache_dir", "dataset_cache")
    data = load_or_create_split(
        cfg["dataset"], cfg.get("split", "random"), cfg.get("seed", 42), cache_dir,
    )

    k_hops = cfg["model"]["num_layers"] - 1
    seed   = cfg.get("seed", 42)
    log.info("Dataset=%s  model=%s  k_hops=%d  seed=%d",
             cfg["dataset"]["name"], cfg["model"]["name"], k_hops, seed)

    N         = data.num_nodes
    y         = data.y.cpu()
    all_deg   = graph_degree(data.edge_index[1], N).cpu()
    test_mask = data.test_mask.cpu()
    test_idx  = test_mask.nonzero(as_tuple=False).view(-1).tolist()
    train_set = set(data.train_mask.cpu().nonzero(as_tuple=False).view(-1).tolist())

    log.info("Computing neighbourhood purity …")
    purity_1 = get_node_purity(data, k=1, node_mask=test_mask)
    purity_2 = get_node_purity(data, k=2, node_mask=test_mask)

    log.info("Computing min-SPL to training nodes …")
    dist_any, dist_same = compute_distances_to_train(data)

    log.info("Computing average SPL to training nodes …")
    avg_spl      = get_avg_spl_to_train(data)
    avg_spl_same = get_avg_spl_to_same_class_train(data)

    log.info("Computing closeness centrality …")
    closeness = get_closeness_centrality(data).cpu()
    log.info("Computing eigenvector centrality …")
    eigenvec  = get_eigenvector_centrality(data).cpu()

    data = data.to(device)

    if checkpoint_path:
        pred, model = load_from_checkpoint(cfg, data, device, checkpoint_path)
    else:
        log.info("No checkpoint — training from scratch (seed=%d)", seed)
        set_seed(seed)
        pred, model = train_model(data, cfg, device)

    pred = pred.cpu()

    if skip_embeddings:
        embeddings = None
    elif hasattr(model, "get_intermediate"):
        log.info("Computing penultimate GCN embeddings …")
        embeddings = model.get_intermediate(data.x, data.edge_index, layer=k_hops).cpu()
    else:
        log.warning("Model has no get_intermediate() — falling back to output logits for embeddings")
        with torch.no_grad():
            embeddings = model(data.x, data.edge_index).cpu()

    log.info("Building per-node feature rows %s…",
             "(skipping Jacobian influence)" if skip_influence else
             "(with Jacobian influence — this may take several minutes)")
    rows = _build_rows(
        data=data, model=model, pred=pred, k_hops=k_hops, device=device,
        train_set=train_set, y=y, all_deg=all_deg, test_idx=test_idx,
        purity_1=purity_1, purity_2=purity_2,
        dist_any=dist_any, dist_same=dist_same,
        avg_spl=avg_spl, avg_spl_same=avg_spl_same,
        closeness=closeness, eigenvec=eigenvec,
        embeddings=embeddings,
        skip_influence=skip_influence,
    )

    df = pd.DataFrame(rows)
    log.info("Feature table: %d rows × %d columns", len(df), len(df.columns))
    log.info("Misclassified: %d / %d (%.1f%%)",
             int((df["correct"] == 0).sum()), len(df),
             100.0 * (df["correct"] == 0).mean())

    if save_dir:
        sub   = os.path.join(save_dir, "node_feature_table")
        os.makedirs(sub, exist_ok=True)
        fname = (
            f"{cfg['dataset']['name']}_{cfg['model']['name']}"
            f"_node_features_seed{seed}.csv"
        )
        path = os.path.join(sub, fname)
        df.to_csv(path, index=False)
        log.info("Saved → %s", path)

    if univariate_auroc:
        _univariate_auroc(df)

    _, _, _, _, p_misc, y_misc = _run_logistic_regression(df, feature_cols=feature_cols)

    if plot_roc:
        log.info("Running baseline suite for ROC/PR plot …")
        curves = {}
        for label, cols in [
            ("Degree only",   ["degree"]),
            ("Purity only",   ["purity_1hop", "purity_2hop"]),
            ("Full features", None),
        ]:
            _, _, _, _, pm, ym = _run_logistic_regression(df, feature_cols=cols)
            curves[label] = (pm, ym)
        baseline_rate = y_misc.mean()
        _plot_roc_pr(curves, baseline_rate,
                     cfg["dataset"]["name"], cfg["model"]["name"],
                     save_dir, show)

    if feature_selection:
        log.info("Running feature subset comparison (single-run) …")
        _run_subset_comparison(df, is_multi_run=False, save_dir=save_dir, show=show,
                               dataset=cfg["dataset"]["name"], model_name=cfg["model"]["name"])

    if compute_shap or shap_nodes:
        log.info("Computing SHAP values (5-fold OOF, LinearExplainer) …")
        shap_values, X_raw, feat_names, base_values, shap_node_idxs = \
            _compute_shap_values(df, feature_cols=feature_cols)
        if compute_shap:
            _plot_shap_beeswarm(
                shap_values, X_raw, feat_names,
                cfg["dataset"]["name"], cfg["model"]["name"],
                save_dir, show,
            )
            _plot_shap_bar(
                shap_values, feat_names,
                cfg["dataset"]["name"], cfg["model"]["name"],
                save_dir, show,
            )

        if shap_nodes:
            _plot_shap_waterfall(
                shap_values, X_raw, base_values, shap_node_idxs,
                feat_names, shap_nodes, df,
                cfg["dataset"]["name"], cfg["model"]["name"],
                save_dir, show,
            )

    return df


def run_multi(cfg, device, save_dir, skip_influence, skip_embeddings=False,
              feature_cols=None, compute_shap=False, shap_nodes=None, show=False,
              feature_selection=False):
    """Multi-run pipeline: average influence across N runs and predict misc_freq.

    Loads all num_runs checkpoints, computes per-run Jacobian influence and
    embedding features, averages them, and trains a Ridge regression to predict
    misclassification frequency (fraction of runs a node is misclassified).
    """
    ctx    = _prepare_topology(cfg, device, "Multi-run mode")
    n_runs = ctx["n_runs"]

    # ── load all runs and compute per-run features ────────────────────────────
    log.info("Loading %d run checkpoints …", n_runs)
    runs = _load_all_runs(cfg, ctx["data"], n_runs, device)
    return _run_multi_after_topology(
        cfg, device, save_dir, skip_influence, skip_embeddings,
        feature_cols, compute_shap, shap_nodes, show, feature_selection,
        ctx, runs,
    )


def _prepare_topology(cfg, device, mode_label):
    """Load the split and compute run-independent topology features once.

    The topology features (degree, purity, training-proximity, centrality, raw
    1-hop cosine similarity) do not depend on which run's checkpoint is loaded,
    so they are computed a single time and reused across all runs. Shared by
    `run_multi` and `run_subset_across_runs`.

    Returns a dict with: ``data`` (moved to `device`), ``topo_df``, ``y`` (cpu),
    ``test_idx``, ``train_set``, ``k_hops``, ``n_runs``.
    """
    n_runs    = cfg.get("num_runs", 1)
    cache_dir = cfg.get("dataset_cache_dir", "dataset_cache")
    data = load_or_create_split(
        cfg["dataset"], cfg.get("split", "random"), cfg.get("seed", 42), cache_dir,
    )

    k_hops = cfg["model"]["num_layers"] - 1
    log.info("%s: n_runs=%d  dataset=%s  model=%s  k_hops=%d",
             mode_label, n_runs, cfg["dataset"]["name"], cfg["model"]["name"], k_hops)

    N         = data.num_nodes
    y         = data.y.cpu()
    all_deg   = graph_degree(data.edge_index[1], N).cpu()
    test_mask = data.test_mask.cpu()
    test_idx  = test_mask.nonzero(as_tuple=False).view(-1).tolist()
    train_set = set(data.train_mask.cpu().nonzero(as_tuple=False).view(-1).tolist())

    # ── topology-fixed features (computed once, same across all runs) ─────────
    log.info("Computing topology-fixed features …")
    purity_1     = get_node_purity(data, k=1, node_mask=test_mask)
    purity_2     = get_node_purity(data, k=2, node_mask=test_mask)
    dist_any, dist_same = compute_distances_to_train(data)
    avg_spl      = get_avg_spl_to_train(data)
    avg_spl_same = get_avg_spl_to_same_class_train(data)
    closeness    = get_closeness_centrality(data).cpu()
    eigenvec     = get_eigenvector_centrality(data).cpu()

    data = data.to(device)

    INF        = N + 1
    edge_index = data.edge_index.cpu()
    x_feat     = data.x.cpu()
    adj        = _build_adj_list(edge_index, N)

    topo_rows = []
    for pos, node_x in enumerate(test_idx):
        true_lbl = int(y[node_x].item())
        S1       = list(adj[node_x])
        subsets2 = k_hop_subsets_exact(node_x, 2, edge_index, N, "cpu")
        S2       = subsets2[2].tolist() if len(subsets2) > 2 else []

        st1 = _hop_ring_stats(S1, train_set, y, true_lbl)
        st2 = _hop_ring_stats(S2, train_set, y, true_lbl)

        if S1:
            focal   = F.normalize(x_feat[node_x].unsqueeze(0), dim=1)
            nbr_mat = F.normalize(x_feat[S1], dim=1)
            cos_sim = float((focal * nbr_mat).sum(dim=1).mean().item())
        else:
            cos_sim = float("nan")

        min_d  = int(dist_any[pos].item())
        min_ds = int(dist_same[pos].item())

        topo_rows.append({
            "node_idx":                     node_x,
            "degree":                       int(all_deg[node_x].item()),
            "min_dist_to_train":            min_d  if min_d  < INF else float("nan"),
            "min_dist_to_same_class_train": min_ds if min_ds < INF else float("nan"),
            "avg_spl_to_train":             float(avg_spl[node_x].item()),
            "avg_spl_to_same_class_train":  float(avg_spl_same[node_x].item()),
            "purity_1hop":                  float(purity_1[pos].item()),
            "purity_2hop":                  float(purity_2[pos].item()),
            "n_same_train_1hop":            st1["n_same"],
            "n_diff_train_1hop":            st1["n_diff"],
            "n_same_train_2hop":            st2["n_same"],
            "n_diff_train_2hop":            st2["n_diff"],
            "same_train_ratio_1hop":        st1["ratio_same"],
            "diff_train_ratio_1hop":        st1["ratio_diff"],
            "same_train_ratio_2hop":        st2["ratio_same"],
            "diff_train_ratio_2hop":        st2["ratio_diff"],
            "mean_cosine_sim_1hop":         cos_sim,
            "closeness_centrality":         float(closeness[node_x].item()),
            "eigenvector_centrality":       float(eigenvec[node_x].item()),
        })
        if (pos + 1) % 100 == 0 or (pos + 1) == len(test_idx):
            log.info("  topology: %d / %d test nodes", pos + 1, len(test_idx))

    topo_df = pd.DataFrame(topo_rows)

    return {
        "data": data, "topo_df": topo_df, "y": y, "test_idx": test_idx,
        "train_set": train_set, "k_hops": k_hops, "n_runs": n_runs,
    }


def _run_multi_after_topology(
    cfg, device, save_dir, skip_influence, skip_embeddings,
    feature_cols, compute_shap, shap_nodes, show, feature_selection,
    ctx, runs,
):
    """Aggregate per-run features into run-averaged columns, fit the Ridge
    regression against misclassification frequency, and emit the multi-run
    outputs (CSV, subset comparison, SHAP). Split out of `run_multi` so the
    topology preparation can be shared with `run_subset_across_runs`.
    """
    data      = ctx["data"]
    topo_df   = ctx["topo_df"]
    y         = ctx["y"]
    test_idx  = ctx["test_idx"]
    train_set = ctx["train_set"]
    k_hops    = ctx["k_hops"]
    n_runs    = ctx["n_runs"]

    run_features_list = []
    for run_id, (pred_cpu, model) in enumerate(runs, 1):
        log.info("Computing run %d/%d features %s…",
                 run_id, n_runs,
                 "(skipping influence)" if skip_influence else "(with Jacobian influence)")
        feats = _compute_run_dependent_features(
            data=data, model=model, pred=pred_cpu, k_hops=k_hops, device=device,
            train_set=train_set, y=y, test_idx=test_idx,
            skip_influence=skip_influence, skip_embeddings=skip_embeddings,
        )
        run_features_list.append(feats)

    # ── aggregate across runs ─────────────────────────────────────────────────
    log.info("Aggregating features across %d runs …", n_runs)
    df = _aggregate_multi_run(topo_df, run_features_list)

    log.info("Multi-run feature table: %d rows × %d columns", len(df), len(df.columns))
    log.info("misc_freq distribution: mean=%.3f  std=%.3f  min=%.2f  max=%.2f",
             df["misc_freq"].mean(), df["misc_freq"].std(),
             df["misc_freq"].min(), df["misc_freq"].max())
    log.info("Always misclassified (freq=1): %d   always correct (freq=0): %d",
             int((df["misc_freq"] == 1.0).sum()),
             int((df["misc_freq"] == 0.0).sum()))

    # ── save CSV ──────────────────────────────────────────────────────────────
    if save_dir:
        sub   = os.path.join(save_dir, "node_feature_table")
        os.makedirs(sub, exist_ok=True)
        fname = (
            f"{cfg['dataset']['name']}_{cfg['model']['name']}"
            f"_node_features_multi.csv"
        )
        path = os.path.join(sub, fname)
        df.to_csv(path, index=False)
        log.info("Saved → %s", path)

    # ── Ridge regression ──────────────────────────────────────────────────────
    _run_ridge_regression(df, feature_cols=feature_cols)

    # ── recursive feature selection ───────────────────────────────────────────
    if feature_selection:
        log.info("Running feature subset comparison (multi-run) …")
        _run_subset_comparison(df, is_multi_run=True, save_dir=save_dir, show=show,
                               dataset=cfg["dataset"]["name"], model_name=cfg["model"]["name"])

    # ── SHAP ─────────────────────────────────────────────────────────────────
    if compute_shap or shap_nodes:
        log.info("Computing SHAP values (Ridge, 5-fold OOF, LinearExplainer) …")
        shap_values, X_raw, feat_names, base_values, shap_node_idxs = \
            _compute_shap_values_multi(df, feature_cols=feature_cols)

        if compute_shap:
            _plot_shap_beeswarm(
                shap_values, X_raw, feat_names,
                cfg["dataset"]["name"], cfg["model"]["name"],
                save_dir, show,
                target_label="misclassification frequency",
                suffix="_multi",
            )
            _plot_shap_bar(
                shap_values, feat_names,
                cfg["dataset"]["name"], cfg["model"]["name"],
                save_dir, show,
                target_label="misclassification frequency",
                suffix="_multi",
            )

        if shap_nodes:
            _plot_shap_waterfall(
                shap_values, X_raw, base_values, shap_node_idxs,
                feat_names, shap_nodes, df,
                cfg["dataset"]["name"], cfg["model"]["name"],
                save_dir, show,
                target_col="misc_freq",
                suffix="_multi",
            )

    return df


def _aggregate_subset_across_runs(per_run_scores, n_runs):
    """Combine per-run ``{label: (pr_auc, n_groups)}`` dicts into a mean ± std table.

    Returns a DataFrame sorted ascending by ``pr_auc_mean`` with columns
    ``label, n_groups, pr_auc_mean, pr_auc_std, n_runs_present,
    pr_auc_run1 … pr_auc_runN``. Per-run columns are aligned to the run index
    (NaN if a subset was absent in that run); mean/std are over present values.
    """
    if not per_run_scores:
        return pd.DataFrame()

    # Preserve first-seen label order across runs.
    labels: list[str] = []
    for d in per_run_scores:
        for lab in d:
            if lab not in labels:
                labels.append(lab)

    rows = []
    for lab in labels:
        present  = [(ri, d[lab][0]) for ri, d in enumerate(per_run_scores, 1) if lab in d]
        vals     = np.asarray([v for _, v in present], dtype=float)
        n_groups = next(d[lab][1] for d in per_run_scores if lab in d)
        row = {
            "label":          lab,
            "n_groups":       n_groups,
            "pr_auc_mean":    float(vals.mean()),
            "pr_auc_std":     float(vals.std()),
            "n_runs_present": len(present),
        }
        for ri in range(1, n_runs + 1):
            row[f"pr_auc_run{ri}"] = next((v for r, v in present if r == ri), float("nan"))
        rows.append(row)

    return (
        pd.DataFrame(rows)
        .sort_values("pr_auc_mean", ascending=True)
        .reset_index(drop=True)
    )


def _plot_subset_across_runs(df_res, save_dir, show, dataset, model_name, n_runs):
    """Horizontal bar chart of mean PR-AUC per subset across runs, ±1 std error bars.

    Reuses the single-run subset-comparison styling (single/combination/full
    colour scheme + degree reference line). Error whiskers are clamped so the
    lower end never crosses 0.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    _C_SINGLE = "#4878CF"   # individual feature group
    _C_COMBO  = "#EE854A"   # multi-group combination
    _C_FULL   = "#333333"   # full model

    def _bar_color(row):
        if row["label"] == "Full model":
            return _C_FULL
        return _C_SINGLE if row["n_groups"] == 1 else _C_COMBO

    colors = [_bar_color(row) for _, row in df_res.iterrows()]
    means  = df_res["pr_auc_mean"].to_numpy()
    stds   = df_res["pr_auc_std"].to_numpy()

    # Asymmetric error bars clamped so the lower whisker never crosses 0.
    lower = np.minimum(stds, means)
    xerr  = np.vstack([lower, stds])

    fig, ax = plt.subplots(figsize=(7, 0.55 * len(df_res) + 1.0))
    bars = ax.barh(df_res["label"], means, color=colors, height=0.6, zorder=2,
                   xerr=xerr, error_kw=dict(ecolor="#555555", elinewidth=1.0, capsize=3))

    x_max = float((means + stds).max())
    x_min = float((means - stds).min())
    span  = (x_max - x_min) or 1.0
    for bar, m, s in zip(bars, means, stds):
        ax.text(bar.get_width() + s + span * 0.015,
                bar.get_y() + bar.get_height() / 2,
                f"{m:.3f}±{s:.3f}", va="center", ha="left", fontsize=7.0)

    # Degree reference line (mean PR-AUC of the degree-only subset)
    degree_rows = df_res[df_res["label"] == "Degree"]
    if not degree_rows.empty:
        deg_score = float(degree_rows["pr_auc_mean"].iloc[0])
        ax.axvline(deg_score, color="#D65F5F", ls="--", lw=1.2, zorder=3)
        ax.text(deg_score, len(df_res) - 0.1, f" degree\n ({deg_score:.3f})",
                color="#D65F5F", fontsize=7.5, va="top", ha="left")

    legend_handles = [
        Patch(facecolor=_C_SINGLE, label="Single group"),
        Patch(facecolor=_C_COMBO,  label="Combination"),
        Patch(facecolor=_C_FULL,   label="Full model"),
    ]
    ax.legend(handles=legend_handles, fontsize=8, loc="lower right")

    ax.set_xlabel("PR-AUC (mean ± std across runs)")
    ax.set_xlim(x_min - span * 0.05, x_max + span * 0.22)
    ax.set_title(f"{dataset} · {model_name} — feature subset comparison "
                 f"(mean ± std, {n_runs} runs)", fontsize=10)
    ax.grid(axis="x", lw=0.5, alpha=0.4, zorder=1)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()

    if save_dir:
        sub  = os.path.join(save_dir, "node_feature_table")
        os.makedirs(sub, exist_ok=True)
        path = os.path.join(sub, "subset_comparison_across_runs.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        log.info("Saved → %s", path)

    if show:
        plt.show()
    else:
        plt.close(fig)


def run_subset_across_runs(cfg, device, save_dir, skip_influence,
                           skip_embeddings=False, show=False):
    """Feature-subset PR-AUC comparison computed for every run, reported as mean ± std.

    Unlike `run_multi` (which averages features across runs and predicts
    misclassification *frequency* via Ridge/Spearman), this reconstructs a
    complete single-run feature table for each checkpoint, runs the standard
    PR-AUC subset ablation (5-fold CV logistic regression) on each, and
    aggregates each subset's PR-AUC across runs.
    """
    ctx       = _prepare_topology(cfg, device, "Subset-across-runs mode")
    data      = ctx["data"]
    topo_df   = ctx["topo_df"]
    y         = ctx["y"]
    test_idx  = ctx["test_idx"]
    train_set = ctx["train_set"]
    k_hops    = ctx["k_hops"]
    n_runs    = ctx["n_runs"]

    log.info("Loading %d run checkpoints …", n_runs)
    runs = _load_all_runs(cfg, data, n_runs, device)

    per_run_scores = []  # list[dict]: label -> (pr_auc, n_groups)
    for run_id, (pred_cpu, model) in enumerate(runs, 1):
        log.info("Run %d/%d: computing features %s…", run_id, n_runs,
                 "(skipping influence)" if skip_influence else "(with Jacobian influence)")
        feats  = _compute_run_dependent_features(
            data=data, model=model, pred=pred_cpu, k_hops=k_hops, device=device,
            train_set=train_set, y=y, test_idx=test_idx,
            skip_influence=skip_influence, skip_embeddings=skip_embeddings,
        )
        run_df = topo_df.merge(pd.DataFrame(feats), on="node_idx")

        def _pr_auc(cols, _df=run_df):
            _, _, pr_auc, *_ = _run_logistic_regression(_df, feature_cols=cols)
            return float(pr_auc)

        rows = _eval_subsets(run_df, _FEATURE_GROUP_MAP, _pr_auc)
        per_run_scores.append({r["label"]: (r["score"], r["n_groups"]) for r in rows})
        log.info("Run %d/%d: subset PR-AUCs computed (%d subsets)", run_id, n_runs, len(rows))

    df_res = _aggregate_subset_across_runs(per_run_scores, n_runs)
    if df_res.empty:
        log.warning("No subsets could be evaluated across runs.")
        return df_res

    log.info("Subset PR-AUC across %d runs (mean ± std):", n_runs)
    for _, r in df_res.iterrows():
        log.info("  %-35s  %.4f ± %.4f", r["label"], r["pr_auc_mean"], r["pr_auc_std"])

    if save_dir:
        sub = os.path.join(save_dir, "node_feature_table")
        os.makedirs(sub, exist_ok=True)
        path = os.path.join(sub, "subset_comparison_across_runs.csv")
        df_res.to_csv(path, index=False)
        log.info("Saved subset comparison (across runs) CSV → %s", path)

    _plot_subset_across_runs(df_res, save_dir, show,
                             cfg["dataset"]["name"], cfg["model"]["name"], n_runs)
    return df_res


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Build a per-test-node feature table and train a logistic regression "
            "to predict GCN misclassification."
        )
    )
    parser.add_argument("--config",    default="config.yaml")
    parser.add_argument("--device",    default=None,
                        help="Device override, e.g. cuda:0 or cpu.")
    parser.add_argument("--save-dir",  default=None,
                        help="Directory to save outputs. Ignored when --run is used "
                             "(outputs always go to the run's exec directory).")
    parser.add_argument("--no-influence", action="store_true",
                        help="Skip Jacobian-L1 influence computation (much faster).")
    parser.add_argument("--no-embeddings", action="store_true",
                        help="Skip penultimate embedding similarity computation.")
    parser.add_argument("--features", default=None,
                        help="Comma-separated list of features to use in the LR "
                             "(e.g. 'degree' or 'purity_1hop,purity_2hop'). "
                             "Defaults to all available features.")
    parser.add_argument("--univariate-auroc", action="store_true",
                        help="Report univariate AUROC for every feature (no LR assumptions).")
    parser.add_argument("--plot-roc", action="store_true",
                        help="Plot ROC and PR curves for degree-only, purity-only, and full LR.")
    parser.add_argument("--show", action="store_true",
                        help="Display plots interactively (requires a display).")
    parser.add_argument("--shap", action="store_true",
                        help="Compute OOF SHAP values and save a beeswarm summary plot.")
    parser.add_argument("--shap-nodes", default=None,
                        help="Comma-separated graph node indices for per-node SHAP waterfall plots "
                             "(e.g. '1362,42'). Implies SHAP computation.")
    parser.add_argument("--feature-selection", action="store_true",
                        help="Compare named feature subsets by PR-AUC (single-run) "
                             "or Spearman r (multi-run) and save a bar chart.")

    ckpt_group = parser.add_mutually_exclusive_group()
    ckpt_group.add_argument("--checkpoint", default=None,
                            help="Path to a saved model checkpoint.")
    ckpt_group.add_argument("--run", type=int, default=None,
                            help="Run index (1-based) whose checkpoint to load.")
    ckpt_group.add_argument("--multi-run", action="store_true",
                            help="Load all num_runs checkpoints, average influence features "
                                 "across runs, and predict misclassification frequency "
                                 "(Ridge regression). Mutually exclusive with --run/--checkpoint.")
    ckpt_group.add_argument("--subset-across-runs", action="store_true",
                            help="Load all num_runs checkpoints, compute the feature-subset "
                                 "PR-AUC ablation per run, and report each subset's PR-AUC as "
                                 "mean ± std across runs. Mutually exclusive with --run/--checkpoint.")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )

    cfg = load_cfg(args.config)

    device = torch.device(
        args.device if args.device
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    log.info("Device: %s", device)

    checkpoint_path = args.checkpoint
    if checkpoint_path is None and args.run is not None:
        checkpoint_path = _resolve_run_checkpoint(cfg, args.run)
        if checkpoint_path is None:
            parser.error(
                f"--run {args.run} could not locate a matching checkpoint under "
                f"{cfg.get('results_dir', './results')}/."
            )

    if args.run is not None and checkpoint_path is not None:
        save_dir = os.path.dirname(os.path.dirname(checkpoint_path))
        log.info("save_dir set to run directory: %s", save_dir)
    elif args.multi_run or args.subset_across_runs:
        if args.save_dir:
            save_dir = args.save_dir
        else:
            run1_ckpt = _resolve_run_checkpoint(cfg, 1)
            save_dir  = os.path.dirname(os.path.dirname(run1_ckpt)) if run1_ckpt else None
        if save_dir:
            log.info("save_dir set to exec directory: %s", save_dir)
    else:
        save_dir = args.save_dir
        if save_dir is None and checkpoint_path is not None:
            save_dir = os.path.dirname(os.path.dirname(checkpoint_path))
            log.info("save_dir inferred from checkpoint: %s", save_dir)

    feature_cols = [f.strip() for f in args.features.split(",")] if args.features else None
    shap_nodes   = [int(n.strip()) for n in args.shap_nodes.split(",")] if args.shap_nodes else None

    if args.subset_across_runs:
        run_subset_across_runs(cfg, device, save_dir, args.no_influence,
                               args.no_embeddings, show=args.show)
    elif args.multi_run:
        run_multi(cfg, device, save_dir, args.no_influence, args.no_embeddings,
                  feature_cols=feature_cols, compute_shap=args.shap,
                  shap_nodes=shap_nodes, show=args.show,
                  feature_selection=args.feature_selection)
    else:
        run(cfg, checkpoint_path, device, save_dir, args.no_influence, args.no_embeddings,
            feature_cols=feature_cols, univariate_auroc=args.univariate_auroc,
            plot_roc=args.plot_roc, show=args.show, compute_shap=args.shap,
            shap_nodes=shap_nodes, feature_selection=args.feature_selection)


if __name__ == "__main__":
    main()
