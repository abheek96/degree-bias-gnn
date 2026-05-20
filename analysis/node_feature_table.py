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
  correct                       — 1 = correctly classified, 0 = misclassified (target)

The influence columns are omitted (set to NaN) when ``--no-influence`` is passed.
Fraction features (same/diff_train_infl_frac_*hop) use total I_x.sum() as the denominator.

Model source (mutually exclusive)
----------------------------------
  --checkpoint PATH   load a saved state_dict
  --run N             resolve results/{exec}/checkpoints/run{N:02d}_*.pt
  (neither)           retrain from scratch with config seed

Usage
-----
  uv run analysis/node_feature_table.py --run 1
  uv run analysis/node_feature_table.py --run 1 --no-influence
  uv run analysis/node_feature_table.py \\
      --checkpoint results/.../checkpoints/run01_seed42.pt --save-dir ./output
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
}


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
            "correct":                      int(int(pred[node_x].item()) == true_lbl),
        })

        if (pos + 1) % 100 == 0 or (pos + 1) == len(test_idx):
            log.info("  processed %d / %d test nodes", pos + 1, len(test_idx))

    return rows


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


def _plot_shap_beeswarm(shap_values, X_raw, feature_names, dataset, model_name, save_dir, show):
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
    fig.axes[0].set_xlabel("SHAP value  (impact on log-odds of misclassification)")
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
        path = os.path.join(sub, f"{dataset}_{model_name}_shap_beeswarm.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        log.info("Saved → %s", path)

    if show:
        plt.show()
    else:
        plt.close(fig)


def _plot_shap_bar(shap_values, feature_names, dataset, model_name, save_dir, show):
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
    fig.axes[0].set_xlabel("mean |SHAP value|  (mean absolute impact on log-odds of misclassification)")
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
        path = os.path.join(sub, f"{dataset}_{model_name}_shap_bar.png")
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
                         dataset, model_name, save_dir, show):
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
            deg     = int(meta_row["degree"].iloc[0])
            pur     = meta_row["purity_1hop"].iloc[0]
            correct = int(meta_row["correct"].iloc[0])
            status  = "correct" if correct else "misclassified"
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
            path = os.path.join(sub, f"{dataset}_{model_name}_shap_waterfall_node{node_idx}.png")
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
        compute_shap=False, shap_nodes=None):
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

    ckpt_group = parser.add_mutually_exclusive_group()
    ckpt_group.add_argument("--checkpoint", default=None,
                            help="Path to a saved model checkpoint.")
    ckpt_group.add_argument("--run", type=int, default=None,
                            help="Run index (1-based) whose checkpoint to load.")
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
    else:
        save_dir = args.save_dir
        if save_dir is None and checkpoint_path is not None:
            save_dir = os.path.dirname(os.path.dirname(checkpoint_path))
            log.info("save_dir inferred from checkpoint: %s", save_dir)

    feature_cols = [f.strip() for f in args.features.split(",")] if args.features else None
    shap_nodes   = [int(n.strip()) for n in args.shap_nodes.split(",")] if args.shap_nodes else None
    run(cfg, checkpoint_path, device, save_dir, args.no_influence, args.no_embeddings,
        feature_cols=feature_cols, univariate_auroc=args.univariate_auroc,
        plot_roc=args.plot_roc, show=args.show, compute_shap=args.shap,
        shap_nodes=shap_nodes)


if __name__ == "__main__":
    main()
