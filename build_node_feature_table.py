"""
build_node_feature_table.py — Build a per-test-node feature table and train a
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
  correct                       — 1 = correctly classified, 0 = misclassified (target)

The influence columns are omitted (set to NaN) when ``--no-influence`` is passed.

Model source (mutually exclusive)
----------------------------------
  --checkpoint PATH   load a saved state_dict
  --run N             resolve results/{exec}/checkpoints/run{N:02d}_*.pt
  (neither)           retrain from scratch with config seed

Usage
-----
  uv run build_node_feature_table.py --run 1 --save-dir ./output
  uv run build_node_feature_table.py --run 1 --save-dir ./output --no-influence
  uv run build_node_feature_table.py \\
      --checkpoint results/.../checkpoints/run01_seed42.pt --save-dir ./output
"""

import argparse
import logging
import os
import sys
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from torch_geometric.utils import degree as graph_degree

from analyse_hop_influence import (
    _build_model,       # noqa: F401 — imported for completeness; used via load_from_checkpoint
    _deep_merge,
    _resolve_run_checkpoint,
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
]


# ── helpers ────────────────────────────────────────────────────────────────────

def _build_adj_list(edge_index_cpu, N: int) -> dict:
    adj = defaultdict(set)
    for src, dst in zip(edge_index_cpu[0].tolist(), edge_index_cpu[1].tolist()):
        adj[src].add(dst)
    return adj


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

        # ── 1-hop ring counts ────────────────────────────────────────────────
        S1   = list(adj[node_x])
        n1   = len(S1)
        n_same_train_1 = sum(
            1 for n in S1 if n in train_set and int(y[n].item()) == true_lbl
        )
        n_diff_train_1 = sum(
            1 for n in S1 if n in train_set and int(y[n].item()) != true_lbl
        )
        same_r1 = n_same_train_1 / n1 if n1 > 0 else float("nan")
        diff_r1 = n_diff_train_1 / n1 if n1 > 0 else float("nan")

        # ── 2-hop ring counts ────────────────────────────────────────────────
        subsets2 = k_hop_subsets_exact(node_x, 2, edge_index, N, device)
        S2       = subsets2[2].tolist() if len(subsets2) > 2 else []
        n2       = len(S2)
        n_same_train_2 = sum(
            1 for n in S2 if n in train_set and int(y[n].item()) == true_lbl
        )
        n_diff_train_2 = sum(
            1 for n in S2 if n in train_set and int(y[n].item()) != true_lbl
        )
        same_r2 = n_same_train_2 / n2 if n2 > 0 else float("nan")
        diff_r2 = n_diff_train_2 / n2 if n2 > 0 else float("nan")

        # ── cosine similarity (raw features, 1-hop) ──────────────────────────
        if S1:
            focal   = F.normalize(x_feat[node_x].unsqueeze(0), dim=1)  # [1, F]
            nbr_mat = F.normalize(x_feat[S1], dim=1)                   # [k, F]
            cos_sim = float((focal * nbr_mat).sum(dim=1).mean().item())
        else:
            cos_sim = float("nan")

        # ── Jacobian-L1 influence (hop-1, expensive) ─────────────────────────
        if skip_influence:
            total_same_infl = float("nan")
            total_diff_infl = float("nan")
        else:
            I_x   = influence_distribution(model, data, node_x, k_hops)
            hop_s = k_hop_subsets_exact(node_x, k_hops, edge_index, N, I_x.device)
            if len(hop_s) > 1:
                S1_i   = hop_s[1].tolist()
                same_i = [n for n in S1_i if int(y[n].item()) == true_lbl]
                diff_i = [n for n in S1_i if int(y[n].item()) != true_lbl]
                total_same_infl = float(I_x[same_i].sum().item()) if same_i else 0.0
                total_diff_infl = float(I_x[diff_i].sum().item()) if diff_i else 0.0
            else:
                total_same_infl = 0.0
                total_diff_infl = 0.0

        # ── SPL / distance ───────────────────────────────────────────────────
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
            "n_same_train_1hop":            n_same_train_1,
            "n_diff_train_1hop":            n_diff_train_1,
            "n_same_train_2hop":            n_same_train_2,
            "n_diff_train_2hop":            n_diff_train_2,
            "same_train_ratio_1hop":        same_r1,
            "diff_train_ratio_1hop":        diff_r1,
            "same_train_ratio_2hop":        same_r2,
            "diff_train_ratio_2hop":        diff_r2,
            "mean_cosine_sim_1hop":         cos_sim,
            "total_infl_same_1hop":         total_same_infl,
            "total_infl_diff_1hop":         total_diff_infl,
            "correct":                      int(int(pred[node_x].item()) == true_lbl),
        })

        if (pos + 1) % 100 == 0 or (pos + 1) == len(test_idx):
            log.info("  processed %d / %d test nodes", pos + 1, len(test_idx))

    return rows


# ── logistic regression ────────────────────────────────────────────────────────

def _run_logistic_regression(df: pd.DataFrame):
    """5-fold stratified CV logistic regression; logs AUROC, accuracy, and coefficients."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    available = [c for c in _FEATURE_COLS if c in df.columns]
    # Exclude columns that are entirely NaN (e.g. influence cols under --no-influence)
    available = [c for c in available if df[c].notna().any()]
    if skipped := set(_FEATURE_COLS) - set(available):
        log.info("Skipping all-NaN feature columns: %s", sorted(skipped))
    df_clean  = df[available + ["correct"]].dropna()
    n_dropped = len(df) - len(df_clean)
    if n_dropped:
        log.warning("Dropped %d rows with NaN features before LR fitting", n_dropped)

    X = df_clean[available].values.astype(float)
    y = df_clean["correct"].values

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
    log.info("  n_train=%d  misclassified=%d (%.1f%%)",
             len(df_clean), int((y == 0).sum()), 100.0 * (y == 0).mean())
    log.info("  AUROC:    %.4f ± %.4f", auroc.mean(), auroc.std())
    log.info("  Accuracy: %.4f ± %.4f", acc.mean(),   acc.std())
    log.info("  Coefficients (|coef| descending):")
    for _, row in coef_df.iterrows():
        log.info("    %-42s  %+.4f", row["feature"], row["coefficient"])
    log.info("────────────────────────────────────────────────────────────────────")

    return auroc, acc, coef_df


# ── main orchestration ─────────────────────────────────────────────────────────

def run(cfg, checkpoint_path, device, save_dir, skip_influence):
    cache_dir = cfg.get("dataset_cache_dir", "dataset_cache")
    data      = load_or_create_split(
        cfg["dataset"], cfg.get("split", "random"), cfg.get("seed", 42), cache_dir,
    ).to(device)
    data_cpu  = data.cpu()   # used for structural feature functions that require CPU tensors

    k_hops = cfg["model"]["num_layers"] - 1
    seed   = cfg.get("seed", 42)
    log.info("Dataset=%s  model=%s  k_hops=%d  seed=%d",
             cfg["dataset"]["name"], cfg["model"]["name"], k_hops, seed)

    if checkpoint_path:
        pred, model = load_from_checkpoint(cfg, data, device, checkpoint_path)
    else:
        log.info("No checkpoint — training from scratch (seed=%d)", seed)
        set_seed(seed)
        pred, model = train_model(data, cfg, device)

    pred      = pred.cpu()
    N         = data.num_nodes
    y         = data.y.cpu()
    all_deg   = graph_degree(data.edge_index[1], N).cpu()
    test_mask = data.test_mask.cpu()
    test_idx  = test_mask.nonzero(as_tuple=False).view(-1).tolist()
    train_set = set(data.train_mask.cpu().nonzero(as_tuple=False).view(-1).tolist())

    # ── vectorised structural features (computed once, indexed by position in test_idx) ──
    log.info("Computing neighbourhood purity …")
    purity_1 = get_node_purity(data_cpu, k=1, node_mask=test_mask)
    purity_2 = get_node_purity(data_cpu, k=2, node_mask=test_mask)

    log.info("Computing min-SPL to training nodes …")
    dist_any, dist_same = compute_distances_to_train(data_cpu)

    log.info("Computing average SPL to training nodes …")
    avg_spl      = get_avg_spl_to_train(data_cpu)       # [N]
    avg_spl_same = get_avg_spl_to_same_class_train(data_cpu)  # [N]

    # ── per-node loop ──────────────────────────────────────────────────────────
    log.info("Building per-node feature rows %s…",
             "(skipping Jacobian influence)" if skip_influence else
             "(with Jacobian influence — this may take several minutes)")
    rows = _build_rows(
        data=data, model=model, pred=pred, k_hops=k_hops, device=device,
        train_set=train_set, y=y, all_deg=all_deg, test_idx=test_idx,
        purity_1=purity_1, purity_2=purity_2,
        dist_any=dist_any, dist_same=dist_same,
        avg_spl=avg_spl, avg_spl_same=avg_spl_same,
        skip_influence=skip_influence,
    )

    df = pd.DataFrame(rows)
    log.info("Feature table: %d rows × %d columns", len(df), len(df.columns))
    log.info("Misclassified: %d / %d (%.1f%%)",
             int((df["correct"] == 0).sum()), len(df),
             100.0 * (df["correct"] == 0).mean())

    # ── save CSV ───────────────────────────────────────────────────────────────
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

    # ── logistic regression ────────────────────────────────────────────────────
    _run_logistic_regression(df)

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
                        help="Directory to save the CSV table.")
    parser.add_argument("--no-influence", action="store_true",
                        help="Skip Jacobian-L1 influence computation (much faster).")

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

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    model_cfg_path = os.path.join(
        "configs", f"{cfg['model']['name']}_{cfg['dataset']['name']}.yaml"
    )
    if os.path.exists(model_cfg_path):
        with open(model_cfg_path) as f:
            cfg = _deep_merge(cfg, yaml.safe_load(f))

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

    run(cfg, checkpoint_path, device, args.save_dir, args.no_influence)


if __name__ == "__main__":
    main()
