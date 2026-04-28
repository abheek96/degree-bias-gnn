"""
analyse_1hop_influence_by_degree.py — Aggregate 1-hop Jacobian-L1 influence
by degree group across all test nodes.

For each test node computes:
  - total_same : sum of I_x[n] over all same-class nodes at exactly hop 1
  - total_diff : sum of I_x[n] over all diff-class nodes at exactly hop 1
  - purity     : fraction of 1-hop neighbours sharing the focal node's class

Aggregates these per degree group and produces a two-panel plot:
  Top    — side-by-side boxplots of total_same (blue) and total_diff (orange)
           per degree group, with jittered scatter overlay and node-count bars.
  Bottom — purity line (median ± IQR shading) and per-degree accuracy line.

Model source
------------
  --checkpoint PATH   load a saved state_dict (no retraining)
  --run N             shorthand; resolves results/{exec}/checkpoints/run{N:02d}_*.pt
  (neither)           retrain from scratch with the config seed

Usage
-----
    uv run analyse_1hop_influence_by_degree.py --run 1 --save-dir ./output
    uv run analyse_1hop_influence_by_degree.py \\
        --checkpoint results/.../checkpoints/run01_seed42.pt --show
"""

import argparse
import logging
import sys
from collections import defaultdict

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import torch
import yaml

from torch_geometric.utils import degree as graph_degree

from analyse_hop_influence import (
    _build_model,
    _deep_merge,
    _resolve_run_checkpoint,
    load_from_checkpoint,
    train_model,
    set_seed,
)
from dataset_utils import load_or_create_split
from influence import influence_distribution, k_hop_subsets_exact
from plot_utils import (
    _ACC_COLOR,
    _BP_KWARGS,
    _PURITY_COLOR,
    _degree_axis,
    _fig_w,
    _save,
    _subdir,
    get_accuracy_deg,
)

import os

log = logging.getLogger(__name__)


# ── computation ────────────────────────────────────────────────────────────────

def _compute_all(model, data, pred, k_hops, device):
    """Compute per-test-node 1-hop influence and purity metrics.

    For each test node returns:
      total_same      — I_x sum over all same-class 1-hop nodes (lbl + unlbl)
      total_diff      — I_x sum over all diff-class 1-hop nodes (lbl + unlbl)
      infl_balance    — total_same - total_diff
      lbl_frac_balance— same_lbl_infl/total_1hop_infl - diff_lbl_infl/total_1hop_infl
      purity_1        — fraction of 1-hop ring that is same-class
      purity_2        — fraction of 2-hop ring that is same-class (structural BFS)
      purity_delta    — purity_2 - purity_1  (NaN if 2-hop ring is empty)
      correct         — whether the model predicts this node correctly

    Skips test nodes with no 1-hop neighbours.
    """
    N          = data.num_nodes
    y          = data.y.cpu()
    pred_cpu   = pred.cpu()
    all_deg    = graph_degree(data.edge_index[1], N).cpu()
    test_mask  = data.test_mask.cpu()
    train_set  = set(data.train_mask.cpu().nonzero(as_tuple=False).view(-1).tolist())
    test_idx   = test_mask.nonzero(as_tuple=False).view(-1).tolist()

    model.eval()
    records = []
    n_test  = len(test_idx)

    for i, node_x in enumerate(test_idx):
        if i % 100 == 0:
            log.info("  %d / %d test nodes processed", i, n_test)

        true_lbl = int(y[node_x].item())

        I_x         = influence_distribution(model, data, node_x, k_hops)
        # 1-hop subsets from model receptive field
        hop_subsets = k_hop_subsets_exact(
            node_x, k_hops, data.edge_index, N, I_x.device,
        )

        if len(hop_subsets) < 2 or len(hop_subsets[1]) == 0:
            continue  # no 1-hop neighbours

        S_1 = hop_subsets[1].tolist()
        same_1 = [n for n in S_1 if int(y[n].item()) == true_lbl]
        diff_1 = [n for n in S_1 if int(y[n].item()) != true_lbl]

        total_same = float(I_x[same_1].sum().item()) if same_1 else 0.0
        total_diff = float(I_x[diff_1].sum().item()) if diff_1 else 0.0

        # labelled (training) fractions of total hop-1 influence
        same_lbl_1   = [n for n in same_1 if n in train_set]
        diff_lbl_1   = [n for n in diff_1 if n in train_set]
        total_1_infl = float(I_x[S_1].sum().item())
        if total_1_infl > 0:
            lbl_frac_balance = (
                float(I_x[same_lbl_1].sum().item()) / total_1_infl
                - float(I_x[diff_lbl_1].sum().item()) / total_1_infl
            )
        else:
            lbl_frac_balance = 0.0

        purity_1 = len(same_1) / len(S_1)

        # 2-hop ring — always computed structurally (BFS, independent of k_hops)
        hop2_subsets = k_hop_subsets_exact(
            node_x, 2, data.edge_index, N, I_x.device,
        )
        if len(hop2_subsets) >= 3 and len(hop2_subsets[2]) > 0:
            S_2      = hop2_subsets[2].tolist()
            same_2   = sum(1 for n in S_2 if int(y[n].item()) == true_lbl)
            purity_2 = same_2 / len(S_2)
        else:
            purity_2 = float("nan")

        records.append({
            "node_idx":         node_x,
            "degree":           int(all_deg[node_x].item()),
            "total_same":       total_same,
            "total_diff":       total_diff,
            "infl_balance":     total_same - total_diff,
            "lbl_frac_balance": lbl_frac_balance,
            "purity_1":         purity_1,
            "purity_2":         purity_2,
            "purity_delta":     purity_2 - purity_1 if not np.isnan(purity_2) else float("nan"),
            "correct":          int(pred_cpu[node_x].item()) == true_lbl,
        })

    log.info("  Done — %d / %d test nodes had 1-hop neighbours", len(records), n_test)
    return records


def _aggregate_by_degree(records):
    """Group records by degree."""
    by_deg = defaultdict(lambda: {
        "total_same": [], "total_diff": [],
        "infl_balance": [], "lbl_frac_balance": [],
        "purity_1": [], "purity_2": [], "purity_delta": [], "correct": [],
    })
    for r in records:
        d = r["degree"]
        by_deg[d]["total_same"].append(r["total_same"])
        by_deg[d]["total_diff"].append(r["total_diff"])
        by_deg[d]["infl_balance"].append(r["infl_balance"])
        by_deg[d]["lbl_frac_balance"].append(r["lbl_frac_balance"])
        by_deg[d]["purity_1"].append(r["purity_1"])
        if not np.isnan(r["purity_2"]):
            by_deg[d]["purity_2"].append(r["purity_2"])
        if not np.isnan(r["purity_delta"]):
            by_deg[d]["purity_delta"].append(r["purity_delta"])
        by_deg[d]["correct"].append(float(r["correct"]))
    return by_deg


# ── plotting ───────────────────────────────────────────────────────────────────

def _plot(by_deg, cfg, seed, save_dir, show):
    all_degrees = sorted(by_deg.keys())
    n_deg       = len(all_degrees)
    pos         = np.arange(n_deg, dtype=float)
    w           = 0.35

    dataset = cfg["dataset"]["name"]
    model   = cfg["model"]["name"]

    fig, ax = plt.subplots(figsize=(_fig_w(n_deg), 5))

    # ── left y-axis: side-by-side boxplots ────────────────────────────────────

    same_data = [by_deg[d]["total_same"] for d in all_degrees]
    diff_data = [by_deg[d]["total_diff"] for d in all_degrees]

    bp_kwargs = {**_BP_KWARGS}
    bp_kwargs["flierprops"] = dict(marker="", markersize=0)  # no fliers

    for data_list, offset, color, label in [
        (same_data, -w / 2, "#1565C0", "Same-class infl. (hop 1)"),
        (diff_data, +w / 2, "#E65100", "Diff-class infl. (hop 1)"),
    ]:
        bp = ax.boxplot(
            data_list,
            positions=pos + offset,
            widths=w * 0.85,
            whis=(0, 100),
            **bp_kwargs,
        )
        for patch in bp["boxes"]:
            patch.set_facecolor(color)
            patch.set_alpha(0.72)
        for component in ("whiskers", "caps", "medians"):
            for line in bp[component]:
                line.set_color(color)

        ax.plot([], [], color=color, linewidth=4, alpha=0.72, label=label)

    ax.set_ylabel("Jacobian-L1 influence (hop 1)", fontsize=11)
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.4)

    # ── right y-axis: purity (median + IQR) and accuracy ─────────────────────

    ax_r = ax.twinx()

    pur_med = np.array([np.median(by_deg[d]["purity_1"]) for d in all_degrees])
    pur_q1  = np.array([np.percentile(by_deg[d]["purity_1"], 25) for d in all_degrees])
    pur_q3  = np.array([np.percentile(by_deg[d]["purity_1"], 75) for d in all_degrees])
    acc     = np.array([np.mean(by_deg[d]["correct"]) for d in all_degrees])

    ax_r.plot(pos, pur_med, color=_PURITY_COLOR, linewidth=1.5,
              label="Purity hop 1 (median ± IQR)")
    ax_r.fill_between(pos, pur_q1, pur_q3, color=_PURITY_COLOR, alpha=0.18)
    ax_r.plot(pos, acc, color=_ACC_COLOR, linewidth=1.5,
              marker="o", markersize=4, label="Accuracy")

    ax_r.set_ylim(0, 1.05)
    ax_r.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))
    ax_r.set_ylabel("Purity / Accuracy", fontsize=11)

    handles_l, labels_l = ax.get_legend_handles_labels()
    handles_r, labels_r = ax_r.get_legend_handles_labels()
    ax.legend(handles_l + handles_r, labels_l + labels_r,
              loc="upper right", fontsize=9, framealpha=0.85)

    ax.set_title(
        f"{dataset} · {model} · 1-hop influence by degree   (seed={seed})",
        fontsize=11,
    )
    _degree_axis(ax, pos, np.array(all_degrees))

    fig.tight_layout()

    prefix    = f"{dataset}_{model}"
    fname     = f"{prefix}_1hop_influence_by_degree_seed{seed}.png"
    save_path = _subdir(save_dir, "1hop_influence_by_degree")
    _save(fig, save_path, fname, show)


def _plot_delta(by_deg, cfg, seed, save_dir, show):
    """Single-panel plot of per-degree influence and purity deltas.

    Left y-axis  — boxplots of (total_same - total_diff) per degree group.
    Right y-axis — median ± IQR lines for:
                     (same_lbl/tot - diff_lbl/tot)  labelled influence balance
                     purity_delta = purity(hop 2) - purity(hop 1)
                   and accuracy line.
    """
    all_degrees = sorted(by_deg.keys())
    n_deg       = len(all_degrees)
    pos         = np.arange(n_deg, dtype=float)

    dataset = cfg["dataset"]["name"]
    model   = cfg["model"]["name"]

    fig, ax = plt.subplots(figsize=(_fig_w(n_deg), 5))

    def _line_iqr(key, color, label):
        vals = [by_deg[d][key] for d in all_degrees]
        med  = np.array([np.median(v) if v else np.nan for v in vals])
        q1   = np.array([np.percentile(v, 25) if v else np.nan for v in vals])
        q3   = np.array([np.percentile(v, 75) if v else np.nan for v in vals])
        ax.plot(pos, med, color=color, linewidth=1.5, label=label)
        ax.fill_between(pos, q1, q3, color=color, alpha=0.18)

    _line_iqr("purity_1", _PURITY_COLOR, "Purity hop 1 (median ± IQR)")
    _line_iqr("purity_2", "#9C27B0",     "Purity hop 2 (median ± IQR)")

    acc = np.array([np.mean(by_deg[d]["correct"]) for d in all_degrees])
    ax.plot(pos, acc, color=_ACC_COLOR, linewidth=1.5,
            marker="o", markersize=4, label="Accuracy")

    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))
    ax.set_ylabel("Value", fontsize=11)
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.4)
    ax.legend(loc="upper right", fontsize=9, framealpha=0.85)

    ax.set_title(
        f"{dataset} · {model} · purity by hop & accuracy by degree   (seed={seed})",
        fontsize=11,
    )
    _degree_axis(ax, pos, np.array(all_degrees))

    fig.tight_layout()

    prefix    = f"{dataset}_{model}"
    fname     = f"{prefix}_1hop_influence_balance_by_degree_seed{seed}.png"
    save_path = _subdir(save_dir, "1hop_influence_by_degree")
    _save(fig, save_path, fname, show)


# ── orchestration ──────────────────────────────────────────────────────────────

def run(cfg, device, checkpoint_path, show, save_dir):
    cache_dir = cfg.get("dataset_cache_dir", "dataset_cache")
    data = load_or_create_split(
        cfg["dataset"], cfg.get("split", "random"), cfg.get("seed", 42), cache_dir,
    ).to(device)

    k_hops = cfg["model"]["num_layers"] - 1
    log.info("Dataset=%s  model=%s  k_hops=%d",
             cfg["dataset"]["name"], cfg["model"]["name"], k_hops)

    if checkpoint_path:
        pred, model = load_from_checkpoint(cfg, data, device, checkpoint_path)
        import re
        m    = re.search(r"seed(\d+)", os.path.basename(checkpoint_path))
        seed = int(m.group(1)) if m else cfg.get("seed", 42)
    else:
        log.info("No checkpoint — training from scratch (seed=%d)", cfg.get("seed", 42))
        set_seed(cfg.get("seed", 42))
        pred, model = train_model(data, cfg, device)
        seed = cfg.get("seed", 42)

    log.info("Computing 1-hop influence for all test nodes ...")
    records = _compute_all(model, data, pred.cpu(), k_hops, device)
    by_deg  = _aggregate_by_degree(records)

    _plot(by_deg, cfg, seed, save_dir, show)
    _plot_delta(by_deg, cfg, seed, save_dir, show)


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate 1-hop Jacobian-L1 influence by degree group."
    )
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--device", default=None)
    parser.add_argument("--save-dir", default=None,
                        help="Directory to save the plot.")
    parser.add_argument("--show", action="store_true",
                        help="Display the plot interactively.")

    ckpt_grp = parser.add_mutually_exclusive_group()
    ckpt_grp.add_argument("--checkpoint", default=None,
                          help="Path to a saved model checkpoint.")
    ckpt_grp.add_argument("--run", type=int, default=None,
                          help="Run index (1-based) to auto-resolve checkpoint.")
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
                f"--run {args.run} could not locate a matching checkpoint. "
                "Run main.py first with save_checkpoints: True."
            )

    run(cfg, device, checkpoint_path, args.show, args.save_dir)


if __name__ == "__main__":
    main()
