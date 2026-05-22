"""
degree_group.py — Query and analyse test nodes by degree group.

Three modes selected with --mode:

  query        Print test-node indices in the degree range (no model needed).
  influence    Influence analysis for misclassified nodes in the degree range
               that have a higher-degree same-class training node in their
               k-hop receptive field. Prints a table to stdout.
  reachability Reachability analysis: for every misclassified test node in the
               degree range, classify it into one of three buckets:
                 no_train       — no training node within k hops
                 no_same_train  — training reachable, no same-class
                 has_same_train — at least one same-class training node reachable
               Use --all-degrees to run over every unique test-node degree and
               produce stacked-bar plots.

Usage
-----
    uv run analysis/degree_group.py --degree 5 --mode query
    uv run analysis/degree_group.py --degree-min 3 --degree-max 8 --mode query
    uv run analysis/degree_group.py --degree 5 --mode influence
    uv run analysis/degree_group.py --degree 1 --mode reachability
    uv run analysis/degree_group.py --all-degrees --mode reachability --save-dir ./output
"""

import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.utils import degree as graph_degree

from checkpoint_utils import (
    _deep_merge,
    load_cfg,
    load_data,
    load_from_checkpoint,
    _resolve_run_checkpoint,
    set_seed,
    train_model,
)
from dataset_utils import load_or_create_split
from plot_utils import _fig_w, _degree_axis, _save, _subdir
from influence import _analyse_node, _khop_distances, _khop_neighbors
from models import get_model
from train import train
from test import evaluate

log = logging.getLogger(__name__)


# ── shared dataset loading ────────────────────────────────────────────────────

def _load_filtered_nodes(cfg, deg_min, deg_max, device):
    """Load split, return (matching_test_indices, data, all_deg)."""
    data = load_data(cfg, device)
    all_deg  = graph_degree(data.edge_index[1], data.num_nodes).cpu()
    test_mask = data.test_mask.cpu()
    test_idx  = test_mask.nonzero(as_tuple=False).view(-1).tolist()
    matching  = [n for n in test_idx if deg_min <= int(all_deg[n].item()) <= deg_max]
    return matching, data, all_deg


# ── mode: query ───────────────────────────────────────────────────────────────

def _run_query(cfg, deg_min, deg_max, device):
    matching, data, _ = _load_filtered_nodes(cfg, deg_min, deg_max, device)
    log.info(
        "Dataset=%s  split=%s  degree=[%d, %d]  test nodes found: %d",
        cfg["dataset"]["name"], cfg.get("split", "random"),
        deg_min, deg_max, len(matching),
    )
    print(" ".join(str(n) for n in sorted(matching)))


# ── mode: influence ───────────────────────────────────────────────────────────

def _build_edge_weight_map(edge_index, num_nodes: int) -> dict:
    norm_ei, norm_ew = gcn_norm(
        edge_index.cpu(), edge_weight=None, num_nodes=num_nodes,
        improved=False, add_self_loops=True, flow="source_to_target",
    )
    return {
        (int(s), int(d)): float(w)
        for s, d, w in zip(norm_ei[0].tolist(), norm_ei[1].tolist(), norm_ew.tolist())
    }


def _find_qualifying_nodes(data, all_deg, pred, edge_weight_map: dict,
                            deg_min: int, deg_max: int, k_hops: int = 1) -> list[dict]:
    """Return misclassified test nodes in [deg_min, deg_max] that have ≥1
    same-class training node within k_hops hops whose degree is strictly
    greater than the test node's own degree (≥ own degree for nodes with degree≥5).
    """
    N          = data.num_nodes
    y          = data.y.cpu()
    pred_cpu   = pred.cpu()
    train_mask = data.train_mask.cpu()
    test_mask  = data.test_mask.cpu()
    deg        = all_deg.cpu()
    test_indices = test_mask.nonzero(as_tuple=False).view(-1).tolist()
    results = []

    for node in test_indices:
        node_deg = int(deg[node].item())
        if not (deg_min <= node_deg <= deg_max):
            continue

        true_lbl = int(y[node].item())
        if int(pred_cpu[node].item()) == true_lbl:
            continue  # skip correctly classified

        hop_dist  = _khop_distances(data.edge_index, node, k_hops, N)
        min_nb_deg = node_deg if node_deg >= 5 else node_deg + 1

        qualifying = []
        for nb, hop in hop_dist.items():
            if not train_mask[nb].item():
                continue
            if int(y[nb].item()) != true_lbl:
                continue
            nb_deg = int(deg[nb].item())
            if nb_deg < min_nb_deg:
                continue
            ew = edge_weight_map.get((nb, node)) if hop == 1 else None
            qualifying.append({
                "node_idx":    nb,
                "degree":      nb_deg,
                "hop":         hop,
                "edge_weight": ew,
            })

        if qualifying:
            qualifying.sort(key=lambda d: (d["hop"], -d["degree"]))
            results.append({
                "node_idx":                   node,
                "degree":                     node_deg,
                "true_label":                 true_lbl,
                "qualifying_train_neighbors": qualifying,
            })

    return results


def _print_khop_neighborhood(node: int, data, all_deg, pred, y,
                              edge_weight_map: dict, k_hops: int):
    N        = data.num_nodes
    true_lbl = int(y[node].item())
    hop_dist = _khop_distances(data.edge_index, node, k_hops, N)
    train_mask = data.train_mask.cpu()
    rows = []
    for nb, hop in hop_dist.items():
        ew = edge_weight_map.get((nb, node)) if hop == 1 else None
        rows.append({
            "neighbor":     nb,
            "degree":       int(all_deg[nb].item()),
            "hop":          hop,
            "in_train_set": bool(train_mask[nb].item()),
            "same_class":   int(y[nb].item()) == true_lbl,
            "correct_pred": bool((pred[nb] == y[nb]).item()),
            "edge_weight":  round(ew, 6) if ew is not None else None,
        })

    df = (
        pd.DataFrame(rows)
        .sort_values(
            ["hop", "same_class", "in_train_set", "degree"],
            ascending=[True, False, False, True],
        )
        .reset_index(drop=True)
    )
    df.index += 1
    df.index.name = "#"
    print(f"\n{k_hops}-hop receptive field of node {node}  "
          f"(class={true_lbl}, degree={int(all_deg[node].item())})\n")
    print(df.to_string())
    print()


def _run_influence(cfg, deg_min, deg_max, device):
    data = load_data(cfg, device)
    all_deg = graph_degree(data.edge_index[1], data.num_nodes).cpu()
    k_hops  = cfg["model"]["num_layers"] - 1
    edge_weight_map = _build_edge_weight_map(data.edge_index, data.num_nodes)

    log.info("Training model (%s, %d layers)...",
             cfg["model"]["name"], cfg["model"]["num_layers"])
    set_seed(cfg.get("seed", 42))
    pred, model = train_model(data, cfg, device)

    log.info("Finding qualifying misclassified test nodes in degree range [%d, %d]...",
             deg_min, deg_max)
    qualifying = _find_qualifying_nodes(
        data, all_deg, pred, edge_weight_map, deg_min, deg_max, k_hops=1
    )

    if not qualifying:
        log.info("No test nodes found matching the condition in degree [%d, %d].",
                 deg_min, deg_max)
        return

    log.info("Found %d misclassified qualifying node(s):", len(qualifying))
    for q in qualifying:
        def _nb_str(nb):
            if nb["hop"] == 1:
                return (f"node {nb['node_idx']} (deg={nb['degree']}, hop=1, "
                        f"ew={nb['edge_weight']:.6f})")
            return f"node {nb['node_idx']} (deg={nb['degree']}, hop={nb['hop']})"
        log.info(
            "  node %d  deg=%d  label=%d  pred=%d  — higher-deg same-class train neighbors: %s",
            q["node_idx"], q["degree"], q["true_label"],
            int(pred[q["node_idx"]].item()),
            ", ".join(_nb_str(nb) for nb in q["qualifying_train_neighbors"]),
        )

    log.info("")
    log.info("=" * 70)
    log.info("Running influence analysis for each qualifying node...")
    log.info("=" * 70)

    y         = data.y.cpu()
    train_set = set(data.train_mask.cpu().nonzero(as_tuple=False).view(-1).tolist())

    for q in qualifying:
        node = q["node_idx"]
        log.info("")
        log.info("─" * 60)
        log.info(
            "Node %d | degree=%d | label=%d | pred=%d | MISCLASSIFIED",
            node, q["degree"], q["true_label"], int(pred[node].item()),
        )

        def _nb_detail(nb):
            if nb["hop"] == 1:
                return (f"node {nb['node_idx']} (deg={nb['degree']}, hop=1, "
                        f"ew={nb['edge_weight']:.6f})")
            return f"node {nb['node_idx']} (deg={nb['degree']}, hop={nb['hop']})"

        log.info(
            "Higher-deg same-class train neighbors (within %d hops): %s",
            k_hops,
            ", ".join(_nb_detail(nb) for nb in q["qualifying_train_neighbors"]),
        )
        result = _analyse_node(model, data, pred, node, k_hops, train_set, y, all_deg)
        if result is None:
            log.info("  (skipped — no training nodes in %d-hop receptive field)", k_hops)
            continue

        log.info(
            "  Influence — same_norm=%.4f  diff_norm=%.4f  "
            "(n_same_train=%d  n_diff_train=%d)",
            result["same_class_influence_norm"],
            result["diff_class_influence_norm"],
            result["n_same_train"],
            result["n_diff_train"],
        )
        log.info("  Training neighbors by influence:")
        for nb in result["neighbors"]:
            if nb["type"] == "non_train":
                continue
            ew = edge_weight_map.get((nb["node_idx"], node))
            ew_str = f"ew={ew:.6f}" if ew is not None else "ew=N/A"
            log.info(
                "    [%s] node %-5d  deg=%-4d  hop=%-2d  %s  norm=%.4f",
                nb["type"], nb["node_idx"], nb["degree"],
                nb["hop_distance"], ew_str, nb["influence_norm"],
            )

        _print_khop_neighborhood(node, data, all_deg, pred, y,
                                 edge_weight_map, k_hops=1)


# ── mode: reachability ────────────────────────────────────────────────────────

def _compute_reachability(test_idx, all_deg, y, pred, train_set,
                           edge_index, k_hops: int, N: int,
                           deg_min: int, deg_max: int) -> dict:
    in_range = [
        node for node in test_idx
        if deg_min <= int(all_deg[node].item()) <= deg_max
    ]

    no_train = no_train_misc = 0
    no_same_train = no_same_train_misc = 0
    has_same_train = has_same_train_misc = 0

    for node in in_range:
        true_lbl  = int(y[node].item())
        is_misc   = int(pred[node].item()) != true_lbl
        neighbors = _khop_neighbors(edge_index, node, k_hops, N)

        train_in_field      = [n for n in neighbors if n in train_set]
        same_train_in_field = [n for n in train_in_field
                               if int(y[n].item()) == true_lbl]

        if not train_in_field:
            no_train += 1
            if is_misc: no_train_misc += 1
        elif not same_train_in_field:
            no_same_train += 1
            if is_misc: no_same_train_misc += 1
        else:
            has_same_train += 1
            if is_misc: has_same_train_misc += 1

    n_misc = no_train_misc + no_same_train_misc + has_same_train_misc
    return {
        "total":               len(in_range),
        "n_misc":              n_misc,
        "no_train":            no_train,
        "no_train_misc":       no_train_misc,
        "no_same_train":       no_same_train,
        "no_same_train_misc":  no_same_train_misc,
        "has_same_train":      has_same_train,
        "has_same_train_misc": has_same_train_misc,
    }


def _misc_rate(misc: int, total: int) -> float:
    return misc / total if total > 0 else float("nan")


def _log_results(results: dict, deg_label: str, k_hops: int):
    n_misc = results["n_misc"]
    log.info("")
    log.info("Degree-%s test nodes : %d total, %d misclassified (%.1f%%)",
             deg_label, results["total"], n_misc,
             100 * n_misc / results["total"] if results["total"] else 0)
    log.info("Receptive field radius: %d hop(s)", k_hops)
    log.info("")
    log.info("── No training node reachable within %d hop(s) ──", k_hops)
    log.info("  Total in bucket   : %d", results["no_train"])
    log.info("  Misclassified     : %d / %d  (%.1f%% misc rate)",
             results["no_train_misc"], results["no_train"],
             100 * _misc_rate(results["no_train_misc"], results["no_train"]))
    log.info("")
    log.info("── Training reachable but NO same-class within %d hop(s) ──", k_hops)
    log.info("  Total in bucket   : %d", results["no_same_train"])
    log.info("  Misclassified     : %d / %d  (%.1f%% misc rate)",
             results["no_same_train_misc"], results["no_same_train"],
             100 * _misc_rate(results["no_same_train_misc"], results["no_same_train"]))
    log.info("")
    log.info("── Same-class training reachable within %d hop(s) ──", k_hops)
    log.info("  Total in bucket   : %d", results["has_same_train"])
    log.info("  Misclassified     : %d / %d  (%.1f%% misc rate)",
             results["has_same_train_misc"], results["has_same_train"],
             100 * _misc_rate(results["has_same_train_misc"], results["has_same_train"]))


_REACH_COLORS = {
    "no_train":       "#D32F2F",
    "no_same_train":  "#E65100",
    "has_same_train": "#1565C0",
}
_REACH_LABELS = {
    "no_train":       "No train node reachable",
    "no_same_train":  "Train reachable, no same-class",
    "has_same_train": "Same-class train reachable",
}


def _plot_reachability_by_degree(all_run_results, k_hops, cfg, save_dir, show):
    degrees = sorted(all_run_results[0].keys())
    degrees = [d for d in degrees
               if any(r[d]["n_misc"] > 0 for r in all_run_results)]
    if not degrees:
        log.info("No misclassified nodes — skipping reachability stacked plot.")
        return

    def _prop(run, d, key):
        n = run[d]["n_misc"]
        return run[d][key] / n if n > 0 else 0.0

    n_runs  = len(all_run_results)
    n_deg   = len(degrees)
    dataset = cfg["dataset"]["name"]
    model   = cfg["model"]["name"]
    subtitle = f"{dataset} · {model} · {k_hops}-hop receptive field · {n_runs} run{'s' if n_runs > 1 else ''}"

    group_centers = np.arange(n_deg)
    bar_w = 0.8 / max(n_runs, 1)

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(_fig_w(n_deg), 7),
        sharex=True, gridspec_kw={"height_ratios": [3, 1]},
    )

    for ri, run_result in enumerate(all_run_results):
        offset = (ri - (n_runs - 1) / 2) * bar_w
        xpos   = group_centers + offset
        nt = [_prop(run_result, d, "no_train_misc")       for d in degrees]
        ns = [_prop(run_result, d, "no_same_train_misc")  for d in degrees]
        hs = [_prop(run_result, d, "has_same_train_misc") for d in degrees]
        bot_ns = nt
        bot_hs = [a + b for a, b in zip(nt, ns)]
        ax_top.bar(xpos, nt, width=bar_w, color="#D32F2F", alpha=0.8)
        ax_top.bar(xpos, ns, width=bar_w, bottom=bot_ns, color="#FF8F00", alpha=0.8)
        ax_top.bar(xpos, hs, width=bar_w, bottom=bot_hs, color="#1565C0", alpha=0.8)

    ax_top.set_ylabel("Proportion of misclassified nodes", fontsize=11)
    ax_top.set_ylim(0, 1.05)
    ax_top.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax_top.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.3)
    ax_top.set_title(
        f"Training-node reachability for misclassified test nodes\n{subtitle}",
        fontsize=11,
    )

    counts = [int(np.median([r[d]["n_misc"]  for r in all_run_results])) for d in degrees]
    totals = [int(np.median([r[d]["total"]   for r in all_run_results])) for d in degrees]
    w = 0.35
    ax_bot.bar([p - w / 2 for p in group_centers], totals, width=w, color="steelblue",  alpha=0.5, label="# total")
    ax_bot.bar([p + w / 2 for p in group_centers], counts, width=w, color="lightgrey", alpha=0.9, label="# misc (median)")
    ax_bot.set_ylabel("# nodes", fontsize=9, color="grey")
    ax_bot.tick_params(axis="y", labelsize=7, colors="grey")
    bot_legend_handles = [
        mpatches.Patch(color="steelblue", alpha=0.5, label="# total"),
        mpatches.Patch(color="lightgrey", alpha=0.9, label="# misc (median)"),
    ]
    ax_bot.legend(handles=bot_legend_handles, fontsize=8, framealpha=0.8)
    _degree_axis(ax_bot, group_centers, degrees)
    ax_bot.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.3)

    legend_handles = [
        mpatches.Patch(color="#D32F2F", label=_REACH_LABELS["no_train"]),
        mpatches.Patch(color="#FF8F00", label=_REACH_LABELS["no_same_train"]),
        mpatches.Patch(color="#1565C0", label=_REACH_LABELS["has_same_train"]),
    ]
    ax_top.legend(handles=legend_handles, loc="upper right",
                  fontsize=9, framealpha=0.9)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.14)
    _save(fig, save_dir, f"{dataset}_{model}_reachability_by_degree_{n_runs}runs.png", show)





def _plot_misc_rate_marginal(all_run_results, k_hops, cfg, save_dir, show):
    """Overall misclassification rate per bucket, collapsed across degree.

    Shows median ± IQR across runs.
    """
    bucket_keys = ["no_train", "no_same_train", "has_same_train"]
    misc_keys   = ["no_train_misc", "no_same_train_misc", "has_same_train_misc"]
    n_runs      = len(all_run_results)
    dataset     = cfg["dataset"]["name"]
    model       = cfg["model"]["name"]

    run_rates = {bkey: [] for bkey in bucket_keys}
    for run_result in all_run_results:
        for bkey, mkey in zip(bucket_keys, misc_keys):
            total = sum(r[bkey] for r in run_result.values())
            misc  = sum(r[mkey] for r in run_result.values())
            run_rates[bkey].append(misc / total if total > 0 else float("nan"))

    med  = [float(np.nanmedian(run_rates[k])) for k in bucket_keys]
    q1   = [float(np.nanpercentile(run_rates[k], 25)) for k in bucket_keys]
    q3   = [float(np.nanpercentile(run_rates[k], 75)) for k in bucket_keys]
    yerr = [[m - lo for m, lo in zip(med, q1)],
            [hi - m  for m, hi in zip(med, q3)]]

    fig, ax = plt.subplots(figsize=(6, 5))
    xpos   = np.arange(len(bucket_keys))
    colors = [_REACH_COLORS[k] for k in bucket_keys]
    ax.bar(xpos, med, color=colors, alpha=0.75, width=0.5,
           edgecolor="white", linewidth=0.5)
    ax.errorbar(xpos, med, yerr=yerr, fmt="none", color="black",
                capsize=4, linewidth=1.2)
    for i, (m, hi) in enumerate(zip(med, q3)):
        ax.text(xpos[i], hi + 0.015, f"{m:.1%}", ha="center", va="bottom",
                fontsize=9, fontweight="bold")

    ax.set_xticks(xpos)
    ax.set_xticklabels([_REACH_LABELS[k] for k in bucket_keys], fontsize=7,
                       rotation=15, ha="right")
    ax.set_ylabel("Misclassification rate", fontsize=11)
    ax.set_ylim(0, 1.15)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.3)
    ax.set_title(
        f"Overall misclassification rate by reachability bucket\n"
        f"{dataset} · {model} · {k_hops}-hop  ·  "
        f"median ± IQR, {n_runs} run{'s' if n_runs > 1 else ''}",
        fontsize=11,
    )
    fig.tight_layout()
    _save(fig, save_dir, f"{dataset}_{model}_misc_rate_marginal_{n_runs}runs.png", show)


def _load_data_and_train(cfg, device, run_id=None):
    from dataset_utils import load_or_create_split
    split     = cfg.get("split", "random")
    seed      = cfg.get("seed", 42)
    cache_dir = cfg.get("dataset_cache_dir", "dataset_cache")
    data      = load_or_create_split(cfg["dataset"], split, seed, cache_dir)
    data = data.to(device)
    k_hops = cfg["model"]["num_layers"] - 1
    if run_id is not None:
        ckpt_path = _resolve_run_checkpoint(cfg, run_id)
        if ckpt_path is None:
            raise ValueError(f"Checkpoint for run {run_id} not found")
        log.info("Loading checkpoint run %d: %s", run_id, ckpt_path)
        pred, _ = load_from_checkpoint(cfg, data, device, ckpt_path)
    else:
        log.info("Training model (%s, %d layers, k_hops=%d)...",
                 cfg["model"]["name"], cfg["model"]["num_layers"], k_hops)
        set_seed(cfg.get("seed", 42))
        pred, _ = train_model(data, cfg, device)
    return data, pred.cpu(), k_hops


def _run_reachability(cfg, deg_min, deg_max, device, all_degrees=False,
                      save_dir=None, show=False, n_runs=1):
    run_ids = list(range(1, n_runs + 1)) if n_runs > 1 else [None]

    # Resolve save_dir from exec_dir of run 1 when loading checkpoints
    if save_dir is None and run_ids[0] is not None:
        ckpt = _resolve_run_checkpoint(cfg, run_ids[0])
        if ckpt:
            save_dir = os.path.dirname(os.path.dirname(ckpt))

    # Load data once (graph topology fixed across runs)
    data, _, k_hops = _load_data_and_train(cfg, device, run_id=run_ids[0])
    N         = data.num_nodes
    all_deg   = graph_degree(data.edge_index[1], N).cpu()
    y         = data.y.cpu()
    train_set = set(data.train_mask.cpu().nonzero(as_tuple=False).view(-1).tolist())
    test_idx  = data.test_mask.cpu().nonzero(as_tuple=False).view(-1).tolist()

    if all_degrees:
        unique_degrees = sorted(all_deg[data.test_mask.cpu()].unique().long().tolist())
        log.info("Computing reachability for %d degrees × %d run(s)...",
                 len(unique_degrees), len(run_ids))
        all_run_results = []
        for run_id in run_ids:
            _, pred, _ = _load_data_and_train(cfg, device, run_id=run_id)
            degree_results = {
                d: _compute_reachability(
                    test_idx, all_deg, y, pred, train_set,
                    data.edge_index, k_hops, N, d, d,
                )
                for d in unique_degrees
            }
            all_run_results.append(degree_results)

        # Log results from first run
        for d in unique_degrees:
            if all_run_results[0][d]["n_misc"] > 0:
                _log_results(all_run_results[0][d], str(d), k_hops)

        reach_dir = _subdir(save_dir, "reachability")
        _plot_reachability_by_degree(all_run_results, k_hops, cfg, reach_dir, show)
        _plot_misc_rate_marginal(all_run_results, k_hops, cfg, reach_dir, show)
    else:
        _, pred, _ = _load_data_and_train(cfg, device, run_id=run_ids[0])
        results = _compute_reachability(
            test_idx, all_deg, y, pred, train_set,
            data.edge_index, k_hops, N, deg_min, deg_max,
        )
        deg_label = str(deg_min) if deg_min == deg_max else f"{deg_min}–{deg_max}"
        _log_results(results, deg_label, k_hops)


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Query and analyse test nodes grouped by degree."
    )
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--device", default=None,
                        help="Device override, e.g. cuda:0 or cpu")
    parser.add_argument("--mode", choices=["query", "influence", "reachability"],
                        required=True, help="Analysis mode.")

    # degree selection
    deg_group = parser.add_mutually_exclusive_group()
    deg_group.add_argument("--all-degrees", action="store_true",
                           help="Run over all unique test-node degrees (reachability mode only)")
    deg_group.add_argument("--degree", type=int, default=None,
                           help="Exact degree value")
    deg_group.add_argument("--degree-min", type=int,
                           help="Lower bound of degree range (inclusive)")
    parser.add_argument("--degree-max", type=int, default=None,
                        help="Upper bound; required with --degree-min")

    # reachability-only flags
    parser.add_argument("--save-dir", default=None,
                        help="Directory to save plots (reachability --all-degrees only)")
    parser.add_argument("--show", action="store_true",
                        help="Display plots interactively (reachability --all-degrees only)")
    parser.add_argument("--n-runs", type=int, default=None,
                        help="Number of checkpoint runs to aggregate (reachability mode only); "
                             "defaults to num_runs from config")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )

    cfg = load_cfg(args.config)
    if args.n_runs is None:
        args.n_runs = cfg.get("num_runs", 1)
    device = torch.device(
        args.device if args.device
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    log.info("Mode: %s  Device: %s", args.mode, device)

    if args.all_degrees:
        if args.mode != "reachability":
            parser.error("--all-degrees is only supported with --mode reachability")
        _run_reachability(cfg, None, None, device, all_degrees=True,
                          save_dir=args.save_dir, show=args.show, n_runs=args.n_runs)
        return

    # Resolve degree range
    if args.degree is not None:
        deg_min = deg_max = args.degree
    elif args.degree_min is not None:
        if args.degree_max is None:
            parser.error("--degree-max is required when --degree-min is used")
        deg_min, deg_max = args.degree_min, args.degree_max
        if deg_min > deg_max:
            parser.error("--degree-min must be ≤ --degree-max")
    else:
        parser.error("Specify --degree, --degree-min/--degree-max, or --all-degrees")

    log.info("Degree range: [%d, %d]", deg_min, deg_max)

    if args.mode == "query":
        _run_query(cfg, deg_min, deg_max, device)
    elif args.mode == "influence":
        _run_influence(cfg, deg_min, deg_max, device)
    elif args.mode == "reachability":
        _run_reachability(cfg, deg_min, deg_max, device,
                          all_degrees=False,
                          save_dir=args.save_dir, show=args.show, n_runs=args.n_runs)


if __name__ == "__main__":
    main()
