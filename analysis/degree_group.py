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
    set_seed,
    train_model,
)
from dataset_utils import load_or_create_split
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


def _plot_reachability_by_degree(degree_results: dict, k_hops: int, cfg: dict,
                                  save_path: str | None, show: bool):
    degrees = sorted(degree_results.keys())
    degrees = [d for d in degrees if degree_results[d]["n_misc"] > 0]
    if not degrees:
        log.info("No misclassified nodes found for any degree — skipping plot.")
        return

    no_train_prop = [degree_results[d]["no_train_misc"]       / degree_results[d]["n_misc"] for d in degrees]
    no_same_prop  = [degree_results[d]["no_same_train_misc"]  / degree_results[d]["n_misc"] for d in degrees]
    has_same_prop = [degree_results[d]["has_same_train_misc"] / degree_results[d]["n_misc"] for d in degrees]
    counts        = [degree_results[d]["n_misc"] for d in degrees]

    x   = np.arange(len(degrees))
    fig, ax = plt.subplots(figsize=(max(8, len(degrees) * 0.5), 5))

    ax.bar(x, no_train_prop,  color="#D32F2F", label="No train node reachable")
    ax.bar(x, no_same_prop,   bottom=no_train_prop,
           color="#FF8F00", label="Train reachable, no same-class")
    ax.bar(x, has_same_prop,
           bottom=[a + b for a, b in zip(no_train_prop, no_same_prop)],
           color="#1565C0", label="Same-class train reachable")

    for i, n in enumerate(counts):
        ax.text(x[i], 1.01, str(n), ha="center", va="bottom", fontsize=7, color="dimgrey")

    ax.set_xticks(x)
    ax.set_xticklabels(degrees, rotation=55, ha="right", fontsize=8)
    ax.set_xlabel("Node degree", fontsize=11)
    ax.set_ylabel("Proportion of misclassified nodes", fontsize=11)
    ax.set_ylim(0, 1.12)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))

    dataset = cfg["dataset"]["name"]
    model   = cfg["model"]["name"]
    ax.set_title(
        f"Training-node reachability for misclassified test nodes\n"
        f"{dataset} · {model} · {k_hops}-hop receptive field",
        fontsize=11,
    )
    fig.legend(loc="upper center", bbox_to_anchor=(0.5, -0.02),
               ncol=3, fontsize=9, framealpha=0.9)
    fig.subplots_adjust(bottom=0.22)
    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        log.info("Plot saved to %s", save_path)
    if show:
        plt.show()
    plt.close(fig)


def _plot_classification_split_by_bucket(degree_results: dict, k_hops: int,
                                          cfg: dict, save_path: str | None,
                                          show: bool):
    degrees = sorted(degree_results.keys())
    degrees = [d for d in degrees if degree_results[d]["total"] > 0]
    if not degrees:
        log.info("No nodes found for any degree — skipping plot.")
        return

    n_deg   = len(degrees)
    x       = np.arange(n_deg)
    w       = 0.22
    offsets = [-w, 0, w]

    buckets = [
        ("no_train",       "no_train_misc",       "#C62828", "#FFCDD2", "No train reachable"),
        ("no_same_train",  "no_same_train_misc",   "#E65100", "#FFE0B2", "Train, no same-class"),
        ("has_same_train", "has_same_train_misc",  "#1565C0", "#BBDEFB", "Same-class train reachable"),
    ]

    fig, ax = plt.subplots(figsize=(max(10, n_deg * 0.75), 5))
    max_count = 0
    legend_handles = []
    for (tot_key, misc_key, dark, light, label), offset in zip(buckets, offsets):
        pos     = x + offset
        correct = []
        misc    = []
        for d in degrees:
            r = degree_results[d]
            total = r[tot_key]
            m     = r[misc_key]
            correct.append(total - m)
            misc.append(m)
            max_count = max(max_count, total)
        correct = np.array(correct, dtype=float)
        misc    = np.array(misc,    dtype=float)
        ax.bar(pos, correct, width=w, color=light, edgecolor=dark, linewidth=0.6)
        ax.bar(pos, misc, width=w, bottom=correct,
               color=dark, edgecolor=dark, linewidth=0.6)
        legend_handles.append(
            mpatches.Patch(facecolor=light, edgecolor=dark, linewidth=0.8,
                           label=f"{label}  (light=correct, dark=misc)")
        )

    ax.set_xticks(x)
    ax.set_xticklabels(degrees, rotation=55, ha="right", fontsize=8)
    ax.set_xlabel("Node degree", fontsize=11)
    ax.set_ylabel("Number of nodes", fontsize=11)
    ax.set_ylim(0, max_count * 1.15)
    ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.3)

    dataset = cfg["dataset"]["name"]
    model   = cfg["model"]["name"]
    ax.set_title(
        f"Correct vs misclassified node counts per reachability bucket\n"
        f"{dataset} · {model} · {k_hops}-hop receptive field  "
        f"(bar height = bucket size)",
        fontsize=11,
    )
    ax.legend(handles=legend_handles, loc="upper center",
              bbox_to_anchor=(0.5, -0.18), ncol=1, fontsize=9, framealpha=0.9)
    fig.subplots_adjust(bottom=0.28)
    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        log.info("Plot saved to %s", save_path)
    if show:
        plt.show()
    plt.close(fig)


def _load_data_and_train(cfg, device):
    from dataset import load_dataset
    from dataset_utils import apply_split
    data = load_dataset(cfg["dataset"])
    split = cfg.get("split", "random")
    if split == "random":
        set_seed(cfg.get("seed", 42))
    data = apply_split(data, split, cfg["dataset"])
    data = data.to(device)
    k_hops = cfg["model"]["num_layers"] - 1
    log.info("Training model (%s, %d layers, k_hops=%d)...",
             cfg["model"]["name"], cfg["model"]["num_layers"], k_hops)
    set_seed(cfg.get("seed", 42))
    pred, _ = train_model(data, cfg, device)
    return data, pred.cpu(), k_hops


def _run_reachability(cfg, deg_min, deg_max, device, all_degrees=False,
                      save_dir=None, show=False):
    data, pred, k_hops = _load_data_and_train(cfg, device)
    N          = data.num_nodes
    all_deg    = graph_degree(data.edge_index[1], N).cpu()
    y          = data.y.cpu()
    train_set  = set(data.train_mask.cpu().nonzero(as_tuple=False).view(-1).tolist())
    test_idx   = data.test_mask.cpu().nonzero(as_tuple=False).view(-1).tolist()

    if all_degrees:
        unique_degrees = sorted(all_deg[data.test_mask.cpu()].unique().long().tolist())
        log.info("Computing reachability for %d unique test-node degrees...",
                 len(unique_degrees))
        degree_results = {}
        for d in unique_degrees:
            r = _compute_reachability(
                test_idx, all_deg, y, pred, train_set,
                data.edge_index, k_hops, N, d, d,
            )
            degree_results[d] = r
            if r["n_misc"] > 0:
                _log_results(r, str(d), k_hops)

        dataset = cfg["dataset"]["name"]
        model   = cfg["model"]["name"]
        save_path_stacked = None
        save_path_split   = None
        if save_dir:
            save_path_stacked = os.path.join(
                save_dir, "reachability",
                f"{dataset}_{model}_reachability_by_degree.png",
            )
            save_path_split = os.path.join(
                save_dir, "reachability",
                f"{dataset}_{model}_classification_split_by_bucket.png",
            )
        _plot_reachability_by_degree(degree_results, k_hops, cfg, save_path_stacked, show)
        _plot_classification_split_by_bucket(degree_results, k_hops, cfg, save_path_split, show)
    else:
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
    log.info("Mode: %s  Device: %s", args.mode, device)

    if args.all_degrees:
        if args.mode != "reachability":
            parser.error("--all-degrees is only supported with --mode reachability")
        _run_reachability(cfg, None, None, device, all_degrees=True,
                          save_dir=args.save_dir, show=args.show)
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
                          save_dir=args.save_dir, show=args.show)


if __name__ == "__main__":
    main()
