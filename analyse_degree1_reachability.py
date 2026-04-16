"""
analyse_degree1_reachability.py — Reachability analysis for misclassified test nodes
in a given degree group.

For all misclassified test nodes in the specified degree range, checks within
the (num_layers - 1)-hop receptive field:
  1. How many have NO training node of any class reachable.
  2. How many have NO same-class training node reachable
     (but may have diff-class training nodes).

Usage
-----
    python analyse_degree1_reachability.py                      # default: degree=1
    python analyse_degree1_reachability.py --degree 3
    python analyse_degree1_reachability.py --degree-min 2 --degree-max 5
    python analyse_degree1_reachability.py --degree 1 --config config.yaml --device cpu
"""

import argparse
import copy
import logging
import os
import random
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from torch_geometric.utils import degree as graph_degree

from dataset import load_dataset
from dataset_utils import apply_split
from influence import _khop_neighbors
from models import get_model
from train import train
from test import evaluate

log = logging.getLogger(__name__)


# ── helpers ────────────────────────────────────────────────────────────────────

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _deep_merge(base, override):
    merged = copy.deepcopy(base)
    for key, val in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(val, dict):
            merged[key] = _deep_merge(merged[key], val)
        else:
            merged[key] = copy.deepcopy(val)
    return merged


def train_model(data, cfg, device):
    """Train one model run and return (pred [N], model)."""
    model_cfg = cfg["model"]
    _standard_keys = {"name", "hidden_dim", "num_layers", "dropout"}
    extra_kwargs = {k: v for k, v in model_cfg.items() if k not in _standard_keys}

    model = get_model(
        model_cfg["name"],
        in_dim=data.num_node_features,
        hidden_dim=model_cfg["hidden_dim"],
        out_dim=int(data.y.max().item()) + 1,
        num_layers=model_cfg["num_layers"],
        dropout=model_cfg["dropout"],
        **extra_kwargs,
    ).to(device)

    train_cfg = cfg["train"]
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=train_cfg["lr"],
        weight_decay=float(train_cfg["weight_decay"]),
    )
    criterion = torch.nn.CrossEntropyLoss()

    best_val_acc = 0.0
    best_state = copy.deepcopy(model.state_dict())
    patience = train_cfg.get("patience", 0)
    patience_counter = 0

    for epoch in range(1, train_cfg["epochs"] + 1):
        loss = train(model, data, optimizer, criterion)
        results = evaluate(model, data)
        if results["val"] > best_val_acc:
            best_val_acc = results["val"]
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        if patience > 0 and patience_counter >= patience:
            log.info("Early stopping at epoch %d", epoch)
            break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pred = model(data.x, data.edge_index).argmax(dim=1)

    return pred, model


# ── core analysis ──────────────────────────────────────────────────────────────

def _compute_reachability(test_idx, all_deg, y, pred, train_set,
                          edge_index, k_hops: int, N: int,
                          deg_min: int, deg_max: int) -> dict:
    """Compute reachability counts for all (and misclassified) test nodes in
    [deg_min, deg_max].

    Every node is assigned to exactly one reachability bucket:
        no_train       — no training node within k hops
        no_same_train  — training reachable, but none same-class
        has_same_train — at least one same-class training node reachable

    For each bucket both the total count and the misclassified count are
    tracked, so the caller can compute:
        - proportion of *misclassified* nodes per bucket  (current view)
        - misclassification *rate* within each bucket     (flipped view)

    Returns
    -------
    dict with keys:
        total, n_misc,
        no_train,            no_train_misc,
        no_same_train,       no_same_train_misc,
        has_same_train,      has_same_train_misc
    """
    in_range = [
        node for node in test_idx
        if deg_min <= int(all_deg[node].item()) <= deg_max
    ]

    no_train           = 0;  no_train_misc      = 0
    no_same_train      = 0;  no_same_train_misc = 0
    has_same_train     = 0;  has_same_train_misc = 0

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
        "total":              len(in_range),
        "n_misc":             n_misc,
        "no_train":           no_train,
        "no_train_misc":      no_train_misc,
        "no_same_train":      no_same_train,
        "no_same_train_misc": no_same_train_misc,
        "has_same_train":     has_same_train,
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
    """Stacked bar chart of misclassified-node reachability proportions per degree.

    Three stacked segments per degree bar (proportions of misclassified nodes):
      - No training node reachable          (red)
      - Train reachable, no same-class      (orange)
      - Same-class training node reachable  (steelblue)

    Degrees with zero misclassified nodes are skipped.
    """
    degrees = sorted(degree_results.keys())
    degrees = [d for d in degrees if degree_results[d]["n_misc"] > 0]

    if not degrees:
        log.info("No misclassified nodes found for any degree — skipping plot.")
        return

    no_train_prop      = []
    no_same_prop       = []
    has_same_prop      = []
    counts             = []

    for d in degrees:
        r      = degree_results[d]
        n      = r["n_misc"]
        no_train_prop.append(r["no_train_misc"]       / n)
        no_same_prop.append(r["no_same_train_misc"]   / n)
        has_same_prop.append(r["has_same_train_misc"] / n)
        counts.append(n)

    x   = np.arange(len(degrees))
    fig, ax = plt.subplots(figsize=(max(8, len(degrees) * 0.5), 5))

    b1 = ax.bar(x, no_train_prop,  color="#D32F2F", label="No train node reachable")
    b2 = ax.bar(x, no_same_prop,   bottom=no_train_prop,
                color="#FF8F00", label="Train reachable, no same-class")
    b3 = ax.bar(x, has_same_prop,
                bottom=[a + b for a, b in zip(no_train_prop, no_same_prop)],
                color="#1565C0", label="Same-class train reachable")

    # Annotate misclassified count above each bar
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


def _plot_misclassification_rate_by_reachability(degree_results: dict, k_hops: int,
                                                  cfg: dict, save_path: str | None,
                                                  show: bool):
    """Line plot of misclassification rate per reachability bucket per degree.

    Flipped view: for each reachability bucket, how many nodes in that bucket
    are misclassified?  This answers "how strongly does reachability predict
    misclassification?" rather than "why are misclassified nodes failing?".

    Three lines:
      - Red   : misclassification rate among nodes with no training reachable
      - Orange: misclassification rate among nodes with training but no same-class
      - Blue  : misclassification rate among nodes with same-class training reachable

    Degrees where a bucket is empty are plotted as gaps (NaN).
    """
    degrees = sorted(degree_results.keys())

    def _rates(bucket_total_key, bucket_misc_key):
        vals = []
        for d in degrees:
            r = degree_results[d]
            total = r[bucket_total_key]
            misc  = r[bucket_misc_key]
            vals.append(misc / total if total > 0 else float("nan"))
        return vals

    rate_no_train  = _rates("no_train",       "no_train_misc")
    rate_no_same   = _rates("no_same_train",  "no_same_train_misc")
    rate_has_same  = _rates("has_same_train", "has_same_train_misc")

    # bucket sizes for annotation
    sizes_no_train = [degree_results[d]["no_train"]       for d in degrees]
    sizes_no_same  = [degree_results[d]["no_same_train"]  for d in degrees]
    sizes_has_same = [degree_results[d]["has_same_train"] for d in degrees]

    x = np.arange(len(degrees))
    fig, ax = plt.subplots(figsize=(max(8, len(degrees) * 0.5), 5))

    def _plot_line(rates, sizes, color, label):
        y = np.array(rates, dtype=float)
        mask = ~np.isnan(y)
        ax.plot(x[mask], y[mask], color=color, lw=1.8, marker="o",
                markersize=4, label=label)

    _plot_line(rate_no_train, sizes_no_train, "#D32F2F",
               "No train node reachable")
    _plot_line(rate_no_same,  sizes_no_same,  "#FF8F00",
               "Train reachable, no same-class")
    _plot_line(rate_has_same, sizes_has_same, "#1565C0",
               "Same-class train reachable")

    ax.set_xticks(x)
    ax.set_xticklabels(degrees, rotation=55, ha="right", fontsize=8)
    ax.set_xlabel("Node degree", fontsize=11)
    ax.set_ylabel("Misclassification rate within bucket", fontsize=11)
    ax.set_ylim(-0.05, 1.10)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.3)

    dataset = cfg["dataset"]["name"]
    model   = cfg["model"]["name"]
    ax.set_title(
        f"Misclassification rate by reachability bucket\n"
        f"{dataset} · {model} · {k_hops}-hop receptive field",
        fontsize=11,
    )
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18),
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


def _load_data_and_train(cfg, device):
    """Load dataset, apply split, train model. Returns (data, pred, k_hops)."""
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


def run_analysis(cfg, deg_min: int, deg_max: int, device: torch.device):
    data, pred, k_hops = _load_data_and_train(cfg, device)

    N          = data.num_nodes
    all_deg    = graph_degree(data.edge_index[1], N).cpu()
    y          = data.y.cpu()
    train_mask = data.train_mask.cpu()
    test_mask  = data.test_mask.cpu()
    train_set  = set(train_mask.nonzero(as_tuple=False).view(-1).tolist())
    test_idx   = test_mask.nonzero(as_tuple=False).view(-1).tolist()

    results = _compute_reachability(
        test_idx, all_deg, y, pred, train_set,
        data.edge_index, k_hops, N, deg_min, deg_max,
    )
    deg_label = str(deg_min) if deg_min == deg_max else f"{deg_min}–{deg_max}"
    _log_results(results, deg_label, k_hops)


def run_all_degrees(cfg, device: torch.device, save_dir: str | None, show: bool):
    data, pred, k_hops = _load_data_and_train(cfg, device)

    N          = data.num_nodes
    all_deg    = graph_degree(data.edge_index[1], N).cpu()
    y          = data.y.cpu()
    train_mask = data.train_mask.cpu()
    test_mask  = data.test_mask.cpu()
    train_set  = set(train_mask.nonzero(as_tuple=False).view(-1).tolist())
    test_idx   = test_mask.nonzero(as_tuple=False).view(-1).tolist()

    unique_degrees = sorted(all_deg[test_mask].unique().long().tolist())
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

    save_path_current = None
    save_path_flipped = None
    if save_dir:
        save_path_current = os.path.join(
            save_dir, "reachability",
            f"{dataset}_{model}_reachability_by_degree.png",
        )
        save_path_flipped = os.path.join(
            save_dir, "reachability",
            f"{dataset}_{model}_misc_rate_by_reachability.png",
        )

    _plot_reachability_by_degree(degree_results, k_hops, cfg, save_path_current, show)
    _plot_misclassification_rate_by_reachability(
        degree_results, k_hops, cfg, save_path_flipped, show
    )


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Reachability analysis for misclassified test nodes in a degree group."
    )
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--device", default=None,
                        help="Device override, e.g. cuda:0 or cpu")
    parser.add_argument("--save-dir", default=None,
                        help="Directory to save the plot (--all-degrees only)")
    parser.add_argument("--show", action="store_true",
                        help="Display the plot interactively (--all-degrees only)")

    deg_group = parser.add_mutually_exclusive_group()
    deg_group.add_argument("--all-degrees", action="store_true",
                           help="Run for every unique test-node degree and plot results")
    deg_group.add_argument("--degree", type=int, default=None,
                           help="Exact degree value (default: 1)")
    deg_group.add_argument("--degree-min", type=int,
                           help="Lower bound of degree range (inclusive)")
    parser.add_argument("--degree-max", type=int, default=None,
                        help="Upper bound of degree range (inclusive); "
                             "required when --degree-min is used")
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

    if args.all_degrees:
        run_all_degrees(cfg, device,
                        save_dir=args.save_dir,
                        show=args.show)
    else:
        if args.degree_min is not None:
            if args.degree_max is None:
                parser.error("--degree-max is required when --degree-min is used")
            deg_min, deg_max = args.degree_min, args.degree_max
            if deg_min > deg_max:
                parser.error("--degree-min must be ≤ --degree-max")
        else:
            d = args.degree if args.degree is not None else 1
            deg_min = deg_max = d

        log.info("Degree range: [%d, %d]", deg_min, deg_max)
        run_analysis(cfg, deg_min, deg_max, device)


if __name__ == "__main__":
    main()
