"""
analyse_degree_group.py — Influence analysis for test nodes in a degree group
that have a higher-degree same-class training node in their 1-hop neighbourhood.

Usage
-----
    python analyse_degree_group.py --degree 3
    python analyse_degree_group.py --degree-min 2 --degree-max 5
    python analyse_degree_group.py --degree 3 --config config.yaml --device cpu

For every test node in the specified degree group the script:
  1. Checks whether the node has ≥1 same-class training node among its
     1-hop neighbours whose degree is strictly greater than the test node's
     own degree.
  2. For each qualifying node, runs the exact Jacobian-based influence
     analysis and reports the same/diff-class training-node breakdown.

Output is written to stdout/log only — no plots are generated.
"""

import argparse
import copy
import logging
import random
import sys

import numpy as np
import torch
import yaml
from torch_geometric.utils import degree as graph_degree

from dataset import load_dataset
from dataset_utils import apply_split
from influence import _analyse_node
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

    for epoch in range(1, train_cfg["epochs"] + 1):
        loss = train(model, data, optimizer, criterion)
        results = evaluate(model, data)
        if results["val"] > best_val_acc:
            best_val_acc = results["val"]
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pred = model(data.x, data.edge_index).argmax(dim=1)

    return pred, model


# ── core analysis ──────────────────────────────────────────────────────────────

def find_qualifying_nodes(data, all_deg, deg_min: int, deg_max: int) -> list[dict]:
    """Return test nodes in [deg_min, deg_max] that have ≥1 same-class training
    neighbour whose degree is strictly greater than the test node's own degree.

    Returns a list of dicts:
        node_idx, degree, true_label,
        qualifying_train_neighbors: list of {node_idx, degree}
    """
    N          = data.num_nodes
    y          = data.y.cpu()
    train_mask = data.train_mask.cpu()
    test_mask  = data.test_mask.cpu()
    deg        = all_deg.cpu()

    src, dst = data.edge_index[0].cpu(), data.edge_index[1].cpu()

    # Build adjacency: dst → list of src (incoming edges = what gets aggregated)
    adj = [[] for _ in range(N)]
    for u, v in zip(src.tolist(), dst.tolist()):
        adj[v].append(u)

    test_indices = test_mask.nonzero(as_tuple=False).view(-1).tolist()
    results = []

    for node in test_indices:
        node_deg   = int(deg[node].item())
        if not (deg_min <= node_deg <= deg_max):
            continue

        true_lbl = int(y[node].item())

        qualifying = []
        for nb in adj[node]:
            if nb == node:
                continue
            if not train_mask[nb].item():
                continue
            if int(y[nb].item()) != true_lbl:
                continue
            nb_deg = int(deg[nb].item())
            if nb_deg > node_deg:
                qualifying.append({"node_idx": nb, "degree": nb_deg})

        if qualifying:
            qualifying.sort(key=lambda d: d["degree"], reverse=True)
            results.append({
                "node_idx":                   node,
                "degree":                     node_deg,
                "true_label":                 true_lbl,
                "qualifying_train_neighbors": qualifying,
            })

    return results


def run_analysis(cfg, deg_min: int, deg_max: int, device: torch.device):
    data = load_dataset(cfg["dataset"])

    split = cfg.get("split", "random")
    if split == "random":
        set_seed(cfg.get("seed", 42))
    data = apply_split(data, split, cfg["dataset"])
    data = data.to(device)

    all_deg = graph_degree(data.edge_index[1], data.num_nodes).cpu()
    k_hops  = cfg["model"]["num_layers"] - 1

    log.info("Training model (%s, %d layers)...",
             cfg["model"]["name"], cfg["model"]["num_layers"])
    set_seed(cfg.get("seed", 42))
    pred, model = train_model(data, cfg, device)

    log.info("Finding qualifying test nodes in degree range [%d, %d]...",
             deg_min, deg_max)
    qualifying = find_qualifying_nodes(data, all_deg, deg_min, deg_max)

    if not qualifying:
        log.info("No test nodes found matching the condition in degree [%d, %d].",
                 deg_min, deg_max)
        return

    log.info("Found %d qualifying node(s):", len(qualifying))
    for q in qualifying:
        nb_str = ", ".join(
            f"node {nb['node_idx']} (deg={nb['degree']})"
            for nb in q["qualifying_train_neighbors"]
        )
        correct = int(pred[q["node_idx"]].item()) == q["true_label"]
        log.info(
            "  node %d  deg=%d  label=%d  %s  — higher-deg same-class train neighbors: %s",
            q["node_idx"], q["degree"], q["true_label"],
            "CORRECT" if correct else "MISCLASSIFIED",
            nb_str,
        )

    log.info("")
    log.info("=" * 70)
    log.info("Running influence analysis for each qualifying node...")
    log.info("=" * 70)

    y         = data.y.cpu()
    train_idx = data.train_mask.cpu().nonzero(as_tuple=False).view(-1).tolist()
    train_set = set(train_idx)

    for q in qualifying:
        node = q["node_idx"]
        log.info("")
        log.info("─" * 60)
        correct = int(pred[node].item()) == q["true_label"]
        log.info(
            "Node %d | degree=%d | label=%d | pred=%d | %s",
            node, q["degree"], q["true_label"],
            int(pred[node].item()),
            "CORRECT" if correct else "MISCLASSIFIED",
        )
        log.info(
            "Higher-deg same-class 1-hop train neighbors: %s",
            ", ".join(
                f"node {nb['node_idx']} (deg={nb['degree']})"
                for nb in q["qualifying_train_neighbors"]
            ),
        )
        result = _analyse_node(
            model, data, pred, node, k_hops, train_set, y, all_deg
        )
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
        log.info("  Top neighbors by influence:")
        for nb in result["neighbors"][:10]:
            if nb["type"] == "non_train":
                continue
            log.info(
                "    [%s] node %-5d  deg=%-4d  hop=%-2d  norm=%.4f",
                nb["type"],
                nb["node_idx"],
                nb["degree"],
                nb["hop_distance"],
                nb["influence_norm"],
            )


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Analyse influence for test nodes connected to higher-degree "
                    "same-class training neighbours, within a given degree group."
    )
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--device", default=None,
                        help="Device override, e.g. cuda:0 or cpu")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--degree", type=int,
                       help="Exact degree value (shorthand for --degree-min=D --degree-max=D)")
    group.add_argument("--degree-min", type=int,
                       help="Lower bound of degree range (inclusive)")
    parser.add_argument("--degree-max", type=int, default=None,
                        help="Upper bound of degree range (inclusive); "
                             "required when --degree-min is used")
    args = parser.parse_args()

    if args.degree is not None:
        deg_min = deg_max = args.degree
    else:
        if args.degree_max is None:
            parser.error("--degree-max is required when --degree-min is used")
        deg_min, deg_max = args.degree_min, args.degree_max
        if deg_min > deg_max:
            parser.error("--degree-min must be ≤ --degree-max")

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    import os
    model_name   = cfg["model"]["name"]
    dataset_name = cfg["dataset"]["name"]
    model_cfg_path = os.path.join("configs", f"{model_name}_{dataset_name}.yaml")
    if os.path.exists(model_cfg_path):
        with open(model_cfg_path) as f:
            model_override = yaml.safe_load(f)
        cfg = _deep_merge(cfg, model_override)

    device = torch.device(
        args.device if args.device
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    log.info("Device: %s", device)
    log.info("Degree range: [%d, %d]", deg_min, deg_max)

    run_analysis(cfg, deg_min, deg_max, device)


if __name__ == "__main__":
    main()
