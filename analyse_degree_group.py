"""
analyse_degree_group.py — Influence analysis for test nodes in a degree group
that have a higher-degree same-class training node within their k-hop receptive
field (k = num_layers − 1).

Usage
-----
    python analyse_degree_group.py --degree 3
    python analyse_degree_group.py --degree-min 2 --degree-max 5
    python analyse_degree_group.py --degree 3 --config config.yaml --device cpu

For every test node in the specified degree group the script:
  1. Checks whether the node has ≥1 same-class training node within its
     k-hop receptive field whose degree is strictly greater than the test
     node's own degree.
  2. Reports whether each qualifying node is correctly classified or not.
  3. For each qualifying node, runs the exact Jacobian-based influence
     analysis and reports the same/diff-class training-node breakdown.
     For 1-hop qualifying neighbours the GCN-normalised edge weight is also
     shown; multi-hop entries report their hop distance instead.

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

import pandas as pd
from torch_geometric.nn.conv.gcn_conv import gcn_norm

from dataset import load_dataset
from dataset_utils import apply_split
from influence import _analyse_node, _khop_distances
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

def _build_edge_weight_map(edge_index, num_nodes: int) -> dict:
    """Return a dict mapping (src, dst) → GCN-normalised edge weight.

    Uses gcn_norm with add_self_loops=True, matching the default GCNConv
    behaviour (symmetric normalisation: w = 1/sqrt((deg_u+1)*(deg_v+1))).
    """
    norm_ei, norm_ew = gcn_norm(
        edge_index.cpu(), edge_weight=None, num_nodes=num_nodes,
        improved=False, add_self_loops=True, flow="source_to_target",
    )
    return {
        (int(s), int(d)): float(w)
        for s, d, w in zip(norm_ei[0].tolist(), norm_ei[1].tolist(), norm_ew.tolist())
    }


def find_qualifying_nodes(data, all_deg, pred, edge_weight_map: dict,
                          deg_min: int, deg_max: int,
                          k_hops: int = 1) -> list[dict]:
    """Return misclassified test nodes in [deg_min, deg_max] that have ≥1
    same-class training node within k_hops hops whose degree is strictly
    greater than the test node's own degree.

    Returns a list of dicts:
        node_idx, degree, true_label,
        qualifying_train_neighbors: list of {node_idx, degree, hop, edge_weight}
            edge_weight is the GCN-normalised weight for the direct edge (hop=1)
            or None for multi-hop neighbours.
    """
    N          = data.num_nodes
    y          = data.y.cpu()
    pred_cpu   = pred.cpu()
    train_mask = data.train_mask.cpu()
    test_mask  = data.test_mask.cpu()
    deg        = all_deg.cpu()
    test_indices    = test_mask.nonzero(as_tuple=False).view(-1).tolist()
    results = []

    for node in test_indices:
        node_deg = int(deg[node].item())
        if not (deg_min <= node_deg <= deg_max):
            continue

        true_lbl  = int(y[node].item())

        # skip correctly classified nodes
        if int(pred_cpu[node].item()) == true_lbl:
            continue

        hop_dist  = _khop_distances(data.edge_index, node, k_hops, N)

        qualifying = []
        for nb, hop in hop_dist.items():
            if not train_mask[nb].item():
                continue
            if int(y[nb].item()) != true_lbl:
                continue
            nb_deg = int(deg[nb].item())
            if nb_deg <= node_deg:
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


def print_khop_neighborhood(node: int, data, all_deg, pred, y,
                             edge_weight_map: dict, k_hops: int):
    """Print a table of all nodes in the k-hop receptive field of `node`.

    Columns: neighbor, degree, hop, in_train_set, same_class, correct_pred,
             edge_weight (GCN-normalised for hop=1 direct edges, else N/A).

    Sorted by hop, then same_class (True first), then in_train_set (True first),
    then degree ascending.
    """
    N        = data.num_nodes
    true_lbl = int(y[node].item())
    hop_dist = _khop_distances(data.edge_index, node, k_hops, N)

    train_mask = data.train_mask.cpu()
    rows = []
    for nb, hop in hop_dist.items():
        ew  = edge_weight_map.get((nb, node)) if hop == 1 else None
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


def run_analysis(cfg, deg_min: int, deg_max: int, device: torch.device):
    data = load_dataset(cfg["dataset"])

    split = cfg.get("split", "random")
    if split == "random":
        set_seed(cfg.get("seed", 42))
    data = apply_split(data, split, cfg["dataset"])
    data = data.to(device)

    all_deg = graph_degree(data.edge_index[1], data.num_nodes).cpu()
    k_hops  = cfg["model"]["num_layers"] - 1

    edge_weight_map = _build_edge_weight_map(data.edge_index, data.num_nodes)

    log.info("Training model (%s, %d layers)...",
             cfg["model"]["name"], cfg["model"]["num_layers"])
    set_seed(cfg.get("seed", 42))
    pred, model = train_model(data, cfg, device)

    log.info("Finding qualifying test nodes in degree range [%d, %d] (k_hops=%d)...",
             deg_min, deg_max, k_hops)
    qualifying = find_qualifying_nodes(
        data, all_deg, pred, edge_weight_map, deg_min, deg_max, k_hops=k_hops
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
    train_idx = data.train_mask.cpu().nonzero(as_tuple=False).view(-1).tolist()
    train_set = set(train_idx)

    for q in qualifying:
        node = q["node_idx"]
        log.info("")
        log.info("─" * 60)
        log.info(
            "Node %d | degree=%d | label=%d | pred=%d | MISCLASSIFIED",
            node, q["degree"], q["true_label"],
            int(pred[node].item()),
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
        log.info("  Training neighbors by influence:")
        for nb in result["neighbors"]:
            if nb["type"] == "non_train":
                continue
            ew = edge_weight_map.get((nb["node_idx"], node))
            ew_str = f"ew={ew:.6f}" if ew is not None else "ew=N/A"
            log.info(
                "    [%s] node %-5d  deg=%-4d  hop=%-2d  %s  norm=%.4f",
                nb["type"],
                nb["node_idx"],
                nb["degree"],
                nb["hop_distance"],
                ew_str,
                nb["influence_norm"],
            )

        print_khop_neighborhood(node, data, all_deg, pred, y,
                                edge_weight_map, k_hops)


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
