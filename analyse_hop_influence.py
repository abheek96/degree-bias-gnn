"""
analyse_hop_influence.py — Per-hop Jacobian-L1 influence breakdown for a
specific test node.

For each hop i in ``[0, num_layers - 1]``, reports:
  * total influence at hop i (sum over all nodes exactly i hops away)
  * raw influence from same-class training nodes at hop i
  * raw influence from diff-class training nodes at hop i
  * fractions (same_inf / total_inf, diff_inf / total_inf) at hop i
  * cardinalities: #same-class train, #diff-class train, #non-train

Table header mirrors the aggregate influence analysis in ``influence._analyse_node``
(adds ``true``, ``pred``, ``degree``, per-hop-field counts).

Model source
------------
  --checkpoint PATH     load a saved state_dict (no retraining)
  --run N               shorthand; loads ``results/{exec}/checkpoints/
                        run{N:02d}_seed{base_seed + N - 1}.pt``
  (neither)             retrain from scratch with the config's seed

Usage
-----
    python analyse_hop_influence.py --node-idx 1362
    python analyse_hop_influence.py --node-idx 1362 2210 \\
           --checkpoint results/GCN_Cora_.../checkpoints/run02_seed43.pt
"""

import argparse
import copy
import logging
import os
import random
import sys

import numpy as np
import torch
import yaml
from torch_geometric.utils import degree as graph_degree

from dataset_utils import load_or_create_split
from influence import (
    _khop_distances,
    influence_distribution,
    k_hop_subsets_exact,
)
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


def _build_model(cfg, data, device):
    model_cfg = cfg["model"]
    _standard_keys = {"name", "hidden_dim", "num_layers", "dropout", "weights_path"}
    extra_kwargs = {k: v for k, v in model_cfg.items() if k not in _standard_keys}
    return get_model(
        model_cfg["name"],
        in_dim=data.num_node_features,
        hidden_dim=model_cfg["hidden_dim"],
        out_dim=int(data.y.max().item()) + 1,
        num_layers=model_cfg["num_layers"],
        dropout=model_cfg["dropout"],
        **extra_kwargs,
    ).to(device)


def train_model(data, cfg, device):
    """Train one model run from scratch, return (pred [N], model)."""
    model = _build_model(cfg, data, device)
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
        train(model, data, optimizer, criterion)
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


def load_from_checkpoint(cfg, data, device, checkpoint_path: str):
    """Rebuild the model from cfg and load a saved state_dict.

    Accepts both raw state_dict files and the dict format saved by main.py
    (``{"model_state": ..., "pred": ..., "seed": ...}``). If ``pred`` is in
    the checkpoint it is used directly; otherwise inference is re-run.
    """
    ckpt = torch.load(checkpoint_path, map_location=device)
    state = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt

    model = _build_model(cfg, data, device)
    model.load_state_dict(state)
    model.eval()

    if isinstance(ckpt, dict) and "pred" in ckpt:
        pred = ckpt["pred"].to(device)
    else:
        with torch.no_grad():
            pred = model(data.x, data.edge_index).argmax(dim=1)

    log.info("Loaded checkpoint: %s", checkpoint_path)
    return pred, model


def _resolve_run_checkpoint(cfg, run_id: int) -> str | None:
    """Locate the checkpoint for ``run_id`` under the latest matching exec_dir.

    main.make_exec_name bakes a timestamp into the directory, so we can't
    reconstruct the exact name. Instead: glob under ``results_dir`` for
    ``{dataset}_{model}_{split}_{CC|noCC}_*`` and pick the most recent.
    """
    import glob

    results_dir = cfg.get("results_dir", "./results")
    dataset = cfg["dataset"]["name"]
    model   = cfg["model"]["name"]
    split   = cfg.get("split", "random")
    cc      = "CC" if cfg["dataset"].get("use_cc", False) else "noCC"
    pattern = os.path.join(results_dir, f"{dataset}_{model}_{split}_{cc}_*")

    candidates = sorted(glob.glob(pattern), key=os.path.getmtime, reverse=True)
    base_seed = cfg.get("seed", 42)
    seed = base_seed + run_id - 1
    fname = f"run{run_id:02d}_seed{seed}.pt"

    for exec_dir in candidates:
        path = os.path.join(exec_dir, "checkpoints", fname)
        if os.path.exists(path):
            log.info("Resolved --run %d → %s (exec_dir=%s)",
                     run_id, path, os.path.basename(exec_dir))
            return path
    return None


# ── per-hop analysis ──────────────────────────────────────────────────────────

def analyse_node_per_hop(model, data, pred, node_x: int, k_hops: int,
                          train_set: set, y, all_deg):
    """Compute and log the per-hop influence table for one node."""
    N        = data.num_nodes
    degree   = int(all_deg[node_x].item())
    true_lbl = int(y[node_x].item())
    pred_lbl = int(pred[node_x].item())
    status   = "correct" if pred_lbl == true_lbl else "misclassified"

    hop_dist       = _khop_distances(data.edge_index, node_x, k_hops, N)
    train_in_field = [t for t in hop_dist if t in train_set]
    same_class_all = {t for t in train_in_field if int(y[t].item()) == true_lbl}
    diff_class_all = {t for t in train_in_field if int(y[t].item()) != true_lbl}

    log.info(
        "influence (per-hop): node %d  degree=%d  true=%d  pred=%d  (%s)",
        node_x, degree, true_lbl, pred_lbl, status,
    )
    log.info(
        "  receptive field (k=%d): same_train=%d  diff_train=%d",
        k_hops, len(same_class_all), len(diff_class_all),
    )

    I_x = influence_distribution(model, data, node_x, k_hops)
    hop_subsets = k_hop_subsets_exact(
        node_x, k_hops, data.edge_index, N, I_x.device,
    )

    import math
    from prettytable import PrettyTable

    focal_deg = degree

    def _deg_weight_list(nodes):
        if not nodes:
            return "—"
        tuples = sorted(
            (int(all_deg[n].item()),
             round(1.0 / math.sqrt((int(all_deg[n].item()) + 1) * (focal_deg + 1)), 4))
            for n in nodes
        )
        return str(tuples)

    table = PrettyTable()
    table.field_names = [
        "hop", "|S_i|", "total_inf", "same_inf", "diff_inf",
        "same/tot", "diff/tot", "#same_tr", "#diff_tr", "#non_tr",
        "purity", "same (deg, w)", "diff (deg, w)",
    ]
    table.align = "r"
    table.align["same (deg, w)"] = "l"
    table.align["diff (deg, w)"] = "l"

    rows = []
    for i, S_i in enumerate(hop_subsets):
        S_set = set(S_i.tolist())
        size  = len(S_set)
        total = float(I_x[S_i].sum().item()) if size > 0 else 0.0

        same_nodes = list(S_set & same_class_all)
        diff_nodes = list(S_set & diff_class_all)
        n_same     = len(same_nodes)
        n_diff     = len(diff_nodes)
        n_non      = size - n_same - n_diff

        same_inf  = float(I_x[same_nodes].sum().item()) if same_nodes else 0.0
        diff_inf  = float(I_x[diff_nodes].sum().item()) if diff_nodes else 0.0
        frac_same = same_inf / total if total > 0 else 0.0
        frac_diff = diff_inf / total if total > 0 else 0.0

        same_label = sum(1 for n in S_set if int(y[n].item()) == true_lbl)
        purity = same_label / size if size > 0 else float("nan")

        same_degs = _deg_weight_list(same_nodes)
        diff_degs = _deg_weight_list(diff_nodes)

        table.add_row([
            i, size,
            f"{total:.4e}", f"{same_inf:.4e}", f"{diff_inf:.4e}",
            f"{frac_same:.4f}", f"{frac_diff:.4f}",
            n_same, n_diff, n_non,
            f"{purity:.3f}",
            same_degs, diff_degs,
        ])
        rows.append({
            "hop": i, "size": size, "total_inf": total,
            "same_inf": same_inf, "diff_inf": diff_inf,
            "frac_same": frac_same, "frac_diff": frac_diff,
            "n_same_train": n_same, "n_diff_train": n_diff, "n_non_train": n_non,
            "purity": purity,
            "same_degs": same_degs, "diff_degs": diff_degs,
        })

    for line in table.get_string().splitlines():
        log.info(line)

    return {
        "node_idx": node_x, "degree": degree,
        "true_label": true_lbl, "pred_label": pred_lbl,
        "n_same_train_field": len(same_class_all),
        "n_diff_train_field": len(diff_class_all),
        "per_hop": rows,
    }


# ── CLI orchestration ─────────────────────────────────────────────────────────

def _load_data(cfg, device):
    cache_dir = cfg.get("dataset_cache_dir", "dataset_cache")
    data = load_or_create_split(
        cfg["dataset"], cfg.get("split", "random"), cfg.get("seed", 42), cache_dir,
    )
    return data.to(device)


def run_analysis(cfg, node_ids: list, device, checkpoint_path: str | None):
    data = _load_data(cfg, device)
    k_hops = cfg["model"]["num_layers"] - 1
    log.info("Dataset=%s  model=%s  num_layers=%d  k_hops=%d",
             cfg["dataset"]["name"], cfg["model"]["name"],
             cfg["model"]["num_layers"], k_hops)

    if checkpoint_path:
        pred, model = load_from_checkpoint(cfg, data, device, checkpoint_path)
    else:
        log.info("No checkpoint provided — training from scratch (seed=%d)",
                 cfg.get("seed", 42))
        set_seed(cfg.get("seed", 42))
        pred, model = train_model(data, cfg, device)

    pred = pred.cpu()
    N          = data.num_nodes
    all_deg    = graph_degree(data.edge_index[1], N).cpu()
    y          = data.y.cpu()
    train_mask = data.train_mask.cpu()
    test_mask  = data.test_mask.cpu()
    train_set  = set(train_mask.nonzero(as_tuple=False).view(-1).tolist())
    test_set   = set(test_mask.nonzero(as_tuple=False).view(-1).tolist())

    for node_x in node_ids:
        if node_x < 0 or node_x >= N:
            log.warning("node_idx=%d out of range [0, %d) — skipping", node_x, N)
            continue
        if node_x not in test_set:
            log.warning("node_idx=%d is not a test node (analysing anyway)", node_x)
        analyse_node_per_hop(model, data, pred, node_x, k_hops,
                             train_set, y, all_deg)


def main():
    parser = argparse.ArgumentParser(
        description="Per-hop Jacobian-L1 influence breakdown for a specific test node."
    )
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--device", default=None,
                        help="Device override, e.g. cuda:0 or cpu")
    parser.add_argument("--node-idx", type=int, nargs="+", required=True,
                        help="One or more (post-CC) test-node indices.")

    ckpt_group = parser.add_mutually_exclusive_group()
    ckpt_group.add_argument("--checkpoint", default=None,
                            help="Path to a saved model checkpoint "
                                 "(skips training).")
    ckpt_group.add_argument("--run", type=int, default=None,
                            help="Run index (1-based) whose checkpoint "
                                 "under {exec_dir}/checkpoints/ should be loaded.")
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
                f"{cfg.get('results_dir', './results')}/. Either run main.py first "
                "(with save_checkpoints: True) or pass --checkpoint explicitly."
            )

    run_analysis(cfg, args.node_idx, device, checkpoint_path)


if __name__ == "__main__":
    main()
