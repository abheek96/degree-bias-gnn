"""
analyse_degree1_reachability.py — Reachability analysis for misclassified degree-1 test nodes.

For all misclassified test nodes with degree=1, checks within the
(num_layers - 1)-hop receptive field:
  1. How many have NO training node of any class reachable.
  2. How many have NO same-class training node reachable
     (but may have diff-class training nodes).

Usage
-----
    python analyse_degree1_reachability.py
    python analyse_degree1_reachability.py --config config.yaml --device cpu
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

def run_analysis(cfg, device: torch.device):
    data = load_dataset(cfg["dataset"])

    split = cfg.get("split", "random")
    if split == "random":
        set_seed(cfg.get("seed", 42))
    data = apply_split(data, split, cfg["dataset"])
    data = data.to(device)

    N      = data.num_nodes
    k_hops = cfg["model"]["num_layers"] - 1

    all_deg    = graph_degree(data.edge_index[1], N).cpu()
    y          = data.y.cpu()
    train_mask = data.train_mask.cpu()
    test_mask  = data.test_mask.cpu()

    train_set  = set(train_mask.nonzero(as_tuple=False).view(-1).tolist())
    test_idx   = test_mask.nonzero(as_tuple=False).view(-1).tolist()

    log.info("Training model (%s, %d layers, k_hops=%d)...",
             cfg["model"]["name"], cfg["model"]["num_layers"], k_hops)
    set_seed(cfg.get("seed", 42))
    pred, _ = train_model(data, cfg, device)
    pred = pred.cpu()

    # Collect misclassified degree-1 test nodes
    misclassified_deg1 = [
        node for node in test_idx
        if int(all_deg[node].item()) == 1
        and int(pred[node].item()) != int(y[node].item())
    ]

    total_deg1 = sum(
        1 for node in test_idx if int(all_deg[node].item()) == 1
    )

    log.info("")
    log.info("Degree-1 test nodes : %d total, %d misclassified (%.1f%%)",
             total_deg1, len(misclassified_deg1),
             100 * len(misclassified_deg1) / total_deg1 if total_deg1 else 0)
    log.info("Receptive field radius: %d hop(s)", k_hops)
    log.info("")

    no_train         = []   # no training node of any class reachable
    no_same_train    = []   # training nodes reachable, but none same-class

    for node in misclassified_deg1:
        true_lbl  = int(y[node].item())
        neighbors = _khop_neighbors(data.edge_index, node, k_hops, N)

        train_in_field      = [n for n in neighbors if n in train_set]
        same_train_in_field = [n for n in train_in_field
                               if int(y[n].item()) == true_lbl]

        if not train_in_field:
            no_train.append(node)
        elif not same_train_in_field:
            no_same_train.append(node)

    n_misc = len(misclassified_deg1)

    log.info("── No training node reachable within %d hop(s) ──", k_hops)
    log.info("  Count : %d / %d  (%.1f%% of misclassified degree-1 nodes)",
             len(no_train), n_misc, 100 * len(no_train) / n_misc if n_misc else 0)
    log.info("  Nodes : %s", no_train if no_train else "none")

    log.info("")
    log.info("── Training node(s) reachable but NO same-class within %d hop(s) ──",
             k_hops)
    log.info("  Count : %d / %d  (%.1f%% of misclassified degree-1 nodes)",
             len(no_same_train), n_misc,
             100 * len(no_same_train) / n_misc if n_misc else 0)
    log.info("  Nodes : %s", no_same_train if no_same_train else "none")

    log.info("")
    log.info("── Summary ──")
    log.info("  No training node at all   : %d / %d", len(no_train), n_misc)
    log.info("  No same-class train node  : %d / %d",
             len(no_train) + len(no_same_train), n_misc)
    log.info("  Has same-class train node : %d / %d",
             n_misc - len(no_train) - len(no_same_train), n_misc)


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Reachability analysis for misclassified degree-1 test nodes."
    )
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--device", default=None,
                        help="Device override, e.g. cuda:0 or cpu")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    import os
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

    run_analysis(cfg, device)


if __name__ == "__main__":
    main()
