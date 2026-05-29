"""
checkpoint_utils.py — Shared helpers for loading / training models in analysis scripts.

Extracted from analyse_hop_influence.py so that other analysis scripts can import
model-loading utilities without depending on a CLI entry point.
"""

import copy
import glob
import logging
import os
import random

import numpy as np
import torch
import yaml

from dataset_utils import load_or_create_split
from models import get_model
from train import train
from test import evaluate

log = logging.getLogger(__name__)


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
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
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


def load_cfg(config_path: str) -> dict:
    """Load config.yaml and auto-merge the per-(model, dataset) override if present."""
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    model_cfg_path = os.path.join(
        "configs", f"{cfg['model']['name']}_{cfg['dataset']['name']}.yaml"
    )
    if os.path.exists(model_cfg_path):
        with open(model_cfg_path) as f:
            cfg = _deep_merge(cfg, yaml.safe_load(f))
    return cfg


def load_data(cfg, device):
    """Load (or create) the cached split and move to device."""
    cache_dir = cfg.get("dataset_cache_dir", "dataset_cache")
    data = load_or_create_split(
        cfg["dataset"], cfg.get("split", "random"), cfg.get("seed", 42), cache_dir,
    )
    return data.to(device)
