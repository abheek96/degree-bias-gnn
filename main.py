import argparse
import logging
import os
import random
from datetime import datetime

import numpy as np
import torch
import yaml
from tqdm import tqdm

from dataset import load_dataset
from dataset_utils import apply_split
from logger import setup_logger
from train import train
from test import evaluate

log = logging.getLogger(__name__)


def make_exec_name(cfg) -> str:
    dataset = cfg["dataset"]["name"]
    model = cfg["model"]["name"]
    split = cfg.get("split", "random")
    cc = "CC" if cfg["dataset"].get("use_cc", False) else "noCC"
    ts = datetime.now().strftime("%d%b%Y_%H%M")  # e.g. 02Mar2026_1430
    return f"{dataset}_{model}_{split}_{cc}_{ts}"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def run(data, cfg, run_id, device):
    from models import get_model

    model_cfg = cfg["model"]
    model = get_model(
        model_cfg["name"],
        in_dim=data.num_node_features,
        hidden_dim=model_cfg["hidden_dim"],
        out_dim=int(data.y.max().item()) + 1,
        num_layers=model_cfg["num_layers"],
        dropout=model_cfg["dropout"],
    ).to(device)

    train_cfg = cfg["train"]
    optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg["lr"], weight_decay=float(train_cfg["weight_decay"]))
    criterion = torch.nn.CrossEntropyLoss()

    best_val_acc = 0.0
    best_test_acc = 0.0
    patience_counter = 0
    patience = train_cfg.get("patience", 0)

    epoch_bar = tqdm(range(1, train_cfg["epochs"] + 1), desc=f"Run {run_id}", leave=False)
    for epoch in epoch_bar:
        loss = train(model, data, optimizer, criterion)
        results = evaluate(model, data)

        if results["val"] > best_val_acc:
            best_val_acc = results["val"]
            best_test_acc = results["test"]
            patience_counter = 0
        else:
            patience_counter += 1

        epoch_bar.set_postfix(loss=f"{loss:.4f}", val=f"{results['val']:.4f}", test=f"{results['test']:.4f}")

        if patience > 0 and patience_counter >= patience:
            epoch_bar.close()
            log.info("  [Run %d] Early stopping at epoch %d", run_id, epoch)
            break

    return best_val_acc, best_test_acc


def main():
    parser = argparse.ArgumentParser(description="Graph Learning Experiments")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    parser.add_argument("--device", type=str, default=None, help="Device override (e.g. cuda:0, cpu)")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    results_dir = cfg.get("results_dir", "./results")
    exec_name = make_exec_name(cfg)
    exec_dir = os.path.join(results_dir, exec_name)
    os.makedirs(exec_dir, exist_ok=True)

    # Console-only logging during setup (before any per-run file handler exists)
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    log.info("Experiment: %s", exec_name)
    log.info("Device: %s", device)

    base_seed = cfg.get("seed", 42)
    num_runs = cfg.get("num_runs", 1)
    split = cfg.get("split", "random")

    # Load dataset once. For random splits, fix the partition with base_seed
    # so all runs share the same train/val/test nodes. For public splits with
    # use_cc=True, apply_split logs the surviving mask counts after CC filtering.
    data = load_dataset(cfg["dataset"])

    if split == "random":
        set_seed(base_seed)
    data = apply_split(data, split, cfg["dataset"])
    data = data.to(device)

    val_accs, test_accs = [], []

    for i in tqdm(range(1, num_runs + 1), desc="Runs"):
        seed = base_seed + i - 1
        set_seed(seed)  # only affects model initialisation
        run_name = f"run{i:02d}_seed{seed}"
        setup_logger(log_dir=exec_dir, run_name=run_name)
        log.info("Experiment: %s", exec_name)
        log.info("Config: %s", cfg)
        log.info("=== Run %d/%d (seed=%d) ===", i, num_runs, seed)
        val_acc, test_acc = run(data, cfg, i, device)
        val_accs.append(val_acc)
        test_accs.append(test_acc)
        log.info("Best Val: %.4f  Test: %.4f", val_acc, test_acc)

    val_mean, val_std = np.mean(val_accs), np.std(val_accs)
    test_mean, test_std = np.mean(test_accs), np.std(test_accs)

    setup_logger(log_dir=exec_dir, run_name="summary")
    log.info("Results over %d runs:", num_runs)
    log.info("  Val:  %.4f +/- %.4f", val_mean, val_std)
    log.info("  Test: %.4f +/- %.4f", test_mean, test_std)


if __name__ == "__main__":
    main()
