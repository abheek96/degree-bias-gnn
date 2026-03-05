import argparse
import copy
import logging
import os
import random
from datetime import datetime

import numpy as np
import torch
import yaml
from tqdm import tqdm
from torch_geometric.utils import degree as graph_degree

from dataset import load_dataset
from dataset_utils import apply_split
from logger import setup_logger
from plot_utils import get_accuracy_deg, plot_acc_vs_degree, plot_dist_vs_degree, plot_combined_vs_degree, plot_amp_dmp_vs_degree
from utils import compute_distances_to_train, get_distance_deg, get_amp_deg, get_dmp_deg, get_node_het
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
    best_state = copy.deepcopy(model.state_dict())
    patience_counter = 0
    patience = train_cfg.get("patience", 0)

    epoch_bar = tqdm(range(1, train_cfg["epochs"] + 1), desc=f"Run {run_id}", leave=False)
    for epoch in epoch_bar:
        loss = train(model, data, optimizer, criterion)
        results = evaluate(model, data)

        if results["val"] > best_val_acc:
            best_val_acc = results["val"]
            best_test_acc = results["test"]
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        epoch_bar.set_postfix(loss=f"{loss:.4f}", val=f"{results['val']:.4f}", test=f"{results['test']:.4f}")

        if patience > 0 and patience_counter >= patience:
            epoch_bar.close()
            log.info("  [Run %d] Early stopping at epoch %d", run_id, epoch)
            break

    # Restore best checkpoint and produce predictions for the full graph
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pred = model(data.x, data.edge_index).argmax(dim=1)

    return best_val_acc, best_test_acc, pred


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

    # Degree is a fixed property of the graph — compute once for test nodes
    test_deg = graph_degree(data.edge_index[1], data.num_nodes)[data.test_mask].cpu()

    # Structural distances are graph-fixed — compute once before the run loop
    dist_to_train, dist_to_same_class = compute_distances_to_train(data)
    dist_deg_data = get_distance_deg(
        test_deg, dist_to_train, dist_to_same_class, num_nodes=data.num_nodes
    )

    # AMP and DMP are graph-fixed — compute once before the run loop.
    # amp_coeff / dmp_coeff set the k-hop neighbourhood radius.
    amp_coeff = cfg["dataset"].get("amp_coeff", 1)
    dmp_coeff = cfg["dataset"].get("dmp_coeff", 1)
    log.info("AMP neighbourhood: %d hop(s)  |  DMP neighbourhood: %d hop(s)",
             amp_coeff, dmp_coeff)

    # AMP: heterogeneity over k-hop neighbourhood for each test node
    node_het = get_node_het(data, k=amp_coeff)             # FloatTensor [num_nodes]
    amp_deg_data = get_amp_deg(test_deg, node_het[data.test_mask.cpu()])

    # DMP-k: a test node is DMP if no same-class training node exists within
    # dmp_coeff hops — equivalent to dist_to_same_class > dmp_coeff.
    # dist_to_same_class is already computed above for all test nodes.
    node_dmp_k = (dist_to_same_class > dmp_coeff).numpy()
    dmp_deg_data = get_dmp_deg(test_deg, node_dmp_k)

    val_accs, test_accs = [], []
    deg_acc_results = []
    run_labels = []

    for i in tqdm(range(1, num_runs + 1), desc="Runs"):
        seed = base_seed + i - 1
        set_seed(seed)  # only affects model initialisation
        run_name = f"run{i:02d}_seed{seed}"
        setup_logger(log_dir=exec_dir, run_name=run_name)
        log.info("Experiment: %s", exec_name)
        log.info("Config: %s", cfg)
        log.info("=== Run %d/%d (seed=%d) ===", i, num_runs, seed)
        val_acc, test_acc, pred = run(data, cfg, i, device)
        val_accs.append(val_acc)
        test_accs.append(test_acc)
        log.info("Best Val: %.4f  Test: %.4f", val_acc, test_acc)

        deg_acc_results.append(
            get_accuracy_deg(test_deg, pred[data.test_mask], data.y[data.test_mask])
        )
        run_labels.append(run_name)

    val_mean, val_std = np.mean(val_accs), np.std(val_accs)
    test_mean, test_std = np.mean(test_accs), np.std(test_accs)

    setup_logger(log_dir=exec_dir, run_name="summary")
    log.info("Results over %d runs:", num_runs)
    log.info("  Val:  %.4f +/- %.4f", val_mean, val_std)
    log.info("  Test: %.4f +/- %.4f", test_mean, test_std)

    plot_cfg = cfg.get("plot", {})
    if plot_cfg.get("acc_vs_degree", False):
        plot_acc_vs_degree(
            deg_acc_results,
            cfg,
            save_dir=exec_dir if plot_cfg.get("save", True) else None,
            show=plot_cfg.get("show", False),
        )

    if plot_cfg.get("acc_vs_distance", False):
        plot_combined_vs_degree(
            deg_acc_results,
            dist_deg_data,
            cfg,
            save_dir=exec_dir if plot_cfg.get("save", True) else None,
            show=plot_cfg.get("show", False),
            run_labels=run_labels,
        )

    if plot_cfg.get("acc_vs_amp", False) or plot_cfg.get("acc_vs_dmp", False):
        plot_amp_dmp_vs_degree(
            amp_deg_data,
            dmp_deg_data,
            cfg,
            save_dir=exec_dir if plot_cfg.get("save", True) else None,
            show=plot_cfg.get("show", False),
        )


if __name__ == "__main__":
    main()
