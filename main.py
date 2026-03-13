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
from plot_utils import get_accuracy_deg, plot_acc_vs_degree, plot_combined_vs_degree, plot_acc_vs_khop_degree, plot_acc_vs_degree_by_layers, plot_acc_trend_by_degree, plot_purity_vs_degree, plot_purity_delta_by_degree, plot_labelling_ratio_vs_degree, plot_acc_and_labelling_ratio_vs_degree, plot_spl_vs_degree
from utils import compute_distances_to_train, get_distance_deg, get_khop_degree, get_node_purity, get_labelling_ratio, get_avg_spl_to_train, get_avg_spl_to_same_class_train
from train import train
from test import evaluate

log = logging.getLogger(__name__)


def _deep_merge(base, override):
    """Recursively merge override into base; override values win. Lists are replaced."""
    merged = copy.deepcopy(base)
    for key, val in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(val, dict):
            merged[key] = _deep_merge(merged[key], val)
        else:
            merged[key] = copy.deepcopy(val)
    return merged


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
    best_loss = float("nan")
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
            best_loss = loss
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        epoch_bar.set_postfix(loss=f"{loss:.4f}", val=f"{results['val']:.4f}", test=f"{results['test']:.4f}")
        interval = max(1, train_cfg["epochs"] // 10)
        if epoch % interval == 0:
            log.info("  [Run %d] Epoch %d  loss=%.4f  val=%.4f  test=%.4f",
                     run_id, epoch, loss, results["val"], results["test"])

        if patience > 0 and patience_counter >= patience:
            epoch_bar.close()
            log.info("  [Run %d] Early stopping at epoch %d", run_id, epoch)
            break

    # Restore best checkpoint and produce predictions for the full graph
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pred = model(data.x, data.edge_index).argmax(dim=1)

    return best_val_acc, best_test_acc, best_loss, pred, model


def main():
    parser = argparse.ArgumentParser(description="Graph Learning Experiments")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to base config file")
    parser.add_argument("--model-config", type=str, default=None, help="Path to model-dataset config override (auto-discovered if omitted)")
    parser.add_argument("--device", type=str, default=None, help="Device override (e.g. cuda:0, cpu)")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # Auto-discover or explicitly load a model-dataset config and deep-merge it
    model_cfg_path = args.model_config
    if model_cfg_path is None:
        model_name  = cfg["model"]["name"]
        dataset_name = cfg["dataset"]["name"]
        model_cfg_path = os.path.join("configs", f"{model_name}_{dataset_name}.yaml")

    if os.path.exists(model_cfg_path):
        with open(model_cfg_path) as f:
            model_cfg = yaml.safe_load(f)
        cfg = _deep_merge(cfg, model_cfg)
        log.info("Loaded model-dataset config: %s", model_cfg_path)
    else:
        log.info("No model-dataset config found at %s; using base config only", model_cfg_path)

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
    all_deg  = graph_degree(data.edge_index[1], data.num_nodes).cpu()

    # Labelling ratio is graph-fixed — compute once, sliced to test nodes
    has_labeled_neighbor = get_labelling_ratio(data)[data.test_mask.cpu()]

    # Average SPL to training nodes is graph-fixed — compute once, sliced to test nodes
    avg_spl            = get_avg_spl_to_train(data)[data.test_mask.cpu()]
    avg_spl_same_class = get_avg_spl_to_same_class_train(data)[data.test_mask.cpu()]

    # Structural distances are graph-fixed — compute once before the run loop
    dist_to_train, dist_to_same_class = compute_distances_to_train(data)
    dist_deg_data = get_distance_deg(
        test_deg, dist_to_train, dist_to_same_class, num_nodes=data.num_nodes
    )

    plot_cfg = cfg.get("plot", {})

    # Apply per-model hyperparameter overrides for the main model
    _main_model = cfg["model"]["name"]
    for hp_key, hp_val in plot_cfg.get("model_hyperparams", {}).get(_main_model, {}).items():
        if hp_key in ("lr", "patience"):
            cfg["train"][hp_key] = hp_val
        elif hp_key in ("hidden_dim", "dropout"):
            cfg["model"][hp_key] = hp_val

    # k-hop degree is graph-fixed — compute once before the run loop
    khop_k        = plot_cfg.get("khop_k", 2)
    khop_deg_test = get_khop_degree(data, k=khop_k)[data.test_mask].cpu()

    # Neighborhood purity is graph-fixed — compute once per k value
    purity_k_max = plot_cfg.get("purity_k_max", 4)
    purity_by_k  = {
        k: get_node_purity(data, k=k)
        for k in range(1, purity_k_max + 1)
    } if plot_cfg.get("purity_vs_degree", False) else {}

    val_accs, test_accs = [], []
    deg_acc_results  = []
    khop_acc_results = []
    run_labels = []

    for i in tqdm(range(1, num_runs + 1), desc="Runs"):
        seed = base_seed + i - 1
        set_seed(seed)  # only affects model initialisation
        run_name = f"run{i:02d}_seed{seed}"
        setup_logger(log_dir=exec_dir, run_name=run_name)
        log.info("=== Run %d/%d (seed=%d) ===", i, num_runs, seed)
        val_acc, test_acc, best_loss, pred, model = run(data, cfg, i, device)
        val_accs.append(val_acc)
        test_accs.append(test_acc)
        log.info("Best Val: %.4f  Test: %.4f  Loss: %.4f", val_acc, test_acc, best_loss)

        deg_acc_results.append(
            get_accuracy_deg(test_deg, pred[data.test_mask], data.y[data.test_mask])
        )
        khop_acc_results.append(
            get_accuracy_deg(khop_deg_test, pred[data.test_mask], data.y[data.test_mask])
        )
        run_labels.append(run_name)

    val_mean, val_std = np.mean(val_accs), np.std(val_accs)
    test_mean, test_std = np.mean(test_accs), np.std(test_accs)

    setup_logger(log_dir=exec_dir, run_name="summary")
    log.info("Dataset: %s  |  Model: %s  |  Layers: %d  |  Split: %s",
             cfg["dataset"]["name"], cfg["model"]["name"],
             cfg["model"]["num_layers"], cfg.get("split", "random"))
    log.info(model)
    log.info("Results over %d runs:", num_runs)
    log.info("  Val:  %.4f +/- %.4f", val_mean, val_std)
    log.info("  Test: %.4f +/- %.4f", test_mean, test_std)

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

    if plot_cfg.get("acc_vs_khop_degree", False):
        plot_acc_vs_khop_degree(
            khop_acc_results,
            cfg,
            k=khop_k,
            save_dir=exec_dir if plot_cfg.get("save", True) else None,
            show=plot_cfg.get("show", False),
        )

    if plot_cfg.get("acc_vs_degree_by_layers", False):
        import copy as _copy
        layer_values       = plot_cfg.get("layer_values", [1, 2, 3, 4, 5])
        model_layer_values = plot_cfg.get("model_layer_values", {})
        model_hyperparams  = plot_cfg.get("model_hyperparams", {})
        compare_models     = plot_cfg.get("compare_models", [cfg["model"]["name"]])
        results_by_label = {}
        for model_name in compare_models:
            for L in model_layer_values.get(model_name, layer_values):
                label = f"{model_name} L={L}"
                log.info("=== Training %s ===", label)
                run_cfg = _copy.deepcopy(cfg)
                run_cfg["model"]["name"]       = model_name
                run_cfg["model"]["num_layers"] = L
                for hp_key, hp_val in model_hyperparams.get(model_name, {}).items():
                    if hp_key in ("lr", "patience"):
                        run_cfg["train"][hp_key] = hp_val
                    elif hp_key in ("hidden_dim", "dropout"):
                        run_cfg["model"][hp_key] = hp_val
                label_deg_results = []
                for i in tqdm(range(1, num_runs + 1), desc=label):
                    seed = base_seed + i - 1
                    set_seed(seed)
                    _, _, _, pred_L, _ = run(data, run_cfg, i, device)
                    label_deg_results.append(
                        get_accuracy_deg(test_deg, pred_L[data.test_mask], data.y[data.test_mask])
                    )
                results_by_label[label] = label_deg_results
        save_dir = exec_dir if plot_cfg.get("save", True) else None
        plot_acc_vs_degree_by_layers(
            results_by_label, cfg, save_dir=save_dir, show=plot_cfg.get("show", False),
        )
        plot_acc_trend_by_degree(
            results_by_label, cfg, save_dir=save_dir, show=plot_cfg.get("show", False),
        )

    if plot_cfg.get("spl_vs_degree", False):
        save_dir = exec_dir if plot_cfg.get("save", True) else None
        plot_spl_vs_degree(
            test_deg, avg_spl, cfg,
            save_dir=save_dir, show=plot_cfg.get("show", False),
        )
        plot_spl_vs_degree(
            test_deg, avg_spl_same_class, cfg,
            save_dir=save_dir, show=plot_cfg.get("show", False), same_class=True,
        )

    if plot_cfg.get("labelling_ratio", False):
        plot_labelling_ratio_vs_degree(
            test_deg, has_labeled_neighbor, cfg,
            save_dir=exec_dir if plot_cfg.get("save", True) else None,
            show=plot_cfg.get("show", False),
        )
        plot_acc_and_labelling_ratio_vs_degree(
            deg_acc_results, test_deg, has_labeled_neighbor, cfg,
            save_dir=exec_dir if plot_cfg.get("save", True) else None,
            show=plot_cfg.get("show", False),
        )

    if plot_cfg.get("purity_vs_degree", False):
        save_dir = exec_dir if plot_cfg.get("save", True) else None
        for k, purity_all in purity_by_k.items():
            plot_purity_vs_degree(
                all_deg, purity_all, cfg, k,
                save_dir=save_dir,
                show=plot_cfg.get("show", False),
            )
        if len(purity_by_k) > 1:
            plot_purity_delta_by_degree(
                all_deg, purity_by_k, cfg,
                save_dir=save_dir,
                show=plot_cfg.get("show", False),
            )


if __name__ == "__main__":
    main()
