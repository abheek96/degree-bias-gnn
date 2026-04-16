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
from plot_utils import get_accuracy_deg, get_accuracy_class, plot_acc_vs_degree, plot_combined_vs_degree, plot_acc_vs_degree_by_layers, plot_acc_trend_by_degree, plot_purity_vs_degree, plot_purity_delta_by_degree, plot_labelling_ratio_vs_degree, plot_acc_and_labelling_ratio_vs_degree, plot_spl_vs_degree, plot_spl_combined_vs_degree, plot_influence_analysis, plot_influence_per_neighbor, plot_influence_disparity_vs_degree, plot_feature_similarity_delta_vs_degree, plot_node_similarity_analysis, plot_train_neighbor_degree_stats, plot_1hop_train_deg_vs_accuracy, plot_neighborhood_cardinality_vs_degree, plot_class_accuracy_and_degree, plot_train_degree_distribution
from influence import compute_influence_analysis, compute_influence_disparity_all
from utils import compute_distances_to_train, get_distance_deg, get_node_purity, get_labelling_ratio, get_class_labelling_ratio, get_avg_spl_to_train, get_avg_spl_to_same_class_train, get_training_neighbor_degree_stats, get_khop_cardinality, get_feature_similarity_delta, compute_node_similarity_analysis
from train import train
from test import evaluate
from models.gcn import inspect_node_aggregation

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
    _standard_keys = {"name", "hidden_dim", "num_layers", "dropout", "weights_path"}
    extra_kwargs = {k: v for k, v in model_cfg.items() if k not in _standard_keys}

    # If a weights file is provided, infer layer_norm / batch_norm from the
    # checkpoint keys so the model is built with the right architecture before
    # load_state_dict is called.
    weights_path = model_cfg.get("weights_path") or None
    if weights_path:
        _ckpt_keys = set(torch.load(weights_path, map_location="cpu").keys())
        extra_kwargs["layer_norm"] = any(k.startswith("lns.") for k in _ckpt_keys)
        extra_kwargs["batch_norm"] = any(k.startswith("bns.") for k in _ckpt_keys)
        log.info("  [Run %d] Inferred from checkpoint — layer_norm=%s  batch_norm=%s",
                 run_id, extra_kwargs["layer_norm"], extra_kwargs["batch_norm"])

    model = get_model(
        model_cfg["name"],
        in_dim=data.num_node_features,
        hidden_dim=model_cfg["hidden_dim"],
        out_dim=int(data.y.max().item()) + 1,
        num_layers=model_cfg["num_layers"],
        dropout=model_cfg["dropout"],
        **extra_kwargs,
    ).to(device)

    # If a pre-trained weights file is provided, load it and skip training
    if weights_path:
        log.info("  [Run %d] Loading weights from %s — skipping training", run_id, weights_path)
        state = torch.load(weights_path, map_location=device)
        model.load_state_dict(state)
        model.eval()
        with torch.no_grad():
            pred = model(data.x, data.edge_index).argmax(dim=1)
        results = evaluate(model, data)
        return results["val"], results["test"], results["train"], float("nan"), pred, model

    train_cfg = cfg["train"]
    optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg["lr"], weight_decay=float(train_cfg["weight_decay"]))
    criterion = torch.nn.CrossEntropyLoss()

    best_val_acc = 0.0
    best_test_acc = 0.0
    best_train_acc = 0.0
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
            best_train_acc = results["train"]
            best_loss = loss
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        epoch_bar.set_postfix(
            loss=f"{loss:.4f}",
            train=f"{results['train']:.4f}",
            val=f"{results['val']:.4f}",
            test=f"{results['test']:.4f}",
        )
        interval = max(1, train_cfg["epochs"] // 10)
        if epoch % interval == 0:
            log.info("  [Run %d] Epoch %d  loss=%.4f  train=%.4f  val=%.4f  test=%.4f",
                     run_id, epoch, loss, results["train"], results["val"], results["test"])

        if patience > 0 and patience_counter >= patience:
            epoch_bar.close()
            log.info("  [Run %d] Early stopping at epoch %d", run_id, epoch)
            break

    # Restore best checkpoint and produce predictions for the full graph
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pred = model(data.x, data.edge_index).argmax(dim=1)

    return best_val_acc, best_test_acc, best_train_acc, best_loss, pred, model


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
    test_deg  = graph_degree(data.edge_index[1], data.num_nodes)[data.test_mask].cpu()
    train_deg = graph_degree(data.edge_index[1], data.num_nodes)[data.train_mask].cpu()
    all_deg   = graph_degree(data.edge_index[1], data.num_nodes).cpu()

    # Class labels for test and train nodes (graph-fixed)
    test_labels  = data.y[data.test_mask].cpu()
    train_labels = data.y[data.train_mask].cpu()

    # Labelling ratio is graph-fixed — compute once, sliced to test nodes
    has_labeled_neighbor = get_labelling_ratio(data)[data.test_mask.cpu()]
    _same_lr, _diff_lr   = get_class_labelling_ratio(data)
    has_same_class_train = _same_lr[data.test_mask.cpu()]
    has_diff_class_train = _diff_lr[data.test_mask.cpu()]

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

    # Neighborhood purity is graph-fixed — compute once per k value.
    # Also needed for the delta-purity overlay in the feature_similarity plot.
    purity_k_max = plot_cfg.get("purity_k_max", 4)
    _need_purity = (plot_cfg.get("purity_vs_degree", False)
                    or plot_cfg.get("feature_similarity", False))
    purity_by_k  = {
        k: get_node_purity(data, k=k, node_mask=data.test_mask)
        for k in range(1, purity_k_max + 1)
    } if _need_purity else {}

    # k_hops = num_layers - 1 because the final layer is nn.Linear (no message passing)
    k_hops = cfg["model"]["num_layers"] - 1

    # Training-neighbor degree stats are graph-fixed — compute once
    train_nb_deg_stats = (
        get_training_neighbor_degree_stats(data, k=k_hops)
        if plot_cfg.get("train_neighbor_degree", False) else None
    )

    # 1-hop training-neighbor degree stats (graph-fixed) — for degree-normalisation plot
    train_1hop_deg_stats = (
        get_training_neighbor_degree_stats(data, k=1)
        if plot_cfg.get("train_1hop_deg_accuracy", False) else None
    )

    # Neighbourhood cardinality (k=1, k=2) is graph-fixed — compute once
    cardinality_by_k = (
        {k: get_khop_cardinality(data, k)[data.test_mask].cpu()
         for k in [1, 2]}
        if plot_cfg.get("neighborhood_cardinality", False) else {}
    )

    val_accs, test_accs, train_accs = [], [], []
    deg_acc_results   = []
    class_acc_results = []
    run_labels = []

    for i in tqdm(range(1, num_runs + 1), desc="Runs"):
        seed = base_seed + i - 1
        set_seed(seed)  # only affects model initialisation
        run_name = f"run{i:02d}_seed{seed}"
        setup_logger(log_dir=exec_dir, run_name=run_name)
        log.info("=== Run %d/%d (seed=%d) ===", i, num_runs, seed)
        val_acc, test_acc, train_acc, best_loss, pred, model = run(data, cfg, i, device)
        val_accs.append(val_acc)
        test_accs.append(test_acc)
        train_accs.append(train_acc)
        log.info("Best  Train: %.4f  Val: %.4f  Test: %.4f  Loss: %.4f",
                 train_acc, val_acc, test_acc, best_loss)

        deg_acc_results.append(
            get_accuracy_deg(test_deg, pred[data.test_mask], data.y[data.test_mask])
        )
        class_acc_results.append(
            get_accuracy_class(pred[data.test_mask], data.y[data.test_mask])
        )
        run_labels.append(run_name)

    val_mean,   val_std   = np.mean(val_accs),   np.std(val_accs)
    test_mean,  test_std  = np.mean(test_accs),  np.std(test_accs)
    train_mean, train_std = np.mean(train_accs), np.std(train_accs)

    setup_logger(log_dir=exec_dir, run_name="summary")
    log.info("Dataset: %s  |  Model: %s  |  Layers: %d  |  Split: %s",
             cfg["dataset"]["name"], cfg["model"]["name"],
             cfg["model"]["num_layers"], cfg.get("split", "random"))
    log.info(model)
    log.info("Results over %d runs:", num_runs)
    log.info("  Train: %.4f +/- %.4f", train_mean, train_std)
    log.info("  Val:   %.4f +/- %.4f", val_mean, val_std)
    log.info("  Test:  %.4f +/- %.4f", test_mean, test_std)

    if plot_cfg.get("acc_vs_degree", False):
        plot_acc_vs_degree(
            deg_acc_results,
            cfg,
            save_dir=exec_dir if plot_cfg.get("save", True) else None,
            show=plot_cfg.get("show", False),
        )

    if plot_cfg.get("neighborhood_cardinality", False) and cardinality_by_k:
        plot_neighborhood_cardinality_vs_degree(
            test_deg, cardinality_by_k, deg_acc_results, cfg,
            all_deg=all_deg,
            purity_by_k=purity_by_k if len(purity_by_k) >= 2 else None,
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

    if plot_cfg.get("acc_vs_degree_by_layers", False):
        import copy as _copy
        layer_values       = plot_cfg.get("layer_values", [1, 2, 3, 4, 5])
        model_layer_values = plot_cfg.get("model_layer_values", {})
        model_hyperparams  = plot_cfg.get("model_hyperparams", {})
        compare_models     = plot_cfg.get("compare_models", [cfg["model"]["name"]])
        results_by_label = {}
        for model_name in compare_models:
            for L in model_layer_values.get(model_name, layer_values):
                label = f"{model_name} L={L - 1}"
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
                    _, _, _, _, pred_L, model_L = run(data, run_cfg, i, device)
                    if i == 1:
                        log.info("%s architecture:\n%s", label, model_L)
                    label_deg_results.append(
                        get_accuracy_deg(test_deg, pred_L[data.test_mask], data.y[data.test_mask])
                    )
                results_by_label[label] = label_deg_results
        save_dir = exec_dir if plot_cfg.get("save", True) else None
        plot_acc_vs_degree(
            deg_acc_results, cfg,
            save_dir=save_dir, show=plot_cfg.get("show", False),
        )
        plot_acc_vs_degree_by_layers(
            results_by_label, cfg, save_dir=save_dir, show=plot_cfg.get("show", False),
        )
        plot_acc_trend_by_degree(
            results_by_label, cfg, save_dir=save_dir, show=plot_cfg.get("show", False),
        )

    if plot_cfg.get("spl_vs_degree", False):
        save_dir = exec_dir if plot_cfg.get("save", True) else None
        plot_spl_combined_vs_degree(
            test_deg, avg_spl, avg_spl_same_class, cfg,
            deg_acc_results=deg_acc_results,
            purity_by_k=purity_by_k if len(purity_by_k) >= 1 else None,
            all_deg=all_deg,
            has_labeled_neighbor=has_labeled_neighbor,
            save_dir=save_dir, show=plot_cfg.get("show", False),
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
        for k, purity_test in purity_by_k.items():
            plot_purity_vs_degree(
                test_deg, purity_test, cfg, k,
                has_same_class_train=has_same_class_train if k == 1 else None,
                has_diff_class_train=has_diff_class_train if k == 1 else None,
                save_dir=save_dir,
                show=plot_cfg.get("show", False),
            )
        if len(purity_by_k) > 1:
            plot_purity_delta_by_degree(
                test_deg, purity_by_k, cfg,
                save_dir=save_dir,
                show=plot_cfg.get("show", False),
            )


    if plot_cfg.get("train_neighbor_degree", False) and train_nb_deg_stats is not None:
        plot_train_neighbor_degree_stats(
            train_nb_deg_stats, test_deg, pred, data, cfg,
            k=k_hops,
            save_dir=exec_dir if plot_cfg.get("save", True) else None,
            show=plot_cfg.get("show", False),
        )

    if plot_cfg.get("train_1hop_deg_accuracy", False) and train_1hop_deg_stats is not None:
        plot_1hop_train_deg_vs_accuracy(
            train_1hop_deg_stats, deg_acc_results, test_deg, cfg,
            save_dir=exec_dir if plot_cfg.get("save", True) else None,
            show=plot_cfg.get("show", False),
        )

    if plot_cfg.get("class_accuracy", False):
        plot_class_accuracy_and_degree(
            class_acc_results,
            test_deg, test_labels,
            train_deg, train_labels,
            cfg,
            save_dir=exec_dir if plot_cfg.get("save", True) else None,
            show=plot_cfg.get("show", False),
        )

    if plot_cfg.get("influence_analysis", False):
        target_degrees = plot_cfg.get("influence_degrees") or []
        target_nodes   = plot_cfg.get("influence_nodes") or []
        influence_results = compute_influence_analysis(
            model, data, pred,
            k_hops=k_hops,
            target_degrees=target_degrees,
            target_nodes=target_nodes,
        )
        plot_influence_analysis(
            influence_results, cfg,
            save_dir=exec_dir if plot_cfg.get("save", True) else None,
            show=plot_cfg.get("show", False),
        )
        plot_influence_per_neighbor(
            influence_results, cfg,
            save_dir=exec_dir if plot_cfg.get("save", True) else None,
            show=plot_cfg.get("show", False),
        )

    if plot_cfg.get("similarity_nodes"):
        sim_node_results = compute_node_similarity_analysis(
            model, data, k_hops=k_hops,
            target_nodes=plot_cfg.get("similarity_nodes"),
        )
        plot_node_similarity_analysis(
            sim_node_results, cfg,
            save_dir=exec_dir if plot_cfg.get("save", True) else None,
            show=plot_cfg.get("show", False),
        )

    if plot_cfg.get("feature_similarity", False):
        log.info("Computing feature similarity delta (raw vs h^(1))...")
        sim_results = get_feature_similarity_delta(data, model, k_hops=k_hops)
        plot_feature_similarity_delta_vs_degree(
            sim_results, cfg, k_hops=k_hops,
            deg_acc_results=deg_acc_results,
            purity_by_k=purity_by_k if len(purity_by_k) >= 2 else None,
            test_deg=test_deg,
            save_dir=exec_dir if plot_cfg.get("save", True) else None,
            show=plot_cfg.get("show", False),
        )

    if plot_cfg.get("influence_disparity", False):
        log.info("Computing influence disparity for all test nodes (one Jacobian per node)...")
        disparity_results = compute_influence_disparity_all(
            model, data, pred, k_hops=k_hops,
        )
        plot_influence_disparity_vs_degree(
            disparity_results, cfg,
            save_dir=exec_dir if plot_cfg.get("save", True) else None,
            show=plot_cfg.get("show", False),
        )

    if plot_cfg.get("train_degree_distribution", False):
        plot_train_degree_distribution(
            data, cfg,
            save_dir=exec_dir if plot_cfg.get("save", True) else None,
            show=plot_cfg.get("show", False),
        )

    for node_idx in plot_cfg.get("inspect_nodes") or []:
        inspect_node_aggregation(
            node_idx=node_idx,
            edge_index=data.edge_index,
            train_mask=data.train_mask,
            y=data.y,
            pred=pred,
            deg=all_deg,
        )


if __name__ == "__main__":
    main()
