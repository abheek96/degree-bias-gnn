"""
export_node_csv.py — Export node-level graph and model characteristics to CSV.

Produces three CSV files and a JSON attribute dictionary:
  all_nodes.csv     — every node in the largest connected component
  test_nodes.csv    — test nodes only
  train_nodes.csv   — training nodes only
  attribute_dictionary.json — column definitions + experimental setup

Every row is one node.  Graph-structural attributes (degree, purity, SPL, …)
are computed once from the fixed graph.  Model predictions are aggregated
across ``num_runs`` independent training runs.

Usage
-----
  python export_node_csv.py [--config config.yaml] [--out-dir ./csv_exports]
                            [--num-runs N] [--model-config path/to/override.yaml]
"""

import argparse
import copy
import json
import logging
import os
import random
from collections import defaultdict, deque

import numpy as np
import pandas as pd
import torch
import yaml
from torch_geometric.utils import degree as graph_degree

from dataset import load_dataset
from dataset_utils import apply_split
from models import get_model
from train import train
from test import evaluate
from utils import (
    get_avg_spl_to_train,
    get_avg_spl_to_same_class_train,
    get_khop_degree,
    get_labelling_ratio,
    get_node_purity,
)

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ── helpers ───────────────────────────────────────────────────────────────────

def _deep_merge(base, override):
    merged = copy.deepcopy(base)
    for key, val in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(val, dict):
            merged[key] = _deep_merge(merged[key], val)
        else:
            merged[key] = copy.deepcopy(val)
    return merged


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def compute_dist_to_train_all_nodes(data):
    """Min-hop distance from every node to the nearest training node.

    Returns two float numpy arrays of shape [N]:
      dist_any   — nearest training node of any class
      dist_same  — nearest training node of the same class as each node

    Training nodes themselves receive distance 0.
    Nodes unreachable within MAX_HOP hops receive NaN.
    """
    MAX_HOP = 10
    N  = data.num_nodes
    y  = data.y.cpu().numpy()

    src, dst = data.edge_index.cpu()
    adj = [[] for _ in range(N)]
    for u, v in zip(src.tolist(), dst.tolist()):
        adj[u].append(v)

    train_idx = data.train_mask.cpu().nonzero(as_tuple=True)[0].tolist()
    train_set = set(train_idx)

    class_train = defaultdict(set)
    for t in train_idx:
        class_train[int(y[t])].add(t)

    # ── dist_any: multi-source BFS from all training nodes simultaneously ──
    dist_any = np.full(N, np.nan)
    visited  = np.full(N, -1, dtype=int)
    queue    = deque()
    for t in train_idx:
        dist_any[t] = 0
        visited[t]  = 0
        queue.append((t, 0))
    while queue:
        u, d = queue.popleft()
        for v in adj[u]:
            if visited[v] == -1:
                visited[v]  = d + 1
                dist_any[v] = d + 1
                if d + 1 < MAX_HOP:
                    queue.append((v, d + 1))

    # ── dist_same: one multi-source BFS per class ──────────────────────────
    dist_same = np.full(N, np.nan)
    for cls, cls_nodes in class_train.items():
        vis = np.full(N, -1, dtype=int)
        q   = deque()
        for t in cls_nodes:
            vis[t] = 0
            q.append((t, 0))
        while q:
            u, d = q.popleft()
            for v in adj[u]:
                if vis[v] == -1:
                    vis[v] = d + 1
                    if d + 1 < MAX_HOP:
                        q.append((v, d + 1))
        # Record distance for nodes whose label matches this class
        for v in range(N):
            if int(y[v]) == cls and vis[v] >= 0:
                d = float(vis[v])
                if np.isnan(dist_same[v]) or d < dist_same[v]:
                    dist_same[v] = d

    return dist_any, dist_same


# ── training ──────────────────────────────────────────────────────────────────

def run_and_collect_preds(data, cfg, num_runs, base_seed, device):
    """Train ``num_runs`` models; return prediction matrix [N, num_runs]."""
    N         = data.num_nodes
    model_cfg = cfg["model"]
    train_cfg = cfg["train"]
    all_preds = []

    for i in range(num_runs):
        seed = base_seed + i
        set_seed(seed)

        model = get_model(
            model_cfg["name"],
            in_dim=data.num_node_features,
            hidden_dim=model_cfg["hidden_dim"],
            out_dim=int(data.y.max().item()) + 1,
            num_layers=model_cfg["num_layers"],
            dropout=model_cfg["dropout"],
        ).to(device)

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=train_cfg["lr"],
            weight_decay=float(train_cfg["weight_decay"]),
        )
        criterion   = torch.nn.CrossEntropyLoss()
        best_val    = 0.0
        best_train  = 0.0
        best_state  = copy.deepcopy(model.state_dict())
        patience    = train_cfg.get("patience", 0)
        p_counter   = 0

        for epoch in range(1, train_cfg["epochs"] + 1):
            loss    = train(model, data, optimizer, criterion)
            results = evaluate(model, data)
            if results["val"] > best_val:
                best_val   = results["val"]
                best_train = results["train"]
                best_state = copy.deepcopy(model.state_dict())
                p_counter  = 0
            else:
                p_counter += 1
            if patience > 0 and p_counter >= patience:
                break

        model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            pred = model(data.x, data.edge_index).argmax(dim=1).cpu().numpy()
        all_preds.append(pred)
        log.info("  Run %d/%d  best_val=%.4f  best_train=%.4f",
                 i + 1, num_runs, best_val, best_train)

    return np.stack(all_preds, axis=1)  # [N, num_runs]


# ── dataframe assembly ────────────────────────────────────────────────────────

def build_dataframe(data, preds_matrix, cfg):
    N = data.num_nodes
    y = data.y.cpu().numpy()

    khop_k = cfg.get("plot", {}).get("khop_k", 2)

    log.info("Computing degree ...")
    deg = graph_degree(data.edge_index[1], N).cpu().numpy().astype(int)

    log.info("Computing %d-hop degree ...", khop_k)
    khop_deg = get_khop_degree(data, k=khop_k).cpu().numpy().astype(int)

    log.info("Computing hop-distances to training nodes (all nodes) ...")
    dist_any, dist_same = compute_dist_to_train_all_nodes(data)

    log.info("Computing average SPL to training nodes ...")
    avg_spl_all  = get_avg_spl_to_train(data).numpy()
    avg_spl_same = get_avg_spl_to_same_class_train(data).numpy()

    log.info("Computing labelling ratio ...")
    has_labelled = get_labelling_ratio(data).numpy().astype(int)

    log.info("Computing neighbourhood purity (k=1, k=2) ...")
    purity_k1    = get_node_purity(data, k=1).numpy()
    purity_k2    = get_node_purity(data, k=2).numpy()
    delta_purity = purity_k2 - purity_k1

    # Model predictions aggregated across runs
    num_classes  = int(y.max()) + 1
    pred_mode    = np.apply_along_axis(
        lambda x: np.bincount(x.astype(int), minlength=num_classes).argmax(),
        axis=1,
        arr=preds_matrix,
    )
    correct_mean = (preds_matrix == y[:, None]).mean(axis=1).round(4)
    correct      = (correct_mean >= 0.5).astype(int)

    train_mask = data.train_mask.cpu().numpy().astype(int)
    val_mask   = data.val_mask.cpu().numpy().astype(int)
    test_mask  = data.test_mask.cpu().numpy().astype(int)

    df = pd.DataFrame({
        "node_idx":                    np.arange(N),
        "true_label":                  y,
        "is_train":                    train_mask,
        "is_val":                      val_mask,
        "is_test":                     test_mask,
        "degree":                      deg,
        f"khop_degree_k{khop_k}":      khop_deg,
        "dist_to_train":               dist_any,
        "dist_to_same_class_train":    dist_same,
        "avg_spl_to_train":            avg_spl_all.round(4),
        "avg_spl_to_same_class_train": avg_spl_same.round(4),
        "has_labelled_neighbour":      has_labelled,
        "purity_k1":                   purity_k1.round(4),
        "purity_k2":                   purity_k2.round(4),
        "delta_purity":                delta_purity.round(4),
        "pred_label_mode":             pred_mode,
        "correct_mean":                correct_mean,
        "correct":                     correct,
    })

    return df


# ── dictionary ────────────────────────────────────────────────────────────────

def write_dictionary(out_dir, cfg, num_runs):
    khop_k       = cfg.get("plot", {}).get("khop_k", 2)
    dataset_name = cfg["dataset"]["name"]
    model_name   = cfg["model"]["name"]

    doc = {
        "setup": {
            "description": (
                "Node-level CSV dataset capturing graph-structural and model-behaviour "
                "characteristics for each node in the largest connected component. "
                "Designed to study degree bias in GNN node classification: "
                "whether and why a node's connectivity affects how accurately a GNN classifies it."
            ),
            "dataset":             dataset_name,
            "use_largest_cc_only": cfg["dataset"].get("use_cc", True),
            "split":               cfg.get("split", "random"),
            "num_train_per_class": cfg["dataset"].get("num_train_per_class"),
            "num_val":             cfg["dataset"].get("num_val"),
            "num_test":            cfg["dataset"].get("num_test"),
            "model":               model_name,
            "num_layers":          cfg["model"]["num_layers"],
            "hidden_dim":          cfg["model"]["hidden_dim"],
            "dropout":             cfg["model"]["dropout"],
            "lr":                  cfg["train"]["lr"],
            "weight_decay":        cfg["train"]["weight_decay"],
            "epochs":              cfg["train"]["epochs"],
            "patience":            cfg["train"].get("patience", 0),
            "num_runs":            num_runs,
            "base_seed":           cfg.get("seed", 42),
            "index_convention":    "Node indices are 0-based (PyTorch Geometric convention).",
            "prediction_note": (
                "pred_label_mode and correct_mean are computed for all nodes. "
                "For training nodes these reflect the model's memorisation of training labels; "
                "for test nodes they reflect generalisation performance."
            ),
        },
        "columns": {
            "node_idx": (
                "0-based integer index of the node within the graph "
                "(after largest-connected-component filtering, if applicable)."
            ),
            "true_label": (
                "Ground-truth class label assigned to the node."
            ),
            "is_train": (
                "Binary (0/1). 1 if the node belongs to the training set, 0 otherwise."
            ),
            "is_val": (
                "Binary (0/1). 1 if the node belongs to the validation set, 0 otherwise."
            ),
            "is_test": (
                "Binary (0/1). 1 if the node belongs to the test set, 0 otherwise. "
                "Nodes that are 0 for all three flags are unlabelled/unassigned nodes."
            ),
            "degree": (
                "1-hop degree: the number of direct neighbours of the node. "
                "Computed as the in-degree from the directed edge_index representation."
            ),
            f"khop_degree_k{khop_k}": (
                f"k-hop degree at k={khop_k}: the total number of distinct nodes "
                f"reachable within at most {khop_k} hops (shortest-path distance <= {khop_k}), "
                f"excluding the node itself. "
                f"This equals the size of the receptive field of a {khop_k}-layer GNN."
            ),
            "dist_to_train": (
                "Minimum hop-distance (shortest path) from this node to the nearest "
                "training node of any class. "
                "Training nodes themselves have distance 0. "
                "NaN if no training node is reachable within 10 hops."
            ),
            "dist_to_same_class_train": (
                "Minimum hop-distance from this node to the nearest training node "
                "that shares the same class label as this node. "
                "Training nodes of the same class have distance 0. "
                "NaN if no same-class training node is reachable within 10 hops."
            ),
            "avg_spl_to_train": (
                "Average shortest path length from this node to all training nodes "
                "that are reachable from it. "
                "A higher value means the training signal must travel further on average "
                "before reaching this node via message passing. "
                "NaN if no training node is reachable."
            ),
            "avg_spl_to_same_class_train": (
                "Average shortest path length from this node to all same-class training "
                "nodes that are reachable from it. "
                "Higher values indicate the class-specific training signal arrives more "
                "diluted. NaN if no same-class training node is reachable."
            ),
            "has_labelled_neighbour": (
                "Binary (0/1). 1 if the node has at least one direct (1-hop) training "
                "node as an immediate neighbour; 0 otherwise. "
                "A node with a labelled neighbour receives direct, undiluted label signal "
                "in the first GNN layer."
            ),
            "purity_k1": (
                "Neighbourhood purity at k=1: the fraction of the node's immediate "
                "(1-hop) neighbours that share its true class label. "
                "A value of 1.0 means all neighbours are of the same class (homophilic); "
                "0.0 means all are of a different class (heterophilic). "
                "NaN for isolated nodes with no neighbours."
            ),
            "purity_k2": (
                "Neighbourhood purity at k=2: the fraction of all nodes reachable "
                "within 2 hops that share this node's true class label. "
                "As k increases, purity typically decreases for high-degree nodes "
                "because the neighbourhood expands to include more diverse classes. "
                "NaN if the 2-hop neighbourhood is empty."
            ),
            "delta_purity": (
                "purity_k2 - purity_k1. "
                "Negative values mean the neighbourhood becomes less class-homogeneous "
                "as the receptive field expands from 1 to 2 hops (class signal degrades "
                "with model depth). "
                "The magnitude tends to be larger for high-degree nodes whose wider "
                "2-hop neighbourhood spans more class boundaries. "
                "NaN if either purity_k1 or purity_k2 is NaN."
            ),
            "pred_label_mode": (
                f"The most frequently predicted class label across {num_runs} independent "
                "training runs (mode of the per-run predictions). "
                "Represents the model's stable prediction for this node."
            ),
            "correct_mean": (
                f"Fraction of the {num_runs} training runs in which the node was "
                "correctly classified (range 0.0 to 1.0). "
                "0.0 = always misclassified; 1.0 = always correctly classified. "
                "Values between 0 and 1 indicate sensitivity to random seed / initialisation."
            ),
            "correct": (
                "Binary (0/1). 1 if correct_mean >= 0.5 (node is correctly classified "
                "in the majority of runs), 0 otherwise."
            ),
        },
        "files": {
            "all_nodes.csv": (
                "All nodes in the largest connected component. "
                "Includes is_train / is_val / is_test flag columns. "
                "Nodes with all three flags = 0 are unassigned (neither train, val, nor test)."
            ),
            "test_nodes.csv": (
                "Subset of all_nodes.csv restricted to test nodes (is_test == 1). "
                "Use this file for degree-bias analysis of model generalisation."
            ),
            "train_nodes.csv": (
                "Subset of all_nodes.csv restricted to training nodes (is_train == 1). "
                "Use this file to inspect properties of training nodes "
                "(e.g. degree distribution, purity of the labelled set)."
            ),
            "attribute_dictionary.json": "This file.",
        },
    }

    path = os.path.join(out_dir, "attribute_dictionary.json")
    with open(path, "w") as f:
        json.dump(doc, f, indent=2)
    log.info("Saved %s", path)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Export node-level graph and model characteristics to CSV"
    )
    parser.add_argument("--config",       type=str, default="config.yaml",
                        help="Path to base config YAML")
    parser.add_argument("--model-config", type=str, default=None,
                        help="Path to model-dataset override YAML (auto-discovered if omitted)")
    parser.add_argument("--out-dir",      type=str, default="./csv_exports",
                        help="Output directory for CSV files and dictionary")
    parser.add_argument("--num-runs",     type=int, default=None,
                        help="Number of training runs (overrides config)")
    args = parser.parse_args()

    # Load and merge configs (mirrors main.py logic)
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    model_cfg_path = args.model_config
    if model_cfg_path is None:
        model_cfg_path = os.path.join(
            "configs", f"{cfg['model']['name']}_{cfg['dataset']['name']}.yaml"
        )
    if os.path.exists(model_cfg_path):
        with open(model_cfg_path) as f:
            cfg = _deep_merge(cfg, yaml.safe_load(f))
        log.info("Loaded model-dataset config: %s", model_cfg_path)

    os.makedirs(args.out_dir, exist_ok=True)

    num_runs  = args.num_runs or cfg.get("num_runs", 5)
    base_seed = cfg.get("seed", 42)
    split     = cfg.get("split", "random")
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    log.info("Dataset: %s | Model: %s | Runs: %d | Device: %s",
             cfg["dataset"]["name"], cfg["model"]["name"], num_runs, device)

    # Load dataset and apply split
    data = load_dataset(cfg["dataset"])
    if split == "random":
        set_seed(base_seed)
    data = apply_split(data, split, cfg["dataset"])
    log.info("Nodes: %d | Edges: %d | Train: %d | Val: %d | Test: %d",
             data.num_nodes, data.edge_index.shape[1],
             data.train_mask.sum().item(),
             data.val_mask.sum().item(),
             data.test_mask.sum().item())

    data = data.to(device)

    # Train and collect predictions
    log.info("Training %d run(s) ...", num_runs)
    preds_matrix = run_and_collect_preds(data, cfg, num_runs, base_seed, device)

    # Move to CPU for metric computation
    data = data.cpu()

    # Build DataFrame
    log.info("Computing node-level attributes ...")
    df = build_dataframe(data, preds_matrix, cfg)

    # Save CSVs
    all_path   = os.path.join(args.out_dir, "all_nodes.csv")
    test_path  = os.path.join(args.out_dir, "test_nodes.csv")
    train_path = os.path.join(args.out_dir, "train_nodes.csv")

    df.to_csv(all_path, index=False)
    df[df["is_test"]  == 1].reset_index(drop=True).to_csv(test_path,  index=False)
    df[df["is_train"] == 1].reset_index(drop=True).to_csv(train_path, index=False)

    log.info("Saved %s  (%d rows)",  all_path,   len(df))
    log.info("Saved %s  (%d rows)",  test_path,  int(df["is_test"].sum()))
    log.info("Saved %s  (%d rows)",  train_path, int(df["is_train"].sum()))

    write_dictionary(args.out_dir, cfg, num_runs)
    log.info("Done. All files written to %s", args.out_dir)


if __name__ == "__main__":
    main()
