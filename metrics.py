"""
metrics.py — Serialisation of per-run computed metrics for post-hoc plot regeneration.

Implements TODO #13: decouple computation from visualisation so that any plot
can be regenerated from saved artifacts without re-running the training loop.

Usage
-----
Saving (called by main.py after training completes):

    from metrics import save_metrics
    save_metrics(bundle, exec_dir)

Loading (called by plots/*.py scripts):

    from metrics import load_metrics
    m = load_metrics(results_dir)
    plot_acc_vs_degree(m["deg_acc_results"], m["cfg"], ...)

Metrics bundle keys
-------------------
cfg                    : dict — merged run configuration
test_deg               : LongTensor   [num_test]       — test-node degrees
all_deg                : LongTensor   [num_nodes]       — all-node degrees
train_deg              : LongTensor   [num_train]       — training-node degrees
test_labels            : LongTensor   [num_test]        — true labels for test nodes
train_labels           : LongTensor   [num_train]       — true labels for train nodes
deg_acc_results        : list[dict]                     — per-run accuracy-by-degree
class_acc_results      : list[dict]                     — per-run accuracy-by-class
run_labels             : list[str]                      — run identifiers
overall_acc            : float                          — mean test accuracy across runs
k_hops                 : int                            — GCN receptive-field radius
purity_by_k            : dict[int, FloatTensor]         — neighborhood purity tensors
avg_spl                : FloatTensor  [num_test]        — avg SPL to any training node
avg_spl_same_class     : FloatTensor  [num_test]        — avg SPL to same-class train
cardinality_by_k       : dict[int, LongTensor]          — k-hop cardinality for test nodes
dist_deg_data          : dict                           — distance-by-degree structure
has_labeled_neighbor   : BoolTensor   [num_test]        — ≥1 labeled 1-hop neighbor
has_khop_labeled_neighbor : BoolTensor [num_test]       — ≥1 labeled k-hop neighbor
has_same_class_train   : BoolTensor   [num_test]        — ≥1 same-class train at 1-hop
has_diff_class_train   : BoolTensor   [num_test]        — ≥1 diff-class train at 1-hop
train_nb_deg_stats     : dict or None                   — training-neighbor degree stats
train_1hop_deg_stats   : dict or None                   — 1-hop training-neighbor degree stats
max_same_train_deg     : LongTensor   [num_test] or None — max same-class train-neighbor degree
"""

import logging
import os

import torch

log = logging.getLogger(__name__)

METRICS_FILENAME = "metrics.pt"


def save_metrics(metrics: dict, exec_dir: str) -> str:
    """Save the metrics bundle to exec_dir/metrics.pt.

    Parameters
    ----------
    metrics  : dict of tensors, scalars, lists, and nested dicts (see module docstring)
    exec_dir : path to the current experiment directory

    Returns
    -------
    path : str — absolute path to the saved file
    """
    os.makedirs(exec_dir, exist_ok=True)
    path = os.path.join(exec_dir, METRICS_FILENAME)
    torch.save(metrics, path)
    log.info("Metrics saved: %s", path)
    return path


def load_metrics(results_dir: str) -> dict:
    """Load a previously saved metrics bundle.

    Parameters
    ----------
    results_dir : path to an experiment directory that contains metrics.pt

    Returns
    -------
    metrics : dict

    Raises
    ------
    FileNotFoundError if metrics.pt is absent — with a hint to re-run main.py.
    """
    path = os.path.join(results_dir, METRICS_FILENAME)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No metrics file found at {path!r}.\n"
            "Run 'uv run main.py' to generate metrics.pt for this experiment directory."
        )
    metrics = torch.load(path, map_location="cpu", weights_only=False)
    log.info("Metrics loaded from %s", path)
    return metrics
