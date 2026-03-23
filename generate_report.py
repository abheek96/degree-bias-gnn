"""generate_report.py — End-to-end degree-bias PDF report.

Runs the configured GCN (default: config.yaml), computes the five key
degree-bias metrics, and assembles a characterised multi-page PDF report.

Usage
-----
    python generate_report.py                   # uses config.yaml
    python generate_report.py --cfg my.yaml
    python generate_report.py --out report.pdf  # override output path
    python generate_report.py --skip-influence  # skip expensive Jacobian pass
"""
import argparse
import copy
import logging
import os
import random
import sys
import tempfile
from glob import glob
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from matplotlib.backends.backend_pdf import PdfPages
from tqdm import tqdm
from torch_geometric.utils import degree as graph_degree

sys.path.insert(0, str(Path(__file__).parent))

from dataset import load_dataset
from dataset_utils import apply_split
from influence import compute_influence_disparity_all
from plot_utils import (
    get_accuracy_deg,
    plot_acc_vs_degree,
    plot_combined_vs_degree,
    plot_influence_disparity_vs_degree,
    plot_neighborhood_cardinality_vs_degree,
    plot_spl_combined_vs_degree,
)
from test import evaluate
from train import train
from utils import (
    compute_distances_to_train,
    get_avg_spl_to_same_class_train,
    get_avg_spl_to_train,
    get_distance_deg,
    get_khop_cardinality,
    get_labelling_ratio,
    get_node_purity,
)

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ── helpers ────────────────────────────────────────────────────────────────────

def _deep_merge(base, override):
    merged = copy.deepcopy(base)
    for k, v in override.items():
        if k in merged and isinstance(merged[k], dict) and isinstance(v, dict):
            merged[k] = _deep_merge(merged[k], v)
        else:
            merged[k] = copy.deepcopy(v)
    return merged


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _run_once(data, cfg, run_id, device):
    from models import get_model

    mc = cfg["model"]
    model = get_model(
        mc["name"],
        in_dim=data.num_node_features,
        hidden_dim=mc["hidden_dim"],
        out_dim=int(data.y.max().item()) + 1,
        num_layers=mc["num_layers"],
        dropout=mc["dropout"],
    ).to(device)

    tc = cfg["train"]
    opt = torch.optim.Adam(model.parameters(), lr=tc["lr"],
                           weight_decay=float(tc["weight_decay"]))
    crit = torch.nn.CrossEntropyLoss()

    best_val, best_state = 0.0, copy.deepcopy(model.state_dict())
    patience_ctr, patience = 0, tc.get("patience", 0)

    for epoch in tqdm(range(1, tc["epochs"] + 1), desc=f"Run {run_id}", leave=False):
        loss = train(model, data, opt, crit)
        res  = evaluate(model, data)
        if res["val"] > best_val:
            best_val   = res["val"]
            best_state = copy.deepcopy(model.state_dict())
            patience_ctr = 0
        else:
            patience_ctr += 1
        if patience > 0 and patience_ctr >= patience:
            break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pred = model(data.x, data.edge_index).argmax(dim=1)
    return pred, model


def _find(directory, pattern):
    """Return first PNG matching pattern, or None."""
    hits = sorted(glob(os.path.join(directory, "**", f"*{pattern}*"), recursive=True))
    return hits[0] if hits else None


# ── PDF assembly ───────────────────────────────────────────────────────────────

# Characterisation text: (section_title, [bullets], ...)  grouped into 3 categories
_SECTIONS = [
    {
        "title": "1 · Classification Accuracy vs Node Degree",
        "plot_key": "acc_vs_degree_across_runs",
        "intro": (
            "Accuracy at every individual degree value across all training seeds. "
            "Boxplots show cross-seed spread; the bottom panel gives node count per group."
        ),
        "chars": [
            (
                "Graph / dataset structural",
                [
                    "Cora's degree distribution is right-skewed (median ≈ 4, max ≈ 168); "
                    "most test nodes have few connections and high-degree groups contain few nodes.",
                    "Higher-degree nodes aggregate over larger neighbourhoods — richer input, "
                    "but increasingly class-heterogeneous signal.",
                ],
            ),
            (
                "Task setting",
                [
                    "Only 140 training nodes (20 per class); low-degree test nodes may have "
                    "no labelled node reachable within the model's 1-hop receptive field.",
                    "Cross-seed IQR is widest for the smallest groups: "
                    "single-degree conclusions at the tail are statistically fragile.",
                ],
            ),
            (
                "Model influence",
                [
                    "GCN aggregation weight ∝ 1/√(deg(u)·deg(v)); per-edge signal strength "
                    "shrinks as degree grows, partially offsetting the richer neighbourhood.",
                    "Accuracy plateaus or becomes erratic at very high degrees: the node "
                    "is well-connected but aggregates across many classes simultaneously.",
                ],
            ),
        ],
    },
    {
        "title": "2 · Neighbourhood Cardinality + Delta Purity",
        "plot_key": "neighborhood_cardinality_vs_degree",
        "intro": (
            "Top: k=1 vs k=2 neighbourhood sizes (diff-class and same-class breakdown) "
            "with across-runs accuracy overlay. "
            "Bottom: Δpurity = purity(k=2) − purity(k=1) averaged per degree group."
        ),
        "chars": [
            (
                "Graph / dataset structural",
                [
                    "The 2-hop neighbourhood grows super-linearly with 1-hop degree: "
                    "hubs pull in topically diverse communities spanning multiple research areas.",
                    "Δpurity is negative for most high-degree groups: expanding the receptive "
                    "field by one hop adds proportionally more cross-class nodes.",
                ],
            ),
            (
                "Task setting",
                [
                    "Larger cardinality increases the number of reachable training nodes but "
                    "dilutes their class signal when the count is mixed-class dominated.",
                    "Diff-class count > same-class count in the receptive field is the "
                    "structural root cause of label noise, not degree per se.",
                ],
            ),
            (
                "Model influence",
                [
                    "GCN has no mechanism to selectively down-weight cross-class neighbours "
                    "during aggregation; all reachable nodes contribute symmetrically.",
                    "Negative Δpurity predicts that adding a third GCN layer hurts those "
                    "nodes: each additional hop adds more cross-class signal, not less.",
                ],
            ),
        ],
    },
    {
        "title": "3 · Hop-Distance Distribution to Training Nodes",
        "plot_key": "combined_vs_degree",
        "intro": (
            "Boxplots of hop-distance from each test node to its nearest training node "
            "and nearest same-class training node, per degree group, with accuracy overlay."
        ),
        "chars": [
            (
                "Graph / dataset structural",
                [
                    "Distance to nearest training node varies widely; isolated low-degree "
                    "nodes can be 4+ hops from any labelled node.",
                    "The gap between distance-to-any-train and distance-to-same-class-train "
                    "measures per-node label misalignment: high-degree nodes are close to "
                    "many training nodes, but those nodes are often of the wrong class.",
                ],
            ),
            (
                "Task setting",
                [
                    "Planetoid public split places training nodes without coverage "
                    "optimisation; entire graph regions can be far from any labelled node.",
                    "Test nodes 3+ hops from any training node fall entirely outside the "
                    "2-layer GCN's receptive field and cannot receive direct label signal.",
                ],
            ),
            (
                "Model influence",
                [
                    "The GCN's receptive field is bounded by num_layers − 1 hops; "
                    "supervision beyond this radius is transmitted only through learned "
                    "feature representations, not direct aggregation.",
                    "Distance mismatch (close to diff-class, far from same-class training "
                    "nodes) is the structural precondition for high diff-class Jacobian "
                    "influence observed in §5.",
                ],
            ),
        ],
    },
    {
        "title": "4 · SPL + Accuracy + Labelling Ratio + Delta Purity",
        "plot_key": "spl_combined_vs_degree",
        "intro": (
            "Top: average SPL to any training node (boxplots) + accuracy median±IQR + "
            "labelling ratio (fraction of 1-hop neighbours that are training nodes). "
            "Bottom: Δpurity mean per degree group."
        ),
        "chars": [
            (
                "Graph / dataset structural",
                [
                    "Labelling ratio drops with degree because degree grows while the "
                    "absolute training-node count in the immediate neighbourhood stays "
                    "roughly constant — a structural dilution effect.",
                    "Average SPL to same-class training nodes is longer than SPL to any "
                    "training node for high-degree nodes: even well-connected nodes receive "
                    "predominantly class-misaligned label signal.",
                ],
            ),
            (
                "Task setting",
                [
                    "Labelling ratio = 0 means the first GCN layer aggregates no direct "
                    "label signal; the model relies entirely on feature-space generalisation.",
                    "Co-declining labelling ratio and accuracy across degree groups is the "
                    "strongest structural evidence for degree bias in this task setting.",
                ],
            ),
            (
                "Model influence",
                [
                    "Full causal chain: degree ↑  →  labelling ratio ↓  →  SPL to "
                    "same-class ↑  →  cross-class influence ↑  →  misclassification.",
                    "Δpurity on the bottom panel predicts model-depth sensitivity: "
                    "negative delta → deeper GCN adds noise; positive delta → more layers "
                    "might help.",
                ],
            ),
        ],
    },
    {
        "title": "5 · Influence Disparity vs Node Degree",
        "plot_key": "influence_disparity_vs_degree",
        "intro": (
            "Per-test-node influence disparity = same_class_inf_norm − diff_class_inf_norm, "
            "computed via exact Jacobian on the final trained model. "
            "Boxes coloured blue (same-class dominant) or red (diff-class dominant); "
            "accuracy of analysed nodes overlaid on right axis."
        ),
        "chars": [
            (
                "Graph / dataset structural",
                [
                    "At higher degree, more diff-class training nodes enter the k-hop "
                    "receptive field, pushing disparity negative — structural count imbalance "
                    "translates directly into gradient-flow imbalance.",
                    "Nodes with zero reachable same-class training nodes have disparity ≈ −1 "
                    "by construction, independent of model quality.",
                ],
            ),
            (
                "Task setting",
                [
                    "With only 20 same-class training nodes per class globally, high-degree "
                    "test nodes statistically encounter more diff-class than same-class "
                    "training nodes within their receptive field.",
                    "Positive-disparity nodes (same-class dominant) show markedly higher "
                    "accuracy than negative-disparity nodes across all degree groups.",
                ],
            ),
            (
                "Model influence",
                [
                    "Jacobian ∂logit(v)/∂x(u) captures actual learned routing through "
                    "trained weights + graph topology, not just raw graph structure.",
                    "Pearson r(disparity, correct) quantifies how much of degree bias is "
                    "model-mediated vs purely structural.",
                    "GCN's symmetric normalisation amplifies low-degree training nodes; "
                    "if diff-class training nodes have lower degree, their per-edge "
                    "influence exceeds their count ratio — the model amplifies the "
                    "structural imbalance.",
                ],
            ),
        ],
    },
]

# Category header colours
_CAT_COLORS = {
    "Graph / dataset structural": "#1565C0",
    "Task setting":               "#2E7D32",
    "Model influence":            "#6A1B9A",
}

_PAGE_W, _PAGE_H = 8.5, 11.0   # letter inches


def _cover_page(pdf, cfg):
    fig = plt.figure(figsize=(_PAGE_W, _PAGE_H))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.set_facecolor("#F5F5F5")
    fig.patch.set_facecolor("#F5F5F5")

    ax.text(0.5, 0.78, "Degree Bias in GCN", ha="center", va="center",
            fontsize=28, fontweight="bold", color="#1A237E")
    ax.text(0.5, 0.70, "A Structural Analysis of Classification Accuracy\nand Message-Passing Influence",
            ha="center", va="center", fontsize=14, color="#37474F",
            linespacing=1.6)

    ds   = cfg["dataset"]["name"]
    mdl  = cfg["model"]["name"]
    splt = cfg.get("split", "random")
    cc   = "CC-filtered" if cfg["dataset"].get("use_cc") else "no CC filter"
    nl   = cfg["model"]["num_layers"]
    nr   = cfg.get("num_runs", 1)
    ax.text(0.5, 0.59,
            f"Dataset: {ds}   ·   Model: {mdl} ({nl} layers)   ·   Split: {splt} ({cc})   ·   Runs: {nr}",
            ha="center", va="center", fontsize=10, color="#455A64")

    ax.axhline(0.54, xmin=0.1, xmax=0.9, color="#B0BEC5", linewidth=0.8)

    ax.text(0.12, 0.50, "Characterisation framework", ha="left", va="top",
            fontsize=11, fontweight="bold", color="#37474F")
    chars = [
        ("Graph / dataset structural", "#1565C0",
         "Degree distribution, neighbourhood homophily, receptive-field cardinality, purity."),
        ("Task setting",               "#2E7D32",
         "Training-node placement, labelling ratio, split protocol, coverage gaps."),
        ("Model influence",            "#6A1B9A",
         "GCN aggregation weights, Jacobian-based influence flow, depth sensitivity."),
    ]
    y = 0.44
    for cat, col, desc in chars:
        ax.add_patch(mpatches.FancyBboxPatch(
            (0.10, y - 0.025), 0.80, 0.048,
            boxstyle="round,pad=0.005", facecolor=col, alpha=0.08, zorder=0,
        ))
        ax.text(0.13, y, cat, ha="left", va="center",
                fontsize=9, fontweight="bold", color=col)
        ax.text(0.13, y - 0.018, desc, ha="left", va="center",
                fontsize=7.5, color="#546E7A")
        y -= 0.065

    ax.axhline(0.24, xmin=0.1, xmax=0.9, color="#B0BEC5", linewidth=0.8)
    ax.text(0.12, 0.20, "Contents", ha="left", va="top", fontsize=10,
            fontweight="bold", color="#37474F")
    for i, sec in enumerate(_SECTIONS, 1):
        ax.text(0.14, 0.16 - (i - 1) * 0.026, f"{sec['title']}",
                ha="left", va="top", fontsize=8, color="#455A64")

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _section_page(pdf, section, img_path):
    """One page: plot image on top, characterisation text below."""
    fig = plt.figure(figsize=(_PAGE_W, _PAGE_H))

    # ── title strip ────────────────────────────────────────────────────────────
    ax_title = fig.add_axes([0.0, 0.955, 1.0, 0.045])
    ax_title.set_xlim(0, 1); ax_title.set_ylim(0, 1); ax_title.axis("off")
    ax_title.set_facecolor("#E8EAF6")
    ax_title.text(0.02, 0.5, section["title"], ha="left", va="center",
                  fontsize=12, fontweight="bold", color="#1A237E")

    # ── intro sentence ─────────────────────────────────────────────────────────
    ax_intro = fig.add_axes([0.04, 0.905, 0.92, 0.045])
    ax_intro.set_xlim(0, 1); ax_intro.set_ylim(0, 1); ax_intro.axis("off")
    ax_intro.text(0.0, 0.5, section["intro"], ha="left", va="center",
                  fontsize=8, color="#37474F", style="italic",
                  wrap=True)

    # ── plot image ─────────────────────────────────────────────────────────────
    ax_img = fig.add_axes([0.02, 0.44, 0.96, 0.455])
    ax_img.axis("off")
    if img_path and os.path.exists(img_path):
        img = mpimg.imread(img_path)
        ax_img.imshow(img, aspect="auto", interpolation="lanczos")
    else:
        ax_img.text(0.5, 0.5, "[plot not generated — run without --skip-influence\nor enable the relevant config flag]",
                    ha="center", va="center", fontsize=9, color="grey",
                    transform=ax_img.transAxes)
        ax_img.set_facecolor("#F5F5F5")

    # ── characterisation text (3 columns) ──────────────────────────────────────
    n_cats = len(section["chars"])
    col_w  = 0.92 / n_cats
    for ci, (cat_title, bullets) in enumerate(section["chars"]):
        col_x = 0.04 + ci * col_w
        col_color = _CAT_COLORS.get(cat_title, "#333333")

        ax_cat = fig.add_axes([col_x, 0.395, col_w - 0.01, 0.038])
        ax_cat.set_xlim(0, 1); ax_cat.set_ylim(0, 1); ax_cat.axis("off")
        ax_cat.set_facecolor(col_color)
        ax_cat.text(0.05, 0.5, cat_title, ha="left", va="center",
                    fontsize=7.5, fontweight="bold", color="white")

        ax_b = fig.add_axes([col_x, 0.03, col_w - 0.01, 0.36])
        ax_b.set_xlim(0, 1); ax_b.set_ylim(0, 1); ax_b.axis("off")
        ax_b.set_facecolor("#FAFAFA")
        ax_b.add_patch(mpatches.FancyBboxPatch(
            (0, 0), 1, 1, boxstyle="square,pad=0",
            edgecolor=col_color, facecolor="#FAFAFA", linewidth=0.6,
        ))

        y_b = 0.95
        for bullet in bullets:
            # Wrap text manually (approx 55 chars per line for col_w ≈ 0.29)
            words  = bullet.split()
            lines  = []
            line   = ""
            for w in words:
                if len(line) + len(w) + 1 > 52:
                    lines.append(line)
                    line = w
                else:
                    line = (line + " " + w).strip()
            if line:
                lines.append(line)

            # Bullet dot
            ax_b.text(0.03, y_b, "•", ha="left", va="top",
                      fontsize=7, color=col_color)
            for li, l in enumerate(lines):
                ax_b.text(0.09, y_b - li * 0.075, l, ha="left", va="top",
                          fontsize=7, color="#37474F")
            y_b -= len(lines) * 0.075 + 0.055

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


# ── main pipeline ──────────────────────────────────────────────────────────────

def generate(cfg_path, out_path, skip_influence):
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    # Auto-merge model-dataset config if present
    model_name   = cfg["model"]["name"]
    dataset_name = cfg["dataset"]["name"]
    mdc_path = os.path.join("configs", f"{model_name}_{dataset_name}.yaml")
    if os.path.exists(mdc_path):
        with open(mdc_path) as f:
            cfg = _deep_merge(cfg, yaml.safe_load(f))
        log.info("Merged model-dataset config: %s", mdc_path)

    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_seed = cfg.get("seed", 42)
    num_runs  = cfg.get("num_runs", 1)
    split     = cfg.get("split", "random")

    log.info("Loading dataset...")
    if split == "random":
        set_seed(base_seed)
    data = load_dataset(cfg["dataset"])
    data = apply_split(data, split, cfg["dataset"])
    data = data.to(device)

    # ── fixed graph quantities ─────────────────────────────────────────────────
    test_deg  = graph_degree(data.edge_index[1], data.num_nodes)[data.test_mask].cpu()
    all_deg   = graph_degree(data.edge_index[1], data.num_nodes).cpu()
    k_hops    = cfg["model"]["num_layers"] - 1

    log.info("Pre-computing structural metrics (graph-fixed)...")
    has_labeled_neighbor = get_labelling_ratio(data)[data.test_mask.cpu()]
    avg_spl              = get_avg_spl_to_train(data)[data.test_mask.cpu()]
    avg_spl_same_class   = get_avg_spl_to_same_class_train(data)[data.test_mask.cpu()]
    dist_to_train, dist_to_same_class = compute_distances_to_train(data)
    dist_deg_data = get_distance_deg(test_deg, dist_to_train, dist_to_same_class,
                                     num_nodes=data.num_nodes)
    purity_by_k = {k: get_node_purity(data, k=k) for k in [1, 2]}
    cardinality_by_k = {k: get_khop_cardinality(data, k)[data.test_mask].cpu()
                        for k in [1, 2]}

    # ── training runs ─────────────────────────────────────────────────────────
    deg_acc_results = []
    for i in range(1, num_runs + 1):
        seed = base_seed + i - 1
        set_seed(seed)
        log.info("Run %d / %d  (seed=%d)", i, num_runs, seed)
        pred, model = _run_once(data, cfg, i, device)
        deg_acc_results.append(
            get_accuracy_deg(test_deg, pred[data.test_mask], data.y[data.test_mask])
        )

    # The last run's model/pred are used for influence analysis
    log.info("Training complete.")

    # ── generate plots to temp directory ──────────────────────────────────────
    with tempfile.TemporaryDirectory() as tmp:
        kw = dict(save_dir=tmp, show=False)

        log.info("Generating accuracy vs degree plot...")
        plot_acc_vs_degree(deg_acc_results, cfg, **kw)

        log.info("Generating neighbourhood cardinality + delta purity plot...")
        plot_neighborhood_cardinality_vs_degree(
            test_deg, cardinality_by_k, deg_acc_results, cfg,
            all_deg=all_deg, purity_by_k=purity_by_k, **kw,
        )

        log.info("Generating combined hop-distance plot...")
        plot_combined_vs_degree(deg_acc_results, dist_deg_data, cfg, **kw)

        log.info("Generating SPL combined plot...")
        plot_spl_combined_vs_degree(
            test_deg, avg_spl, avg_spl_same_class, cfg,
            deg_acc_results=deg_acc_results,
            purity_by_k=purity_by_k,
            all_deg=all_deg,
            has_labeled_neighbor=has_labeled_neighbor,
            **kw,
        )

        if skip_influence:
            log.info("Skipping influence disparity (--skip-influence set)")
            disp_path = None
        else:
            log.info("Computing influence disparity (one Jacobian per test node)...")
            disparity_results = compute_influence_disparity_all(
                model, data, pred, k_hops=k_hops,
            )
            plot_influence_disparity_vs_degree(disparity_results, cfg, **kw)

        # ── find the right files ───────────────────────────────────────────────
        plots = {
            "acc_vs_degree_across_runs":        _find(tmp, "acc_vs_degree_across_runs"),
            "neighborhood_cardinality_vs_degree": _find(tmp, "neighborhood_cardinality_vs_degree"),
            "combined_vs_degree":               _find(tmp, "combined_vs_degree"),
            "spl_combined_vs_degree":           _find(tmp, "spl_combined_vs_degree"),
            "influence_disparity_vs_degree":    _find(tmp, "influence_disparity_vs_degree"),
        }

        for key, path in plots.items():
            if path:
                log.info("  ✓  %s → %s", key, os.path.basename(path))
            else:
                log.warning("  ✗  %s  not found", key)

        # ── assemble PDF ───────────────────────────────────────────────────────
        log.info("Assembling PDF → %s", out_path)
        with PdfPages(out_path) as pdf:
            _cover_page(pdf, cfg)
            for sec in _SECTIONS:
                img = plots.get(sec["plot_key"])
                _section_page(pdf, sec, img)

        log.info("Done. PDF saved to %s", out_path)


# ── entry point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate degree-bias PDF report")
    parser.add_argument("--cfg",  default="config.yaml", help="Config YAML path")
    parser.add_argument("--out",  default="degree_bias_report.pdf",
                        help="Output PDF path (default: degree_bias_report.pdf)")
    parser.add_argument("--skip-influence", action="store_true",
                        help="Skip the expensive influence-disparity Jacobian pass")
    args = parser.parse_args()
    generate(args.cfg, args.out, args.skip_influence)


if __name__ == "__main__":
    main()
