import logging
import os

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import torch

log = logging.getLogger(__name__)


# ── shared helpers ─────────────────────────────────────────────────────────────

def get_accuracy_deg(deg, pred, true):
    """Map node predictions to their degree and compute per-degree accuracy.

    Works with degrees from the original graph or the largest connected
    component subgraph — pass whichever degree tensor corresponds to the
    nodes in pred/true.

    Parameters
    ----------
    deg  : 1-D LongTensor of node degrees (one entry per node in the split).
    pred : 1-D LongTensor of predicted class indices (same length as deg).
    true : 1-D LongTensor of ground-truth class indices (same length as deg).

    Returns
    -------
    dict mapping degree (int) -> {
        'preds'  : predicted labels for nodes of that degree (cpu tensor),
        'labels' : true labels for nodes of that degree (cpu tensor),
        'acc'    : classification accuracy as a Python float,
    }
    """
    deg, pred, true = deg.cpu(), pred.cpu(), true.cpu()
    result = {}
    for d in deg.unique():
        idx = (deg == d).nonzero(as_tuple=False).view(-1)
        p, t = pred[idx], true[idx]
        result[d.item()] = {
            "preds":  p,
            "labels": t,
            "acc":    (p == t).float().mean().item(),
        }
    return result


def _fname_prefix(cfg):
    """Filename-safe experiment prefix, e.g. 'Cora_GCN_random_CC'."""
    dataset = cfg["dataset"]["name"]
    model   = cfg["model"]["name"]
    split   = cfg.get("split", "random")
    cc      = "CC" if cfg["dataset"].get("use_cc", False) else "noCC"
    return f"{dataset}_{model}_{split}_{cc}"


def _subtitle(cfg, n_test, n_degrees):
    """Descriptive subtitle encoding the full experimental context."""
    dataset = cfg["dataset"]["name"]
    model   = cfg["model"]["name"]
    split   = cfg.get("split", "random")
    cc      = "CC" if cfg["dataset"].get("use_cc", False) else "noCC"
    return (f"{dataset} · {model} · {split} · {cc}"
            f"   |   {n_test:,} test nodes  ·  {n_degrees} unique degrees")


def _collect(run_results):
    """Convert per-run get_accuracy_deg dicts into a degree-keyed structure.

    Returns
    -------
    all_degrees : sorted list of all unique degree values across every run.
    deg_data    : dict {degree: [node_acc_array_run0, node_acc_array_run1, …]}
                  where each array holds per-node 0/1 correctness values.
                  An empty array signals that degree was absent in that run.
    """
    all_degrees = sorted({d for r in run_results for d in r})
    deg_data = {}
    for d in all_degrees:
        arrays = []
        for r in run_results:
            if d in r:
                arrays.append((r[d]["preds"] == r[d]["labels"]).float().numpy())
            else:
                arrays.append(np.array([]))
        deg_data[d] = arrays
    return all_degrees, deg_data


def _fig_w(n_deg, n_runs=1):
    """Figure width that scales with the number of degree groups and runs."""
    return max(10, min(n_deg * max(0.5, 0.35 * n_runs), 48))


_BP_KWARGS = dict(
    patch_artist=True,
    medianprops=dict(color="black", linewidth=1.5),
    whiskerprops=dict(linewidth=0.9),
    capprops=dict(linewidth=0.9),
    flierprops=dict(marker="o", markersize=3, alpha=0.35, markeredgewidth=0),
)


def _save(fig, save_dir, filename, show):
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, filename)
        fig.savefig(path, dpi=300, bbox_inches="tight")
        log.info("Saved %s", path)
    if show:
        plt.show()
    plt.close(fig)


def _subdir(save_dir, name):
    """Return save_dir/name if save_dir is set, else None."""
    return os.path.join(save_dir, name) if save_dir else None


def _degree_axis(ax, pos, all_degrees):
    """Set x-axis label and ticks; thin labels when there are many degrees."""
    ax.set_xlabel("Node degree", fontsize=11)
    ax.set_xlim(pos[0] - 0.6, pos[-1] + 0.6)
    step = max(1, len(all_degrees) // 30)
    ax.set_xticks(pos[::step])
    ax.set_xticklabels(all_degrees[::step], rotation=55, ha="right", fontsize=8)


def _count_bars(ax_main, pos, counts):
    """Add a secondary y-axis with light grey count bars behind the main plot.

    Bars are scaled to occupy the bottom ~25 % of the axis so they don't
    obscure the accuracy values.
    """
    ax2 = ax_main.twinx()
    ax2.bar(pos, counts, color="lightgrey", alpha=0.35, width=0.55, zorder=0)
    ax2.set_ylabel("# test nodes per degree", fontsize=8, color="grey")
    ax2.tick_params(axis="y", labelsize=7, colors="grey", length=3)
    ax2.spines["right"].set_color("lightgrey")
    ax2.set_ylim(0, max(counts) * 4)
    return ax2


# ── public entry points ────────────────────────────────────────────────────────

def plot_acc_vs_degree(run_results, cfg, save_dir=None, show=False):
    """Plot test-node accuracy vs. node degree.

    Single run  — scatter (bubble size ∝ node count) with WLS trend and Δ subplot.
    Multi-run   — one box per degree showing cross-seed variability, WLS trend,
                  count bars, and Δ subplot.

    Parameters
    ----------
    run_results : list[dict]
        One entry per run; output of ``get_accuracy_deg``.
    cfg : dict
    save_dir : str or None
    show : bool
    """
    n_runs      = len(run_results)
    all_degrees, deg_data = _collect(run_results)
    pos         = list(range(len(all_degrees)))
    prefix      = _fname_prefix(cfg)
    n_test      = sum(len(deg_data[d][0]) for d in all_degrees)
    subtitle    = _subtitle(cfg, n_test, len(all_degrees))

    if n_runs == 1:
        _plot_single(all_degrees, pos, deg_data, subtitle, prefix, save_dir, show)
    else:
        _plot_across_runs(all_degrees, pos, deg_data, n_runs, subtitle, prefix, save_dir, show)


# ── single run: scatter + OLS trend + Δ subplot ────────────────────────────────

def _plot_single(all_degrees, pos, deg_data, subtitle, prefix, save_dir, show):
    counts   = [len(deg_data[d][0]) for d in all_degrees]
    mean_acc = [
        float(deg_data[d][0].mean()) if counts[i] > 0 else np.nan
        for i, d in enumerate(all_degrees)
    ]
    n_test  = sum(counts)
    overall = (sum(a * c for a, c in zip(mean_acc, counts) if not np.isnan(a))
               / n_test)

    max_count   = max(counts) or 1
    bubble_size = [max(30, 700 * c / max_count) for c in counts]

    fig, ax_main = plt.subplots(
        figsize=(_fig_w(len(all_degrees)), 5),
    )

    # Scatter: one point per degree, size encodes node count
    ax_main.scatter(pos, mean_acc,
                    s=bubble_size, c="#3498db", alpha=0.78,
                    edgecolors="white", linewidths=0.6, zorder=3)

    # Bubble-size legend using three representative counts
    ref_counts = sorted({min(counts), int(np.median(counts)), max(counts)})
    for rc in ref_counts:
        ax_main.scatter([], [], s=max(30, 700 * rc / max_count),
                        c="#3498db", alpha=0.65, edgecolors="white",
                        label=f"n = {rc}")

    ax_main.axhline(overall, color="dimgrey", lw=1.0, ls=":",
                    label=f"Mean test acc ({overall:.1%})", zorder=2)

    ax_main.set_ylabel("Accuracy  (test nodes)", fontsize=11)
    ax_main.set_ylim(-0.05, 1.10)
    ax_main.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))
    ax_main.legend(loc="upper left", fontsize=8, framealpha=0.85,
                   title="Node count", title_fontsize=8)
    ax_main.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.4)
    ax_main.set_title(
        f"Accuracy vs. Node Degree  —  single run\n{subtitle}", fontsize=11
    )

    _degree_axis(ax_main, pos, all_degrees)

    fig.tight_layout()
    _save(fig, _subdir(save_dir, "acc_vs_degree"), f"{prefix}_acc_vs_degree_single_run.png", show)


# ── multiple runs: seed-to-seed variability per degree ─────────────────────────

def _plot_across_runs(all_degrees, pos, deg_data, n_runs, subtitle, prefix, save_dir, show):
    counts = [len(deg_data[d][0]) for d in all_degrees]
    n_test = sum(counts)

    per_run_means = []
    for d in all_degrees:
        means = [float(a.mean()) for a in deg_data[d] if len(a) > 0]
        per_run_means.append(means if means else [np.nan])

    median_accs = [float(np.median(m)) for m in per_run_means]
    overall     = (sum(a * c for a, c in zip(median_accs, counts) if not np.isnan(a))
                   / n_test)

    fig, ax_main = plt.subplots(
        figsize=(_fig_w(len(all_degrees)), 5),
    )

    bp = ax_main.boxplot(per_run_means, positions=pos, widths=0.6, **_BP_KWARGS)
    for patch in bp["boxes"]:
        patch.set_facecolor("#5b9bd5")
        patch.set_alpha(0.72)

    _count_bars(ax_main, pos, counts)
    ax_main.axhline(overall, color="dimgrey", lw=1.0, ls=":",
                    label=f"Mean test acc ({overall:.1%})", zorder=2)

    ax_main.set_ylabel(f"Mean accuracy per run  ({n_runs} seeds)", fontsize=11)
    ax_main.set_ylim(-0.05, 1.10)
    ax_main.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))
    ax_main.legend(loc="upper left", fontsize=8, framealpha=0.85)
    ax_main.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.4)
    ax_main.set_title(
        f"Accuracy vs. Node Degree  —  {n_runs} seeds\n{subtitle}", fontsize=11
    )

    _degree_axis(ax_main, pos, all_degrees)

    fig.tight_layout()
    _save(fig, _subdir(save_dir, "acc_vs_degree"), f"{prefix}_acc_vs_degree_across_runs.png", show)


# ── combined accuracy + distance vs degree ─────────────────────────────────────

def plot_combined_vs_degree(run_results, dist_deg_data, cfg,
                            save_dir=None, show=False, run_labels=None):
    """One combined figure per run: accuracy scatter paired with distances.

    Each figure has two side-by-side panels sharing the x-axis (node degree):
      Left  — Accuracy scatter (blue, left axis) + dist to any train node
               (orange, right axis, median ± IQR).
      Right — Accuracy scatter (blue, left axis) + dist to same-class train
               node (green, right axis, median ± IQR).

    Distance signals are graph-fixed (identical across runs).  Legends are
    anchored below each panel to avoid overlapping the data.  Files are named
    per run label, e.g. ``{prefix}_combined_vs_degree_run01_seed42.png``.

    Parameters
    ----------
    run_results : list[dict]
        One entry per run; output of ``get_accuracy_deg``.
    dist_deg_data : dict
        Output of ``utils.get_distance_deg``.
    cfg : dict
    save_dir : str or None
    show : bool
    run_labels : list[str] or None
    """
    num_layers = cfg["model"]["num_layers"]
    n_runs = len(run_results)
    all_degrees, deg_data = _collect(run_results)

    all_degrees = sorted(set(all_degrees) & set(dist_deg_data.keys()))
    pos    = list(range(len(all_degrees)))
    counts = [len(deg_data[d][0]) for d in all_degrees]
    n_test = sum(counts)
    prefix   = _fname_prefix(cfg)
    subtitle = _subtitle(cfg, n_test, len(all_degrees))

    # ── Distance signals — computed once, reused for every run ────────────────
    def _dist_stats(key):
        med, lo, hi = [], [], []
        for d in all_degrees:
            arr   = dist_deg_data[d][key]
            clean = arr[~np.isnan(arr)]
            if len(clean) == 0:
                med.append(np.nan); lo.append(np.nan); hi.append(np.nan)
            else:
                med.append(float(np.median(clean)))
                lo.append(float(np.percentile(clean, 25)))
                hi.append(float(np.percentile(clean, 75)))
        return np.array(med), np.array(lo), np.array(hi)

    d_tr_med, d_tr_lo, d_tr_hi = _dist_stats("dist_to_train")
    d_sc_med, d_sc_lo, d_sc_hi = _dist_stats("dist_to_same_class")

    # ── Drawing helpers ────────────────────────────────────────────────────────
    def _draw_distance(ax, d_med, d_lo, d_hi, color):
        ax_d = ax.twinx()
        ax_d.plot(pos, d_med, color=color, lw=2.2, zorder=4)
        ax_d.fill_between(pos, d_lo, d_hi, color=color, alpha=0.18, zorder=2)
        ax_d.axhline(num_layers, color="#e74c3c", lw=1.5, ls="--", zorder=6)
        ymax = max(np.nanmax(d_hi) if not np.all(np.isnan(d_hi)) else num_layers,
                   num_layers) * 1.3
        ax_d.set_ylim(0, ymax)
        ax_d.set_ylabel("Hop distance", fontsize=10)
        ax_d.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        ax_d.tick_params(axis="y", labelsize=8)

    # ── One figure per run ─────────────────────────────────────────────────────
    fw = max(_fig_w(len(all_degrees)), 12)

    for run_idx in range(n_runs):
        label   = run_labels[run_idx] if run_labels else f"run{run_idx + 1:02d}"
        acc     = np.array([
            float(deg_data[d][run_idx].mean()) if len(deg_data[d][run_idx]) > 0
            else np.nan
            for d in all_degrees
        ])
        overall = sum(a * c for a, c in zip(acc, counts) if not np.isnan(a)) / n_test

        fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(fw, 5), sharex=True)
        fig.suptitle(
            f"Accuracy & Hop Distance to Training Nodes vs. Node Degree  —  {label}"
            f"\n{subtitle}",
            fontsize=11, y=1.02,
        )

        def _draw_accuracy(ax, _acc=acc, _overall=overall):
            ax.scatter(pos, _acc, s=40, c="#3498db", alpha=0.9,
                       edgecolors="white", linewidths=0.5, zorder=4)
            ax.plot(pos, _acc, color="#3498db", lw=1.3, alpha=0.5, zorder=3)
            ax.axhline(_overall, color="dimgrey", lw=1.0, ls=":", zorder=2)
            ax.set_ylim(-0.05, 1.10)
            ax.set_ylabel("Accuracy", fontsize=10)
            ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))
            ax.tick_params(axis="y", labelsize=8)
            ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.3)

        def _below_legend(ax, dist_color, dist_label, _overall=overall):
            handles = [
                plt.Line2D([0], [0], color="#3498db", lw=2.0,
                           label="Accuracy  (per-degree mean)"),
                plt.Line2D([0], [0], color="dimgrey",  lw=1.0, ls=":",
                           label=f"Mean acc  {_overall:.1%}"),
                plt.Line2D([0], [0], color=dist_color, lw=2.0, label=dist_label),
                plt.Line2D([0], [0], color="#e74c3c",  lw=1.5, ls="--",
                           label=f"Model depth  ({num_layers} layers)"),
            ]
            ax.legend(handles=handles, loc="upper center",
                      bbox_to_anchor=(0.5, -0.22),
                      ncol=2, fontsize=8.5, framealpha=0.9, borderpad=0.8)

        _draw_accuracy(ax_l)
        _draw_distance(ax_l, d_tr_med, d_tr_lo, d_tr_hi, "#e67e22")
        ax_l.set_title("vs. Nearest Training Node  (any class)", fontsize=10, pad=6)
        _below_legend(ax_l, "#e67e22", "Dist to nearest train node  (median ± IQR)")
        _degree_axis(ax_l, pos, all_degrees)

        _draw_accuracy(ax_r)
        _draw_distance(ax_r, d_sc_med, d_sc_lo, d_sc_hi, "#27ae60")
        ax_r.set_title("vs. Nearest Same-Class Training Node", fontsize=10, pad=6)
        _below_legend(ax_r, "#27ae60",
                      "Dist to nearest same-class train node  (median ± IQR)")
        _degree_axis(ax_r, pos, all_degrees)

        fig.tight_layout()
        fig.subplots_adjust(bottom=0.26, wspace=0.5)
        _save(fig, _subdir(save_dir, "acc_vs_distance"), f"{prefix}_combined_vs_degree_{label}.png", show)


# ── accuracy vs cumulative k-hop degree ────────────────────────────────────────

def plot_acc_vs_khop_degree(run_results, cfg, k, save_dir=None, show=False):
    """Plot test-node accuracy vs. cumulative k-hop degree.

    Each unique k-hop degree value forms its own group on the x-axis —
    no binning applied.

    Single run  — scatter plot; bubble size ∝ node count per degree value.
    Multi-run   — box plot per degree value showing cross-seed variability.

    Parameters
    ----------
    run_results : list[dict]
        One entry per run; output of ``get_accuracy_deg`` called with the
        k-hop degree tensor in place of the standard 1-hop degree.
    cfg : dict
    k : int
        Neighbourhood radius used when computing the k-hop degree.
    save_dir : str or None
    show : bool
    """
    n_runs = len(run_results)
    all_degrees, deg_data = _collect(run_results)
    pos    = list(range(len(all_degrees)))
    prefix   = _fname_prefix(cfg)
    n_test   = sum(len(deg_data[d][0]) for d in all_degrees)
    subtitle = _subtitle(cfg, n_test, len(all_degrees))

    if n_runs == 1:
        _plot_khop_single(all_degrees, pos, deg_data, k, subtitle, prefix, save_dir, show)
    else:
        _plot_khop_across_runs(all_degrees, pos, deg_data, n_runs, k, subtitle, prefix, save_dir, show)


def _plot_khop_single(all_degrees, pos, deg_data, k, subtitle, prefix, save_dir, show):
    counts   = [len(deg_data[d][0]) for d in all_degrees]
    mean_acc = [
        float(deg_data[d][0].mean()) if counts[i] > 0 else np.nan
        for i, d in enumerate(all_degrees)
    ]
    n_test  = sum(counts)
    overall = (sum(a * c for a, c in zip(mean_acc, counts) if not np.isnan(a)) / n_test)

    max_count   = max(counts) or 1
    bubble_size = [max(30, 700 * c / max_count) for c in counts]

    fig, ax = plt.subplots(figsize=(_fig_w(len(all_degrees)), 5))

    ax.scatter(pos, mean_acc, s=bubble_size, c="#3498db", alpha=0.78,
               edgecolors="white", linewidths=0.6, zorder=3)

    ref_counts = sorted({min(counts), int(np.median(counts)), max(counts)})
    for rc in ref_counts:
        ax.scatter([], [], s=max(30, 700 * rc / max_count),
                   c="#3498db", alpha=0.65, edgecolors="white", label=f"n = {rc}")

    ax.axhline(overall, color="dimgrey", lw=1.0, ls=":",
               label=f"Mean test acc ({overall:.1%})", zorder=2)
    ax.set_ylabel("Accuracy  (test nodes)", fontsize=11)
    ax.set_ylim(-0.05, 1.10)
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))
    ax.legend(loc="upper left", fontsize=8, framealpha=0.85,
              title="Node count", title_fontsize=8)
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.4)
    ax.set_title(
        f"Accuracy vs. {k}-Hop Degree  —  single run\n{subtitle}", fontsize=11
    )
    ax.set_xlabel(f"Cumulative {k}-hop degree", fontsize=11)
    ax.set_xlim(pos[0] - 0.6, pos[-1] + 0.6)
    ax.set_xticks(pos)
    ax.set_xticklabels(all_degrees, rotation=55, ha="right", fontsize=8)

    fig.tight_layout()
    _save(fig, _subdir(save_dir, "acc_vs_khop_degree"), f"{prefix}_acc_vs_{k}hop_degree_single_run.png", show)


def _plot_khop_across_runs(all_degrees, pos, deg_data, n_runs, k, subtitle, prefix, save_dir, show):
    counts = [len(deg_data[d][0]) for d in all_degrees]
    n_test = sum(counts)

    per_run_means = []
    for d in all_degrees:
        means = [float(a.mean()) for a in deg_data[d] if len(a) > 0]
        per_run_means.append(means if means else [np.nan])

    median_accs = [float(np.median(m)) for m in per_run_means]
    overall     = (sum(a * c for a, c in zip(median_accs, counts) if not np.isnan(a)) / n_test)

    fig, ax = plt.subplots(figsize=(_fig_w(len(all_degrees)), 5))

    bp = ax.boxplot(per_run_means, positions=pos, widths=0.6, **_BP_KWARGS)
    for patch in bp["boxes"]:
        patch.set_facecolor("#5b9bd5")
        patch.set_alpha(0.72)

    _count_bars(ax, pos, counts)
    ax.axhline(overall, color="dimgrey", lw=1.0, ls=":",
               label=f"Mean test acc ({overall:.1%})", zorder=2)
    ax.set_ylabel(f"Mean accuracy per run  ({n_runs} seeds)", fontsize=11)
    ax.set_ylim(-0.05, 1.10)
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))
    ax.legend(loc="upper left", fontsize=8, framealpha=0.85)
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.4)
    ax.set_title(
        f"Accuracy vs. {k}-Hop Degree  —  {n_runs} seeds\n{subtitle}", fontsize=11
    )
    ax.set_xlabel(f"Cumulative {k}-hop degree", fontsize=11)
    ax.set_xlim(pos[0] - 0.6, pos[-1] + 0.6)
    ax.set_xticks(pos)
    ax.set_xticklabels(all_degrees, rotation=55, ha="right", fontsize=8)

    fig.tight_layout()
    _save(fig, _subdir(save_dir, "acc_vs_khop_degree"), f"{prefix}_acc_vs_{k}hop_degree_across_runs.png", show)


# ── accuracy vs. 1-hop degree: grouped boxplots by num_layers ──────────────────

# Colorblind-friendly, high-contrast palette (tab10 subset, visually distinct)
_LAYER_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
                 "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]


def plot_acc_vs_degree_by_layers(results_by_label, cfg, save_dir=None, show=False):
    """Grouped boxplot of accuracy vs. 1-hop degree, one box per (model, num_layers).

    For each degree group on the x-axis there is one boxplot per label (e.g.
    ``"GCN L=2"`` or ``"GCNII L=3"``).  Each boxplot shows the distribution of
    per-run mean accuracies across seeds.

    Parameters
    ----------
    results_by_label : dict[str, list[dict]]
        Ordered mapping from a human-readable label to a list of per-run
        ``get_accuracy_deg`` dicts.  Insertion order determines box order.
    cfg : dict
    save_dir : str or None
    show : bool
    """
    labels   = list(results_by_label.keys())
    n_labels = len(labels)
    if n_labels == 0:
        return

    # Collect all unique degree values across every configuration
    all_degrees = sorted({d for results in results_by_label.values()
                          for run in results for d in run})
    n_deg  = len(all_degrees)
    pos    = list(range(n_deg))
    prefix = _fname_prefix(cfg)

    # Node counts are graph-fixed — use the first label's results
    _, first_deg_data = _collect(results_by_label[labels[0]])
    counts = [len(first_deg_data[d][0]) if d in first_deg_data else 0
              for d in all_degrees]
    n_test   = sum(counts)
    subtitle = _subtitle(cfg, n_test, n_deg)
    n_runs   = len(results_by_label[labels[0]])

    # Per-label, per-degree: list of per-run mean accuracies
    label_deg_means = {}
    for lbl in labels:
        _, deg_data = _collect(results_by_label[lbl])
        label_deg_means[lbl] = {}
        for d in all_degrees:
            if d in deg_data:
                means = [float(a.mean()) for a in deg_data[d] if len(a) > 0]
                label_deg_means[lbl][d] = means if means else [np.nan]
            else:
                label_deg_means[lbl][d] = [np.nan]

    # Box geometry
    group_w  = 0.8
    box_w    = group_w / n_labels
    offsets  = np.linspace(-group_w / 2 + box_w / 2,
                            group_w / 2 - box_w / 2, n_labels)

    fig, ax = plt.subplots(figsize=(_fig_w(n_deg, n_labels), 5))

    for i, lbl in enumerate(labels):
        bpos  = [p + offsets[i] for p in pos]
        data  = [label_deg_means[lbl][d] for d in all_degrees]
        color = _LAYER_COLORS[i % len(_LAYER_COLORS)]

        bp = ax.boxplot(data, positions=bpos, widths=box_w * 0.85, **_BP_KWARGS)
        for patch in bp["boxes"]:
            patch.set_facecolor(color)
            patch.set_alpha(0.65)

        ax.scatter([], [], color=color, s=30, alpha=0.85, label=lbl)

    _count_bars(ax, pos, counts)

    ax.set_ylabel(f"Mean accuracy per run  ({n_runs} seeds)", fontsize=11)
    ax.set_ylim(-0.05, 1.10)
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), borderaxespad=0,
              fontsize=8, framealpha=0.9)
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.4)
    ax.set_title(
        f"Accuracy vs. Node Degree  —  model × layers  ({n_runs} seeds)\n{subtitle}",
        fontsize=11,
    )

    ax.set_xlabel("Node degree", fontsize=11)
    ax.set_xlim(pos[0] - 0.6, pos[-1] + 0.6)
    step = max(1, n_deg // 30)
    ax.set_xticks(pos[::step])
    ax.set_xticklabels(all_degrees[::step], rotation=55, ha="right", fontsize=8)

    fig.tight_layout()
    models_str = "_".join(dict.fromkeys(lbl.split()[0] for lbl in labels))
    _save(fig, _subdir(save_dir, "acc_vs_degree_by_layers"), f"{prefix}_acc_vs_degree_by_layers_{models_str}.png", show)


# ── neighborhood purity vs. 1-hop degree ───────────────────────────────────────

def plot_purity_vs_degree(test_deg, purity_test, cfg, k, save_dir=None, show=False):
    """Scatter plot of mean neighborhood purity vs. 1-hop degree.

    Nodes are grouped by their 1-hop degree.  For each group the mean purity
    across test nodes is plotted; bubble size encodes node count.

    purity(v) = |same-class nodes in N_k(v)| / |N_k(v)|

    Parameters
    ----------
    test_deg    : 1-D LongTensor of 1-hop degrees for test nodes.
    purity_test : 1-D FloatTensor of purity values for test nodes (NaN-safe).
    cfg         : dict
    k           : neighbourhood radius used to compute purity.
    save_dir    : str or None
    show        : bool
    """
    deg    = test_deg.cpu()
    purity = purity_test.cpu()

    unique_degrees = sorted(deg.unique().tolist())
    pos    = list(range(len(unique_degrees)))
    prefix = _fname_prefix(cfg)

    counts    = []
    mean_purs = []
    for d in unique_degrees:
        mask  = deg == d
        vals  = purity[mask]
        valid = vals[~torch.isnan(vals)]
        counts.append(int(mask.sum()))
        mean_purs.append(float(valid.mean()) if len(valid) > 0 else float("nan"))

    n_test   = sum(counts)
    n_deg    = len(unique_degrees)
    subtitle = _subtitle(cfg, n_test, n_deg)

    max_count   = max(counts) or 1
    bubble_size = [max(30, 700 * c / max_count) for c in counts]

    fig, ax = plt.subplots(figsize=(_fig_w(n_deg), 5))

    ax.scatter(pos, mean_purs, s=bubble_size, c="#e67e22", alpha=0.78,
               edgecolors="white", linewidths=0.6, zorder=3)

    ref_counts = sorted({min(counts), int(np.median(counts)), max(counts)})
    for rc in ref_counts:
        ax.scatter([], [], s=max(30, 700 * rc / max_count),
                   c="#e67e22", alpha=0.65, edgecolors="white", label=f"n = {rc}")

    # Overall mean purity (weighted by node count)
    valid_pairs = [(m, c) for m, c in zip(mean_purs, counts) if not np.isnan(m)]
    if valid_pairs:
        overall = sum(m * c for m, c in valid_pairs) / sum(c for _, c in valid_pairs)
        ax.axhline(overall, color="dimgrey", lw=1.0, ls=":",
                   label=f"Mean purity ({overall:.1%})", zorder=2)

    _count_bars(ax, pos, counts)

    ax.set_ylabel(f"Mean neighborhood purity  (k={k})", fontsize=11)
    ax.set_ylim(-0.05, 1.10)
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))
    ax.legend(loc="upper left", fontsize=8, framealpha=0.85,
              title="Node count", title_fontsize=8)
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.4)
    ax.set_title(
        f"Neighborhood Purity vs. Node Degree  (k={k})\n{subtitle}", fontsize=11
    )

    ax.set_xlabel("Node degree", fontsize=11)
    ax.set_xlim(pos[0] - 0.6, pos[-1] + 0.6)
    step = max(1, n_deg // 30)
    ax.set_xticks(pos[::step])
    ax.set_xticklabels(unique_degrees[::step], rotation=55, ha="right", fontsize=8)

    fig.tight_layout()
    _save(fig, _subdir(save_dir, "purity_vs_degree"), f"{prefix}_purity_vs_degree_k{k}.png", show)


# ── purity evolution vs. k, one line per degree group ─────────────────────────

def plot_purity_vs_k_by_degree(test_deg, purity_by_k, cfg,
                                save_dir=None, show=False, min_count=3):
    """Line plot showing how neighborhood purity evolves with k, per degree group.

    X-axis is the neighbourhood radius k; each line represents nodes that share
    the same 1-hop degree.  Degree groups with fewer than ``min_count`` test
    nodes are omitted to avoid noisy / unrepresentative lines.

    Parameters
    ----------
    test_deg    : 1-D LongTensor of 1-hop degrees for test nodes.
    purity_by_k : dict { k (int) -> 1-D FloatTensor of purity values }
                  Purity tensors must be aligned with test_deg (same ordering).
    cfg         : dict
    save_dir    : str or None
    show        : bool
    min_count   : int — skip degree groups smaller than this (default 3).
    """
    deg      = test_deg.cpu()
    k_values = sorted(purity_by_k.keys())

    unique_degrees = sorted(deg.unique().tolist())
    # Filter to groups with enough nodes
    unique_degrees = [d for d in unique_degrees
                      if int((deg == d).sum()) >= min_count]

    prefix   = _fname_prefix(cfg)
    n_test   = int(len(deg))
    subtitle = _subtitle(cfg, n_test, len(unique_degrees))

    cmap   = plt.get_cmap("viridis", max(len(unique_degrees), 1))
    fig, ax = plt.subplots(figsize=(7, 5))

    for i, d in enumerate(unique_degrees):
        mask  = (deg == d).numpy()
        count = int(mask.sum())

        mean_purs = []
        for k in k_values:
            vals  = purity_by_k[k].cpu().numpy()[mask]
            valid = vals[~np.isnan(vals)]
            mean_purs.append(float(valid.mean()) if len(valid) > 0 else float("nan"))

        ax.plot(k_values, mean_purs,
                marker="o", linewidth=1.8, markersize=5,
                color=cmap(i),
                label=f"deg={d}  (n={count})")

    ax.set_xlabel("Neighbourhood radius  k", fontsize=11)
    ax.set_ylabel("Mean neighbourhood purity", fontsize=11)
    ax.set_ylim(-0.05, 1.10)
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))
    ax.set_xticks(k_values)
    ax.grid(axis="both", linestyle="--", linewidth=0.5, alpha=0.4)
    ax.legend(loc="best", fontsize=7, framealpha=0.85, ncol=2,
              title="Degree group", title_fontsize=7)
    ax.set_title(
        f"Purity Evolution by Degree  (k = {k_values[0]}…{k_values[-1]})\n{subtitle}",
        fontsize=11,
    )

    fig.tight_layout()
    k_range = f"k{k_values[0]}-{k_values[-1]}"
    _save(fig, _subdir(save_dir, "purity_vs_degree"),
          f"{prefix}_purity_vs_k_by_degree_{k_range}.png", show)


# # ── AMP heterogeneity distribution + DMP counts vs degree ──────────────────────
# 
# def plot_amp_dmp_vs_degree(amp_deg_data, dmp_deg_data, cfg,
#                            save_dir=None, show=False):
#     """Two-panel figure: AMP heterogeneity and DMP rate vs node degree.
# 
#     Each unique degree is shown as its own tick — no binning.
# 
#     Left  — Boxplot of neighbour-heterogeneity ratios per degree.
#             A dashed line marks the AMP threshold (het > threshold ⇒ AMP).
# 
#     Right — Line chart of DMP rate (% of nodes lacking a same-class training
#             node within dmp_coeff hops) per degree.
#     """
#     all_degrees = sorted(set(amp_deg_data.keys()) & set(dmp_deg_data.keys()))
#     raw_counts  = [amp_deg_data[d]["count"] for d in all_degrees]
#     n_test      = sum(raw_counts)
#     prefix      = _fname_prefix(cfg)
#     amp_coeff   = cfg["dataset"].get("amp_coeff", 1)
#     dmp_coeff   = cfg["dataset"].get("dmp_coeff", 1)
#     amp_thr     = cfg["dataset"].get("amp_threshold", 0.5)
#     subtitle    = _subtitle(cfg, n_test, len(all_degrees))
# 
#     pos = list(range(len(all_degrees)))
# 
#     def _clean(arr):
#         a = arr[~np.isnan(arr)]
#         return a if len(a) > 0 else np.array([np.nan])
# 
#     het_per_deg = [_clean(amp_deg_data[d]["het_values"]) for d in all_degrees]
# 
#     dmp_rate = np.array([
#         dmp_deg_data[d]["count_1"] / dmp_deg_data[d]["count"]
#         if dmp_deg_data[d]["count"] > 0 else np.nan
#         for d in all_degrees
#     ])
# 
#     # ── Figure ─────────────────────────────────────────────────────────────────
#     n_deg = len(all_degrees)
#     fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(_fig_w(n_deg) + 4, 5))
#     fig.suptitle(
#         f"AMP ({amp_coeff}-hop) Heterogeneity  &  DMP ({dmp_coeff}-hop) Rate  vs. Degree\n"
#         f"{subtitle}",
#         fontsize=11, y=1.02,
#     )
# 
#     # ── Left: AMP heterogeneity boxplots ───────────────────────────────────────
#     bp = ax_l.boxplot(het_per_deg, positions=pos, widths=0.55, **_BP_KWARGS)
#     for patch in bp["boxes"]:
#         patch.set_facecolor("#e67e22")
#         patch.set_alpha(0.80)
# 
#     ax_l.axhline(amp_thr, color="#c0392b", lw=1.4, ls="--", zorder=6,
#                  label=f"AMP threshold ({amp_thr:.0%})")
#     ax_l.set_ylabel("Neighbour heterogeneity ratio", fontsize=10)
#     ax_l.set_ylim(-0.05, 1.10)
#     ax_l.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))
#     ax_l.tick_params(axis="y", labelsize=8)
#     ax_l.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.4)
#     ax_l.set_title("AMP: neighbour heterogeneity per degree\n"
#                    "(box = IQR, centre line = median)",
#                    fontsize=10, pad=6)
#     ax_l.legend(loc="upper left", bbox_to_anchor=(0, -0.18),
#                 borderaxespad=0, fontsize=9, framealpha=0.85, ncol=1)
#     _degree_axis(ax_l, pos, all_degrees)
# 
#     # ── Right: DMP rate per degree ──────────────────────────────────────────────
#     ax_r.plot(pos, dmp_rate, color="#8e44ad", lw=2.2, marker="o",
#               markersize=5, zorder=4, label=f"DMP rate ({dmp_coeff}-hop)")
#     ax_r.fill_between(pos, 0, dmp_rate, color="#8e44ad", alpha=0.15, zorder=2)
# 
#     ax_r.set_ylim(0, 1.10)
#     ax_r.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))
#     ax_r.tick_params(axis="y", labelsize=8)
#     ax_r.set_ylabel("% of nodes with DMP", fontsize=10)
#     ax_r.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.4)
#     ax_r.set_title("DMP: fraction lacking a nearby\nsame-class training node",
#                    fontsize=10, pad=6)
#     ax_r.legend(loc="upper left", bbox_to_anchor=(0, -0.18),
#                 borderaxespad=0, fontsize=9, framealpha=0.85, ncol=1)
#     _degree_axis(ax_r, pos, all_degrees)
# 
#     fig.tight_layout()
#     fig.subplots_adjust(bottom=0.25, wspace=0.45)
#     _save(fig, save_dir, f"{prefix}_amp_dmp_vs_degree.png", show)
# 
# 
# # ── accuracy by AMP × DMP group ────────────────────────────────────────────────
# 
# def plot_acc_by_amp_dmp_group(group_acc_per_run, group_names, group_counts,
#                                cfg, save_dir=None, show=False):
#     """Compare classification accuracy across four AMP × DMP groups.
# 
#     Groups (x-axis, left to right):
#       0 – Low AMP + No DMP   (structurally easy nodes)
#       1 – Low AMP + DMP
#       2 – High AMP + No DMP
#       3 – High AMP + DMP     (structurally hard nodes)
# 
#     Single run  — horizontal bar per group, annotated with node count.
#     Multi-run   — boxplot per group (distribution across seeds), with node
#                   count shown below each box.
# 
#     The two extreme groups (0 and 3) are highlighted to make the easy vs
#     hard comparison immediately visible.
# 
#     Parameters
#     ----------
#     group_acc_per_run : list[list[float]]
#         Outer list: one entry per group (4 total).
#         Inner list: one accuracy value per run.
#     group_names : list[str]
#         Human-readable label for each group (4 total).
#     group_counts : list[int]
#         Number of test nodes per group.
#     cfg : dict
#     save_dir : str or None
#     show : bool
#     """
#     n_runs   = len(group_acc_per_run[0])
#     prefix   = _fname_prefix(cfg)
#     amp_coeff = cfg["dataset"].get("amp_coeff", 1)
#     dmp_coeff = cfg["dataset"].get("dmp_coeff", 1)
#     amp_thr   = cfg["dataset"].get("amp_threshold", 0.5)
# 
#     # Colours: highlight groups 0 (easy, green) and 3 (hard, red); others grey
#     GROUP_COLORS = ["#27ae60", "#95a5a6", "#95a5a6", "#e74c3c"]
#     GROUP_EDGE   = ["#1e8449", "#7f8c8d", "#7f8c8d", "#c0392b"]
# 
#     n_test   = sum(group_counts)
#     subtitle = (f"{cfg['dataset']['name']} · {cfg['model']['name']} · "
#                 f"{cfg.get('split','random')} · "
#                 f"{'CC' if cfg['dataset'].get('use_cc') else 'noCC'}"
#                 f"   |   AMP {amp_coeff}-hop  thr={amp_thr}  ·  DMP {dmp_coeff}-hop"
#                 f"   |   {n_test:,} test nodes")
# 
#     pos = list(range(4))
# 
#     fig, ax = plt.subplots(figsize=(8, 5))
# 
#     if n_runs == 1:
#         accs = [vals[0] for vals in group_acc_per_run]
#         bars = ax.bar(pos, accs, color=GROUP_COLORS, edgecolor=GROUP_EDGE,
#                       linewidth=1.2, alpha=0.85, zorder=3)
#         for bar, acc, cnt in zip(bars, accs, group_counts):
#             ax.text(bar.get_x() + bar.get_width() / 2,
#                     bar.get_height() + 0.015,
#                     f"{acc:.1%}", ha="center", va="bottom", fontsize=9,
#                     fontweight="bold")
#             ax.text(bar.get_x() + bar.get_width() / 2,
#                     -0.04, f"n={cnt}", ha="center", va="top",
#                     fontsize=8, color="dimgrey")
#     else:
#         bp = ax.boxplot(group_acc_per_run, positions=pos, widths=0.55,
#                         **_BP_KWARGS)
#         for patch, color in zip(bp["boxes"], GROUP_COLORS):
#             patch.set_facecolor(color)
#             patch.set_alpha(0.80)
#         for cnt, x in zip(group_counts, pos):
#             ax.text(x, -0.06, f"n={cnt}", ha="center", va="top",
#                     fontsize=8, color="dimgrey",
#                     transform=ax.get_xaxis_transform())
# 
#     # Annotate the two extremes
#     ax.annotate("← easy nodes", xy=(0, 0.02), xycoords=("data", "axes fraction"),
#                 fontsize=8, color="#1e8449", ha="center")
#     ax.annotate("hard nodes →", xy=(3, 0.02), xycoords=("data", "axes fraction"),
#                 fontsize=8, color="#c0392b", ha="center")
# 
#     overall = (sum(a[0] * c for a, c in zip(group_acc_per_run, group_counts))
#                / n_test)
#     ax.axhline(overall, color="dimgrey", lw=1.0, ls=":",
#                label=f"Overall test acc  {overall:.1%}", zorder=2)
# 
#     ax.set_xticks(pos)
#     ax.set_xticklabels(group_names, fontsize=10)
#     ax.set_ylabel("Accuracy", fontsize=11)
#     ax.set_ylim(-0.05, 1.15)
#     ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))
#     ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.4)
#     ax.legend(loc="upper right", fontsize=9, framealpha=0.85)
#     ax.set_title(
#         f"Accuracy by AMP × DMP Group  "
#         f"({'single run' if n_runs == 1 else f'{n_runs} seeds'})\n{subtitle}",
#         fontsize=10,
#     )
# 
#     fig.tight_layout()
#     _save(fig, save_dir, f"{prefix}_acc_by_amp_dmp_group.png", show)
# 
# 
# # ── accuracy by AMP × DMP group, stratified by degree ──────────────────────────
# 
# def plot_acc_by_amp_dmp_group_vs_degree(group_deg_acc, group_names, cfg,
#                                          save_dir=None, show=False):
#     """Accuracy per AMP × DMP group vs binned node degree — 4 overlaid lines.
# 
#     Degrees are binned into equal-count buckets so each point on the x-axis
#     represents a comparable number of nodes.  For multi-run experiments each
#     line shows the cross-seed mean with a ±1σ shaded band.
# 
#     Answers: does the group ranking (Low AMP + No DMP best, High AMP + DMP
#     worst) hold consistently at every degree, or does it collapse/reverse for
#     very low- or very high-degree nodes?
#     """
#     GROUP_COLORS  = ["#27ae60", "#5d6d7e", "#a569bd", "#e74c3c"]
#     GROUP_MARKERS = ["o", "s", "^", "D"]
#     GROUP_LS      = ["-", "--", "-.", ":"]
# 
#     prefix    = _fname_prefix(cfg)
#     amp_coeff = cfg["dataset"].get("amp_coeff", 1)
#     dmp_coeff = cfg["dataset"].get("dmp_coeff", 1)
#     amp_thr   = cfg["dataset"].get("amp_threshold", 0.5)
#     dataset   = cfg["dataset"]["name"]
#     model     = cfg["model"]["name"]
#     split     = cfg.get("split", "random")
#     cc        = "CC" if cfg["dataset"].get("use_cc") else "noCC"
# 
#     all_degrees = sorted({d for g in range(4) for d in group_deg_acc[g]})
#     if not all_degrees:
#         return
# 
#     # Use total node count across all groups for degree binning
#     deg_total = {}
#     for d in all_degrees:
#         deg_total[d] = sum(len(group_deg_acc[g].get(d, [])) for g in range(4))
#     raw_counts  = [deg_total[d] for d in all_degrees]
#     bin_of_deg, bin_labels, n_bins = _make_degree_bins(all_degrees, raw_counts)
#     pos = list(range(n_bins))
# 
#     n_runs = max((len(v) for g in range(4) for v in group_deg_acc[g].values()),
#                  default=1)
# 
#     # Aggregate per-run accuracy into bins: mean over degrees in each bin
#     fig, ax = plt.subplots(figsize=(max(10, n_bins * 1.5 + 3), 5))
# 
#     for g, (color, marker, ls, name) in enumerate(
#         zip(GROUP_COLORS, GROUP_MARKERS, GROUP_LS, group_names)
#     ):
#         bin_means, bin_stds = [], []
#         for b in range(n_bins):
#             degs_in_bin = [d for d in all_degrees if bin_of_deg[d] == b]
#             # Collect all per-run accuracy values for this (group, bin)
#             run_accs = []
#             for run_i in range(n_runs):
#                 vals = [group_deg_acc[g][d][run_i]
#                         for d in degs_in_bin
#                         if d in group_deg_acc[g] and run_i < len(group_deg_acc[g][d])
#                         and not np.isnan(group_deg_acc[g][d][run_i])]
#                 if vals:
#                     run_accs.append(float(np.mean(vals)))
#             if run_accs:
#                 bin_means.append(float(np.mean(run_accs)))
#                 bin_stds.append(float(np.std(run_accs)))
#             else:
#                 bin_means.append(np.nan)
#                 bin_stds.append(np.nan)
# 
#         xs    = np.array(pos, dtype=float)
#         means = np.array(bin_means)
#         stds  = np.array(bin_stds)
#         label = name.replace("\n", " ")
# 
#         valid = ~np.isnan(means)
#         ax.plot(xs[valid], means[valid], color=color, lw=2.2, ls=ls,
#                 marker=marker, markersize=7, zorder=4, label=label)
#         if n_runs > 1:
#             ax.fill_between(xs[valid],
#                             (means - stds)[valid], (means + stds)[valid],
#                             color=color, alpha=0.13, zorder=2)
# 
#     ax.set_ylabel("Accuracy", fontsize=11)
#     ax.set_ylim(-0.05, 1.10)
#     ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))
#     ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.4)
#     ax.set_xticks(pos)
#     ax.set_xticklabels(bin_labels, rotation=40, ha="right", fontsize=9)
#     ax.set_xlabel("Node degree (binned)", fontsize=10)
# 
#     run_str = "single run" if n_runs == 1 else f"{n_runs} seeds  (mean ± 1σ)"
#     ax.set_title(
#         f"Does the accuracy gap between AMP × DMP groups persist across degrees?  "
#         f"—  {run_str}\n"
#         f"{dataset} · {model} · {split} · {cc}"
#         f"   |   AMP {amp_coeff}-hop  thr={amp_thr}  ·  DMP {dmp_coeff}-hop",
#         fontsize=10,
#     )
#     ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22),
#               ncol=4, fontsize=9.5, framealpha=0.9, borderpad=0.8)
# 
#     fig.tight_layout()
#     fig.subplots_adjust(bottom=0.24)
#     _save(fig, save_dir, f"{prefix}_acc_by_amp_dmp_group_vs_degree.png", show)
# 
# 
# 
# # ── Totoro advantage vs degree/AMP for High AMP / No DMP nodes ─────────────────
# 
# def plot_totoro_advantage_group2(node_data, cfg, save_dir=None, show=False):
#     """Two-panel node-wise scatter for Group 2 (High AMP, No DMP) test nodes.
# 
#     For each node the Totoro advantage is defined as:
#         advantage = mean_same_class_Totoro − mean_diff_class_Totoro
# 
#     A positive advantage means the same-class training neighbours have higher
#     Totoro scores (they receive more cross-class PPR confusion themselves) —
#     the correct-class signal is *noisier* than the wrong-class signal.
#     A negative advantage means the correct-class signal is cleaner.
# 
#     The key question is whether this advantage, together with degree, can
#     explain whether high AMP (heterogeneous neighbourhood) is compensated.
# 
#     Left  — Advantage vs node degree. Each dot is one node, coloured by its
#             continuous AMP score (heterogeneity ratio).  Horizontal dashed
#             line at zero separates the two regimes.
# 
#     Right — Advantage vs AMP score (het ratio). Each dot coloured by degree.
#             Vertical dashed line at the AMP threshold; horizontal at zero.
# 
#     Nodes without diff-class training neighbours are excluded (noted in title).
#     """
#     amp_coeff = cfg["dataset"].get("amp_coeff", 1)
#     amp_thr   = cfg["dataset"].get("amp_threshold", 0.5)
#     prefix    = _fname_prefix(cfg)
#     subtitle  = (
#         f"{cfg['dataset']['name']} · {cfg['model']['name']} · "
#         f"{cfg.get('split', 'random')} · "
#         f"{'CC' if cfg['dataset'].get('use_cc') else 'noCC'}"
#         f"   |   {amp_coeff}-hop   |   Group 2: High AMP, No DMP"
#     )
# 
#     deg  = node_data['degree']
#     het  = node_data['het']
#     same = node_data['same_totoro']
#     diff = node_data['diff_totoro']
# 
#     # Require both same-class and diff-class neighbours to compute advantage
#     valid      = ~(np.isnan(same) | np.isnan(diff))
#     n_total    = len(deg)
#     n_valid    = int(valid.sum())
#     n_excluded = n_total - n_valid
# 
#     deg_v  = deg[valid]
#     het_v  = het[valid]
#     adv_v  = same[valid] - diff[valid]
# 
#     if n_valid == 0:
#         log.warning("No Group 2 nodes with both same- and diff-class neighbours.")
#         return
# 
#     pct_positive = 100 * (adv_v > 0).sum() / n_valid
# 
#     fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(13, 5))
#     fig.suptitle(
#         f"Totoro advantage per node  (same − diff class Totoro)  —  Group 2\n"
#         f"positive = correct signal noisier,  negative = correct signal cleaner\n"
#         f"{subtitle}   |   {n_valid} nodes  ({n_excluded} excluded, no diff-class neighbour)",
#         fontsize=10, y=1.03,
#     )
# 
#     # ── Left: advantage vs degree, coloured by het ─────────────────────────────
#     sc_l = ax_l.scatter(deg_v, adv_v, c=het_v, cmap="YlOrRd",
#                         vmin=amp_thr, vmax=1.0,
#                         s=30, alpha=0.7, linewidths=0, zorder=3)
#     ax_l.axhline(0, color="dimgrey", lw=1.2, ls="--", zorder=2)
#     ax_l.text(0.98, 0.02, f"{pct_positive:.1f}% nodes: same > diff",
#               transform=ax_l.transAxes, ha="right", va="bottom", fontsize=8.5,
#               color="dimgrey")
#     ax_l.set_xlabel("Node degree", fontsize=10)
#     ax_l.set_ylabel("Totoro advantage  (same − diff)", fontsize=10)
#     ax_l.set_title("Advantage vs degree\n(colour = AMP heterogeneity score)",
#                    fontsize=9, pad=5)
#     ax_l.grid(linestyle="--", linewidth=0.4, alpha=0.4)
#     cb_l = fig.colorbar(sc_l, ax=ax_l, pad=0.02)
#     cb_l.set_label("Heterogeneity ratio (AMP score)", fontsize=8)
#     cb_l.ax.tick_params(labelsize=7)
# 
#     # ── Right: advantage vs het, coloured by degree ─────────────────────────────
#     sc_r = ax_r.scatter(het_v, adv_v, c=deg_v, cmap="viridis_r",
#                         s=30, alpha=0.7, linewidths=0, zorder=3)
#     ax_r.axhline(0, color="dimgrey", lw=1.2, ls="--", zorder=2)
#     ax_r.axvline(amp_thr, color="#c0392b", lw=1.2, ls="--", zorder=2,
#                  label=f"AMP threshold ({amp_thr:.0%})")
#     ax_r.set_xlabel("Heterogeneity ratio (AMP score)", fontsize=10)
#     ax_r.set_ylabel("Totoro advantage  (same − diff)", fontsize=10)
#     ax_r.set_title("Advantage vs AMP score\n(colour = node degree)",
#                    fontsize=9, pad=5)
#     ax_r.grid(linestyle="--", linewidth=0.4, alpha=0.4)
#     ax_r.legend(loc="upper left", bbox_to_anchor=(0, -0.14),
#                 borderaxespad=0, fontsize=9, framealpha=0.85)
#     cb_r = fig.colorbar(sc_r, ax=ax_r, pad=0.02)
#     cb_r.set_label("Node degree", fontsize=8)
#     cb_r.ax.tick_params(labelsize=7)
# 
#     fig.tight_layout()
#     fig.subplots_adjust(bottom=0.18, wspace=0.40)
#     _save(fig, save_dir, f"{prefix}_totoro_advantage_group2.png", show)
# 



# ── Unused functions ──────────────────────────────────────────────────────────

# def _make_degree_bins(all_degrees, counts, n_bins=8):
#     """Bin degree values into n_bins groups each containing roughly equal numbers
#     of test nodes (equal-count / quantile binning).
#
#     Parameters
#     ----------
#     all_degrees : sorted list[int]
#     counts      : list[int], test-node count for each degree (same order)
#     n_bins      : target number of bins
#
#     Returns
#     -------
#     bin_of_degree : dict {degree -> bin_index}
#     bin_labels    : list[str] – human-readable ranges, e.g. "1", "2–3", "≥16"
#     n_actual      : int – actual number of bins produced
#     """
#     arr  = np.array(all_degrees, dtype=int)
#     cnts = np.array(counts, dtype=float)
#
#     if len(arr) <= n_bins:
#         return ({int(d): i for i, d in enumerate(arr)},
#                 [str(d) for d in arr],
#                 len(arr))
#
#     cum     = np.cumsum(cnts)
#     targets = np.linspace(0, cum[-1], n_bins + 1)[1:-1]
#     bounds  = sorted({int(arr[min(int(np.searchsorted(cum, t)), len(arr) - 1)])
#                       for t in targets})
#
#     bin_of_degree = {int(d): int(np.searchsorted(bounds, d, side="right"))
#                      for d in arr}
#
#     n_actual = max(bin_of_degree.values()) + 1
#     bin_labels = []
#     for b in range(n_actual):
#         in_b = sorted(d for d, bi in bin_of_degree.items() if bi == b)
#         lo, hi = in_b[0], in_b[-1]
#         if lo == hi:
#             bin_labels.append(str(lo))
#         elif b == n_actual - 1:
#             bin_labels.append(f"≥{lo}")
#         else:
#             bin_labels.append(f"{lo}–{hi}")
#     return bin_of_degree, bin_labels, n_actual
#
# def _add_trend(ax, all_degrees, pos, accs, counts=None):
#     """Fit WLS on (degree, accuracy) weighted by node count and overlay as a
#     dashed line with a shaded 95 % confidence band.
#
#     Using WLS (rather than plain OLS) down-weights degree buckets that contain
#     very few test nodes, whose accuracy estimates are noisier.  The slope is in
#     units of accuracy-change per unit degree — a positive slope means
#     higher-degree nodes tend to be classified more accurately.
#
#     The 95 % CI band is derived from the standard error of the WLS fit
#     evaluated at each interpolated degree value.
#     """
#     pairs = [(d, p, a, c) for d, p, a, c in
#              zip(all_degrees, pos, accs, counts if counts is not None else [1] * len(all_degrees))
#              if not np.isnan(a)]
#     if len(pairs) < 2:
#         return
#     degs, poss, vals, wts = map(list, zip(*pairs))
#     degs = np.array(degs, dtype=float)
#     vals = np.array(vals, dtype=float)
#     wts  = np.array(wts,  dtype=float)
#
#     # WLS via weighted polyfit
#     z = np.polyfit(degs, vals, 1, w=wts)
#
#     # Residual standard error for the confidence band
#     y_hat = np.polyval(z, degs)
#     resid = vals - y_hat
#     # Weighted residual sum of squares
#     wrss  = np.sum(wts * resid ** 2)
#     dof   = max(len(degs) - 2, 1)
#     s2    = wrss / dof
#
#     x_deg = np.linspace(degs.min(), degs.max(), 200)
#     x_pos = np.interp(x_deg, degs, poss)
#     y_fit = np.polyval(z, x_deg)
#
#     # Leverage for each interpolated point: var(ŷ) = s² * x'(X'WX)⁻¹x
#     W   = np.diag(wts)
#     X   = np.column_stack([degs, np.ones_like(degs)])
#     XtW = X.T @ W
#     try:
#         cov = np.linalg.inv(XtW @ X) * s2
#         X_new = np.column_stack([x_deg, np.ones_like(x_deg)])
#         se_fit = np.sqrt(np.einsum("ij,jk,ik->i", X_new, cov, X_new))
#         ci = 1.96 * se_fit
#         ax.fill_between(x_pos, y_fit - ci, y_fit + ci,
#                         color="black", alpha=0.10, zorder=4)
#     except np.linalg.LinAlgError:
#         pass  # skip CI if matrix is singular
#
#     ax.plot(x_pos, y_fit,
#             color="black", lw=1.5, ls="--", zorder=5,
#             label=f"WLS trend  ({z[0]:+.4f} acc / degree)")
#
#
# def _diff_subplot(ax, pos, accs, overall):
#     """Bar chart of per-degree accuracy minus the overall test accuracy.
#
#     Green bars = above average for that degree; red = below average.
#     This makes the degree-bias trend immediately legible even when absolute
#     accuracy values are tightly clustered.
#     """
#     diffs  = [a - overall for a in accs]
#     colors = ["#27ae60" if d >= 0 else "#e74c3c" for d in diffs]
#     ax.bar(pos, diffs, color=colors, width=0.6, alpha=0.82, zorder=3)
#     ax.axhline(0, color="black", linewidth=0.9, zorder=4)
#     ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))
#     ax.set_ylabel("Δ from mean\ntest acc", fontsize=9)
#     ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.4)
#
# # ── distance-to-train vs degree ────────────────────────────────────────────────
#
# def plot_dist_vs_degree(dist_deg_data, cfg, save_dir=None, show=False):
#     """Plot hop-distance distributions per degree group.
#
#     Two stacked subplots, sharing the x-axis (node degree):
#       • Top    – dist to *any* training node   (blue boxes)
#       • Bottom – dist to *same-class* training node (green boxes)
#
#     A horizontal dashed red line marks the model's receptive field
#     (``cfg['model']['num_layers']`` hops).  NaN entries (unreachable nodes)
#     are silently omitted from each boxplot.
#
#     Parameters
#     ----------
#     dist_deg_data : dict
#         Output of ``utils.get_distance_deg``.
#     cfg : dict
#         Experiment config (used for ``num_layers``, titles, filenames).
#     save_dir : str or None
#     show : bool
#     """
#     num_layers  = cfg["model"]["num_layers"]
#     all_degrees = sorted(dist_deg_data.keys())
#     pos         = list(range(len(all_degrees)))
#     counts      = [dist_deg_data[d]["count"] for d in all_degrees]
#     n_test      = sum(counts)
#     prefix      = _fname_prefix(cfg)
#     subtitle    = _subtitle(cfg, n_test, len(all_degrees))
#
#     def _clean(arr):
#         """Drop NaN, return at least [NaN] so boxplot doesn't crash."""
#         a = arr[~np.isnan(arr)]
#         return a if len(a) > 0 else np.array([np.nan])
#
#     data_train = [_clean(dist_deg_data[d]["dist_to_train"])     for d in all_degrees]
#     data_same  = [_clean(dist_deg_data[d]["dist_to_same_class"]) for d in all_degrees]
#
#     fig, (ax_top, ax_bot) = plt.subplots(
#         2, 1,
#         figsize=(_fig_w(len(all_degrees)), 8),
#         sharex=True,
#         gridspec_kw={"height_ratios": [1, 1]},
#     )
#     fig.subplots_adjust(hspace=0.08)
#
#     # ── top: dist to any train node ──
#     bp1 = ax_top.boxplot(data_train, positions=pos, widths=0.6, **_BP_KWARGS)
#     for patch in bp1["boxes"]:
#         patch.set_facecolor("#5b9bd5")
#         patch.set_alpha(0.72)
#     _count_bars(ax_top, pos, counts)
#     ax_top.axhline(
#         num_layers, color="#e74c3c", lw=1.8, ls="--", zorder=6,
#         label=f"Model receptive field  ({num_layers} layers)",
#     )
#     ax_top.set_ylabel("Hops to nearest\ntraining node", fontsize=10)
#     ax_top.legend(loc="upper left", fontsize=9, framealpha=0.85)
#     ax_top.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.4)
#     ax_top.set_title(
#         f"Distance to Training Node vs. Node Degree\n{subtitle}", fontsize=11
#     )
#
#     # ── bottom: dist to same-class train node ──
#     bp2 = ax_bot.boxplot(data_same, positions=pos, widths=0.6, **_BP_KWARGS)
#     for patch in bp2["boxes"]:
#         patch.set_facecolor("#27ae60")
#         patch.set_alpha(0.72)
#     ax_bot.axhline(
#         num_layers, color="#e74c3c", lw=1.8, ls="--", zorder=6,
#         label=f"Model receptive field  ({num_layers} layers)",
#     )
#     ax_bot.set_ylabel("Hops to nearest\nsame-class training node", fontsize=10)
#     ax_bot.legend(loc="upper left", fontsize=9, framealpha=0.85)
#     ax_bot.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.4)
#     _degree_axis(ax_bot, pos, all_degrees)
#
#     fig.tight_layout()
#     _save(fig, save_dir, f"{prefix}_dist_vs_degree.png", show)
#
