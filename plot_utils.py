import logging
import os

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

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


def _degree_axis(ax, pos, all_degrees):
    """Set x-axis label and ticks; thin labels when there are many degrees."""
    ax.set_xlabel("Node degree", fontsize=11)
    ax.set_xlim(pos[0] - 0.6, pos[-1] + 0.6)
    step = max(1, len(all_degrees) // 30)
    ax.set_xticks(pos[::step])
    ax.set_xticklabels(all_degrees[::step], rotation=55, ha="right", fontsize=8)


def _add_trend(ax, all_degrees, pos, accs):
    """Fit OLS on (degree, accuracy) and overlay as a dashed line.

    The slope is in units of accuracy-change per unit degree — a positive
    slope means higher-degree nodes tend to be classified more accurately.
    """
    pairs = [(d, p, a) for d, p, a in zip(all_degrees, pos, accs)
             if not np.isnan(a)]
    if len(pairs) < 2:
        return
    degs, poss, vals = map(list, zip(*pairs))
    z     = np.polyfit(degs, vals, 1)
    x_deg = np.linspace(min(degs), max(degs), 200)
    x_pos = np.interp(x_deg, degs, poss)
    ax.plot(x_pos, np.polyval(z, x_deg),
            color="black", lw=1.5, ls="--", zorder=5,
            label=f"OLS trend  ({z[0]:+.4f} acc / degree)")


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


def _diff_subplot(ax, pos, accs, overall):
    """Bar chart of per-degree accuracy minus the overall test accuracy.

    Green bars = above average for that degree; red = below average.
    This makes the degree-bias trend immediately legible even when absolute
    accuracy values are tightly clustered.
    """
    diffs  = [a - overall for a in accs]
    colors = ["#27ae60" if d >= 0 else "#e74c3c" for d in diffs]
    ax.bar(pos, diffs, color=colors, width=0.6, alpha=0.82, zorder=3)
    ax.axhline(0, color="black", linewidth=0.9, zorder=4)
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))
    ax.set_ylabel("Δ from mean\ntest acc", fontsize=9)
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.4)


# ── public entry point ─────────────────────────────────────────────────────────

def plot_acc_vs_degree(run_results, cfg, save_dir=None, show=False):
    """Plot test-node accuracy vs. node degree.

    Single run
        Scatter plot (bubble size ∝ number of test nodes with that degree)
        with an OLS trend line, an overall-accuracy reference, and a Δ-from-
        mean subplot below — making it easy to see whether higher-degree nodes
        are classified better or worse.

    Multiple runs
        Figure 1 – *across-runs*: one box per degree (neutral colour) showing
        how the per-run mean accuracy varies across seeds, with an OLS trend
        through the medians, count bars on a secondary axis, and the same Δ
        subplot.  No degree-based colour grading.

        Figure 2 – *per-run*: grouped boxes with one box per run per degree
        (tab10 colours), count bars on a secondary axis.  Lets you verify that
        the degree-bias direction is consistent across individual runs.

    Parameters
    ----------
    run_results : list[dict]
        One entry per run; each is the output of ``get_accuracy_deg`` called
        on test nodes only (keys = degree ints, values contain 'preds'/'labels').
    cfg : dict
        Experiment config — used for axis labels, titles, and filenames.
    save_dir : str or None
        Directory to write PDF figures.  Skipped when None.
    show : bool
        Call ``plt.show()`` after each figure.
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
        _plot_per_run(all_degrees, pos, deg_data, n_runs, subtitle, prefix, save_dir, show)


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

    fig, (ax_main, ax_diff) = plt.subplots(
        2, 1,
        figsize=(_fig_w(len(all_degrees)), 7),
        gridspec_kw={"height_ratios": [3, 1]},
        sharex=True,
    )
    fig.subplots_adjust(hspace=0.08)

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
    _add_trend(ax_main, all_degrees, pos, mean_acc)

    ax_main.set_ylabel("Accuracy  (test nodes)", fontsize=11)
    ax_main.set_ylim(-0.05, 1.10)
    ax_main.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))
    ax_main.legend(loc="upper left", fontsize=8, framealpha=0.85,
                   title="Node count", title_fontsize=8)
    ax_main.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.4)
    ax_main.set_title(
        f"Accuracy vs. Node Degree  —  single run\n{subtitle}", fontsize=11
    )

    _diff_subplot(ax_diff, pos, mean_acc, overall)
    _degree_axis(ax_diff, pos, all_degrees)

    fig.tight_layout()
    _save(fig, save_dir, f"{prefix}_acc_vs_degree_single_run.png", show)


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

    fig, (ax_main, ax_diff) = plt.subplots(
        2, 1,
        figsize=(_fig_w(len(all_degrees)), 7),
        gridspec_kw={"height_ratios": [3, 1]},
        sharex=True,
    )
    fig.subplots_adjust(hspace=0.08)

    bp = ax_main.boxplot(per_run_means, positions=pos, widths=0.6, **_BP_KWARGS)
    for patch in bp["boxes"]:
        patch.set_facecolor("#5b9bd5")
        patch.set_alpha(0.72)

    _count_bars(ax_main, pos, counts)
    ax_main.axhline(overall, color="dimgrey", lw=1.0, ls=":",
                    label=f"Mean test acc ({overall:.1%})", zorder=2)
    _add_trend(ax_main, all_degrees, pos, median_accs)

    ax_main.set_ylabel(f"Mean accuracy per run  ({n_runs} seeds)", fontsize=11)
    ax_main.set_ylim(-0.05, 1.10)
    ax_main.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))
    ax_main.legend(loc="upper left", fontsize=8, framealpha=0.85)
    ax_main.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.4)
    ax_main.set_title(
        f"Accuracy vs. Node Degree  —  {n_runs} seeds\n{subtitle}", fontsize=11
    )

    _diff_subplot(ax_diff, pos, median_accs, overall)
    _degree_axis(ax_diff, pos, all_degrees)

    fig.tight_layout()
    _save(fig, save_dir, f"{prefix}_acc_vs_degree_across_runs.png", show)


# ── multiple runs: per-run node-level distributions ────────────────────────────

def _plot_per_run(all_degrees, pos, deg_data, n_runs, subtitle, prefix, save_dir, show):
    counts    = [len(deg_data[d][0]) for d in all_degrees]
    box_w     = 0.82 / n_runs
    run_colors = [plt.cm.tab10(i / 10) for i in range(min(n_runs, 10))]

    fig, ax_main = plt.subplots(figsize=(_fig_w(len(all_degrees), n_runs), 5))

    handles = []
    for run_idx in range(n_runs):
        offsets = [p + (run_idx - (n_runs - 1) / 2) * box_w for p in pos]
        data = [
            deg_data[d][run_idx] if len(deg_data[d][run_idx]) > 0 else np.array([np.nan])
            for d in all_degrees
        ]
        c  = run_colors[run_idx % 10]
        bp = ax_main.boxplot(data, positions=offsets, widths=box_w * 0.88,
                             manage_ticks=False, **_BP_KWARGS)
        for patch in bp["boxes"]:
            patch.set_facecolor(c)
            patch.set_alpha(0.72)
        handles.append(plt.Rectangle((0, 0), 1, 1, fc=c, alpha=0.75,
                                      label=f"Run {run_idx + 1}"))

    _count_bars(ax_main, pos, counts)

    ax_main.set_ylabel("Accuracy  (test nodes)", fontsize=11)
    ax_main.set_ylim(-0.05, 1.10)
    ax_main.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))
    ax_main.legend(handles=handles, title="Run (seed)", loc="upper left",
                   framealpha=0.85, fontsize=9)
    ax_main.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.4)
    ax_main.set_title(
        f"Accuracy vs. Node Degree  —  per-run distributions  ({n_runs} runs)\n{subtitle}",
        fontsize=11,
    )
    _degree_axis(ax_main, pos, all_degrees)

    fig.tight_layout()
    _save(fig, save_dir, f"{prefix}_acc_vs_degree_per_run.png", show)
