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



def _make_degree_bins(all_degrees, counts, n_bins=6):
    """Bin degree values into n_bins groups each containing roughly equal numbers
    of test nodes (equal-count / quantile binning).

    Parameters
    ----------
    all_degrees : sorted list[int]
    counts      : list[int], test-node count for each degree (same order)
    n_bins      : target number of bins

    Returns
    -------
    bin_of_degree : dict {degree -> bin_index}
    bin_labels    : list[str] – human-readable ranges, e.g. "1", "2–3", "≥16"
    n_actual      : int – actual number of bins produced
    """
    arr  = np.array(all_degrees, dtype=int)
    cnts = np.array(counts, dtype=float)

    if len(arr) <= n_bins:
        return ({int(d): i for i, d in enumerate(arr)},
                [str(d) for d in arr],
                len(arr))

    # Find degree boundaries that split the cumulative node count equally
    cum     = np.cumsum(cnts)
    targets = np.linspace(0, cum[-1], n_bins + 1)[1:-1]   # interior splits
    bounds  = sorted({int(arr[min(int(np.searchsorted(cum, t)), len(arr) - 1)])
                      for t in targets})

    bin_of_degree = {int(d): int(np.searchsorted(bounds, d, side="right"))
                     for d in arr}

    n_actual = max(bin_of_degree.values()) + 1
    bin_labels = []
    for b in range(n_actual):
        in_b = sorted(d for d, bi in bin_of_degree.items() if bi == b)
        lo, hi = in_b[0], in_b[-1]
        if lo == hi:
            bin_labels.append(str(lo))
        elif b == n_actual - 1:
            bin_labels.append(f"≥{lo}")
        else:
            bin_labels.append(f"{lo}–{hi}")
    return bin_of_degree, bin_labels, n_actual


def _degree_axis(ax, pos, all_degrees):
    """Set x-axis label and ticks; thin labels when there are many degrees."""
    ax.set_xlabel("Node degree", fontsize=11)
    ax.set_xlim(pos[0] - 0.6, pos[-1] + 0.6)
    step = max(1, len(all_degrees) // 30)
    ax.set_xticks(pos[::step])
    ax.set_xticklabels(all_degrees[::step], rotation=55, ha="right", fontsize=8)


def _add_trend(ax, all_degrees, pos, accs, counts=None):
    """Fit WLS on (degree, accuracy) weighted by node count and overlay as a
    dashed line with a shaded 95 % confidence band.

    Using WLS (rather than plain OLS) down-weights degree buckets that contain
    very few test nodes, whose accuracy estimates are noisier.  The slope is in
    units of accuracy-change per unit degree — a positive slope means
    higher-degree nodes tend to be classified more accurately.

    The 95 % CI band is derived from the standard error of the WLS fit
    evaluated at each interpolated degree value.
    """
    pairs = [(d, p, a, c) for d, p, a, c in
             zip(all_degrees, pos, accs, counts if counts is not None else [1] * len(all_degrees))
             if not np.isnan(a)]
    if len(pairs) < 2:
        return
    degs, poss, vals, wts = map(list, zip(*pairs))
    degs = np.array(degs, dtype=float)
    vals = np.array(vals, dtype=float)
    wts  = np.array(wts,  dtype=float)

    # WLS via weighted polyfit
    z = np.polyfit(degs, vals, 1, w=wts)

    # Residual standard error for the confidence band
    y_hat = np.polyval(z, degs)
    resid = vals - y_hat
    # Weighted residual sum of squares
    wrss  = np.sum(wts * resid ** 2)
    dof   = max(len(degs) - 2, 1)
    s2    = wrss / dof

    x_deg = np.linspace(degs.min(), degs.max(), 200)
    x_pos = np.interp(x_deg, degs, poss)
    y_fit = np.polyval(z, x_deg)

    # Leverage for each interpolated point: var(ŷ) = s² * x'(X'WX)⁻¹x
    W   = np.diag(wts)
    X   = np.column_stack([degs, np.ones_like(degs)])
    XtW = X.T @ W
    try:
        cov = np.linalg.inv(XtW @ X) * s2
        X_new = np.column_stack([x_deg, np.ones_like(x_deg)])
        se_fit = np.sqrt(np.einsum("ij,jk,ik->i", X_new, cov, X_new))
        ci = 1.96 * se_fit
        ax.fill_between(x_pos, y_fit - ci, y_fit + ci,
                        color="black", alpha=0.10, zorder=4)
    except np.linalg.LinAlgError:
        pass  # skip CI if matrix is singular

    ax.plot(x_pos, y_fit,
            color="black", lw=1.5, ls="--", zorder=5,
            label=f"WLS trend  ({z[0]:+.4f} acc / degree)")


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
    _add_trend(ax_main, all_degrees, pos, mean_acc, counts=counts)

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
    _add_trend(ax_main, all_degrees, pos, median_accs, counts=counts)

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




# ── distance-to-train vs degree ────────────────────────────────────────────────

def plot_dist_vs_degree(dist_deg_data, cfg, save_dir=None, show=False):
    """Plot hop-distance distributions per degree group.

    Two stacked subplots, sharing the x-axis (node degree):
      • Top    – dist to *any* training node   (blue boxes)
      • Bottom – dist to *same-class* training node (green boxes)

    A horizontal dashed red line marks the model's receptive field
    (``cfg['model']['num_layers']`` hops).  NaN entries (unreachable nodes)
    are silently omitted from each boxplot.

    Parameters
    ----------
    dist_deg_data : dict
        Output of ``utils.get_distance_deg``.
    cfg : dict
        Experiment config (used for ``num_layers``, titles, filenames).
    save_dir : str or None
    show : bool
    """
    num_layers  = cfg["model"]["num_layers"]
    all_degrees = sorted(dist_deg_data.keys())
    pos         = list(range(len(all_degrees)))
    counts      = [dist_deg_data[d]["count"] for d in all_degrees]
    n_test      = sum(counts)
    prefix      = _fname_prefix(cfg)
    subtitle    = _subtitle(cfg, n_test, len(all_degrees))

    def _clean(arr):
        """Drop NaN, return at least [NaN] so boxplot doesn't crash."""
        a = arr[~np.isnan(arr)]
        return a if len(a) > 0 else np.array([np.nan])

    data_train = [_clean(dist_deg_data[d]["dist_to_train"])     for d in all_degrees]
    data_same  = [_clean(dist_deg_data[d]["dist_to_same_class"]) for d in all_degrees]

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1,
        figsize=(_fig_w(len(all_degrees)), 8),
        sharex=True,
        gridspec_kw={"height_ratios": [1, 1]},
    )
    fig.subplots_adjust(hspace=0.08)

    # ── top: dist to any train node ──
    bp1 = ax_top.boxplot(data_train, positions=pos, widths=0.6, **_BP_KWARGS)
    for patch in bp1["boxes"]:
        patch.set_facecolor("#5b9bd5")
        patch.set_alpha(0.72)
    _count_bars(ax_top, pos, counts)
    ax_top.axhline(
        num_layers, color="#e74c3c", lw=1.8, ls="--", zorder=6,
        label=f"Model receptive field  ({num_layers} layers)",
    )
    ax_top.set_ylabel("Hops to nearest\ntraining node", fontsize=10)
    ax_top.legend(loc="upper left", fontsize=9, framealpha=0.85)
    ax_top.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.4)
    ax_top.set_title(
        f"Distance to Training Node vs. Node Degree\n{subtitle}", fontsize=11
    )

    # ── bottom: dist to same-class train node ──
    bp2 = ax_bot.boxplot(data_same, positions=pos, widths=0.6, **_BP_KWARGS)
    for patch in bp2["boxes"]:
        patch.set_facecolor("#27ae60")
        patch.set_alpha(0.72)
    ax_bot.axhline(
        num_layers, color="#e74c3c", lw=1.8, ls="--", zorder=6,
        label=f"Model receptive field  ({num_layers} layers)",
    )
    ax_bot.set_ylabel("Hops to nearest\nsame-class training node", fontsize=10)
    ax_bot.legend(loc="upper left", fontsize=9, framealpha=0.85)
    ax_bot.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.4)
    _degree_axis(ax_bot, pos, all_degrees)

    fig.tight_layout()
    _save(fig, save_dir, f"{prefix}_dist_vs_degree.png", show)


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
        _save(fig, save_dir, f"{prefix}_combined_vs_degree_{label}.png", show)


# ── AMP heterogeneity distribution + DMP counts vs degree ──────────────────────

def plot_amp_dmp_vs_degree(amp_deg_data, dmp_deg_data, cfg,
                           save_dir=None, show=False, run_results=None):
    """Two-panel figure: AMP heterogeneity and DMP rate vs node degree.

    Each unique degree is shown as its own tick — no binning.

    Left  — Boxplot of neighbour-heterogeneity ratios per degree.
            A dashed line marks the AMP threshold (het > threshold ⇒ AMP).
            If run_results is provided, mean accuracy (± 1 σ band for
            multi-run) is overlaid on a right-hand axis.

    Right — Line chart of DMP rate (% of nodes lacking a same-class training
            node within dmp_coeff hops) per degree.  Accuracy is overlaid
            the same way if run_results is given.
    """
    all_degrees = sorted(set(amp_deg_data.keys()) & set(dmp_deg_data.keys()))
    raw_counts  = [amp_deg_data[d]["count"] for d in all_degrees]
    n_test      = sum(raw_counts)
    prefix      = _fname_prefix(cfg)
    amp_coeff   = cfg["dataset"].get("amp_coeff", 1)
    dmp_coeff   = cfg["dataset"].get("dmp_coeff", 1)
    amp_thr     = cfg["dataset"].get("amp_threshold", 0.5)
    subtitle    = _subtitle(cfg, n_test, len(all_degrees))

    pos = list(range(len(all_degrees)))

    def _clean(arr):
        a = arr[~np.isnan(arr)]
        return a if len(a) > 0 else np.array([np.nan])

    # Per-degree het values (no binning)
    het_per_deg = [_clean(amp_deg_data[d]["het_values"]) for d in all_degrees]

    # Per-degree DMP rate
    dmp_rate = np.array([
        dmp_deg_data[d]["count_1"] / dmp_deg_data[d]["count"]
        if dmp_deg_data[d]["count"] > 0 else np.nan
        for d in all_degrees
    ])

    # Accuracy per degree (optional)
    acc_mean = acc_std = None
    if run_results is not None:
        _, deg_data = _collect(run_results)
        _am, _as = [], []
        for d in all_degrees:
            arrs = [a for a in deg_data.get(d, []) if len(a) > 0]
            if arrs:
                per_run = [a.mean() for a in arrs]
                _am.append(float(np.mean(per_run)))
                _as.append(float(np.std(per_run)))
            else:
                _am.append(np.nan)
                _as.append(0.0)
        acc_mean = np.array(_am)
        acc_std  = np.array(_as)

    # ── Figure ─────────────────────────────────────────────────────────────────
    n_deg = len(all_degrees)
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(_fig_w(n_deg) + 4, 5))
    fig.suptitle(
        f"AMP ({amp_coeff}-hop) Heterogeneity  &  DMP ({dmp_coeff}-hop) Rate  vs. Degree\n"
        f"{subtitle}",
        fontsize=11, y=1.02,
    )

    ACC_COLOR = "#2980b9"
    ann_step  = max(1, n_deg // 15)   # sparse annotations to avoid clutter

    # ── Left: AMP heterogeneity boxplots ───────────────────────────────────────
    bp = ax_l.boxplot(het_per_deg, positions=pos, widths=0.55, **_BP_KWARGS)
    for patch in bp["boxes"]:
        patch.set_facecolor("#e67e22")
        patch.set_alpha(0.80)

    ax_l.axhline(amp_thr, color="#c0392b", lw=1.4, ls="--", zorder=6,
                 label=f"AMP threshold  ({amp_thr:.0%})")
    ax_l.set_ylabel("Neighbour heterogeneity ratio", fontsize=10)
    ax_l.set_ylim(-0.05, 1.10)
    ax_l.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))
    ax_l.tick_params(axis="y", labelsize=8)
    ax_l.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.4)
    ax_l.set_title("AMP: how ambivalent are node neighbourhoods?\n"
                   "(box = IQR, centre line = median, dots = outliers)",
                   fontsize=10, pad=6)

    for i, n in enumerate(raw_counts):
        if i % ann_step == 0:
            ax_l.text(i, -0.08, f"n={n}", ha="center", va="top",
                      fontsize=6.5, color="dimgrey",
                      transform=ax_l.get_xaxis_transform())

    if acc_mean is not None:
        ax_l2 = ax_l.twinx()
        valid = ~np.isnan(acc_mean)
        xs, ym, ys = np.array(pos)[valid], acc_mean[valid], acc_std[valid]
        ax_l2.plot(xs, ym, color=ACC_COLOR, lw=1.8, marker="s",
                   markersize=4, zorder=5, label="Accuracy (mean)")
        if len(run_results) > 1:
            ax_l2.fill_between(xs, ym - ys, ym + ys,
                               color=ACC_COLOR, alpha=0.15, zorder=3)
        ax_l2.set_ylim(0, 1.05)
        ax_l2.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))
        ax_l2.set_ylabel("Accuracy", fontsize=10, color=ACC_COLOR)
        ax_l2.tick_params(axis="y", labelsize=8, colors=ACC_COLOR)
        ax_l2.spines["right"].set_edgecolor(ACC_COLOR)
        h1, l1 = ax_l.get_legend_handles_labels()
        h2, l2 = ax_l2.get_legend_handles_labels()
        ax_l.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=8, framealpha=0.85)
    else:
        ax_l.legend(loc="upper left", fontsize=9, framealpha=0.85)

    _degree_axis(ax_l, pos, all_degrees)

    # ── Right: DMP rate per degree ──────────────────────────────────────────────
    ax_r.plot(pos, dmp_rate, color="#8e44ad", lw=2.2, marker="o",
              markersize=5, zorder=4)
    ax_r.fill_between(pos, 0, dmp_rate, color="#8e44ad", alpha=0.15, zorder=2)

    for i, rate in enumerate(dmp_rate):
        if not np.isnan(rate) and i % ann_step == 0:
            ax_r.text(i, rate + 0.03, f"{rate:.0%}", ha="center", va="bottom",
                      fontsize=7.5, color="#6c3483", fontweight="bold")

    for i, n in enumerate(raw_counts):
        if i % ann_step == 0:
            ax_r.text(i, -0.08, f"n={n}", ha="center", va="top",
                      fontsize=6.5, color="dimgrey",
                      transform=ax_r.get_xaxis_transform())

    ax_r.set_ylim(0, 1.10)
    ax_r.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))
    ax_r.tick_params(axis="y", labelsize=8)
    ax_r.set_ylabel("% of nodes with DMP", fontsize=10)
    ax_r.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.4)
    ax_r.set_title("DMP: what fraction of nodes lack a nearby\nsame-class training node?",
                   fontsize=10, pad=6)

    if acc_mean is not None:
        ax_r2 = ax_r.twinx()
        valid = ~np.isnan(acc_mean)
        xs, ym, ys = np.array(pos)[valid], acc_mean[valid], acc_std[valid]
        ax_r2.plot(xs, ym, color=ACC_COLOR, lw=1.8, marker="s",
                   markersize=4, zorder=5, label="Accuracy (mean)")
        if len(run_results) > 1:
            ax_r2.fill_between(xs, ym - ys, ym + ys,
                               color=ACC_COLOR, alpha=0.15, zorder=3)
        ax_r2.set_ylim(0, 1.05)
        ax_r2.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))
        ax_r2.set_ylabel("Accuracy", fontsize=10, color=ACC_COLOR)
        ax_r2.tick_params(axis="y", labelsize=8, colors=ACC_COLOR)
        ax_r2.spines["right"].set_edgecolor(ACC_COLOR)
        ax_r2.legend(loc="upper right", fontsize=8, framealpha=0.85)

    _degree_axis(ax_r, pos, all_degrees)

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.20, wspace=0.55)
    _save(fig, save_dir, f"{prefix}_amp_dmp_vs_degree.png", show)


# ── accuracy by AMP × DMP group ────────────────────────────────────────────────

def plot_acc_by_amp_dmp_group(group_acc_per_run, group_names, group_counts,
                               cfg, save_dir=None, show=False):
    """Compare classification accuracy across four AMP × DMP groups.

    Groups (x-axis, left to right):
      0 – Low AMP + No DMP   (structurally easy nodes)
      1 – Low AMP + DMP
      2 – High AMP + No DMP
      3 – High AMP + DMP     (structurally hard nodes)

    Single run  — horizontal bar per group, annotated with node count.
    Multi-run   — boxplot per group (distribution across seeds), with node
                  count shown below each box.

    The two extreme groups (0 and 3) are highlighted to make the easy vs
    hard comparison immediately visible.

    Parameters
    ----------
    group_acc_per_run : list[list[float]]
        Outer list: one entry per group (4 total).
        Inner list: one accuracy value per run.
    group_names : list[str]
        Human-readable label for each group (4 total).
    group_counts : list[int]
        Number of test nodes per group.
    cfg : dict
    save_dir : str or None
    show : bool
    """
    n_runs   = len(group_acc_per_run[0])
    prefix   = _fname_prefix(cfg)
    amp_coeff = cfg["dataset"].get("amp_coeff", 1)
    dmp_coeff = cfg["dataset"].get("dmp_coeff", 1)
    amp_thr   = cfg["dataset"].get("amp_threshold", 0.5)

    # Colours: highlight groups 0 (easy, green) and 3 (hard, red); others grey
    GROUP_COLORS = ["#27ae60", "#95a5a6", "#95a5a6", "#e74c3c"]
    GROUP_EDGE   = ["#1e8449", "#7f8c8d", "#7f8c8d", "#c0392b"]

    n_test   = sum(group_counts)
    subtitle = (f"{cfg['dataset']['name']} · {cfg['model']['name']} · "
                f"{cfg.get('split','random')} · "
                f"{'CC' if cfg['dataset'].get('use_cc') else 'noCC'}"
                f"   |   AMP {amp_coeff}-hop  thr={amp_thr}  ·  DMP {dmp_coeff}-hop"
                f"   |   {n_test:,} test nodes")

    pos = list(range(4))

    fig, ax = plt.subplots(figsize=(8, 5))

    if n_runs == 1:
        accs = [vals[0] for vals in group_acc_per_run]
        bars = ax.bar(pos, accs, color=GROUP_COLORS, edgecolor=GROUP_EDGE,
                      linewidth=1.2, alpha=0.85, zorder=3)
        for bar, acc, cnt in zip(bars, accs, group_counts):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.015,
                    f"{acc:.1%}", ha="center", va="bottom", fontsize=9,
                    fontweight="bold")
            ax.text(bar.get_x() + bar.get_width() / 2,
                    -0.04, f"n={cnt}", ha="center", va="top",
                    fontsize=8, color="dimgrey")
    else:
        bp = ax.boxplot(group_acc_per_run, positions=pos, widths=0.55,
                        **_BP_KWARGS)
        for patch, color in zip(bp["boxes"], GROUP_COLORS):
            patch.set_facecolor(color)
            patch.set_alpha(0.80)
        for cnt, x in zip(group_counts, pos):
            ax.text(x, -0.06, f"n={cnt}", ha="center", va="top",
                    fontsize=8, color="dimgrey",
                    transform=ax.get_xaxis_transform())

    # Annotate the two extremes
    ax.annotate("← easy nodes", xy=(0, 0.02), xycoords=("data", "axes fraction"),
                fontsize=8, color="#1e8449", ha="center")
    ax.annotate("hard nodes →", xy=(3, 0.02), xycoords=("data", "axes fraction"),
                fontsize=8, color="#c0392b", ha="center")

    overall = (sum(a[0] * c for a, c in zip(group_acc_per_run, group_counts))
               / n_test)
    ax.axhline(overall, color="dimgrey", lw=1.0, ls=":",
               label=f"Overall test acc  {overall:.1%}", zorder=2)

    ax.set_xticks(pos)
    ax.set_xticklabels(group_names, fontsize=10)
    ax.set_ylabel("Accuracy", fontsize=11)
    ax.set_ylim(-0.05, 1.15)
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.4)
    ax.legend(loc="upper right", fontsize=9, framealpha=0.85)
    ax.set_title(
        f"Accuracy by AMP × DMP Group  "
        f"({'single run' if n_runs == 1 else f'{n_runs} seeds'})\n{subtitle}",
        fontsize=10,
    )

    fig.tight_layout()
    _save(fig, save_dir, f"{prefix}_acc_by_amp_dmp_group.png", show)


# ── accuracy by AMP × DMP group, stratified by degree ──────────────────────────

def plot_acc_by_amp_dmp_group_vs_degree(group_deg_acc, group_names, cfg,
                                         save_dir=None, show=False):
    """Accuracy per AMP × DMP group vs binned node degree — 4 overlaid lines.

    Degrees are binned into equal-count buckets so each point on the x-axis
    represents a comparable number of nodes.  For multi-run experiments each
    line shows the cross-seed mean with a ±1σ shaded band.

    Answers: does the group ranking (Low AMP + No DMP best, High AMP + DMP
    worst) hold consistently at every degree, or does it collapse/reverse for
    very low- or very high-degree nodes?
    """
    GROUP_COLORS  = ["#27ae60", "#5d6d7e", "#a569bd", "#e74c3c"]
    GROUP_MARKERS = ["o", "s", "^", "D"]
    GROUP_LS      = ["-", "--", "-.", ":"]

    prefix    = _fname_prefix(cfg)
    amp_coeff = cfg["dataset"].get("amp_coeff", 1)
    dmp_coeff = cfg["dataset"].get("dmp_coeff", 1)
    amp_thr   = cfg["dataset"].get("amp_threshold", 0.5)
    dataset   = cfg["dataset"]["name"]
    model     = cfg["model"]["name"]
    split     = cfg.get("split", "random")
    cc        = "CC" if cfg["dataset"].get("use_cc") else "noCC"

    all_degrees = sorted({d for g in range(4) for d in group_deg_acc[g]})
    if not all_degrees:
        return

    # Use total node count across all groups for degree binning
    deg_total = {}
    for d in all_degrees:
        deg_total[d] = sum(len(group_deg_acc[g].get(d, [])) for g in range(4))
    raw_counts  = [deg_total[d] for d in all_degrees]
    bin_of_deg, bin_labels, n_bins = _make_degree_bins(all_degrees, raw_counts)
    pos = list(range(n_bins))

    n_runs = max((len(v) for g in range(4) for v in group_deg_acc[g].values()),
                 default=1)

    # Aggregate per-run accuracy into bins: mean over degrees in each bin
    fig, ax = plt.subplots(figsize=(max(10, n_bins * 1.5 + 3), 5))

    for g, (color, marker, ls, name) in enumerate(
        zip(GROUP_COLORS, GROUP_MARKERS, GROUP_LS, group_names)
    ):
        bin_means, bin_stds = [], []
        for b in range(n_bins):
            degs_in_bin = [d for d in all_degrees if bin_of_deg[d] == b]
            # Collect all per-run accuracy values for this (group, bin)
            run_accs = []
            for run_i in range(n_runs):
                vals = [group_deg_acc[g][d][run_i]
                        for d in degs_in_bin
                        if d in group_deg_acc[g] and run_i < len(group_deg_acc[g][d])
                        and not np.isnan(group_deg_acc[g][d][run_i])]
                if vals:
                    run_accs.append(float(np.mean(vals)))
            if run_accs:
                bin_means.append(float(np.mean(run_accs)))
                bin_stds.append(float(np.std(run_accs)))
            else:
                bin_means.append(np.nan)
                bin_stds.append(np.nan)

        xs    = np.array(pos, dtype=float)
        means = np.array(bin_means)
        stds  = np.array(bin_stds)
        label = name.replace("\n", " ")

        valid = ~np.isnan(means)
        ax.plot(xs[valid], means[valid], color=color, lw=2.2, ls=ls,
                marker=marker, markersize=7, zorder=4, label=label)
        if n_runs > 1:
            ax.fill_between(xs[valid],
                            (means - stds)[valid], (means + stds)[valid],
                            color=color, alpha=0.13, zorder=2)

    ax.set_ylabel("Accuracy", fontsize=11)
    ax.set_ylim(-0.05, 1.10)
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.4)
    ax.set_xticks(pos)
    ax.set_xticklabels(bin_labels, rotation=40, ha="right", fontsize=9)
    ax.set_xlabel("Node degree (binned)", fontsize=10)

    run_str = "single run" if n_runs == 1 else f"{n_runs} seeds  (mean ± 1σ)"
    ax.set_title(
        f"Does the accuracy gap between AMP × DMP groups persist across degrees?  "
        f"—  {run_str}\n"
        f"{dataset} · {model} · {split} · {cc}"
        f"   |   AMP {amp_coeff}-hop  thr={amp_thr}  ·  DMP {dmp_coeff}-hop",
        fontsize=10,
    )
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22),
              ncol=4, fontsize=9.5, framealpha=0.9, borderpad=0.8)

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.24)
    _save(fig, save_dir, f"{prefix}_acc_by_amp_dmp_group_vs_degree.png", show)


# ── group cardinality + hop distances vs degree ─────────────────────────────────

def plot_group_cardinality_and_distance(group_deg_counts, dist_deg_data,
                                         group_names, cfg,
                                         save_dir=None, show=False):
    """Two-panel figure: AMP × DMP group composition and hop distances vs degree.

    Degrees are grouped into equal-count bins.

    Left  — 100 % normalised stacked bar showing the *proportion* of nodes in
            each AMP × DMP group at each degree bin.  Using proportions (not
            raw counts) makes the composition shift across degrees immediately
            visible regardless of how many nodes are in each bin.  The total
            node count is annotated on top of each bar for context.

    Right — Median ± IQR lines for dist_to_train (orange) and
            dist_to_same_class (green) per bin.  A red dashed line marks the
            model's receptive field (num_layers hops) — nodes whose nearest
            same-class training node is above this line cannot receive correct
            label signal within the model's reach.
    """
    GROUP_COLORS = ["#27ae60", "#5d6d7e", "#a569bd", "#e74c3c"]

    num_layers = cfg["model"]["num_layers"]
    amp_coeff  = cfg["dataset"].get("amp_coeff", 1)
    dmp_coeff  = cfg["dataset"].get("dmp_coeff", 1)
    amp_thr    = cfg["dataset"].get("amp_threshold", 0.5)
    prefix     = _fname_prefix(cfg)

    all_degrees = sorted(set(group_deg_counts) & set(dist_deg_data))
    raw_counts  = [sum(group_deg_counts[d].values()) for d in all_degrees]
    n_test      = sum(raw_counts)
    subtitle    = (
        f"{cfg['dataset']['name']} · {cfg['model']['name']} · "
        f"{cfg.get('split','random')} · "
        f"{'CC' if cfg['dataset'].get('use_cc') else 'noCC'}"
        f"   |   AMP {amp_coeff}-hop  thr={amp_thr}  ·  DMP {dmp_coeff}-hop"
        f"   |   {n_test:,} test nodes"
    )

    # ── Degree binning ─────────────────────────────────────────────────────────
    bin_of_deg, bin_labels, n_bins = _make_degree_bins(all_degrees, raw_counts)
    pos = list(range(n_bins))

    # Per-bin node count and group counts
    bin_total   = np.zeros(n_bins)
    bin_g_count = np.zeros((4, n_bins))
    for d in all_degrees:
        b = bin_of_deg[d]
        for g in range(4):
            cnt = group_deg_counts[d].get(g, 0)
            bin_g_count[g, b] += cnt
            bin_total[b]      += cnt

    # Per-bin distance stats (aggregate raw arrays across degrees in bin)
    def _bin_dist_stats(key):
        med, lo, hi = [], [], []
        for b in range(n_bins):
            arrays = [dist_deg_data[d][key]
                      for d in all_degrees if bin_of_deg[d] == b]
            if arrays:
                arr   = np.concatenate(arrays)
                clean = arr[~np.isnan(arr)]
            else:
                clean = np.array([])
            if len(clean) == 0:
                med.append(np.nan); lo.append(np.nan); hi.append(np.nan)
            else:
                med.append(float(np.median(clean)))
                lo.append(float(np.percentile(clean, 25)))
                hi.append(float(np.percentile(clean, 75)))
        return np.array(med), np.array(lo), np.array(hi)

    d_tr_med, d_tr_lo, d_tr_hi = _bin_dist_stats("dist_to_train")
    d_sc_med, d_sc_lo, d_sc_hi = _bin_dist_stats("dist_to_same_class")

    # ── Figure ─────────────────────────────────────────────────────────────────
    fw = max(10, n_bins * 1.4 + 4)
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(fw, 5))
    fig.suptitle(
        f"AMP × DMP Group Composition  &  Hop Distances to Training Nodes  "
        f"vs. Node Degree\n{subtitle}",
        fontsize=11, y=1.02,
    )

    # Left — 100 % normalised stacked bar (composition per bin)
    safe_total = np.where(bin_total > 0, bin_total, 1)
    bottoms = np.zeros(n_bins)
    for g in range(4):
        props = bin_g_count[g] / safe_total * 100
        ax_l.bar(pos, props, bottom=bottoms, width=0.65,
                 color=GROUP_COLORS[g], alpha=0.88, zorder=3,
                 label=group_names[g].replace("\n", " "))
        # Label each section if it's large enough to read
        for b in range(n_bins):
            if props[b] >= 8:
                ax_l.text(b, bottoms[b] + props[b] / 2,
                          f"{props[b]:.0f}%", ha="center", va="center",
                          fontsize=7.5, color="white", fontweight="bold")
        bottoms += props

    # Annotate total n= on top of each bar
    for b, n in enumerate(bin_total):
        ax_l.text(b, 101, f"n={int(n)}", ha="center", va="bottom",
                  fontsize=7.5, color="dimgrey")

    ax_l.set_ylim(0, 115)
    ax_l.set_ylabel("% of test nodes in degree bin", fontsize=10)
    ax_l.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:.0f}%"))
    ax_l.tick_params(axis="y", labelsize=8)
    ax_l.set_title("Which structural regime dominates at each degree?\n"
                   "(green = easy: Low AMP + No DMP  ·  red = hard: High AMP + DMP)",
                   fontsize=10, pad=6)
    ax_l.set_xticks(pos)
    ax_l.set_xticklabels(bin_labels, rotation=40, ha="right", fontsize=9)
    ax_l.set_xlabel("Node degree (binned)", fontsize=10)
    ax_l.legend(loc="upper center", bbox_to_anchor=(0.5, -0.20),
                ncol=2, fontsize=8.5, framealpha=0.9, borderpad=0.8)

    # Right — distance lines + IQR bands per bin
    ax_r.plot(pos, d_tr_med, color="#e67e22", lw=2.2, marker="o",
              markersize=6, zorder=4, label="Any training node  (median)")
    ax_r.fill_between(pos, d_tr_lo, d_tr_hi,
                      color="#e67e22", alpha=0.18, zorder=2, label="IQR")

    ax_r.plot(pos, d_sc_med, color="#2980b9", lw=2.2, marker="s",
              markersize=6, zorder=4, label="Same-class training node  (median)")
    ax_r.fill_between(pos, d_sc_lo, d_sc_hi,
                      color="#2980b9", alpha=0.18, zorder=2)

    ax_r.axhline(num_layers, color="#e74c3c", lw=1.6, ls="--", zorder=6,
                 label=f"Model receptive field  ({num_layers} hops)")

    ymax = max(np.nanmax(d_sc_hi) if not np.all(np.isnan(d_sc_hi)) else num_layers,
               num_layers) * 1.3
    ax_r.set_ylim(0, ymax)
    ax_r.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax_r.tick_params(axis="y", labelsize=8)
    ax_r.set_ylabel("Hop distance  (median ± IQR)", fontsize=10)
    ax_r.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.4)
    ax_r.set_title("How far is the nearest (same-class) training node?\n"
                   "Nodes above the red line cannot be reached by the model",
                   fontsize=10, pad=6)
    ax_r.set_xticks(pos)
    ax_r.set_xticklabels(bin_labels, rotation=40, ha="right", fontsize=9)
    ax_r.set_xlabel("Node degree (binned)", fontsize=10)
    ax_r.legend(loc="upper center", bbox_to_anchor=(0.5, -0.20),
                ncol=2, fontsize=8.5, framealpha=0.9, borderpad=0.8)

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.22, wspace=0.42)
    _save(fig, save_dir, f"{prefix}_group_cardinality_and_distance.png", show)


# ── accuracy by Totoro neighbourhood group ──────────────────────────────────────

def plot_acc_by_totoro_group(group_acc_per_run, group_names, group_counts,
                              neighborhood_stats, cfg,
                              save_dir=None, show=False):
    """Two-panel figure comparing accuracy across Totoro neighbourhood groups.

    Test nodes are grouped by how their k-hop training neighbourhood compares
    along two dimensions — count and mean Totoro score of same-class vs
    different-class training neighbours.

    Left panel  — Accuracy per group (bars for single run, boxplots for
                  multi-run).  Group 0 (same class wins both) is coloured
                  green; group 3 (diff class wins / no same-class) is red.
                  Node counts and group definitions are annotated.

    Right panel — For each group, side-by-side bars showing the mean Totoro
                  score of same-class (blue) vs different-class (red) training
                  neighbours and mean neighbour counts (overlaid line).
                  Makes the neighbourhood imbalance that defines each group
                  directly visible.

    Parameters
    ----------
    group_acc_per_run : list[list[float]]
        Per-run accuracy for each group (4 groups, n_runs values each).
    group_names : list[str]
    group_counts : list[int]
        Total test nodes per group.
    neighborhood_stats : dict
        Output of ``utils.get_totoro_neighborhood_groups`` — per-test-node
        arrays: same_count, diff_count, same_totoro, diff_totoro.
    cfg : dict
    save_dir : str or None
    show : bool
    """
    GROUP_COLORS = ["#27ae60", "#5d6d7e", "#a569bd", "#e74c3c"]
    GROUP_EDGE   = ["#1e8449", "#4a5568", "#6c3483", "#c0392b"]

    n_runs  = len(group_acc_per_run[0])
    prefix  = _fname_prefix(cfg)
    n_test  = sum(group_counts)
    k       = cfg["dataset"].get("amp_coeff", 2)   # reuse or default to 2
    subtitle = (
        f"{cfg['dataset']['name']} · {cfg['model']['name']} · "
        f"{cfg.get('split','random')} · "
        f"{'CC' if cfg['dataset'].get('use_cc') else 'noCC'}"
        f"   |   {n_test:,} test nodes  ·  {k}-hop neighbourhood"
    )

    pos = list(range(4))
    g_labels = [n.replace("\n", " ") for n in group_names]

    # ── Per-group neighbourhood stats (mean across test nodes in each group) ──
    group_labels_arr = np.concatenate([
        np.full(group_counts[g], g) for g in range(4)
    ])
    same_tot  = neighborhood_stats["same_totoro"]
    diff_tot  = neighborhood_stats["diff_totoro"]
    same_cnt  = neighborhood_stats["same_count"]
    diff_cnt  = neighborhood_stats["diff_count"]

    mean_same_tot = [same_tot[group_labels_arr == g].mean() if group_counts[g] > 0
                     else 0.0 for g in range(4)]
    mean_diff_tot = [diff_tot[group_labels_arr == g].mean() if group_counts[g] > 0
                     else 0.0 for g in range(4)]
    mean_same_cnt = [same_cnt[group_labels_arr == g].mean() if group_counts[g] > 0
                     else 0.0 for g in range(4)]
    mean_diff_cnt = [diff_cnt[group_labels_arr == g].mean() if group_counts[g] > 0
                     else 0.0 for g in range(4)]

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        f"Accuracy by Totoro Neighbourhood Group\n{subtitle}",
        fontsize=11, y=1.02,
    )

    # ── Left: accuracy per group ───────────────────────────────────────────────
    if n_runs == 1:
        accs = [vals[0] for vals in group_acc_per_run]
        bars = ax_l.bar(pos, accs, color=GROUP_COLORS, edgecolor=GROUP_EDGE,
                        linewidth=1.2, alpha=0.87, zorder=3, width=0.6)
        for bar, acc, cnt in zip(bars, accs, group_counts):
            ax_l.text(bar.get_x() + bar.get_width() / 2,
                      bar.get_height() + 0.015,
                      f"{acc:.1%}", ha="center", va="bottom",
                      fontsize=9.5, fontweight="bold")
            ax_l.text(bar.get_x() + bar.get_width() / 2,
                      -0.06, f"n={cnt}", ha="center", va="top",
                      fontsize=8, color="dimgrey",
                      transform=ax_l.get_xaxis_transform())
    else:
        bp = ax_l.boxplot(group_acc_per_run, positions=pos, widths=0.55,
                          **_BP_KWARGS)
        for patch, color in zip(bp["boxes"], GROUP_COLORS):
            patch.set_facecolor(color)
            patch.set_alpha(0.82)
        for cnt, x in zip(group_counts, pos):
            ax_l.text(x, -0.06, f"n={cnt}", ha="center", va="top",
                      fontsize=8, color="dimgrey",
                      transform=ax_l.get_xaxis_transform())

    overall = (sum(a[0] * c for a, c in zip(group_acc_per_run, group_counts)
                   if not np.isnan(a[0])) / n_test)
    ax_l.axhline(overall, color="dimgrey", lw=1.0, ls=":",
                 label=f"Overall acc  {overall:.1%}", zorder=2)

    ax_l.annotate("← easiest", xy=(0, 0.02), xycoords=("data","axes fraction"),
                  fontsize=8, color="#1e8449", ha="center")
    ax_l.annotate("hardest →", xy=(3, 0.02), xycoords=("data","axes fraction"),
                  fontsize=8, color="#c0392b", ha="center")

    ax_l.set_xticks(pos)
    ax_l.set_xticklabels(g_labels, fontsize=9)
    ax_l.set_ylabel("Accuracy", fontsize=11)
    ax_l.set_ylim(-0.05, 1.15)
    ax_l.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))
    ax_l.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.4)
    ax_l.legend(loc="upper right", fontsize=9, framealpha=0.85)
    ax_l.set_title(
        f"Accuracy per group\n({'single run' if n_runs == 1 else f'{n_runs} seeds'})",
        fontsize=10, pad=6,
    )

    # ── Right: mean same vs diff Totoro score + count per group ───────────────
    bw = 0.3
    xs = np.array(pos, dtype=float)

    ax_r.bar(xs - bw / 2, mean_same_tot, width=bw, color="#2980b9",
             alpha=0.85, label="Same-class train neighbours  (Totoro)", zorder=3)
    ax_r.bar(xs + bw / 2, mean_diff_tot, width=bw, color="#e74c3c",
             alpha=0.85, label="Diff-class train neighbours  (Totoro)", zorder=3)

    # Annotate Totoro values
    for x, sv, dv in zip(xs, mean_same_tot, mean_diff_tot):
        ax_r.text(x - bw / 2, sv + 0.002, f"{sv:.3f}",
                  ha="center", va="bottom", fontsize=7.5, color="#1a5276")
        ax_r.text(x + bw / 2, dv + 0.002, f"{dv:.3f}",
                  ha="center", va="bottom", fontsize=7.5, color="#922b21")

    ax_r.set_ylabel("Mean Totoro score", fontsize=10)
    ax_r.tick_params(axis="y", labelsize=8)
    ax_r.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.4)

    # Overlay mean neighbour counts as a twin axis line
    ax_r2 = ax_r.twinx()
    ax_r2.plot(xs, mean_same_cnt, color="#2980b9", lw=1.8, ls="--",
               marker="o", markersize=5, alpha=0.7, label="Same-class count")
    ax_r2.plot(xs, mean_diff_cnt, color="#e74c3c", lw=1.8, ls="--",
               marker="s", markersize=5, alpha=0.7, label="Diff-class count")
    ax_r2.set_ylabel("Mean # training neighbours", fontsize=9, color="dimgrey")
    ax_r2.tick_params(axis="y", labelsize=7, colors="dimgrey")
    ax_r2.spines["right"].set_color("lightgrey")

    ax_r.set_xticks(pos)
    ax_r.set_xticklabels(g_labels, fontsize=9)
    ax_r.set_title(
        "Neighbourhood profile per group\n"
        "(bars = Totoro scores · dashed lines = neighbour counts)",
        fontsize=10, pad=6,
    )

    # Combined legend below right panel
    handles_bar  = [plt.Rectangle((0,0),1,1, color="#2980b9", alpha=0.85,
                                  label="Same-class Totoro (bar)"),
                    plt.Rectangle((0,0),1,1, color="#e74c3c", alpha=0.85,
                                  label="Diff-class Totoro (bar)")]
    handles_line = [plt.Line2D([0],[0], color="#2980b9", lw=1.8, ls="--",
                               marker="o", markersize=5, label="Same-class count (line)"),
                    plt.Line2D([0],[0], color="#e74c3c", lw=1.8, ls="--",
                               marker="s", markersize=5, label="Diff-class count (line)")]
    ax_r.legend(handles=handles_bar + handles_line,
                loc="upper center", bbox_to_anchor=(0.5, -0.16),
                ncol=2, fontsize=8, framealpha=0.9)

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.20, wspace=0.52)
    _save(fig, save_dir, f"{prefix}_acc_by_totoro_group.png", show)
