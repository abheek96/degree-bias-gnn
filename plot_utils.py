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
                           save_dir=None, show=False):
    """Two-panel figure showing AMP and DMP distributions per node degree.

    Left panel  — Boxplot of neighbor-heterogeneity ratios per degree bucket
                  (AMP: Ambivalent Message Passing).  Each box shows the spread
                  of het values across test nodes at that degree.

    Right panel — Stacked bar chart of DMP node counts per degree bucket
                  (DMP: Distant Message Passing).  Green = non-DMP (label match),
                  purple = DMP (label mismatch with nearest training node).

    Both plots are graph-fixed (independent of training seed / run).

    Parameters
    ----------
    amp_deg_data : dict
        Output of ``utils.get_amp_deg`` — per-degree het_values arrays.
    dmp_deg_data : dict
        Output of ``utils.get_dmp_deg`` — per-degree count_0 / count_1.
    cfg : dict
    save_dir : str or None
    show : bool
    """
    all_degrees = sorted(set(amp_deg_data.keys()) & set(dmp_deg_data.keys()))
    pos    = list(range(len(all_degrees)))
    counts = [amp_deg_data[d]["count"] for d in all_degrees]
    n_test = sum(counts)
    prefix   = _fname_prefix(cfg)
    subtitle = _subtitle(cfg, n_test, len(all_degrees))

    amp_coeff = cfg["dataset"].get("amp_coeff", 1)
    dmp_coeff = cfg["dataset"].get("dmp_coeff", 1)

    fw = max(_fig_w(len(all_degrees)), 12)
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(fw, 5))
    fig.suptitle(
        f"AMP ({amp_coeff}-hop) Heterogeneity Distribution  &  "
        f"DMP ({dmp_coeff}-hop) Counts vs. Node Degree"
        f"\n{subtitle}",
        fontsize=11, y=1.02,
    )

    # ── Left: boxplot of het values per degree ─────────────────────────────────
    def _clean(arr):
        a = arr[~np.isnan(arr)]
        return a if len(a) > 0 else np.array([np.nan])

    het_data = [_clean(amp_deg_data[d]["het_values"]) for d in all_degrees]

    bp = ax_l.boxplot(het_data, positions=pos, widths=0.6, **_BP_KWARGS)
    for patch in bp["boxes"]:
        patch.set_facecolor("#e67e22")
        patch.set_alpha(0.72)
    _count_bars(ax_l, pos, counts)
    ax_l.set_ylabel("Neighbor heterogeneity ratio", fontsize=10)
    ax_l.set_ylim(-0.05, 1.10)
    ax_l.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))
    ax_l.tick_params(axis="y", labelsize=8)
    ax_l.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.4)
    ax_l.set_title("AMP: Heterogeneity Distribution\n(higher = more ambivalent neighbours)",
                   fontsize=10, pad=6)
    handles_l = [
        plt.Rectangle((0, 0), 1, 1, facecolor="#e67e22", alpha=0.72,
                       label="Het ratio (box = IQR, line = median)"),
        plt.Rectangle((0, 0), 1, 1, facecolor="lightgrey", alpha=0.55,
                       label="# test nodes  (right axis)"),
    ]
    ax_l.legend(handles=handles_l, loc="upper center",
                bbox_to_anchor=(0.5, -0.18), ncol=2,
                fontsize=8.5, framealpha=0.9, borderpad=0.8)
    _degree_axis(ax_l, pos, all_degrees)

    # ── Right: stacked bar of DMP counts per degree ────────────────────────────
    cnt0 = np.array([dmp_deg_data[d]["count_0"] for d in all_degrees])
    cnt1 = np.array([dmp_deg_data[d]["count_1"] for d in all_degrees])

    ax_r.bar(pos, cnt0, width=0.6, color="#27ae60", alpha=0.82, label="Non-DMP  (label match)",  zorder=3)
    ax_r.bar(pos, cnt1, width=0.6, color="#8e44ad", alpha=0.82, label="DMP  (label mismatch)",
             bottom=cnt0, zorder=3)
    ax_r.set_ylabel("# test nodes", fontsize=10)
    ax_r.tick_params(axis="y", labelsize=8)
    ax_r.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.4)
    ax_r.set_title("DMP: Node Counts per Degree\n(purple = nearest train label mismatches true label)",
                   fontsize=10, pad=6)
    ax_r.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18),
                ncol=2, fontsize=8.5, framealpha=0.9, borderpad=0.8)
    _degree_axis(ax_r, pos, all_degrees)

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.24, wspace=0.45)
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
    """Compare AMP × DMP group accuracy across node degrees on one set of axes.

    Draws four lines (one per group) overlaid on the same panel:
      - x-axis : node degree
      - y-axis : accuracy
      - Single run  : scatter + connecting line per group
      - Multi-run   : mean line with ±1 std shaded band per group

    This reveals whether the group ranking (Low AMP + No DMP > … > High AMP +
    DMP) is consistent across all degree values or inverts/collapses at certain
    degrees.  Degree buckets with fewer than 2 nodes in a group are omitted
    (NaN) to avoid misleading single-node accuracy values.

    Parameters
    ----------
    group_deg_acc : dict {group_id (int) -> {degree (int) -> list[float]}}
        Per-run accuracy for each (group, degree) cell.  Inner list length
        equals the number of runs.
    group_names : list[str]
        Human-readable label for each of the 4 groups.
    cfg : dict
    save_dir : str or None
    show : bool
    """
    GROUP_COLORS  = ["#27ae60", "#5d6d7e", "#a569bd", "#e74c3c"]
    GROUP_MARKERS = ["o", "s", "^", "D"]

    prefix    = _fname_prefix(cfg)
    amp_coeff = cfg["dataset"].get("amp_coeff", 1)
    dmp_coeff = cfg["dataset"].get("dmp_coeff", 1)
    amp_thr   = cfg["dataset"].get("amp_threshold", 0.5)
    dataset   = cfg["dataset"]["name"]
    model     = cfg["model"]["name"]
    split     = cfg.get("split", "random")
    cc        = "CC" if cfg["dataset"].get("use_cc") else "noCC"

    # Collect all degrees present in any group
    all_degrees = sorted({d for g in range(4) for d in group_deg_acc[g]})
    pos = list(range(len(all_degrees)))
    deg_to_pos = {d: p for d, p in zip(all_degrees, pos)}

    n_runs = max(len(v) for g in range(4)
                 for v in group_deg_acc[g].values()) if all_degrees else 1

    fw = max(_fig_w(len(all_degrees)), 12)
    fig, ax = plt.subplots(figsize=(fw, 5))

    for g, (color, marker, name) in enumerate(
        zip(GROUP_COLORS, GROUP_MARKERS, group_names)
    ):
        # Build mean and std across runs for each degree
        means, stds, xs = [], [], []
        for d in all_degrees:
            vals = group_deg_acc[g].get(d, [])
            valid = [v for v in vals if not np.isnan(v)]
            if len(valid) < 1:
                continue
            xs.append(deg_to_pos[d])
            means.append(float(np.mean(valid)))
            stds.append(float(np.std(valid)) if len(valid) > 1 else 0.0)

        if not xs:
            continue

        xs    = np.array(xs)
        means = np.array(means)
        stds  = np.array(stds)
        label = name.replace("\n", " ")

        ax.plot(xs, means, color=color, lw=2.0, marker=marker,
                markersize=5, zorder=4, label=label)
        if n_runs > 1:
            ax.fill_between(xs, means - stds, means + stds,
                            color=color, alpha=0.15, zorder=2)

    ax.set_ylabel("Accuracy", fontsize=11)
    ax.set_ylim(-0.05, 1.10)
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.4)
    _degree_axis(ax, pos, all_degrees)

    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18),
              ncol=4, fontsize=9, framealpha=0.9, borderpad=0.8)

    run_str = "single run" if n_runs == 1 else f"{n_runs} seeds  (mean ± 1σ)"
    ax.set_title(
        f"Accuracy by AMP × DMP Group vs. Node Degree  —  {run_str}\n"
        f"{dataset} · {model} · {split} · {cc}"
        f"   |   AMP {amp_coeff}-hop  thr={amp_thr}  ·  DMP {dmp_coeff}-hop",
        fontsize=10,
    )

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.22)
    _save(fig, save_dir, f"{prefix}_acc_by_amp_dmp_group_vs_degree.png", show)


# ── group cardinality + hop distances vs degree ─────────────────────────────────

def plot_group_cardinality_and_distance(group_deg_counts, dist_deg_data,
                                         group_names, cfg,
                                         save_dir=None, show=False):
    """Two-panel figure: group cardinality and hop distances per degree bucket.

    Left panel — Stacked bar chart showing how many test nodes at each degree
                 fall into each AMP × DMP group.  Reveals which structural
                 regime dominates at low vs high degrees.

    Right panel — Median ± IQR lines for dist_to_train (orange) and
                  dist_to_same_class (green) per degree, with a red dashed
                  reference line at ``num_layers`` (model receptive field).

    Both panels share no axis — they are sized independently so each panel
    is easy to read on its own.

    Parameters
    ----------
    group_deg_counts : dict {degree (int) -> {group (int): count (int)}}
        Output of ``utils.get_group_deg_counts``.
    dist_deg_data : dict
        Output of ``utils.get_distance_deg``.
    group_names : list[str]
        Human-readable labels for groups 0-3.
    cfg : dict
    save_dir : str or None
    show : bool
    """
    GROUP_COLORS = ["#27ae60", "#5d6d7e", "#a569bd", "#e74c3c"]

    num_layers  = cfg["model"]["num_layers"]
    amp_coeff   = cfg["dataset"].get("amp_coeff", 1)
    dmp_coeff   = cfg["dataset"].get("dmp_coeff", 1)
    amp_thr     = cfg["dataset"].get("amp_threshold", 0.5)
    prefix      = _fname_prefix(cfg)

    all_degrees = sorted(set(group_deg_counts) & set(dist_deg_data))
    pos         = list(range(len(all_degrees)))
    n_test      = sum(sum(group_deg_counts[d].values()) for d in all_degrees)
    subtitle    = (
        f"{cfg['dataset']['name']} · {cfg['model']['name']} · "
        f"{cfg.get('split','random')} · "
        f"{'CC' if cfg['dataset'].get('use_cc') else 'noCC'}"
        f"   |   AMP {amp_coeff}-hop  thr={amp_thr}  ·  DMP {dmp_coeff}-hop"
        f"   |   {n_test:,} test nodes"
    )

    fw = max(_fig_w(len(all_degrees)), 14)
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(fw, 5))
    fig.suptitle(
        f"AMP × DMP Group Cardinality & Hop Distances vs. Node Degree\n{subtitle}",
        fontsize=11, y=1.02,
    )

    # ── Left: stacked bar of group counts per degree ────────────────────────────
    bottoms = np.zeros(len(all_degrees))
    for g in range(4):
        counts_g = np.array([group_deg_counts[d].get(g, 0) for d in all_degrees],
                             dtype=float)
        ax_l.bar(pos, counts_g, bottom=bottoms, width=0.7,
                 color=GROUP_COLORS[g], alpha=0.85, zorder=3,
                 label=group_names[g].replace("\n", " "))
        bottoms += counts_g

    ax_l.set_ylabel("# test nodes", fontsize=10)
    ax_l.tick_params(axis="y", labelsize=8)
    ax_l.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.4)
    ax_l.set_title("Group Cardinality per Degree Bucket", fontsize=10, pad=6)
    ax_l.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18),
                ncol=2, fontsize=8.5, framealpha=0.9, borderpad=0.8)
    _degree_axis(ax_l, pos, all_degrees)

    # ── Right: distance medians + IQR bands per degree ─────────────────────────
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

    ax_r.plot(pos, d_tr_med, color="#e67e22", lw=2.2, marker="o",
              markersize=4, zorder=4, label="Nearest train node  (any class)")
    ax_r.fill_between(pos, d_tr_lo, d_tr_hi, color="#e67e22", alpha=0.18, zorder=2)

    ax_r.plot(pos, d_sc_med, color="#27ae60", lw=2.2, marker="s",
              markersize=4, zorder=4, label="Nearest same-class train node")
    ax_r.fill_between(pos, d_sc_lo, d_sc_hi, color="#27ae60", alpha=0.18, zorder=2)

    ax_r.axhline(num_layers, color="#e74c3c", lw=1.5, ls="--", zorder=6,
                 label=f"Model depth  ({num_layers} layers)")

    ymax = max(
        np.nanmax(d_sc_hi) if not np.all(np.isnan(d_sc_hi)) else num_layers,
        num_layers
    ) * 1.25
    ax_r.set_ylim(0, ymax)
    ax_r.set_ylabel("Hop distance  (median ± IQR)", fontsize=10)
    ax_r.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax_r.tick_params(axis="y", labelsize=8)
    ax_r.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.4)
    ax_r.set_title("Hop Distances to Training Nodes per Degree", fontsize=10, pad=6)
    ax_r.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18),
                ncol=2, fontsize=8.5, framealpha=0.9, borderpad=0.8)
    _degree_axis(ax_r, pos, all_degrees)

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.24, wspace=0.45)
    _save(fig, save_dir, f"{prefix}_group_cardinality_and_distance.png", show)
