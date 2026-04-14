import logging
import os

import matplotlib.patches as mpatches
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


def nodes_in_degree_group(deg, degree, mask=None):
    """Return the node indices that belong to a given degree group.

    Parameters
    ----------
    deg    : 1-D LongTensor of node degrees, one entry per node in the graph
             (length == data.num_nodes).
    degree : int  — exact degree value, or
             tuple (lo, hi) — inclusive range [lo, hi].
    mask   : optional 1-D BoolTensor of length num_nodes; when provided, only
             nodes where mask is True are considered (e.g. train_mask).

    Returns
    -------
    1-D LongTensor of node indices (into the original node set).
    """
    deg = deg.cpu()
    if isinstance(degree, (tuple, list)):
        lo, hi = degree
        hit = (deg >= lo) & (deg <= hi)
    else:
        hit = deg == degree

    if mask is not None:
        hit = hit & mask.cpu()

    return hit.nonzero(as_tuple=False).view(-1)


def get_accuracy_class(pred, true):
    """Group per-node predictions by true class label.

    Mirrors get_accuracy_deg but keyed by class instead of degree.

    Parameters
    ----------
    pred : 1-D LongTensor of predicted class indices (test nodes).
    true : 1-D LongTensor of ground-truth class indices (test nodes).

    Returns
    -------
    dict mapping class_label (int) -> {
        'preds'  : predicted labels for nodes of that class (cpu tensor),
        'labels' : true labels for nodes of that class (cpu tensor),
        'acc'    : classification accuracy as a Python float,
    }
    """
    pred, true = pred.cpu(), true.cpu()
    result = {}
    for c in true.unique():
        idx = (true == c).nonzero(as_tuple=False).view(-1)
        p, t = pred[idx], true[idx]
        result[c.item()] = {
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


# Shared plot colours
_ACC_COLOR     = "#388E3C"                 # dark green — accuracy
_PURITY_COLOR  = "#7B1FA2"                 # purple     — purity
_ANOMALY_COLOR = "#C62828"                 # dark crimson — anomaly highlights
_CARD_COLORS   = {1: "#2196F3", 2: "#FF5722"}  # blue, deep-orange — cardinality

# SPL combined plot colours
_SPL_ALL_COLOR = "#1976D2"   # blue   — any training node
_SPL_SC_COLOR  = "#E53935"   # red    — same-class training node
_SPL_ACC_COLOR = _ACC_COLOR
_SPL_PUR_COLOR = _PURITY_COLOR

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
    Multi-run   — across-runs boxplot, PLUS one per-run scatter for each seed.

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
        _plot_single(all_degrees, pos, deg_data, subtitle, prefix,
                     save_dir, show, run_idx=0)
    else:
        _plot_across_runs(all_degrees, pos, deg_data, n_runs, subtitle, prefix, save_dir, show)
        base_seed = cfg.get("seed", 42)
        for r in range(n_runs):
            seed = base_seed + r
            _plot_single(all_degrees, pos, deg_data, subtitle, prefix,
                         save_dir, show, run_idx=r, seed=seed)


# ── single run: scatter + OLS trend + Δ subplot ────────────────────────────────

def _plot_single(all_degrees, pos, deg_data, subtitle, prefix, save_dir, show,
                 run_idx=0, seed=None):
    counts   = [len(deg_data[d][run_idx]) for d in all_degrees]
    mean_acc = [
        float(deg_data[d][run_idx].mean()) if counts[i] > 0 else np.nan
        for i, d in enumerate(all_degrees)
    ]
    n_test  = sum(counts)
    overall = (sum(a * c for a, c in zip(mean_acc, counts) if not np.isnan(a))
               / n_test)

    max_count   = max(counts) or 1
    bubble_size = [max(30, 700 * c / max_count) for c in counts]

    run_label = f"run {run_idx + 1}" + (f"  (seed {seed})" if seed is not None else "")
    fname_tag = f"run{run_idx + 1:02d}" + (f"_seed{seed}" if seed is not None else "")

    fig, ax_main = plt.subplots(figsize=(_fig_w(len(all_degrees)), 5))

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
        f"Accuracy vs. Node Degree  —  {run_label}\n{subtitle}", fontsize=11
    )

    _degree_axis(ax_main, pos, all_degrees)

    fig.tight_layout()
    _save(fig, _subdir(save_dir, "acc_vs_degree"),
          f"{prefix}_acc_vs_degree_{fname_tag}.png", show)


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
    """Single panel: accuracy (median ± IQR across all runs) with both distances.

    Left y-axis  — Accuracy median ± IQR across runs (green).
    Right y-axis — Dist to nearest train node (orange) and dist to nearest
                   same-class train node (teal), both median ± IQR across nodes.
                   A dashed red line marks the model depth (receptive field limit).

    Both distance signals share the right y-axis so the gap between them —
    the extra hops to reach a same-class label — is directly readable.
    Distance signals are graph-fixed (identical across runs).  Accuracy is
    aggregated by concatenating per-node correctness values from all runs for
    each degree group, then computing median and interquartile range.

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
        Unused; kept for backward compatibility.
    """
    num_layers = cfg["model"]["num_layers"] - 1   # GCNConv layers only; last layer is nn.Linear
    n_runs = len(run_results)
    all_degrees, deg_data = _collect(run_results)

    all_degrees = sorted(set(all_degrees) & set(dist_deg_data.keys()))
    pos    = list(range(len(all_degrees)))
    n_test = sum(len(deg_data[d][0]) for d in all_degrees)
    prefix   = _fname_prefix(cfg)
    subtitle = _subtitle(cfg, n_test, len(all_degrees))

    # ── Accuracy: per-run means → median / IQR across runs ───────────────────
    # Compute the mean accuracy for each degree within each run first, then
    # take median and IQR across those per-run means.  Concatenating raw 0/1
    # values before taking the median would collapse the binary array and give
    # 100% for any group where more than half the (node × run) pairs are correct.
    acc_median, acc_q1, acc_q3 = [], [], []
    for d in all_degrees:
        means = [float(deg_data[d][r].mean()) for r in range(n_runs)
                 if len(deg_data[d][r]) > 0]
        if not means:
            acc_median.append(np.nan); acc_q1.append(np.nan); acc_q3.append(np.nan)
        else:
            acc_median.append(float(np.median(means)))
            acc_q1.append(float(np.percentile(means, 25)))
            acc_q3.append(float(np.percentile(means, 75)))
    acc_median = np.array(acc_median)
    acc_q1     = np.array(acc_q1)
    acc_q3     = np.array(acc_q3)
    overall    = float(np.nanmean(acc_median))

    # ── Distance signals — graph-fixed, computed once ─────────────────────────
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

    # ── Figure ────────────────────────────────────────────────────────────────
    fw = max(_fig_w(len(all_degrees)), 8)
    fig, ax = plt.subplots(figsize=(fw, 5))
    fig.suptitle(
        f"Accuracy & Hop Distance to Training Nodes vs. Node Degree"
        f"\n{subtitle}  ({n_runs} runs aggregated)",
        fontsize=11, y=1.02,
    )

    # Left axis — accuracy
    ax.plot(pos, acc_median, color=_ACC_COLOR, lw=1.8, marker="s",
            markersize=4, zorder=4)
    ax.fill_between(pos, acc_q1, acc_q3, color=_ACC_COLOR, alpha=0.15, zorder=3)
    ax.axhline(overall, color="dimgrey", lw=1.0, ls=":", zorder=2)
    ax.set_ylim(-0.05, 1.10)
    ax.set_ylabel("Accuracy", fontsize=10, color=_ACC_COLOR)
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))
    ax.tick_params(axis="y", labelsize=8, colors=_ACC_COLOR)
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.3)

    # Right axis — both distance signals
    ax_d = ax.twinx()
    _D_ANY   = "#e67e22"   # orange  — any train node
    _D_SAME  = "#0097A7"   # teal    — same-class train node
    ymax = max(
        np.nanmax(d_tr_hi) if not np.all(np.isnan(d_tr_hi)) else num_layers,
        np.nanmax(d_sc_hi) if not np.all(np.isnan(d_sc_hi)) else num_layers,
        num_layers,
    ) * 1.3
    ax_d.plot(pos, d_tr_med, color=_D_ANY,  lw=2.0, zorder=4)
    ax_d.fill_between(pos, d_tr_lo, d_tr_hi, color=_D_ANY,  alpha=0.18, zorder=2)
    ax_d.plot(pos, d_sc_med, color=_D_SAME, lw=2.0, zorder=4)
    ax_d.fill_between(pos, d_sc_lo, d_sc_hi, color=_D_SAME, alpha=0.18, zorder=2)
    ax_d.axhline(num_layers, color="#e74c3c", lw=1.5, ls="--", zorder=6)
    ax_d.set_ylim(0, ymax)
    ax_d.set_ylabel("Hop distance  (median ± IQR)", fontsize=10)
    ax_d.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax_d.tick_params(axis="y", labelsize=8)

    # Legend
    handles = [
        plt.Line2D([0], [0], color=_ACC_COLOR, lw=2.0, marker="s", markersize=4,
                   label=f"Accuracy  (median ± IQR, {n_runs} runs)"),
        plt.Line2D([0], [0], color="dimgrey", lw=1.0, ls=":",
                   label=f"Mean acc  {overall:.1%}"),
        plt.Line2D([0], [0], color=_D_ANY,  lw=2.0,
                   label="Dist to nearest train node  (median ± IQR)"),
        plt.Line2D([0], [0], color=_D_SAME, lw=2.0,
                   label="Dist to nearest same-class train node  (median ± IQR)"),
        plt.Line2D([0], [0], color="#e74c3c", lw=1.5, ls="--",
                   label=f"Model depth  ({num_layers} layers)"),
    ]
    ax.legend(handles=handles, loc="upper center",
              bbox_to_anchor=(0.5, -0.18),
              ncol=2, fontsize=8.5, framealpha=0.9, borderpad=0.8)

    _degree_axis(ax, pos, all_degrees)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.24)
    _save(fig, _subdir(save_dir, "acc_vs_distance"),
          f"{prefix}_combined_vs_degree.png", show)


# ── neighbourhood cardinality vs degree ────────────────────────────────────────

def plot_neighborhood_cardinality_vs_degree(
    test_deg, cardinality_by_k, deg_acc_results, cfg,
    all_deg=None, purity_by_k=None,
    save_dir=None, show=False,
):
    """Neighbourhood cardinality (k=1, k=2), accuracy, and Δ purity vs. 1-hop degree.

    Top panel
        Left y-axis  — mean neighbourhood cardinality ± 1 std for k=1 (blue)
                       and k=2 (orange), grouped by 1-hop degree.
        Right y-axis — median classification accuracy ± IQR across runs (green).

    Bottom panel  (when ``all_deg`` and ``purity_by_k`` are provided)
        Mean Δ purity (purity[k_max] − purity[k_min]) ± 1 std per degree group
        (purple), with a zero reference line.  Uses ALL graph nodes.

    Anomaly highlights
        Degree groups where ALL three structural signals align — accuracy below
        the cross-degree median, k=2 cardinality above average, and Δ purity
        more negative than average — are flagged with a dashed crimson vertical
        line spanning both panels, a ★ marker on the accuracy curve, and a
        degree label at the top of the upper panel.  Up to five anomalies are
        shown (the worst by accuracy).  Requires ≥ 3 test nodes per group to
        suppress single-node noise.  When purity data is absent, only the
        first two criteria are used.

    Parameters
    ----------
    test_deg         : LongTensor [N_test]
    cardinality_by_k : dict {k: LongTensor [N_test]}
    deg_acc_results  : list[dict]  — output of get_accuracy_deg per run.
    cfg              : dict
    all_deg          : LongTensor [N_all], optional
    purity_by_k      : dict {k: FloatTensor [N_all]}, optional
    save_dir         : str or None
    show             : bool
    """

    deg = test_deg.cpu()
    all_degrees = sorted(deg.unique().tolist())
    pos    = list(range(len(all_degrees)))
    n_test = len(deg)

    # ── cardinality stats per degree group ────────────────────────────────────
    card_stats = {}
    k_values   = sorted(cardinality_by_k.keys())
    for k, card_tensor in cardinality_by_k.items():
        card = card_tensor.cpu().float()
        means, stds = [], []
        for d in all_degrees:
            vals = card[deg == d]
            means.append(float(vals.mean()))
            stds.append(float(vals.std()) if len(vals) > 1 else 0.0)
        card_stats[k] = {"means": np.array(means), "stds": np.array(stds)}

    # ── accuracy stats per degree group ──────────────────────────────────────
    _, deg_data = _collect(deg_acc_results)
    n_runs = len(deg_acc_results)
    acc_median, acc_q1, acc_q3 = [], [], []
    for d in all_degrees:
        run_means = [float(a.mean()) for a in deg_data.get(d, []) if len(a) > 0]
        if run_means:
            acc_median.append(float(np.median(run_means)))
            acc_q1.append(float(np.percentile(run_means, 25)))
            acc_q3.append(float(np.percentile(run_means, 75)))
        else:
            acc_median.append(float("nan"))
            acc_q1.append(float("nan"))
            acc_q3.append(float("nan"))
    acc_median = np.array(acc_median)
    acc_q1, acc_q3 = np.array(acc_q1), np.array(acc_q3)

    # ── delta purity stats per degree group (uses ALL nodes) ─────────────────
    has_purity = (all_deg is not None and purity_by_k is not None
                  and len(purity_by_k) >= 2)
    if has_purity:
        full_deg = all_deg.cpu()
        k_ks     = sorted(purity_by_k.keys())
        k_lo, k_hi = k_ks[0], k_ks[-1]
        delta_all = purity_by_k[k_hi].cpu().float() - purity_by_k[k_lo].cpu().float()

        dp_means = []
        for d in all_degrees:
            vals  = delta_all[full_deg == d]
            valid = vals[~torch.isnan(vals)]
            dp_means.append(float(valid.mean()) if len(valid) > 0 else float("nan"))
        dp_means = np.array(dp_means)

    # ── anomaly detection ─────────────────────────────────────────────────────
    # Two complementary criteria, both require ≥ 3 test nodes:
    #
    # Criterion A — three-signal conjunction:
    #   1. Accuracy below the cross-degree median
    #   2. k=2 cardinality above the cross-degree mean
    #   3. Δ purity more negative than the cross-degree mean  (purity only)
    #
    # Criterion B — sharp structural deterioration:
    #   1. Accuracy below the cross-degree median
    #   2. Δ purity drops sharply step-to-step (diff < mean − 1 std)  (purity only)
    #
    # Anomaly set = union of both criteria, sorted by accuracy, capped at 5.
    counts = np.array([(deg == d).sum().item() for d in all_degrees])
    anomaly_idx = []
    if len(all_degrees) > 3:
        enough_nodes = counts >= 3
        low_acc   = acc_median < np.nanmedian(acc_median)
        k2_means  = card_stats.get(2, card_stats[max(k_values)])["means"]
        high_card = k2_means > np.nanmean(k2_means)

        if has_purity:
            # Criterion A
            neg_dp   = dp_means < np.nanmean(dp_means)
            crit_a   = low_acc & high_card & neg_dp & enough_nodes
            # Criterion B — step-to-step drop in Δpurity
            dp_diff  = np.concatenate([[0.0], np.diff(dp_means)])
            sharp_dp = dp_diff < (np.nanmean(dp_diff) - np.nanstd(dp_diff))
            crit_b   = low_acc & sharp_dp & enough_nodes
            combined = crit_a | crit_b
        else:
            combined = low_acc & high_card & enough_nodes

        candidates = np.where(combined)[0]
        if len(candidates) > 0:
            # Sort by accuracy (most anomalous = lowest accuracy first), cap at 5
            order = np.argsort(acc_median[candidates])
            anomaly_idx = candidates[order][:5].tolist()

    # ── figure layout ─────────────────────────────────────────────────────────
    fig_w = _fig_w(len(all_degrees))
    if has_purity:
        fig, (ax_card, ax_dp) = plt.subplots(
            2, 1, figsize=(fig_w, 7),
            sharex=True, gridspec_kw={"height_ratios": [3, 1.4]},
        )
    else:
        fig, ax_card = plt.subplots(figsize=(fig_w, 5))
        ax_dp = None

    # ── top panel: anomaly vertical lines (behind all other elements) ─────────
    all_axes = [ax_card] + ([ax_dp] if ax_dp is not None else [])
    for ai in anomaly_idx:
        for ax in all_axes:
            ax.axvline(pos[ai], color=_ANOMALY_COLOR, lw=0.9,
                       ls="--", alpha=0.40, zorder=1)

    # ── top panel: cardinality lines ─────────────────────────────────────────
    for k in k_values:
        color = _CARD_COLORS.get(k, "#9C27B0")
        m, s  = card_stats[k]["means"], card_stats[k]["stds"]
        ax_card.plot(pos, m, color=color, linewidth=1.8,
                     marker="o", markersize=4, zorder=3,
                     label=f"{k}-hop cardinality (mean ± 1 std)")
        ax_card.fill_between(pos, m - s, m + s,
                             color=color, alpha=0.18, zorder=2)

    ax_card.set_ylabel("Neighbourhood cardinality  (# nodes)", fontsize=11)
    ax_card.set_ylim(bottom=0)
    ax_card.yaxis.set_major_formatter(
        ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax_card.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.3)
    # Show degree labels on top panel even when sharex=True suppresses them
    if has_purity:
        _degree_axis(ax_card, pos, all_degrees)
        plt.setp(ax_card.get_xticklabels(), visible=True)

    # ── top panel: accuracy twin axis ────────────────────────────────────────
    ax_acc = ax_card.twinx()
    # Regular accuracy line
    ax_acc.plot(pos, acc_median, color=_ACC_COLOR, linewidth=1.8,
                marker="s", markersize=4, zorder=4,
                label=f"Accuracy (median ± IQR, {n_runs} runs)")
    ax_acc.fill_between(pos, acc_q1, acc_q3,
                        color=_ACC_COLOR, alpha=0.15, zorder=3)
    # Anomaly accent markers on the accuracy line
    if anomaly_idx:
        ax_acc.scatter([pos[ai] for ai in anomaly_idx],
                       [acc_median[ai] for ai in anomaly_idx],
                       color=_ANOMALY_COLOR, marker="*", s=120,
                       zorder=6, label="_nolegend_")
    ax_acc.set_ylabel("Classification accuracy", fontsize=11, color=_ACC_COLOR)
    ax_acc.tick_params(axis="y", colors=_ACC_COLOR, labelsize=8)
    ax_acc.set_ylim(-0.05, 1.10)
    ax_acc.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))

    # Degree labels at the top of the upper panel for each anomaly
    for ai in anomaly_idx:
        ax_card.text(pos[ai], 1.01, f"d={all_degrees[ai]}",
                     transform=ax_card.get_xaxis_transform(),
                     ha="center", va="bottom", fontsize=7,
                     color=_ANOMALY_COLOR, rotation=90)

    # ── top panel: combined legend ────────────────────────────────────────────
    handles_card = [
        plt.Line2D([0], [0], color=_CARD_COLORS.get(k, "#9C27B0"), lw=2,
                   marker="o", markersize=4,
                   label=f"{k}-hop cardinality (mean ± 1 std)")
        for k in k_values
    ]
    handles_acc = [
        plt.Line2D([0], [0], color=_ACC_COLOR, lw=2, marker="s", markersize=4,
                   label=f"Accuracy (median ± IQR, {n_runs} runs)"),
    ]
    handles_anom = (
        [plt.Line2D([0], [0], color=_ANOMALY_COLOR, lw=0, marker="*",
                    markersize=9, label="Anomaly: acc↓  (card↑ + Δpur↓)  or  Δpur sharp drop")]
        if anomaly_idx else []
    )
    ax_card.legend(handles=handles_card + handles_acc + handles_anom,
                   loc="upper left", fontsize=8, framealpha=0.88)

    prefix   = _fname_prefix(cfg)
    subtitle = _subtitle(cfg, n_test, len(all_degrees))
    title    = "Neighbourhood Cardinality (k=1, 2) & Accuracy vs. Node Degree"
    if has_purity:
        title += "  +  Δ Purity"
    ax_card.set_title(f"{title}\n{subtitle}", fontsize=11)

    # ── bottom panel: delta purity ────────────────────────────────────────────
    if has_purity and ax_dp is not None:
        ax_dp.plot(pos, dp_means, color=_PURITY_COLOR, linewidth=1.8,
                   marker="o", markersize=4, zorder=3,
                   label=f"Δ purity  (k={k_hi}−k={k_lo})  mean")
        ax_dp.axhline(0, color="dimgrey", lw=1.0, ls="--", zorder=2,
                      label="No change")
        # Anomaly accent markers on the delta purity line
        if anomaly_idx:
            ax_dp.scatter([pos[ai] for ai in anomaly_idx],
                          [dp_means[ai] for ai in anomaly_idx],
                          color=_ANOMALY_COLOR, marker="*", s=120,
                          zorder=6)
        ax_dp.set_ylabel(f"Δ purity\n(k={k_hi}−k={k_lo})", fontsize=10)
        ax_dp.yaxis.set_major_formatter(
            ticker.FuncFormatter(lambda x, _: f"{x:+.2f}"))
        ax_dp.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.3)
        ax_dp.legend(loc="upper right", fontsize=7, framealpha=0.85)
        _degree_axis(ax_dp, pos, all_degrees)
    else:
        _degree_axis(ax_card, pos, all_degrees)

    fig.tight_layout()
    _save(fig, _subdir(save_dir, "neighborhood_cardinality"),
          f"{prefix}_neighborhood_cardinality_vs_degree.png", show)


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


# ── accuracy trend with depth per degree group ─────────────────────────────────

def plot_acc_trend_by_degree(results_by_label, cfg, save_dir=None, show=False):
    """Linear trend of accuracy with depth, plotted per degree group.

    For each degree group, fits a line to mean accuracy vs. layer count
    and plots the slope on the y-axis.  Positive slope = more layers help
    nodes of that degree; negative = over-smoothing hurts them.  One line
    per model, making cross-degree and cross-model comparisons direct.

    Labels in results_by_label must follow "<ModelName> L=<int>" so that
    layer counts can be parsed.

    Parameters
    ----------
    results_by_label : dict[str, list[dict]]
        Same format as plot_acc_vs_degree_by_layers.
    cfg : dict
    save_dir : str or None
    show : bool
    """
    labels = list(results_by_label.keys())
    if not labels:
        return

    # Group labels by model name; parse layer count from "ModelName L=<int>"
    model_labels = {}
    for lbl in labels:
        parts = lbl.split(" L=")
        if len(parts) != 2:
            continue
        model_name = parts[0]
        try:
            layer_count = int(parts[1])
        except ValueError:
            continue
        model_labels.setdefault(model_name, []).append((layer_count, lbl))
    for model_name in model_labels:
        model_labels[model_name].sort(key=lambda x: x[0])

    all_degrees = sorted({d for results in results_by_label.values()
                          for run in results for d in run})
    n_deg  = len(all_degrees)
    pos    = list(range(n_deg))
    prefix = _fname_prefix(cfg)

    _, first_deg_data = _collect(results_by_label[labels[0]])
    counts = [len(first_deg_data[d][0]) if d in first_deg_data else 0
              for d in all_degrees]
    n_test   = sum(counts)
    subtitle = _subtitle(cfg, n_test, n_deg)

    # Mean accuracy per (label, degree), averaged over runs
    label_deg_mean = {}
    for lbl in labels:
        _, deg_data = _collect(results_by_label[lbl])
        label_deg_mean[lbl] = {}
        for d in all_degrees:
            if d in deg_data:
                run_means = [float(a.mean()) for a in deg_data[d] if len(a) > 0]
                label_deg_mean[lbl][d] = float(np.mean(run_means)) if run_means else float("nan")
            else:
                label_deg_mean[lbl][d] = float("nan")

    fig, ax = plt.subplots(figsize=(_fig_w(n_deg), 5))

    for i, (model_name, layer_list) in enumerate(model_labels.items()):
        layer_counts = [lc for lc, _ in layer_list]
        color = _LAYER_COLORS[i % len(_LAYER_COLORS)]

        slopes = []
        for d in all_degrees:
            accs  = [label_deg_mean[lbl][d] for _, lbl in layer_list]
            valid = [(lc, a) for lc, a in zip(layer_counts, accs) if not np.isnan(a)]
            if len(valid) >= 2:
                xs, ys = zip(*valid)
                slopes.append(float(np.polyfit(xs, ys, 1)[0]))
            else:
                slopes.append(float("nan"))

        ax.plot(pos, slopes, marker="o", linewidth=1.8, markersize=5,
                color=color, label=model_name, zorder=3)

    ax.axhline(0, color="dimgrey", lw=1.0, ls="--", zorder=2, label="No trend")

    _count_bars(ax, pos, counts)

    ax.set_ylabel("Accuracy slope  (Δ acc / layer)", fontsize=11)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:+.3f}"))
    ax.legend(loc="upper left", fontsize=8, framealpha=0.85)
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.4)
    ax.set_title(
        f"Accuracy Trend with Depth by Degree\n{subtitle}", fontsize=11,
    )

    ax.set_xlabel("Node degree", fontsize=11)
    ax.set_xlim(pos[0] - 0.6, pos[-1] + 0.6)
    step = max(1, n_deg // 30)
    ax.set_xticks(pos[::step])
    ax.set_xticklabels(all_degrees[::step], rotation=55, ha="right", fontsize=8)

    fig.tight_layout()
    models_str = "_".join(model_labels.keys())
    _save(fig, _subdir(save_dir, "acc_vs_degree_by_layers"),
          f"{prefix}_acc_trend_by_degree_{models_str}.png", show)


# ── average shortest path length vs. degree ───────────────────────────────────

def plot_spl_vs_degree(test_deg, avg_spl, cfg, save_dir=None, show=False,
                       same_class=False):
    """Boxplots of average shortest path length to training nodes, by degree.

    For each degree group, the distribution of per-node average SPL across
    all test nodes at that degree is shown as a boxplot.  Bottom panel shows
    node count per degree.

    Parameters
    ----------
    test_deg   : 1-D LongTensor of degrees for test nodes.
    avg_spl    : 1-D FloatTensor of average SPL values for test nodes (NaN-safe).
    cfg        : dict
    save_dir   : str or None
    show       : bool
    same_class : bool — if True, labels reflect same-class training nodes only.
    """
    deg = test_deg.cpu()
    spl = avg_spl.cpu().numpy()

    unique_degrees = sorted(deg.unique().tolist())
    pos    = list(range(len(unique_degrees)))
    prefix = _fname_prefix(cfg)
    n_test = int(len(deg))
    subtitle = _subtitle(cfg, n_test, len(unique_degrees))

    bp_data = []
    counts  = []
    for d in unique_degrees:
        mask  = (deg == d).numpy()
        vals  = spl[mask]
        valid = vals[~np.isnan(vals)]
        bp_data.append(valid if len(valid) > 0 else np.array([float("nan")]))
        counts.append(int(mask.sum()))

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(_fig_w(len(unique_degrees)), 7),
        sharex=True, gridspec_kw={"height_ratios": [3, 1]},
    )

    bp = ax_top.boxplot(bp_data, positions=pos, widths=0.55, **_BP_KWARGS)
    for patch in bp["boxes"]:
        patch.set_facecolor("#8e44ad")
        patch.set_alpha(0.65)

    target = "Same-Class Training Nodes" if same_class else "Training Nodes"
    ax_top.set_ylabel(f"Avg. shortest path length to {target.lower()}", fontsize=11)
    ax_top.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax_top.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.4)
    ax_top.set_title(
        f"Avg. SPL to {target} vs. Degree\n{subtitle}",
        fontsize=11,
    )

    ax_bot.bar(pos, counts, color="lightgrey", alpha=0.7, width=0.6)
    ax_bot.set_ylabel("# test nodes", fontsize=9, color="grey")
    ax_bot.tick_params(axis="y", labelsize=7, colors="grey")
    ax_bot.set_xlabel("Node degree", fontsize=11)
    step = max(1, len(unique_degrees) // 30)
    ax_bot.set_xticks(pos[::step])
    ax_bot.set_xticklabels(unique_degrees[::step], rotation=55, ha="right", fontsize=8)
    ax_bot.set_xlim(pos[0] - 0.6, pos[-1] + 0.6)
    ax_bot.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.3)

    fig.tight_layout()
    suffix = "same_class" if same_class else "all_train"
    _save(fig, _subdir(save_dir, "spl_vs_degree"),
          f"{prefix}_spl_vs_degree_{suffix}.png", show)


def plot_spl_combined_vs_degree(
    test_deg, avg_spl, avg_spl_same_class, cfg,
    deg_acc_results=None, purity_by_k=None, all_deg=None,
    has_labeled_neighbor=None,
    save_dir=None, show=False,
):
    """Two combined SPL figures with side-by-side boxplots per degree group.

    Figure 1 — base:
        Side-by-side boxplots of avg SPL to any training node (blue) and to
        same-class training nodes only (red), per degree group.
        Bottom panel: node count per degree.

    Figure 2 — with overlays:
        Top panel  — all-train SPL boxplots (left axis) + accuracy median ± IQR
                     (green, right axis) + labelling ratio per degree (orange,
                     right axis).  Both overlay signals share the right axis
                     (0–1 range, percent-formatted).
        Bottom panel — Δ purity mean per degree group (purple) with a zero
                     reference line.

    Parameters
    ----------
    test_deg              : 1-D LongTensor — degrees of test nodes.
    avg_spl               : 1-D FloatTensor — avg SPL to any training node.
    avg_spl_same_class    : 1-D FloatTensor — avg SPL to same-class training node.
    cfg                   : dict
    deg_acc_results       : list[dict] — per-run output of get_accuracy_deg.
    purity_by_k           : dict {k: FloatTensor[N_all]} — node purity for all nodes.
    all_deg               : 1-D LongTensor — degrees of all graph nodes.
    has_labeled_neighbor  : 1-D BoolTensor[num_test_nodes] — True if the test node
                            has at least one training neighbour.
    save_dir              : str or None
    show                  : bool
    """
    deg     = test_deg.cpu()
    spl_all = avg_spl.cpu().numpy()
    spl_sc  = avg_spl_same_class.cpu().numpy()

    unique_degrees = sorted(deg.unique().tolist())
    n_deg  = len(unique_degrees)
    pos    = np.array(list(range(n_deg)))
    prefix   = _fname_prefix(cfg)
    n_test   = int(len(deg))
    subtitle = _subtitle(cfg, n_test, n_deg)

    # ── per-degree boxplot data ────────────────────────────────────────────────
    bp_all, bp_sc, counts = [], [], []
    for d in unique_degrees:
        mask = (deg == d).numpy()
        counts.append(int(mask.sum()))
        for spl, lst in [(spl_all, bp_all), (spl_sc, bp_sc)]:
            v     = spl[mask]
            valid = v[~np.isnan(v)]
            lst.append(valid if len(valid) > 0 else np.array([float("nan")]))

    pos_all = pos - 0.2
    pos_sc  = pos + 0.2
    box_w   = 0.35

    def _draw_boxplots(ax):
        bpa = ax.boxplot(bp_all, positions=pos_all, widths=box_w, **_BP_KWARGS)
        for patch in bpa["boxes"]:
            patch.set_facecolor(_SPL_ALL_COLOR); patch.set_alpha(0.65)
        bps = ax.boxplot(bp_sc, positions=pos_sc, widths=box_w, **_BP_KWARGS)
        for patch in bps["boxes"]:
            patch.set_facecolor(_SPL_SC_COLOR); patch.set_alpha(0.65)
        ax.set_ylabel("Avg. shortest path length", fontsize=11)
        ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.4)
        ax.set_xlim(pos[0] - 0.6, pos[-1] + 0.6)

    def _x_axis(ax):
        step = max(1, n_deg // 30)
        ax.set_xticks(pos[::step])
        ax.set_xticklabels(unique_degrees[::step], rotation=55, ha="right", fontsize=8)
        ax.set_xlabel("Node degree", fontsize=11)

    spl_handles = [
        mpatches.Patch(color=_SPL_ALL_COLOR, alpha=0.65,
                       label="Avg. SPL to any train node"),
        mpatches.Patch(color=_SPL_SC_COLOR,  alpha=0.65,
                       label="Avg. SPL to same-class train node"),
    ]

    fig_w = _fig_w(n_deg)

    # ── Figure 1: base ────────────────────────────────────────────────────────
    fig1, (ax_top1, ax_bot1) = plt.subplots(
        2, 1, figsize=(fig_w, 7), sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
    )
    _draw_boxplots(ax_top1)
    ax_top1.set_title(f"Avg. SPL to Training Nodes vs. Degree\n{subtitle}", fontsize=11)
    ax_top1.legend(handles=spl_handles, fontsize=9, loc="upper left", framealpha=0.9)
    plt.setp(ax_top1.get_xticklabels(), visible=True)

    ax_bot1.bar(pos, counts, color="lightgrey", alpha=0.7, width=0.6)
    ax_bot1.set_ylabel("# test nodes", fontsize=9, color="grey")
    ax_bot1.tick_params(axis="y", labelsize=7, colors="grey")
    ax_bot1.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.3)
    _x_axis(ax_bot1)

    fig1.tight_layout()
    _save(fig1, _subdir(save_dir, "spl_vs_degree"),
          f"{prefix}_spl_combined_vs_degree.png", show)

    # ── Figure 2: two-panel — SPL + accuracy + labelling ratio / Δ purity ──────
    has_acc    = deg_acc_results is not None
    has_purity = (purity_by_k is not None and all_deg is not None
                  and len(purity_by_k) >= 2)
    has_lr     = has_labeled_neighbor is not None
    if not (has_acc or has_purity or has_lr):
        return

    # ── pre-compute Δ purity ──────────────────────────────────────────────────
    dp_means = None
    k_lo = k_hi = None
    if has_purity:
        full_deg  = all_deg.cpu() if hasattr(all_deg, "cpu") else torch.as_tensor(all_deg)
        k_keys    = sorted(purity_by_k.keys())
        k_lo, k_hi = k_keys[0], k_keys[-1]
        delta_all = purity_by_k[k_hi].cpu().float() - purity_by_k[k_lo].cpu().float()
        dp_means  = np.array([
            float(delta_all[full_deg == d][~torch.isnan(delta_all[full_deg == d])].mean())
            if (full_deg == d).any() else float("nan")
            for d in unique_degrees
        ])

    # ── pre-compute accuracy ───────────────────────────────────────────────────
    acc_median = acc_q1 = acc_q3 = n_runs = None
    if has_acc:
        _, deg_data = _collect(deg_acc_results)
        n_runs = len(deg_acc_results)
        _med, _q1, _q3 = [], [], []
        for d in unique_degrees:
            means = [float(deg_data[d][r].mean()) for r in range(n_runs)
                     if len(deg_data[d][r]) > 0]
            if not means:
                _med.append(np.nan); _q1.append(np.nan); _q3.append(np.nan)
            else:
                _med.append(float(np.median(means)))
                _q1.append(float(np.percentile(means, 25)))
                _q3.append(float(np.percentile(means, 75)))
        acc_median = np.array(_med)
        acc_q1     = np.array(_q1)
        acc_q3     = np.array(_q3)

    # ── pre-compute labelling ratio ───────────────────────────────────────────
    ratio_vals = None
    if has_lr:
        hlr = has_labeled_neighbor.cpu()
        ratio_vals = np.array([
            float(hlr[deg == d].float().mean()) if (deg == d).any() else float("nan")
            for d in unique_degrees
        ])

    # ── layout ────────────────────────────────────────────────────────────────
    n_panels = 2 if dp_means is not None else 1
    height_ratios = [3, 1.4] if n_panels == 2 else [1]
    fig2, axes2 = plt.subplots(
        n_panels, 1, figsize=(fig_w, 7 if n_panels == 2 else 5),
        sharex=True, gridspec_kw={"height_ratios": height_ratios},
    )
    ax_top2 = axes2[0] if n_panels == 2 else axes2
    ax_bot2 = axes2[1] if n_panels == 2 else None

    ax_top2.set_title(
        f"Avg. SPL  +  Accuracy & Labelling Ratio vs. Degree\n{subtitle}", fontsize=11)

    # All-train SPL boxplots (centred, wider)
    bpa2 = ax_top2.boxplot(bp_all, positions=pos, widths=0.55, **_BP_KWARGS)
    for patch in bpa2["boxes"]:
        patch.set_facecolor(_SPL_ALL_COLOR); patch.set_alpha(0.65)
    ax_top2.set_ylabel("Avg. SPL to any training node", fontsize=11)
    ax_top2.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax_top2.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.4)
    ax_top2.set_xlim(pos[0] - 0.6, pos[-1] + 0.6)

    # Right twin axis — accuracy + labelling ratio (both 0–1 range)
    ax_ov2 = ax_top2.twinx()
    ax_ov2.set_ylim(-0.05, 1.10)
    ax_ov2.set_ylabel("Accuracy  /  Labelling ratio", fontsize=10)
    ax_ov2.tick_params(axis="y", labelsize=8)
    ax_ov2.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:.0%}"))

    ov2_handles = [mpatches.Patch(color=_SPL_ALL_COLOR, alpha=0.65,
                                  label="Avg. SPL to any train node")]

    if acc_median is not None:
        ax_ov2.plot(pos, acc_median, color=_SPL_ACC_COLOR, lw=1.8, marker="s",
                    markersize=4, zorder=5)
        ax_ov2.fill_between(pos, acc_q1, acc_q3, color=_SPL_ACC_COLOR,
                            alpha=0.15, zorder=4)
        ov2_handles.append(plt.Line2D(
            [0], [0], color=_SPL_ACC_COLOR, lw=2, marker="s", markersize=4,
            label=f"Accuracy  (median ± IQR, {n_runs} runs)"))

    _LR_COLOR = "#E65100"
    if ratio_vals is not None:
        ax_ov2.plot(pos, ratio_vals, color=_LR_COLOR, lw=1.6, marker="^",
                    markersize=4, ls="--", zorder=5)
        ov2_handles.append(plt.Line2D(
            [0], [0], color=_LR_COLOR, lw=2, marker="^", markersize=4, ls="--",
            label="Labelling ratio"))

    ax_top2.legend(handles=ov2_handles, fontsize=9, loc="upper left", framealpha=0.9)

    # Bottom panel — Δ purity
    if ax_bot2 is not None and dp_means is not None:
        ax_bot2.plot(pos, dp_means, color=_SPL_PUR_COLOR, lw=1.8, marker="o",
                     markersize=4, zorder=5)
        ax_bot2.axhline(0, color="dimgrey", lw=1.0, ls="--", zorder=2)
        ax_bot2.set_ylabel(f"Δ purity\n(k={k_hi}−k={k_lo})", fontsize=9,
                           color=_SPL_PUR_COLOR)
        ax_bot2.tick_params(axis="y", labelsize=8, colors=_SPL_PUR_COLOR)
        ax_bot2.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.3)
        _x_axis(ax_bot2)
    else:
        # single-panel: x-axis labels on the only panel
        _x_axis(ax_top2)

    fig2.tight_layout()
    _save(fig2, _subdir(save_dir, "spl_vs_degree"),
          f"{prefix}_spl_combined_with_overlays_vs_degree.png", show)


# ── accuracy + labelling ratio vs. degree ─────────────────────────────────────

def plot_acc_and_labelling_ratio_vs_degree(run_results, test_deg,
                                           has_labeled_neighbor, cfg,
                                           save_dir=None, show=False):
    """Accuracy (test nodes) and labelling ratio (test nodes) vs. degree.

    Left y-axis  — median per-degree accuracy across runs (blue).
    Right y-axis — labelling ratio per degree (orange): fraction of test nodes
                   at that degree that have at least one training neighbor.

    Parameters
    ----------
    run_results          : list[dict]  — output of get_accuracy_deg per run.
    test_deg             : 1-D LongTensor of degrees for test nodes.
    has_labeled_neighbor : 1-D BoolTensor, shape [num_test_nodes].
    cfg                  : dict
    save_dir             : str or None
    show                 : bool
    """
    # ── accuracy side ──────────────────────────────────────────────────────────
    _, deg_data   = _collect(run_results)
    acc_degrees   = sorted(deg_data.keys())

    # median accuracy across runs per degree
    acc_by_deg = {}
    for d in acc_degrees:
        run_means = [float(a.mean()) for a in deg_data[d] if len(a) > 0]
        acc_by_deg[d] = float(np.median(run_means)) if run_means else float("nan")

    # ── labelling ratio side ───────────────────────────────────────────────────
    deg = test_deg.cpu()
    has = has_labeled_neighbor.cpu()
    all_unique = sorted(deg.unique().tolist())
    ratio_by_deg = {}
    counts_by_deg = {}
    for d in all_unique:
        mask = (deg == d)
        ratio_by_deg[d]  = float(has[mask].float().mean())
        counts_by_deg[d] = int(mask.sum())

    # ── common degree axis ─────────────────────────────────────────────────────
    all_degrees = sorted(set(acc_degrees) | set(all_unique))
    pos = list(range(len(all_degrees)))

    acc_vals   = [acc_by_deg.get(d,   float("nan")) for d in all_degrees]
    ratio_vals = [ratio_by_deg.get(d, float("nan")) for d in all_degrees]
    counts     = [counts_by_deg.get(d, 0)           for d in all_degrees]

    prefix   = _fname_prefix(cfg)
    n_runs   = len(run_results)
    n_test   = sum(len(deg_data[d][0]) for d in acc_degrees)
    subtitle = _subtitle(cfg, n_test, len(all_degrees))

    fig, ax_acc = plt.subplots(figsize=(_fig_w(len(all_degrees)), 5))

    # Accuracy line
    ax_acc.plot(pos, acc_vals, color="#3498db", linewidth=1.8,
                marker="o", markersize=4, zorder=3, label="Accuracy (median)")
    ax_acc.set_ylabel("Accuracy", fontsize=11, color="#3498db")
    ax_acc.tick_params(axis="y", colors="#3498db", labelsize=8)
    ax_acc.set_ylim(-0.05, 1.10)
    ax_acc.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))
    ax_acc.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.3)

    # Labelling ratio on twin axis
    ax_lr = ax_acc.twinx()
    ax_lr.plot(pos, ratio_vals, color="#e67e22", linewidth=1.8,
               marker="s", markersize=4, zorder=3, label="Labelling ratio")
    ax_lr.set_ylabel("Labelling ratio", fontsize=11, color="#e67e22")
    ax_lr.tick_params(axis="y", colors="#e67e22", labelsize=8)
    ax_lr.set_ylim(-0.05, 1.10)
    ax_lr.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))

    # Combined legend
    handles = [
        plt.Line2D([0], [0], color="#3498db", lw=2, marker="o", markersize=4,
                   label=f"Accuracy  ({n_runs} run{'s' if n_runs > 1 else ''}, median)"),
        plt.Line2D([0], [0], color="#e67e22", lw=2, marker="s", markersize=4,
                   label="Labelling ratio  (test nodes)"),
    ]
    ax_acc.legend(handles=handles, loc="upper left", fontsize=8, framealpha=0.85)

    ax_acc.set_title(
        f"Accuracy & Labelling Ratio vs. Node Degree\n{subtitle}", fontsize=11,
    )
    _degree_axis(ax_acc, pos, all_degrees)

    fig.tight_layout()
    _save(fig, _subdir(save_dir, "labelling_ratio"),
          f"{prefix}_acc_and_labelling_ratio_vs_degree.png", show)


# ── labelling ratio vs. degree ────────────────────────────────────────────────

def plot_labelling_ratio_vs_degree(all_deg, has_labeled_neighbor, cfg,
                                   save_dir=None, show=False):
    """Fraction of nodes with at least one labeled neighbor, grouped by degree.

    For each degree group d:
        labelling_ratio(d) = |{v : deg(v)=d, ∃ u∈N_1(v) s.t. u∈train}|
                             / |{v : deg(v)=d}|

    Parameters
    ----------
    all_deg              : 1-D LongTensor of degrees for all nodes.
    has_labeled_neighbor : 1-D BoolTensor, shape [num_nodes].
    cfg                  : dict
    save_dir             : str or None
    show                 : bool
    """
    deg  = all_deg.cpu()
    has  = has_labeled_neighbor.cpu()

    unique_degrees = sorted(deg.unique().tolist())
    pos    = list(range(len(unique_degrees)))
    prefix = _fname_prefix(cfg)
    n_all  = int(len(deg))
    subtitle = _subtitle(cfg, n_all, len(unique_degrees))

    counts = []
    ratios = []
    for d in unique_degrees:
        mask = (deg == d)
        counts.append(int(mask.sum()))
        ratios.append(float(has[mask].float().mean()))

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(_fig_w(len(unique_degrees)), 7),
        sharex=True, gridspec_kw={"height_ratios": [3, 1]},
    )

    ax_top.plot(pos, ratios, marker="o", linewidth=1.8, markersize=5,
                color="#2980b9", zorder=3)
    ax_top.set_ylabel("Labelling ratio", fontsize=11)
    ax_top.set_ylim(-0.05, 1.10)
    ax_top.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))
    ax_top.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.4)
    ax_top.set_title(
        f"Labelling Ratio vs. Node Degree\n{subtitle}", fontsize=11,
    )

    ax_bot.bar(pos, counts, color="lightgrey", alpha=0.7, width=0.6)
    ax_bot.set_ylabel("# nodes", fontsize=9, color="grey")
    ax_bot.tick_params(axis="y", labelsize=7, colors="grey")
    ax_bot.set_xlabel("Node degree", fontsize=11)
    step = max(1, len(unique_degrees) // 30)
    ax_bot.set_xticks(pos[::step])
    ax_bot.set_xticklabels(unique_degrees[::step], rotation=55, ha="right", fontsize=8)
    ax_bot.set_xlim(pos[0] - 0.6, pos[-1] + 0.6)
    ax_bot.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.3)

    fig.tight_layout()
    _save(fig, _subdir(save_dir, "labelling_ratio"),
          f"{prefix}_labelling_ratio_vs_degree.png", show)


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


# ── delta purity across k transitions per degree ───────────────────────────────

def plot_purity_delta_by_degree(test_deg, purity_by_k, cfg,
                                save_dir=None, show=False):
    """Overall purity delta (purity[k_max] - purity[k_min]) per degree group.

    Top panel  — single line: y = mean(purity[k_max]) - mean(purity[k_min])
                 for nodes of that degree.  A zero line marks no change.
    Bottom panel — node count per degree (bar chart).

    Parameters
    ----------
    test_deg    : 1-D LongTensor of 1-hop degrees for test nodes.
    purity_by_k : dict { k (int) -> 1-D FloatTensor of purity values }
    cfg         : dict
    save_dir    : str or None
    show        : bool
    """
    deg      = test_deg.cpu()
    k_values = sorted(purity_by_k.keys())
    if len(k_values) < 2:
        return

    k_min, k_max = k_values[0], k_values[-1]
    unique_degrees = sorted(deg.unique().tolist())
    pos    = list(range(len(unique_degrees)))
    prefix = _fname_prefix(cfg)
    n_test = int(len(deg))
    subtitle = _subtitle(cfg, n_test, len(unique_degrees))

    counts = []
    deltas = []
    for d in unique_degrees:
        mask = (deg == d).numpy()
        counts.append(int(mask.sum()))

        def _mean(k):
            vals  = purity_by_k[k].cpu().numpy()[mask]
            valid = vals[~np.isnan(vals)]
            return float(valid.mean()) if len(valid) > 0 else float("nan")

        deltas.append(_mean(k_max) - _mean(k_min))

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(_fig_w(len(unique_degrees)), 7),
        sharex=True, gridspec_kw={"height_ratios": [3, 1]},
    )

    ax_top.plot(pos, deltas, marker="o", linewidth=1.8, markersize=5,
                color="#e67e22", zorder=3)
    ax_top.axhline(0, color="dimgrey", lw=1.0, ls="--", zorder=2, label="No change")
    ax_top.set_ylabel(f"Δ mean purity  (k={k_max} − k={k_min})", fontsize=11)
    ax_top.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:+.2f}"))
    ax_top.legend(loc="upper left", fontsize=8, framealpha=0.85)
    ax_top.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.4)
    ax_top.set_title(
        f"Overall Purity Change (k={k_min}→{k_max}) by Degree\n{subtitle}", fontsize=11,
    )

    ax_bot.bar(pos, counts, color="lightgrey", alpha=0.7, width=0.6)
    ax_bot.set_ylabel("# test nodes", fontsize=9, color="grey")
    ax_bot.tick_params(axis="y", labelsize=7, colors="grey")
    ax_bot.set_xlabel("Node degree", fontsize=11)
    step = max(1, len(unique_degrees) // 30)
    ax_bot.set_xticks(pos[::step])
    ax_bot.set_xticklabels(unique_degrees[::step], rotation=55, ha="right", fontsize=8)
    ax_bot.set_xlim(pos[0] - 0.6, pos[-1] + 0.6)
    ax_bot.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.3)

    fig.tight_layout()
    k_range = f"k{k_min}-{k_max}"
    _save(fig, _subdir(save_dir, "purity_vs_degree"),
          f"{prefix}_purity_delta_by_degree_{k_range}.png", show)


# ── influence analysis: same-class vs diff-class training nodes ────────────────

def plot_influence_analysis(results, cfg, save_dir=None, show=False):
    """Grouped bar chart comparing same-class vs different-class training node
    influence for high-degree misclassified test nodes.

    Nodes are sorted by degree on the x-axis.  For each node, two bars show
    the total influence attributable to same-class and different-class training
    nodes within the model's receptive field.  A secondary line shows the
    fraction of influence coming from different-class nodes.

    Parameters
    ----------
    results  : list of dicts from influence.compute_influence_analysis.
    cfg      : dict
    save_dir : str or None
    show     : bool
    """
    if not results:
        return

    results = sorted(results, key=lambda r: r["degree"])

    labels    = [f"n{r['node_idx']}\ndeg={r['degree']}" for r in results]
    # Use scores normalised to total training-node influence (sum to 1 per node)
    same_inf  = np.array([r["same_class_influence_norm"] for r in results])
    diff_inf  = np.array([r["diff_class_influence_norm"] for r in results])
    total     = same_inf + diff_inf   # should be 1.0 wherever both exist
    diff_frac = np.divide(diff_inf, total, out=np.full_like(total, float("nan")), where=total > 0)

    x     = np.arange(len(results))
    width = 0.35
    prefix = _fname_prefix(cfg)

    fig, ax = plt.subplots(figsize=(max(10, len(results) * 0.9), 5))

    bars_same = ax.bar(x - width / 2, same_inf, width,
                       label="Same-class train nodes", color="#1f78b4", alpha=0.9)
    bars_diff = ax.bar(x + width / 2, diff_inf, width,
                       label="Diff-class train nodes", color="#ff7f00", alpha=0.9)

    # Value labels above every bar
    for bar in bars_same:
        h = bar.get_height()
        if np.isfinite(h) and h > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, h,
                    f"{h:.3f}", ha="center", va="bottom", fontsize=6, color="#1f78b4")
    for bar in bars_diff:
        h = bar.get_height()
        if np.isfinite(h) and h > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, h,
                    f"{h:.3f}", ha="center", va="bottom", fontsize=6, color="#b35806")

    # Fraction of diff-class influence as a line on twin axis
    ax2 = ax.twinx()
    ax2.plot(x, diff_frac, color="#b35806", lw=1.5, ls="--",
             marker="o", markersize=4, zorder=4,
             label="Diff-class fraction")
    ax2.set_ylabel("Diff-class influence fraction", fontsize=10, color="#b35806")
    ax2.tick_params(axis="y", colors="#b35806", labelsize=8)
    ax2.set_ylim(-0.05, 1.10)
    ax2.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))

    ax.set_ylabel("Normalised influence score\n(fraction of total training-node influence)", fontsize=10)
    ax.set_xlabel("Test node  (sorted by degree)", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.4)

    handles1, _ = ax.get_legend_handles_labels()
    handles2, _ = ax2.get_legend_handles_labels()
    ax.legend(handles=handles1 + handles2, loc="upper left",
              fontsize=8, framealpha=0.85)

    ax.set_title(
        "Influence of Training Nodes on High-Degree Misclassified Test Nodes\n"
        f"{_fname_prefix(cfg)}  —  same-class vs. diff-class within receptive field",
        fontsize=11,
    )

    fig.tight_layout()
    _save(fig, _subdir(save_dir, "influence"),
          f"{prefix}_influence_analysis.png", show)


def plot_influence_per_neighbor(results, cfg, save_dir=None, show=False):
    """One figure per selected node: influence scores of training nodes in
    the k-hop receptive field, colored by class relationship to the target.

    Colors:
        green — same-class training node
        red   — different-class training node

    Training nodes are sorted left-to-right by descending influence score.

    Parameters
    ----------
    results  : list of dicts from influence.compute_influence_analysis,
               each containing a "neighbors" key (training nodes only).
    cfg      : dict
    save_dir : str or None
    show     : bool
    """
    if not results:
        return

    TYPE_COLOR = {
        "same_train": "#1f78b4",   # blue
        "diff_train": "#ff7f00",   # orange
        "non_train":  "#bdbdbd",   # light grey
    }
    TYPE_LABEL = {
        "same_train": "Same-class train",
        "diff_train": "Diff-class train",
        "non_train":  "Non-training",
    }

    prefix = _fname_prefix(cfg)
    subdir = _subdir(save_dir, "influence")

    for r in results:
        # Only show training nodes in the per-neighbor plot
        neighbors = [nb for nb in r.get("neighbors", [])
                     if nb["type"] in ("same_train", "diff_train")]
        if not neighbors:
            continue

        node_x   = r["node_idx"]
        degree   = r["degree"]
        true_lbl = r["true_label"]
        pred_lbl = r["pred_label"]

        infl   = np.array([nb["influence_norm"] for nb in neighbors])
        types  = [nb["type"] for nb in neighbors]
        colors = [TYPE_COLOR[t] for t in types]
        x      = np.arange(len(neighbors))

        fig, ax = plt.subplots(figsize=(max(8, len(neighbors) * 0.4 + 2), 4))
        ax.bar(x, infl, color=colors, alpha=0.88, width=0.7, zorder=3)

        # Legend — only show types that are present
        seen_types = set(types)
        legend_patches = [
            plt.Rectangle((0, 0), 1, 1, color=TYPE_COLOR[t], alpha=0.88,
                          label=TYPE_LABEL[t])
            for t in ("same_train", "diff_train", "non_train") if t in seen_types
        ]
        ax.legend(handles=legend_patches, fontsize=9, framealpha=0.85)

        correct = "correct" if true_lbl == pred_lbl else "misclassified"
        n_same = r["n_same_train"]
        n_diff = r["n_diff_train"]
        n_non  = sum(1 for t in types if t == "non_train")
        ax.set_title(
            f"k-hop neighbor influence — node {node_x}  [{correct}]\n"
            f"degree={degree}  true={true_lbl}  pred={pred_lbl}  "
            f"same-class train={n_same}  diff-class train={n_diff}  non-train={n_non}  "
            f"({cfg['dataset']['name']} / {cfg['model']['name']})",
            fontsize=9,
        )
        ax.set_xlabel("k-hop neighbor node index  (deg = 1-hop degree)", fontsize=10)
        ax.set_ylabel("Normalised influence score\n(fraction of total training-node influence)", fontsize=9)
        ax.set_xticks(x)
        ax.set_xticklabels(
            [f"{nb['node_idx']}\n(deg {nb['degree']})" for nb in neighbors],
            rotation=0, fontsize=6, ha="center",
        )
        ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.4)
        ax.set_xlim(-0.7, len(neighbors) - 0.3)

        fig.tight_layout()
        fname = f"{prefix}_influence_neighbors_node{node_x}_deg{degree}.png"
        _save(fig, subdir, fname, show)


_SIM_COLORS = {
    "same_train": "#1f78b4",   # blue
    "diff_train": "#ff7f00",   # orange
    "non_train":  "#bdbdbd",   # light grey
}
_SIM_LABELS = {
    "same_train": "Same-class train",
    "diff_train": "Diff-class train",
    "non_train":  "Non-training",
}


def plot_node_similarity_analysis(results, cfg, save_dir=None, show=False):
    """Per-node bar charts: cosine similarity to each 1-hop neighbor at
    every representation level (raw features → h^1 → … → h^k_hops).

    One figure per node, one panel per layer.  Bars are colored by neighbor
    type (same_train / diff_train / non_train) and sorted by raw-feature
    similarity (descending) — the sort order is fixed across all panels so
    changes are easy to track visually.

    Parameters
    ----------
    results  : list of dicts from compute_node_similarity_analysis.
    cfg      : dict
    save_dir : str or None
    show     : bool
    """
    if not results:
        log.warning("plot_node_similarity_analysis: no results to plot")
        return

    prefix = _fname_prefix(cfg)
    subdir = _subdir(save_dir, "node_similarity")

    layer_names = ["Raw (x)"]      # first panel label

    for result in results:
        node_x    = result["node_idx"]
        degree    = result["degree"]
        true_lbl  = result["true_label"]
        pred_lbl  = result["pred_label"]
        k_hops    = result["k_hops"]
        neighbors = result["neighbors"]

        if not neighbors:
            continue

        # Build layer names lazily from k_hops
        all_layer_names = ["Raw (x)"] + [f"h^{l}" for l in range(1, k_hops + 1)]
        n_layers = len(all_layer_names)   # k_hops + 1
        n_nbs    = len(neighbors)
        x_pos    = np.arange(n_nbs)

        colors    = [_SIM_COLORS[nb["type"]] for nb in neighbors]
        tick_lbls = [f"{nb['node_idx']}\n(deg {nb['degree']})" for nb in neighbors]
        correct   = true_lbl == pred_lbl

        fig, axes = plt.subplots(
            n_layers, 1,
            figsize=(_fig_w(n_nbs), 3.2 * n_layers),
            sharex=True,
        )
        if n_layers == 1:
            axes = [axes]

        for li, (ax, lname) in enumerate(zip(axes, all_layer_names)):
            sims = [nb["similarities"][li] for nb in neighbors]
            bars = ax.bar(x_pos, sims, color=colors, width=0.65,
                          edgecolor="white", linewidth=0.4, zorder=2)
            ax.axhline(0, color="grey", linewidth=0.6, linestyle="--", zorder=1)
            ax.set_ylim(-1.05, 1.05)
            ax.set_ylabel(f"Cosine sim\n{lname}", fontsize=9)
            ax.grid(axis="y", linestyle="--", linewidth=0.4, alpha=0.35, zorder=0)
            ax.set_title(lname, fontsize=9, loc="right", color="#555555")

            # Annotate bar tops with sim value
            for bar, s in zip(bars, sims):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    max(s, 0) + 0.03,
                    f"{s:.2f}",
                    ha="center", va="bottom", fontsize=5.5, color="#333333",
                )

        # x-ticks on last panel only
        axes[-1].set_xticks(x_pos)
        axes[-1].set_xticklabels(tick_lbls, fontsize=6, ha="center")
        axes[-1].set_xlabel(
            "1-hop neighbor  (sorted by raw-feature similarity, descending)",
            fontsize=9,
        )

        # Legend
        seen_types = {nb["type"] for nb in neighbors}
        legend_handles = [
            mpatches.Patch(facecolor=_SIM_COLORS[t], label=_SIM_LABELS[t])
            for t in ("same_train", "diff_train", "non_train") if t in seen_types
        ]
        axes[0].legend(handles=legend_handles, fontsize=8,
                       framealpha=0.85, loc="upper right")

        verdict = "✓ correct" if correct else "✗ wrong"
        fig.suptitle(
            f"Node {node_x}  |  degree={degree}  |  "
            f"true={true_lbl}  pred={pred_lbl}  {verdict}\n"
            f"Cosine similarity to 1-hop neighbors across representation layers",
            fontsize=10, y=1.01,
        )

        fig.tight_layout()
        fname = f"{prefix}_node_similarity_node{node_x}_deg{degree}.png"
        _save(fig, subdir, fname, show)


# Influence disparity plot colours
_DISP_POS_COLOR = "#1976D2"   # blue  — same-class dominant
_DISP_NEG_COLOR = "#E53935"   # red   — diff-class dominant
_DISP_BOX_COLOR = "#546E7A"   # blue-grey — neutral boxplot fill


def plot_influence_disparity_vs_degree(
    disparity_results, cfg, save_dir=None, show=False
):
    """Boxplots of per-node influence disparity vs test-node degree, with
    accuracy of the analysed nodes overlaid on a twin right axis.

    Influence disparity  =  same_class_influence_norm − diff_class_influence_norm
    Range [-1, 1]:  +1 → all training influence from same-class;
                    -1 → all training influence from diff-class.
    A dashed horizontal line marks y=0 (equal balance).

    Only test nodes that have at least one training node in their k-hop
    receptive field are included (computed by compute_influence_disparity_all).

    Parameters
    ----------
    disparity_results : list of dicts from compute_influence_disparity_all.
    cfg      : dict
    save_dir : str or None
    show     : bool
    """
    if not disparity_results:
        log.warning("plot_influence_disparity_vs_degree: no data to plot")
        return

    # ── group by degree ───────────────────────────────────────────────────────
    from collections import defaultdict
    disp_by_deg = defaultdict(list)
    acc_by_deg  = defaultdict(list)
    for r in disparity_results:
        d = r["degree"]
        disp_by_deg[d].append(r["disparity"])
        acc_by_deg[d].append(1.0 if r["correct"] else 0.0)

    degrees  = sorted(disp_by_deg)
    pos      = list(range(len(degrees)))
    counts   = [len(disp_by_deg[d]) for d in degrees]
    bp_data  = [disp_by_deg[d]      for d in degrees]
    mean_acc = [float(np.mean(acc_by_deg[d])) for d in degrees]

    prefix   = _fname_prefix(cfg)
    n_nodes  = len(disparity_results)
    subtitle = (
        f"{cfg['dataset']['name']} · {cfg['model']['name']} · "
        f"{cfg.get('split','random')} · "
        f"{'CC' if cfg['dataset'].get('use_cc') else 'noCC'}"
        f"   |   {n_nodes} nodes analysed  ·  {len(degrees)} unique degrees"
        f"   (last-run model)"
    )

    fig, ax = plt.subplots(figsize=(_fig_w(len(degrees)), 4.5))

    # ── disparity boxplots ────────────────────────────────────────────────────
    bp = ax.boxplot(
        bp_data, positions=pos, widths=0.55,
        **{**_BP_KWARGS, "boxprops": dict(facecolor=_DISP_BOX_COLOR, alpha=0.55)},
    )

    # Colour each box: blue if median > 0, red if median < 0
    for patch, data in zip(bp["boxes"], bp_data):
        med = float(np.median(data))
        patch.set_facecolor(_DISP_POS_COLOR if med >= 0 else _DISP_NEG_COLOR)
        patch.set_alpha(0.55)

    # Zero reference
    ax.axhline(0, color="grey", linestyle="--", linewidth=0.9, zorder=1,
               label="Neutral (disparity = 0)")

    ax.set_ylabel("Influence disparity  (same − diff)", fontsize=10)
    ax.set_ylim(-1.1, 1.1)
    ax.set_yticks([-1.0, -0.5, 0.0, 0.5, 1.0])
    ax.grid(axis="y", linestyle="--", linewidth=0.4, alpha=0.35, zorder=0)

    # ── accuracy overlay (right axis) ─────────────────────────────────────────
    ax2 = ax.twinx()
    ax2.plot(pos, mean_acc, color=_ACC_COLOR, linewidth=1.8, marker="o",
             markersize=4, zorder=4, label="Accuracy (analysed nodes)")
    ax2.set_ylabel("Accuracy", fontsize=10, color=_ACC_COLOR)
    ax2.yaxis.set_major_formatter(ticker.PercentFormatter(1.0))
    ax2.tick_params(axis="y", labelcolor=_ACC_COLOR)
    ax2.set_ylim(0, 1.05)

    # ── count bars (light grey, behind everything) ────────────────────────────
    ax3 = ax.twinx()
    ax3.spines["right"].set_position(("axes", 1.12))
    ax3.bar(pos, counts, color="lightgrey", alpha=0.3, width=0.55, zorder=0)
    ax3.set_ylabel("# nodes", fontsize=7, color="grey")
    ax3.tick_params(axis="y", labelsize=6, colors="grey", length=2)
    ax3.spines["right"].set_color("lightgrey")
    ax3.set_ylim(0, max(counts) * 4)

    # ── legend ────────────────────────────────────────────────────────────────
    handles = [
        mpatches.Patch(facecolor=_DISP_POS_COLOR, alpha=0.7,
                       label="Median ≥ 0  (same-class dominant)"),
        mpatches.Patch(facecolor=_DISP_NEG_COLOR, alpha=0.7,
                       label="Median < 0  (diff-class dominant)"),
        plt.Line2D([0], [0], color="grey", linestyle="--", linewidth=0.9,
                   label="Neutral (0)"),
        plt.Line2D([0], [0], color=_ACC_COLOR, linewidth=1.8, marker="o",
                   markersize=4, label="Accuracy"),
    ]
    ax.legend(handles=handles, fontsize=8, framealpha=0.85, loc="upper left")

    # ── x-axis + title ────────────────────────────────────────────────────────
    _degree_axis(ax, pos, degrees)
    ax.set_title(
        f"Influence disparity vs node degree\n{subtitle}", fontsize=9
    )

    fig.tight_layout()
    _save(fig, _subdir(save_dir, "influence"),
          f"{prefix}_influence_disparity_vs_degree.png", show)


_FSIM_RAW_COLOR   = "#1976D2"   # blue  — raw input features
_FSIM_H1_COLOR    = "#E53935"   # red   — after 1-hop message passing
_FSIM_DELTA_COLOR = "#7B1FA2"   # purple — delta feature sim (h^k − raw)
_FSIM_DPUR_COLOR  = "#00838F"   # teal   — delta label purity overlay


def plot_feature_similarity_delta_vs_degree(
    sim_results, cfg, k_hops: int = 1,
    deg_acc_results=None,
    purity_by_k=None,
    test_deg=None,
    save_dir=None, show=False,
):
    """Two-panel figure: cosine similarity to same-class training 1-hop
    neighbors, in raw feature space vs after k_hops GCNConv steps.

    Top panel   — mean ± std of sim_raw (blue) and sim_hk (red) per degree
                  group; count bars on secondary axis.
    Bottom panel — mean ± std of Δ feature sim (purple); optionally overlaid
                  with Δ label purity (teal, left axis) and accuracy of the
                  last run (green, twin right axis).  A dashed zero line marks
                  no change.

    Parameters
    ----------
    sim_results     : list of dicts from get_feature_similarity_delta.
    cfg             : dict
    k_hops          : number of GCNConv layers (= num_layers - 1).
    deg_acc_results : list of per-run get_accuracy_deg dicts; last entry used
                      for accuracy overlay.  Pass None to omit.
    purity_by_k     : {k (int): FloatTensor} for all graph nodes; needs at
                      least keys 1 and 2 for delta purity overlay.  Pass None
                      to omit.
    test_deg        : 1-D LongTensor of test-node degrees; required when
                      purity_by_k is provided.
    save_dir        : str or None
    show            : bool
    """
    if not sim_results:
        log.warning("plot_feature_similarity_delta_vs_degree: no data to plot")
        return

    from collections import defaultdict
    raw_by_deg   = defaultdict(list)
    hk_by_deg    = defaultdict(list)
    delta_by_deg = defaultdict(list)

    for r in sim_results:
        d = r["degree"]
        raw_by_deg[d].append(r["sim_raw"])
        hk_by_deg[d].append(r["sim_hk"])
        delta_by_deg[d].append(r["delta"])

    degrees  = sorted(raw_by_deg)
    pos      = list(range(len(degrees)))
    pa       = np.array(pos)

    mean_raw   = np.array([np.mean(raw_by_deg[d])   for d in degrees])
    std_raw    = np.array([np.std(raw_by_deg[d])    for d in degrees])
    mean_hk    = np.array([np.mean(hk_by_deg[d])    for d in degrees])
    std_hk     = np.array([np.std(hk_by_deg[d])     for d in degrees])
    mean_delta = np.array([np.mean(delta_by_deg[d]) for d in degrees])
    std_delta  = np.array([np.std(delta_by_deg[d])  for d in degrees])
    counts     = [len(raw_by_deg[d]) for d in degrees]

    prefix   = _fname_prefix(cfg)
    n_nodes  = len(sim_results)
    subtitle = (
        f"{cfg['dataset']['name']} · {cfg['model']['name']} · "
        f"{cfg.get('split','random')} · "
        f"{'CC' if cfg['dataset'].get('use_cc') else 'noCC'}"
        f"   |   {n_nodes} nodes analysed  ·  {len(degrees)} unique degrees"
        f"   (last-run model)"
    )

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(_fig_w(len(degrees)), 7),
        sharex=True, gridspec_kw={"height_ratios": [3, 2]},
    )

    # ── top panel: raw sim + hk sim ───────────────────────────────────────────
    ax_top.plot(pa, mean_raw, color=_FSIM_RAW_COLOR, linewidth=1.8, marker="o",
                markersize=4, label="Raw features (x)", zorder=3)
    ax_top.fill_between(pa, mean_raw - std_raw, mean_raw + std_raw,
                        color=_FSIM_RAW_COLOR, alpha=0.15, zorder=2)

    ax_top.plot(pa, mean_hk, color=_FSIM_H1_COLOR, linewidth=1.8, marker="s",
                markersize=4, label=f"After {k_hops}-hop (h^{k_hops})", zorder=3)
    ax_top.fill_between(pa, mean_hk - std_hk, mean_hk + std_hk,
                        color=_FSIM_H1_COLOR, alpha=0.15, zorder=2)

    ax_top.set_ylabel("Mean cosine similarity\nto same-class train neighbors", fontsize=10)
    ax_top.set_ylim(-0.05, 1.05)
    ax_top.grid(axis="y", linestyle="--", linewidth=0.4, alpha=0.35)
    ax_top.legend(fontsize=9, framealpha=0.85, loc="upper left")
    ax_top.set_title(
        f"Feature similarity to same-class training neighbors vs node degree\n{subtitle}",
        fontsize=9,
    )

    ax_cnt = ax_top.twinx()
    ax_cnt.bar(pa, counts, color="lightgrey", alpha=0.3, width=0.55, zorder=0)
    ax_cnt.set_ylabel("# nodes", fontsize=7, color="grey")
    ax_cnt.tick_params(axis="y", labelsize=6, colors="grey", length=2)
    ax_cnt.spines["right"].set_color("lightgrey")
    ax_cnt.set_ylim(0, max(counts) * 4)

    # ── bottom panel: Δ feature sim ───────────────────────────────────────────
    ax_bot.plot(pa, mean_delta, color=_FSIM_DELTA_COLOR, linewidth=1.8,
                marker="D", markersize=4, zorder=3,
                label=f"Δ feature sim  (h^{k_hops} − raw)")
    ax_bot.fill_between(pa, mean_delta - std_delta, mean_delta + std_delta,
                        color=_FSIM_DELTA_COLOR, alpha=0.15, zorder=2)

    # ── Δ purity overlay (left axis, teal) ────────────────────────────────────
    if purity_by_k and test_deg is not None and len(purity_by_k) >= 2:
        k_vals = sorted(purity_by_k.keys())
        k_lo, k_hi = k_vals[0], k_vals[-1]
        deg_cpu = test_deg.cpu()
        dp_mean = []
        for d in degrees:
            mask = (deg_cpu == d).numpy()
            def _pur_mean(k):
                v = purity_by_k[k].cpu().numpy()[mask]
                v = v[~np.isnan(v)]
                return float(v.mean()) if len(v) else float("nan")
            dp_mean.append(_pur_mean(k_hi) - _pur_mean(k_lo))
        dp_mean = np.array(dp_mean)
        ax_bot.plot(pa, dp_mean, color=_FSIM_DPUR_COLOR, linewidth=1.8,
                    marker="^", markersize=4, zorder=3, linestyle="--",
                    label=f"Δ label purity  (k={k_hi}−k={k_lo})")

    ax_bot.axhline(0, color="grey", linestyle=":", linewidth=0.9, zorder=1)
    ax_bot.set_ylabel(f"Delta  (h^{k_hops} − raw  /  purity k_hi−k_lo)", fontsize=9)
    ax_bot.grid(axis="y", linestyle="--", linewidth=0.4, alpha=0.35)

    # ── accuracy overlay (right axis, green) ──────────────────────────────────
    if deg_acc_results:
        last_run = deg_acc_results[-1]
        acc_mean = []
        for d in degrees:
            if d in last_run:
                r = last_run[d]
                acc_mean.append((r["preds"] == r["labels"]).float().mean().item())
            else:
                acc_mean.append(float("nan"))
        ax_acc = ax_bot.twinx()
        ax_acc.plot(pa, acc_mean, color=_ACC_COLOR, linewidth=1.8, marker="o",
                    markersize=4, zorder=4, label="Accuracy (last run)")
        ax_acc.set_ylabel("Accuracy", fontsize=10, color=_ACC_COLOR)
        ax_acc.yaxis.set_major_formatter(ticker.PercentFormatter(1.0))
        ax_acc.tick_params(axis="y", labelcolor=_ACC_COLOR)
        ax_acc.set_ylim(0, 1.05)
        ax_acc.legend(fontsize=8, framealpha=0.85, loc="upper right")

    ax_bot.legend(fontsize=8, framealpha=0.85, loc="upper left")
    _degree_axis(ax_bot, pos, degrees)

    fig.tight_layout()
    _save(fig, _subdir(save_dir, "feature_similarity"),
          f"{prefix}_feature_similarity_delta_vs_degree.png", show)


def plot_train_neighbor_degree_stats(
    stats, test_deg, pred, data, cfg, k=2, save_dir=None, show=False
):
    """Two-panel figure: degree of same-class vs diff-class training neighbors.

    Investigates whether the relative degree of same-class vs diff-class
    training nodes in a test node's k-hop neighborhood correlates with
    classification outcome.

    Panel 1 (top): For each test-node degree group, side-by-side boxplots of
    the mean degree of same-class (blue) and diff-class (orange) training nodes
    within the k-hop receptive field.

    Panel 2 (bottom): For each test-node degree group, the "degree advantage"
    (mean_deg_same − mean_deg_diff) for correctly classified (green) vs
    misclassified (red) test nodes. A positive advantage means same-class
    training nodes have higher average degree; negative means diff-class do.
    In GCN, higher-degree nodes contribute less per edge (1/sqrt(deg) weight),
    so a positive advantage implies the same-class signal is more diluted.

    Parameters
    ----------
    stats    : dict from get_training_neighbor_degree_stats
    test_deg : LongTensor [num_test_nodes]
    pred     : LongTensor [num_nodes] — predicted labels for all nodes
    data     : PyG Data
    cfg      : dict
    k        : receptive field radius used to compute stats
    save_dir : str or None
    show     : bool
    """
    test_mask = data.test_mask.cpu()
    true_y = data.y.cpu()[test_mask]
    pred_y = pred.cpu()[test_mask]
    correct = (pred_y == true_y).numpy()

    test_deg_np = test_deg.cpu().numpy().astype(int)
    all_degrees = sorted(set(test_deg_np.tolist()))
    pos = list(range(len(all_degrees)))

    same_md = stats["same_mean_deg"]
    diff_md = stats["diff_mean_deg"]

    same_by_deg      = {d: [] for d in all_degrees}
    diff_by_deg      = {d: [] for d in all_degrees}
    adv_corr_by_deg  = {d: [] for d in all_degrees}
    adv_wrong_by_deg = {d: [] for d in all_degrees}

    for i, d in enumerate(test_deg_np.tolist()):
        s, f = same_md[i], diff_md[i]
        if not np.isnan(s):
            same_by_deg[d].append(s)
        if not np.isnan(f):
            diff_by_deg[d].append(f)
        if not np.isnan(s) and not np.isnan(f):
            adv = s - f
            if correct[i]:
                adv_corr_by_deg[d].append(adv)
            else:
                adv_wrong_by_deg[d].append(adv)

    def _clean(arr):
        return arr if arr else [float("nan")]

    prefix   = _fname_prefix(cfg)
    n_test   = len(test_deg_np)
    subtitle = _subtitle(cfg, n_test, len(all_degrees))

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1,
        figsize=(_fig_w(len(all_degrees)), 9),
        gridspec_kw={"height_ratios": [1.2, 1]},
    )
    fig.subplots_adjust(hspace=0.38)

    # ── Panel 1: mean degree of same vs diff training neighbors ─────────────
    offset = 0.22
    same_data = [_clean(same_by_deg[d]) for d in all_degrees]
    diff_data = [_clean(diff_by_deg[d]) for d in all_degrees]
    pos_same  = [p - offset for p in pos]
    pos_diff  = [p + offset for p in pos]

    bp_s = ax_top.boxplot(same_data, positions=pos_same, widths=0.35, **_BP_KWARGS)
    for patch in bp_s["boxes"]:
        patch.set_facecolor("#1f78b4")
        patch.set_alpha(0.80)

    bp_d = ax_top.boxplot(diff_data, positions=pos_diff, widths=0.35, **_BP_KWARGS)
    for patch in bp_d["boxes"]:
        patch.set_facecolor("#ff7f00")
        patch.set_alpha(0.80)

    ax_top.legend(
        handles=[
            plt.Rectangle((0, 0), 1, 1, color="#1f78b4", alpha=0.80,
                           label="Same-class train neighbors"),
            plt.Rectangle((0, 0), 1, 1, color="#ff7f00", alpha=0.80,
                           label="Diff-class train neighbors"),
        ],
        fontsize=9, framealpha=0.85, loc="upper left",
        title=f"{k}-hop neighborhood", title_fontsize=8,
    )
    ax_top.set_ylabel("Mean degree of k-hop training neighbors", fontsize=10)
    ax_top.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.4)
    ax_top.set_title(
        f"Mean degree of same-class vs diff-class training nodes in {k}-hop neighborhood\n"
        f"{subtitle}",
        fontsize=10,
    )
    _degree_axis(ax_top, pos, all_degrees)

    # ── Panel 2: degree advantage split by correct / misclassified ─────────
    corr_data  = [_clean(adv_corr_by_deg[d])  for d in all_degrees]
    wrong_data = [_clean(adv_wrong_by_deg[d]) for d in all_degrees]
    pos_corr  = [p - offset for p in pos]
    pos_wrong = [p + offset for p in pos]

    bp_c = ax_bot.boxplot(corr_data, positions=pos_corr, widths=0.35, **_BP_KWARGS)
    for patch in bp_c["boxes"]:
        patch.set_facecolor("#00b4d8")   # vivid cyan-blue
        patch.set_alpha(0.88)

    bp_w = ax_bot.boxplot(wrong_data, positions=pos_wrong, widths=0.35, **_BP_KWARGS)
    for patch in bp_w["boxes"]:
        patch.set_facecolor("#e63946")   # vivid crimson
        patch.set_alpha(0.88)

    ax_bot.axhline(0, color="black", linewidth=1.0, linestyle="--", zorder=5)
    ax_bot.legend(
        handles=[
            plt.Rectangle((0, 0), 1, 1, color="#00b4d8", alpha=0.88,
                           label="Correctly classified"),
            plt.Rectangle((0, 0), 1, 1, color="#e63946", alpha=0.88,
                           label="Misclassified"),
            plt.Line2D([0], [0], color="black", linewidth=1.0, linestyle="--",
                       label="Advantage = 0  (same = diff degree)"),
        ],
        fontsize=9, framealpha=0.85, loc="upper left",
    )
    ax_bot.set_ylabel("Degree advantage\n(mean_deg_same − mean_deg_diff)", fontsize=10)
    ax_bot.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.4)
    ax_bot.set_title(
        "Degree advantage of same-class vs diff-class training neighbors: "
        "correct vs misclassified",
        fontsize=10,
    )
    _degree_axis(ax_bot, pos, all_degrees)

    fig.tight_layout()
    _save(fig, _subdir(save_dir, "train_neighbor_degree"),
          f"{prefix}_train_neighbor_degree_k{k}.png", show)


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


# ── class-wise accuracy + degree distribution ──────────────────────────────────

_CLS_TEST_COLOR  = "#1976D2"   # blue  — test node degrees
_CLS_TRAIN_COLOR = "#E53935"   # red   — train node degrees


def plot_class_accuracy_and_degree(
    class_acc_results,
    test_deg, test_labels,
    train_deg, train_labels,
    cfg,
    save_dir=None, show=False,
):
    """Class-wise accuracy and degree distributions — two single-panel figures.

    Figure 1 — single panel, twin axes:
        Left  — boxplots of test node degree per class (blue).
        Right — classification accuracy line (median ± IQR shading, green).
                n_test annotated below each class tick.

    Figure 2 — single panel, twin axes:
        Left  — boxplots of training node degree per class (red).
        Right — classification accuracy line (median ± IQR shading, green).
                n_train annotated below each class tick.

    Both figures share the same accuracy overlay so the reader can compare
    whether test-node or training-node degree structure better explains the
    class-wise performance disparity.

    Parameters
    ----------
    class_acc_results : list[dict]  — per-run output of get_accuracy_class.
    test_deg          : 1-D LongTensor — degrees of test nodes.
    test_labels       : 1-D LongTensor — true class labels of test nodes.
    train_deg         : 1-D LongTensor — degrees of training nodes.
    train_labels      : 1-D LongTensor — true class labels of training nodes.
    cfg               : dict
    save_dir          : str or None
    show              : bool
    """
    test_deg_np     = test_deg.cpu().numpy().astype(float)
    test_labels_np  = test_labels.cpu().numpy()
    train_deg_np    = train_deg.cpu().numpy().astype(float)
    train_labels_np = train_labels.cpu().numpy()

    all_classes = sorted({int(c) for c in test_labels_np} | {int(c) for c in train_labels_np})
    n_cls    = len(all_classes)
    pos      = np.arange(n_cls)
    n_runs   = len(class_acc_results)
    prefix   = _fname_prefix(cfg)
    n_test   = int(len(test_deg_np))
    subtitle = _subtitle(cfg, n_test, n_cls)

    # ── per-class accuracy stats across runs ──────────────────────────────────
    acc_median, acc_q1, acc_q3 = [], [], []
    n_test_per_cls = []
    for c in all_classes:
        run_accs = []
        for r in class_acc_results:
            if c in r and len(r[c]["preds"]) > 0:
                run_accs.append(float((r[c]["preds"] == r[c]["labels"]).float().mean()))
        acc_median.append(float(np.median(run_accs)) if run_accs else float("nan"))
        acc_q1.append(float(np.percentile(run_accs, 25)) if run_accs else float("nan"))
        acc_q3.append(float(np.percentile(run_accs, 75)) if run_accs else float("nan"))
        n_test_per_cls.append(int((test_labels_np == c).sum()))
    acc_median = np.array(acc_median)
    acc_q1     = np.array(acc_q1)
    acc_q3     = np.array(acc_q3)

    # ── per-class degree data ─────────────────────────────────────────────────
    test_deg_by_cls  = [test_deg_np[test_labels_np == c]   for c in all_classes]
    train_deg_by_cls = [train_deg_np[train_labels_np == c] for c in all_classes]
    n_train_per_cls  = [int((train_labels_np == c).sum())  for c in all_classes]

    fig_w    = max(8, n_cls * 1.4)

    def _cls_xaxis(ax, counts, label=""):
        """Set class tick labels with per-class node count below."""
        tick_labels = [f"Class {c}\n(n={n})" for c, n in zip(all_classes, counts)]
        ax.set_xticks(pos)
        ax.set_xticklabels(tick_labels, rotation=30, ha="right", fontsize=9)
        ax.set_xlabel(label, fontsize=11)

    def _acc_overlay(ax_deg):
        """Add accuracy line + IQR shading on a twin right axis."""
        ax_ov = ax_deg.twinx()
        ax_ov.plot(pos, acc_median, color=_ACC_COLOR, lw=2.0,
                   marker="s", markersize=5, zorder=5,
                   label=f"Accuracy (median ± IQR, {n_runs} runs)")
        ax_ov.fill_between(pos, acc_q1, acc_q3,
                           color=_ACC_COLOR, alpha=0.18, zorder=4)
        ax_ov.axhline(float(np.nanmean(acc_median)), color="dimgrey",
                      lw=1.0, ls="--", zorder=3,
                      label=f"Mean  ({np.nanmean(acc_median):.1%})")
        ax_ov.set_ylim(-0.05, 1.15)
        ax_ov.set_ylabel("Classification accuracy", fontsize=11, color=_ACC_COLOR)
        ax_ov.tick_params(axis="y", colors=_ACC_COLOR, labelsize=8)
        ax_ov.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))
        return ax_ov

    # ── Figure 1: test degree boxplots + accuracy overlay ────────────────────
    fig1, ax1 = plt.subplots(figsize=(fig_w, 5))
    ax1.set_title(f"Class-wise Test Node Degree & Accuracy\n{subtitle}", fontsize=11)

    bp1 = ax1.boxplot(test_deg_by_cls, positions=pos, widths=0.55, **_BP_KWARGS)
    for patch in bp1["boxes"]:
        patch.set_facecolor(_CLS_TEST_COLOR)
        patch.set_alpha(0.65)
    ax1.set_ylabel("Test node degree", fontsize=11)
    ax1.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax1.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.4)
    ax1.set_xlim(pos[0] - 0.6, pos[-1] + 0.6)

    ax_ov1 = _acc_overlay(ax1)
    _cls_xaxis(ax1, n_test_per_cls)

    handles = [
        mpatches.Patch(color=_CLS_TEST_COLOR, alpha=0.65, label="Test node degree"),
        plt.Line2D([0], [0], color=_ACC_COLOR, lw=2, marker="s", markersize=5,
                   label=f"Accuracy (median ± IQR, {n_runs} runs)"),
        plt.Line2D([0], [0], color="dimgrey", lw=1, ls="--",
                   label=f"Mean accuracy  ({np.nanmean(acc_median):.1%})"),
    ]
    ax1.legend(handles=handles, fontsize=8, loc="upper left", framealpha=0.88)

    fig1.tight_layout()
    _save(fig1, _subdir(save_dir, "class_accuracy"),
          f"{prefix}_class_accuracy_and_test_degree.png", show)

    # ── Figure 2: training degree boxplots + accuracy overlay ────────────────
    fig2, ax2 = plt.subplots(figsize=(fig_w, 5))
    ax2.set_title(f"Class-wise Training Node Degree & Accuracy\n{subtitle}", fontsize=11)

    bp2 = ax2.boxplot(train_deg_by_cls, positions=pos, widths=0.55, **_BP_KWARGS)
    for patch in bp2["boxes"]:
        patch.set_facecolor(_CLS_TRAIN_COLOR)
        patch.set_alpha(0.65)
    ax2.set_ylabel("Training node degree", fontsize=11)
    ax2.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax2.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.4)
    ax2.set_xlim(pos[0] - 0.6, pos[-1] + 0.6)

    ax_ov2 = _acc_overlay(ax2)
    _cls_xaxis(ax2, n_train_per_cls)

    handles2 = [
        mpatches.Patch(color=_CLS_TRAIN_COLOR, alpha=0.65, label="Training node degree"),
        plt.Line2D([0], [0], color=_ACC_COLOR, lw=2, marker="s", markersize=5,
                   label=f"Accuracy (median ± IQR, {n_runs} runs)"),
        plt.Line2D([0], [0], color="dimgrey", lw=1, ls="--",
                   label=f"Mean accuracy  ({np.nanmean(acc_median):.1%})"),
    ]
    ax2.legend(handles=handles2, fontsize=8, loc="upper left", framealpha=0.88)

    fig2.tight_layout()
    _save(fig2, _subdir(save_dir, "class_accuracy"),
          f"{prefix}_class_accuracy_and_train_degree.png", show)


def plot_train_degree_distribution(
    data,
    cfg,
    save_dir=None,
    show=False,
):
    """Degree distribution of training nodes only.

    Single panel: histogram of training node degrees (counts).

    Parameters
    ----------
    data     : PyG Data object with train_mask, y, and edge_index set.
    cfg      : experiment config dict.
    save_dir : str or None
    show     : bool
    """
    from torch_geometric.utils import degree as pyg_degree

    deg = pyg_degree(data.edge_index[0], num_nodes=data.num_nodes).long()
    train_mask = data.train_mask
    train_deg = deg[train_mask].cpu().numpy().astype(float)
    n_train = len(train_deg)
    prefix = _fname_prefix(cfg)
    dataset = cfg["dataset"]["name"]
    model   = cfg["model"]["name"]
    split   = cfg.get("split", "random")
    cc      = "CC" if cfg["dataset"].get("use_cc", False) else "noCC"
    subtitle = f"{dataset} · {model} · {split} · {cc}   |   {n_train:,} training nodes"

    fig, ax_hist = plt.subplots(figsize=(10, 5))
    fig.suptitle(f"Training Node Degree Distribution\n{subtitle}", fontsize=11)

    max_deg = int(train_deg.max())
    bins = np.arange(0, max_deg + 2) - 0.5
    ax_hist.hist(train_deg, bins=bins, color=_CLS_TRAIN_COLOR, alpha=0.65,
                 edgecolor="white", linewidth=0.4)

    ax_hist.set_xlabel("Degree", fontsize=11)
    ax_hist.set_ylabel("Count", fontsize=11)
    ax_hist.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax_hist.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.4)

    fig.tight_layout()
    _save(fig, _subdir(save_dir, "train_degree_dist"),
          f"{prefix}_train_degree_distribution.png", show)
