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


def _exp_label(cfg):
    """One-line experiment descriptor used in figure titles."""
    dataset = cfg["dataset"]["name"]
    model   = cfg["model"]["name"]
    split   = cfg.get("split", "random")
    cc      = "CC" if cfg["dataset"].get("use_cc", False) else "noCC"
    return f"{dataset} | {model} | {split} | {cc}"


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
        fig.savefig(path, dpi=150, bbox_inches="tight")
        log.info("Saved %s", path)
    if show:
        plt.show()
    plt.close(fig)


# ── public entry point ─────────────────────────────────────────────────────────

def plot_acc_vs_degree(run_results, cfg, save_dir=None, show=False):
    """Plot test-node accuracy vs. node degree as box plots.

    Single run
        One figure: a box per degree showing the distribution of per-node
        correctness (0/1) across all test nodes with that degree.  Boxes are
        coloured low→high degree using the YlOrRd colormap.

    Multiple runs
        Figure 1 – *across-runs*: a box per degree whose whiskers reflect how
        the *per-run mean* accuracy varies across the different random seeds.
        Useful for seeing where degree-bias is consistent vs. seed-sensitive.

        Figure 2 – *per-run*: a group of boxes per degree, one box per run,
        showing the full node-level accuracy distribution inside each run.
        Runs are colour-coded (tab10); the legend maps colour to run/seed.

    Parameters
    ----------
    run_results : list[dict]
        One entry per run; each is the output of ``get_accuracy_deg`` restricted
        to test nodes (keys = degree ints, values contain 'preds'/'labels').
    cfg : dict
        Experiment config — used to build axis labels and figure titles.
    save_dir : str or None
        Directory to write PDF figures.  Skipped when None.
    show : bool
        Call ``plt.show()`` after each figure.
    """
    n_runs = len(run_results)
    all_degrees, deg_data = _collect(run_results)
    pos = list(range(len(all_degrees)))
    label = _exp_label(cfg)

    if n_runs == 1:
        _plot_single(all_degrees, pos, deg_data, label, save_dir, show)
    else:
        _plot_across_runs(all_degrees, pos, deg_data, n_runs, label, save_dir, show)
        _plot_per_run(all_degrees, pos, deg_data, n_runs, label, save_dir, show)


# ── single run ─────────────────────────────────────────────────────────────────

def _plot_single(all_degrees, pos, deg_data, label, save_dir, show):
    data = [
        deg_data[d][0] if len(deg_data[d][0]) > 0 else np.array([np.nan])
        for d in all_degrees
    ]
    max_d = max(all_degrees) or 1
    cmap  = plt.cm.YlOrRd
    colors = [cmap(0.25 + 0.75 * d / max_d) for d in all_degrees]

    fig, ax = plt.subplots(figsize=(_fig_w(len(all_degrees)), 5))
    bp = ax.boxplot(data, positions=pos, widths=0.6, **_BP_KWARGS)
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.78)

    _degree_axis(ax, pos, all_degrees)
    ax.set_ylabel("Accuracy", fontsize=11)
    ax.set_ylim(-0.05, 1.10)
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))
    ax.set_title(f"Accuracy vs. Degree  —  test nodes (1 run)\n{label}", fontsize=11)
    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.5)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=max_d))
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label="Node degree", pad=0.02, fraction=0.025)

    fig.tight_layout()
    _save(fig, save_dir, "acc_vs_degree_single_run.pdf", show)


# ── multiple runs: distribution of per-run means across seeds ─────────────────

def _plot_across_runs(all_degrees, pos, deg_data, n_runs, label, save_dir, show):
    data = []
    for d in all_degrees:
        means = [a.mean() for a in deg_data[d] if len(a) > 0]
        data.append(np.array(means) if means else np.array([np.nan]))

    max_d  = max(all_degrees) or 1
    cmap   = plt.cm.YlOrRd
    colors = [cmap(0.25 + 0.75 * d / max_d) for d in all_degrees]

    fig, ax = plt.subplots(figsize=(_fig_w(len(all_degrees)), 5))
    bp = ax.boxplot(data, positions=pos, widths=0.6, **_BP_KWARGS)
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.78)

    _degree_axis(ax, pos, all_degrees)
    ax.set_ylabel(f"Mean accuracy per run  ({n_runs} seeds)", fontsize=11)
    ax.set_ylim(-0.05, 1.10)
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))
    ax.set_title(
        f"Accuracy vs. Degree  —  seed-to-seed variability  ({n_runs} runs)\n{label}",
        fontsize=11,
    )
    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.5)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=max_d))
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label="Node degree", pad=0.02, fraction=0.025)

    fig.tight_layout()
    _save(fig, save_dir, "acc_vs_degree_across_runs.pdf", show)


# ── multiple runs: per-run node-level distribution, grouped by degree ──────────

def _plot_per_run(all_degrees, pos, deg_data, n_runs, label, save_dir, show):
    run_colors = [plt.cm.tab10(i / 10) for i in range(min(n_runs, 10))]
    box_w = 0.82 / n_runs

    fig, ax = plt.subplots(figsize=(_fig_w(len(all_degrees), n_runs), 5))

    handles = []
    for run_idx in range(n_runs):
        offsets = [p + (run_idx - (n_runs - 1) / 2) * box_w for p in pos]
        data = [
            deg_data[d][run_idx] if len(deg_data[d][run_idx]) > 0 else np.array([np.nan])
            for d in all_degrees
        ]
        c  = run_colors[run_idx % 10]
        bp = ax.boxplot(data, positions=offsets, widths=box_w * 0.88,
                        manage_ticks=False, **_BP_KWARGS)
        for patch in bp["boxes"]:
            patch.set_facecolor(c)
            patch.set_alpha(0.72)
        handles.append(
            plt.Rectangle((0, 0), 1, 1, fc=c, alpha=0.75,
                           label=f"Run {run_idx + 1}")
        )

    _degree_axis(ax, pos, all_degrees)
    ax.set_ylabel("Accuracy  (test nodes)", fontsize=11)
    ax.set_ylim(-0.05, 1.10)
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))
    ax.set_title(
        f"Accuracy vs. Degree  —  per-run distributions  ({n_runs} runs)\n{label}",
        fontsize=11,
    )
    ax.legend(handles=handles, title="Run (seed)", loc="upper right",
              framealpha=0.85, fontsize=9)
    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.5)

    fig.tight_layout()
    _save(fig, save_dir, "acc_vs_degree_per_run.pdf", show)


# ── axis helpers ───────────────────────────────────────────────────────────────

def _degree_axis(ax, pos, all_degrees):
    """Set x-axis ticks; thin out labels when there are many degrees."""
    ax.set_xlabel("Node degree", fontsize=11)
    ax.set_xlim(pos[0] - 0.6, pos[-1] + 0.6)
    n = len(all_degrees)
    # Show every k-th label so they don't overlap at high degree counts
    step = max(1, n // 30)
    tick_pos    = pos[::step]
    tick_labels = all_degrees[::step]
    ax.set_xticks(tick_pos)
    ax.set_xticklabels(tick_labels, rotation=55, ha="right", fontsize=8)
