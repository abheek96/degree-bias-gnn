"""Regenerate all plots for a given experiment directory.

Reads metrics.pt produced by main.py and regenerates every plot that was
enabled in the original run's config, without re-running training.

Usage
-----
    uv run plots/plot_all.py --results-dir results/<exec>
    uv run plots/plot_all.py --results-dir results/<exec> --show

Each individual plot can also be regenerated in isolation:

    uv run plots/acc_vs_degree.py --results-dir results/<exec>
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from _load import base_parser, load
from plot_utils import (
    plot_1hop_train_deg_vs_accuracy,
    plot_acc_and_labelling_ratio_vs_degree,
    plot_acc_vs_degree,
    plot_class_accuracy_and_degree,
    plot_combined_vs_degree,
    plot_max_same_train_deg_vs_degree,
    plot_neighborhood_cardinality_vs_degree,
    plot_purity_boxplots_vs_degree,
    plot_purity_delta_by_degree,
    plot_purity_vs_degree,
    plot_spl_combined_vs_degree,
    plot_train_neighbor_degree_stats,
)


def main():
    args = base_parser("Regenerate all plots from saved metrics").parse_args()
    m    = load(args)
    cfg  = m["cfg"]
    plot_cfg   = cfg.get("plot", {})
    save_dir   = args.results_dir
    show       = args.show
    purity_by_k = m["purity_by_k"]

    if plot_cfg.get("acc_vs_degree", False):
        plot_acc_vs_degree(
            m["deg_acc_results"], cfg,
            save_dir=save_dir, show=show,
            overall_acc=m["overall_acc"],
        )

    if plot_cfg.get("neighborhood_cardinality", False) and m["cardinality_by_k"]:
        purity = purity_by_k if len(purity_by_k) >= 2 else None
        plot_neighborhood_cardinality_vs_degree(
            m["test_deg"], m["cardinality_by_k"], m["deg_acc_results"], cfg,
            all_deg=m["all_deg"], purity_by_k=purity,
            save_dir=save_dir, show=show,
        )

    if plot_cfg.get("acc_vs_distance", False):
        plot_combined_vs_degree(
            m["deg_acc_results"], m["dist_deg_data"], cfg,
            save_dir=save_dir, show=show,
            run_labels=m["run_labels"],
        )

    if plot_cfg.get("spl_vs_degree", False):
        purity = purity_by_k if len(purity_by_k) >= 1 else None
        plot_spl_combined_vs_degree(
            m["test_deg"], m["avg_spl"], m["avg_spl_same_class"], cfg,
            deg_acc_results=m["deg_acc_results"],
            purity_by_k=purity,
            all_deg=m["all_deg"],
            has_labeled_neighbor=m["has_labeled_neighbor"],
            save_dir=save_dir, show=show,
        )

    if plot_cfg.get("labelling_ratio", False):
        plot_acc_and_labelling_ratio_vs_degree(
            m["deg_acc_results"], m["test_deg"], m["has_labeled_neighbor"], cfg,
            has_khop_labeled_neighbor=m["has_khop_labeled_neighbor"],
            k_hops=m["k_hops"],
            save_dir=save_dir, show=show,
        )

    if plot_cfg.get("purity_vs_degree", False) and purity_by_k:
        for k, purity_test in purity_by_k.items():
            plot_purity_vs_degree(
                m["test_deg"], purity_test, cfg, k,
                has_labeled_neighbor=m["has_labeled_neighbor"] if k == 1 else None,
                has_same_class_train=m["has_same_class_train"] if k == 1 else None,
                has_diff_class_train=m["has_diff_class_train"] if k == 1 else None,
                save_dir=save_dir, show=show,
            )
        if len(purity_by_k) > 1:
            plot_purity_delta_by_degree(
                m["test_deg"], purity_by_k, cfg,
                save_dir=save_dir, show=show,
            )
        if len(purity_by_k) >= 2 and 1 in purity_by_k and 2 in purity_by_k:
            plot_purity_boxplots_vs_degree(
                m["test_deg"], purity_by_k, m["deg_acc_results"], cfg,
                save_dir=save_dir, show=show,
            )

    if plot_cfg.get("train_neighbor_degree", False) and m["train_nb_deg_stats"] is not None:
        plot_train_neighbor_degree_stats(
            m["train_nb_deg_stats"], m["test_deg"], None, None, cfg,
            k=m["k_hops"],
            save_dir=save_dir, show=show,
        )

    if plot_cfg.get("train_1hop_deg_accuracy", False) and m["train_1hop_deg_stats"] is not None:
        plot_1hop_train_deg_vs_accuracy(
            m["train_1hop_deg_stats"], m["deg_acc_results"], m["test_deg"], cfg,
            save_dir=save_dir, show=show,
        )

    if plot_cfg.get("max_same_train_deg", False) and m["max_same_train_deg"] is not None:
        plot_max_same_train_deg_vs_degree(
            m["max_same_train_deg"], m["test_deg"], cfg,
            save_dir=save_dir, show=show,
        )

    if plot_cfg.get("class_accuracy", False):
        plot_class_accuracy_and_degree(
            m["class_acc_results"],
            m["test_deg"], m["test_labels"],
            m["train_deg"], m["train_labels"],
            cfg,
            save_dir=save_dir, show=show,
        )


if __name__ == "__main__":
    main()
