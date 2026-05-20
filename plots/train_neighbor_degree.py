"""Regenerate training-neighbor-degree plots from saved metrics.pt.

Usage
-----
    uv run plots/train_neighbor_degree.py --results-dir results/<exec>
"""

from _load import base_parser, load
from plot_utils import plot_1hop_train_deg_vs_accuracy, plot_max_same_train_deg_vs_degree, plot_train_neighbor_degree_stats


def main():
    args = base_parser("Regenerate training-neighbor-degree plots").parse_args()
    m = load(args)
    cfg = m["cfg"]

    if m["train_nb_deg_stats"] is not None:
        plot_train_neighbor_degree_stats(
            m["train_nb_deg_stats"],
            m["test_deg"],
            None,   # pred not stored in metrics; skips per-node overlay
            None,   # data not stored
            cfg,
            k=m["k_hops"],
            save_dir=args.results_dir,
            show=args.show,
        )

    if m["train_1hop_deg_stats"] is not None:
        plot_1hop_train_deg_vs_accuracy(
            m["train_1hop_deg_stats"],
            m["deg_acc_results"],
            m["test_deg"],
            cfg,
            save_dir=args.results_dir,
            show=args.show,
        )

    if m["max_same_train_deg"] is not None:
        plot_max_same_train_deg_vs_degree(
            m["max_same_train_deg"],
            m["test_deg"],
            cfg,
            save_dir=args.results_dir,
            show=args.show,
        )


if __name__ == "__main__":
    main()
