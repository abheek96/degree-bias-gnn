"""Regenerate purity-vs-degree plots from saved metrics.pt.

Usage
-----
    uv run plots/purity_vs_degree.py --results-dir results/<exec>
"""

from _load import base_parser, load
from plot_utils import plot_purity_boxplots_vs_degree, plot_purity_delta_by_degree, plot_purity_vs_degree


def main():
    args = base_parser("Regenerate purity-vs-degree plots").parse_args()
    m = load(args)
    purity_by_k = m["purity_by_k"]
    if not purity_by_k:
        print("No purity data in metrics (run with purity_vs_degree: true in config).")
        return
    for k, purity_test in purity_by_k.items():
        plot_purity_vs_degree(
            m["test_deg"],
            purity_test,
            m["cfg"],
            k,
            has_labeled_neighbor=m["has_labeled_neighbor"] if k == 1 else None,
            has_same_class_train=m["has_same_class_train"] if k == 1 else None,
            has_diff_class_train=m["has_diff_class_train"] if k == 1 else None,
            save_dir=args.results_dir,
            show=args.show,
        )
    if len(purity_by_k) > 1:
        plot_purity_delta_by_degree(
            m["test_deg"],
            purity_by_k,
            m["cfg"],
            save_dir=args.results_dir,
            show=args.show,
        )
    if len(purity_by_k) >= 2 and 1 in purity_by_k and 2 in purity_by_k:
        plot_purity_boxplots_vs_degree(
            m["test_deg"],
            purity_by_k,
            m["deg_acc_results"],
            m["cfg"],
            save_dir=args.results_dir,
            show=args.show,
        )


if __name__ == "__main__":
    main()
