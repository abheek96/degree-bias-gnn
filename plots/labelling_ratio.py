"""Regenerate labelling-ratio plots from saved metrics.pt.

Usage
-----
    uv run plots/labelling_ratio.py --results-dir results/<exec>
"""

from _load import base_parser, load
from plot_utils import plot_acc_and_labelling_ratio_vs_degree


def main():
    args = base_parser("Regenerate labelling-ratio plots").parse_args()
    m = load(args)
    plot_acc_and_labelling_ratio_vs_degree(
        m["deg_acc_results"],
        m["test_deg"],
        m["has_labeled_neighbor"],
        m["cfg"],
        has_khop_labeled_neighbor=m["has_khop_labeled_neighbor"],
        k_hops=m["k_hops"],
        save_dir=args.results_dir,
        show=args.show,
    )


if __name__ == "__main__":
    main()
