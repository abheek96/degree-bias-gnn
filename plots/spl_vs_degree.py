"""Regenerate SPL-vs-degree plot from saved metrics.pt.

Usage
-----
    uv run plots/spl_vs_degree.py --results-dir results/<exec>
"""

from _load import base_parser, load
from plot_utils import plot_spl_combined_vs_degree


def main():
    args = base_parser("Regenerate SPL-vs-degree plot").parse_args()
    m = load(args)
    purity = m["purity_by_k"] if len(m["purity_by_k"]) >= 1 else None
    plot_spl_combined_vs_degree(
        m["test_deg"],
        m["avg_spl"],
        m["avg_spl_same_class"],
        m["cfg"],
        deg_acc_results=m["deg_acc_results"],
        purity_by_k=purity,
        all_deg=m["all_deg"],
        has_labeled_neighbor=m["has_labeled_neighbor"],
        save_dir=args.results_dir,
        show=args.show,
    )


if __name__ == "__main__":
    main()
