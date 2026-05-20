"""Regenerate neighborhood-cardinality plot from saved metrics.pt.

Usage
-----
    uv run plots/neighborhood_cardinality.py --results-dir results/<exec>
"""

from _load import base_parser, load
from plot_utils import plot_neighborhood_cardinality_vs_degree


def main():
    args = base_parser("Regenerate neighborhood-cardinality plot").parse_args()
    m = load(args)
    if not m["cardinality_by_k"]:
        print("No cardinality data in metrics (run with neighborhood_cardinality: true in config).")
        return
    purity = m["purity_by_k"] if len(m["purity_by_k"]) >= 2 else None
    plot_neighborhood_cardinality_vs_degree(
        m["test_deg"],
        m["cardinality_by_k"],
        m["deg_acc_results"],
        m["cfg"],
        all_deg=m["all_deg"],
        purity_by_k=purity,
        save_dir=args.results_dir,
        show=args.show,
    )


if __name__ == "__main__":
    main()
