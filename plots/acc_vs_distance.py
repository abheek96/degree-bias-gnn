"""Regenerate accuracy-vs-distance plot from saved metrics.pt.

Usage
-----
    uv run plots/acc_vs_distance.py --results-dir results/<exec>
"""

from _load import base_parser, load
from plot_utils import plot_combined_vs_degree


def main():
    args = base_parser("Regenerate accuracy-vs-distance plot").parse_args()
    m = load(args)
    plot_combined_vs_degree(
        m["deg_acc_results"],
        m["dist_deg_data"],
        m["cfg"],
        save_dir=args.results_dir,
        show=args.show,
        run_labels=m["run_labels"],
    )


if __name__ == "__main__":
    main()
