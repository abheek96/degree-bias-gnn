"""Regenerate accuracy-vs-degree plot from saved metrics.pt.

Usage
-----
    uv run plots/acc_vs_degree.py --results-dir results/<exec>
"""

from _load import base_parser, load
from plot_utils import plot_acc_vs_degree


def main():
    args = base_parser("Regenerate accuracy-vs-degree plot").parse_args()
    m = load(args)
    plot_acc_vs_degree(
        m["deg_acc_results"],
        m["cfg"],
        save_dir=args.results_dir,
        show=args.show,
        overall_acc=m["overall_acc"],
    )


if __name__ == "__main__":
    main()
