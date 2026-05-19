"""Regenerate class-accuracy plot from saved metrics.pt.

Usage
-----
    uv run plots/class_accuracy.py --results-dir results/<exec>
"""

from _load import base_parser, load
from plot_utils import plot_class_accuracy_and_degree


def main():
    args = base_parser("Regenerate class-accuracy plot").parse_args()
    m = load(args)
    plot_class_accuracy_and_degree(
        m["class_acc_results"],
        m["test_deg"],
        m["test_labels"],
        m["train_deg"],
        m["train_labels"],
        m["cfg"],
        save_dir=args.results_dir,
        show=args.show,
    )


if __name__ == "__main__":
    main()
