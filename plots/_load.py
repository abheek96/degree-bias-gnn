"""Shared argument parsing and metrics loading for per-plot scripts."""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from metrics import load_metrics


def base_parser(description: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--results-dir", required=True,
                   help="Path to experiment directory containing metrics.pt")
    p.add_argument("--show", action="store_true",
                   help="Display plots interactively in addition to saving")
    return p


def load(args) -> dict:
    """Load metrics from args.results_dir and return the bundle dict."""
    return load_metrics(args.results_dir)
