"""
query_degree_group.py — List test-node indices for a given degree group.

Loads the dataset exactly as main.py does (via the cached split) and prints
the post-CC test-node indices whose degree falls in the specified range.

Usage
-----
    python query_degree_group.py --degree 5
    python query_degree_group.py --degree-min 3 --degree-max 8
    python query_degree_group.py --degree 5 --config config.yaml
"""

import argparse
import copy
import logging
import os
import sys

import torch
import yaml
from torch_geometric.utils import degree as graph_degree

from dataset_utils import load_or_create_split

log = logging.getLogger(__name__)


def _deep_merge(base, override):
    merged = copy.deepcopy(base)
    for key, val in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(val, dict):
            merged[key] = _deep_merge(merged[key], val)
        else:
            merged[key] = copy.deepcopy(val)
    return merged


def query_test_nodes_by_degree(cfg, deg_min: int, deg_max: int,
                                device: torch.device) -> list[int]:
    cache_dir = cfg.get("dataset_cache_dir", "dataset_cache")
    data = load_or_create_split(
        cfg["dataset"], cfg.get("split", "random"), cfg.get("seed", 42), cache_dir,
    ).to(device)

    all_deg   = graph_degree(data.edge_index[1], data.num_nodes).cpu()
    test_mask = data.test_mask.cpu()
    test_idx  = test_mask.nonzero(as_tuple=False).view(-1).tolist()

    matching = [n for n in test_idx if deg_min <= int(all_deg[n].item()) <= deg_max]
    return matching


def main():
    parser = argparse.ArgumentParser(
        description="Print test-node indices for a specific degree group."
    )
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--device", default=None,
                        help="Device override, e.g. cuda:0 or cpu")

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--degree", type=int,
                       help="Exact degree value.")
    group.add_argument("--degree-min", type=int,
                       help="Lower bound of degree range (inclusive).")
    parser.add_argument("--degree-max", type=int, default=None,
                        help="Upper bound of degree range (inclusive); "
                             "required with --degree-min.")
    args = parser.parse_args()

    if args.degree is not None:
        deg_min = deg_max = args.degree
    else:
        if args.degree_max is None:
            parser.error("--degree-max is required when --degree-min is used")
        deg_min, deg_max = args.degree_min, args.degree_max
        if deg_min > deg_max:
            parser.error("--degree-min must be <= --degree-max")

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
    )

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    model_cfg_path = os.path.join(
        "configs", f"{cfg['model']['name']}_{cfg['dataset']['name']}.yaml"
    )
    if os.path.exists(model_cfg_path):
        with open(model_cfg_path) as f:
            cfg = _deep_merge(cfg, yaml.safe_load(f))

    device = torch.device(
        args.device if args.device
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    matching = query_test_nodes_by_degree(cfg, deg_min, deg_max, device)

    log.info(
        "Dataset=%s  split=%s  degree=[%d, %d]  test nodes found: %d",
        cfg["dataset"]["name"], cfg.get("split", "random"),
        deg_min, deg_max, len(matching),
    )
    print(" ".join(str(n) for n in sorted(matching)))


if __name__ == "__main__":
    main()
