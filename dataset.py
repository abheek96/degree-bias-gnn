import logging

import torch
from torch_geometric.datasets import Planetoid, Amazon, Coauthor, WikiCS
from torch_geometric.transforms import LargestConnectedComponents

log = logging.getLogger(__name__)

DATASET_REGISTRY = {
    "Cora": lambda root: Planetoid(root=root, name="Cora"),
    "CiteSeer": lambda root: Planetoid(root=root, name="CiteSeer"),
    "PubMed": lambda root: Planetoid(root=root, name="PubMed"),
    "Computers": lambda root: Amazon(root=root, name="Computers"),
    "Photo": lambda root: Amazon(root=root, name="Photo"),
    "CS": lambda root: Coauthor(root=root, name="CS"),
    "Physics": lambda root: Coauthor(root=root, name="Physics"),
    "WikiCS": lambda root: WikiCS(root=root),
}


def load_dataset(dataset_cfg):
    name = dataset_cfg["name"]
    root = dataset_cfg.get("root", "./data")

    if name not in DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset '{name}'. Choose from: {list(DATASET_REGISTRY.keys())}")

    dataset = DATASET_REGISTRY[name](root)
    data = dataset[0]

    if dataset_cfg.get("use_cc", False):
        data = LargestConnectedComponents()(data)
        log.info("Filtered to largest connected component: %d nodes", data.num_nodes)

    log.info("Dataset: %s  Nodes: %d  Edges: %d  Features: %d  Classes: %d",
             name, data.num_nodes, data.num_edges, data.num_node_features, dataset.num_classes)

    return data

