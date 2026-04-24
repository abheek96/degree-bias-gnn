import logging

import torch
from torch_geometric.datasets import Planetoid, Amazon, Coauthor, WikiCS, WikipediaNetwork, Reddit
from torch_geometric.transforms import LargestConnectedComponents
from torch_geometric.utils import to_networkx
import networkx as nx

log = logging.getLogger(__name__)

DATASET_REGISTRY = {
    "Cora": lambda root: Planetoid(root=root, name="Cora"),
    "CiteSeer": lambda root: Planetoid(root=root, name="CiteSeer"),
    "PubMed": lambda root: Planetoid(root=root, name="PubMed"),
    "Computers": lambda root: Amazon(root=root, name="Computers"),
    "Photo": lambda root: Amazon(root=root, name="Photo"),
    "CS": lambda root: Coauthor(root=root, name="CS"),
    "Physics": lambda root: Coauthor(root=root, name="Physics"),
    "WikiCS":    lambda root: WikiCS(root=root),
    "Reddit":    lambda root: Reddit(root=root),
    "Chameleon": lambda root: WikipediaNetwork(root=root, name="chameleon",
                                               geom_gcn_preprocess=True),
    "Squirrel":  lambda root: WikipediaNetwork(root=root, name="squirrel",
                                               geom_gcn_preprocess=True),
}


def load_dataset(dataset_cfg):
    name = dataset_cfg["name"]
    root = dataset_cfg.get("root", "./data")

    if name not in DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset '{name}'. Choose from: {list(DATASET_REGISTRY.keys())}")

    dataset = DATASET_REGISTRY[name](root)
    data = dataset[0]

    # geom_gcn_preprocess=True gives 2D masks [N, 10] (10 fixed splits).
    # Flatten to 1D by selecting column 0 so downstream code sees bool vectors.
    for attr in ("train_mask", "val_mask", "test_mask"):
        mask = getattr(data, attr, None)
        if mask is not None and mask.dim() == 2:
            setattr(data, attr, mask[:, 0])

    G = to_networkx(data, to_undirected=True)
    comps = sorted(nx.connected_components(G), key=len, reverse=True)
    if len(comps) > 1:
        sizes = [len(c) for c in comps]
        log.info(
            "Connected components: %d  (sizes: LCC=%d, others=%s)",
            len(comps), sizes[0], sizes[1:] if len(sizes[1:]) <= 10 else sizes[1:10] + ["..."],
        )
    else:
        log.info("Graph is fully connected (%d nodes)", data.num_nodes)

    if dataset_cfg.get("use_cc", False):
        data = LargestConnectedComponents()(data)
        log.info("Filtered to largest connected component: %d nodes", data.num_nodes)

    norm = dataset_cfg.get("normalize_features", None)
    if norm == "l1":
        data.x = data.x / data.x.sum(dim=1, keepdim=True).clamp(min=1e-8)
        log.info("Feature normalization: L1 (row-sum)")
    elif norm == "l2":
        data.x = data.x / data.x.norm(dim=1, keepdim=True).clamp(min=1e-8)
        log.info("Feature normalization: L2 (row-norm)")

    log.info("Dataset: %s  Nodes: %d  Edges: %d  Features: %d  Classes: %d",
             name, data.num_nodes, data.num_edges, data.num_node_features, dataset.num_classes)

    return data

