from torch_geometric.datasets import Planetoid, Amazon, Coauthor, WikiCS

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

    print(f"Dataset: {name}")
    print(f"  Nodes: {data.num_nodes}  Edges: {data.num_edges}  "
          f"Features: {data.num_node_features}  Classes: {dataset.num_classes}")

    return data
