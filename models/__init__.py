from .gcn import GCN
from .gat import GAT
from .graphsage import GraphSAGE
from .gcnii import GCNII

MODEL_REGISTRY = {
    "GCN": GCN,
    "GAT": GAT,
    "GraphSAGE": GraphSAGE,
    "GCNII": GCNII,
}


def get_model(name, **kwargs):
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Choose from: {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[name](**kwargs)
