"""
Shared pytest fixtures for DegBias tests.

Fixtures
--------
tiny_graph  : small synthetic graph — fast unit tests, no network I/O
cora_data   : real Cora with CC + random split (seed=42) — slow integration tests
trained_gcn : 2-layer GCN trained on cora_data — slow integration tests
"""

import os
import random
import sys

import numpy as np
import pytest
import torch
from torch_geometric.data import Data

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ── deterministic seeding ──────────────────────────────────────────────────────

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ── tiny synthetic graph ───────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def tiny_graph() -> Data:
    """
    7-node undirected graph (stored bidirectionally):

        0 — 1 — 3 — 4 — 5
            |       |
            2       6

    Edges: 0-1, 1-2, 1-3, 3-4, 4-5, 4-6

    Classes: 0=[0,1,2,3]  1=[4,5,6]
    Train nodes: {0, 4}   (one per class)
    Test  nodes: {2, 3, 5, 6}

    Known BFS from node 3 (incoming = all symmetric neighbors):
        k=1 neighbors: {1, 4}
        k=2 neighbors: {0, 2, 5, 6}
    """
    set_seed(42)
    edges = [(0, 1), (1, 2), (1, 3), (3, 4), (4, 5), (4, 6)]
    # Make undirected (both directions)
    src = [u for u, v in edges] + [v for u, v in edges]
    dst = [v for u, v in edges] + [u for u, v in edges]
    edge_index = torch.tensor([src, dst], dtype=torch.long)

    y = torch.tensor([0, 0, 0, 0, 1, 1, 1], dtype=torch.long)

    num_nodes = 7
    # Simple bag-of-words style features (identity-like)
    x = torch.eye(num_nodes, dtype=torch.float)

    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    train_mask[[0, 4]] = True

    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask[[2, 3, 5, 6]] = True

    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask[[1]] = True

    return Data(
        x=x,
        edge_index=edge_index,
        y=y,
        train_mask=train_mask,
        test_mask=test_mask,
        val_mask=val_mask,
        num_nodes=num_nodes,
    )


# ── Cora (slow) ────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
@pytest.mark.slow
def cora_data() -> Data:
    """Real Cora with CC filtering and random split (seed=42), matching main.py."""
    from torch_geometric.datasets import Planetoid
    from torch_geometric.transforms import LargestConnectedComponents

    from dataset_utils import apply_split

    data = Planetoid(root="./data", name="Cora")[0]
    data = LargestConnectedComponents()(data)

    set_seed(42)
    dataset_cfg = {
        "name": "Cora",
        "use_cc": True,
        "num_train_per_class": 20,
        "num_val": 500,
    }
    return apply_split(data, split="random", dataset_cfg=dataset_cfg)


@pytest.fixture(scope="session")
@pytest.mark.slow
def trained_gcn(cora_data):
    """2-layer GCN trained for 5 epochs on Cora. Returns (model, pred)."""
    from models import get_model
    from train import train as train_step
    from test import evaluate

    set_seed(42)
    device = torch.device("cpu")
    data = cora_data.to(device)

    model = get_model(
        "GCN",
        in_dim=data.num_node_features,
        hidden_dim=64,
        out_dim=int(data.y.max().item()) + 1,
        num_layers=3,
        dropout=0.5,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    criterion = torch.nn.CrossEntropyLoss()

    for _ in range(5):
        train_step(model, data, optimizer, criterion)

    model.eval()
    with torch.no_grad():
        pred = model(data.x, data.edge_index).argmax(dim=1)

    return model, pred
