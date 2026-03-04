"""
utils.py – graph-structural features for studying degree bias in node classification.

Each function takes a PyG Data object (with train_mask, test_mask, y, edge_index)
and returns a tensor aligned to the *test* nodes (same order as data.test_mask.nonzero()).
"""

from collections import deque

import numpy as np
import torch


def _build_adj(num_nodes: int, edge_index: torch.Tensor) -> list[list[int]]:
    """Build an undirected adjacency list from a COO edge_index."""
    row = torch.cat([edge_index[0], edge_index[1]])
    col = torch.cat([edge_index[1], edge_index[0]])
    adj: list[list[int]] = [[] for _ in range(num_nodes)]
    for u, v in zip(row.tolist(), col.tolist()):
        adj[u].append(v)
    return adj


def _multisource_bfs(sources: list[int], adj: list[list[int]], num_nodes: int) -> list[int]:
    """
    BFS from multiple source nodes simultaneously.
    Returns a list of length num_nodes where entry i is the shortest-hop
    distance from node i to the nearest source, or -1 if unreachable.
    """
    dist = [-1] * num_nodes
    queue: deque[int] = deque()
    for src in sources:
        dist[src] = 0
        queue.append(src)
    while queue:
        u = queue.popleft()
        for v in adj[u]:
            if dist[v] == -1:
                dist[v] = dist[u] + 1
                queue.append(v)
    return dist


def compute_distances_to_train(data) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute hop-distance features for every test node.

    Parameters
    ----------
    data : torch_geometric.data.Data
        Graph with attributes: edge_index, train_mask, test_mask, y.

    Returns
    -------
    dist_to_train : LongTensor, shape [num_test_nodes]
        Minimum hop distance to the nearest training node (any class).
        Unreachable nodes get value num_nodes + 1.

    dist_to_same_class_train : LongTensor, shape [num_test_nodes]
        Minimum hop distance to the nearest training node whose label matches
        the test node's true label.
        Unreachable nodes (or classes with no training example) get num_nodes + 1.
    """
    num_nodes = data.num_nodes
    INF = num_nodes + 1

    adj = _build_adj(num_nodes, data.edge_index)

    train_indices = data.train_mask.nonzero(as_tuple=True)[0]  # LongTensor
    test_indices = data.test_mask.nonzero(as_tuple=True)[0]    # LongTensor
    train_nodes = train_indices.tolist()
    test_nodes = test_indices.tolist()
    test_labels = data.y[data.test_mask].tolist()

    # ------------------------------------------------------------------
    # 1. Distance to nearest training node (all classes)
    # ------------------------------------------------------------------
    dist_all = _multisource_bfs(train_nodes, adj, num_nodes)
    dist_to_train = torch.tensor(
        [dist_all[n] if dist_all[n] != -1 else INF for n in test_nodes],
        dtype=torch.long,
    )

    # ------------------------------------------------------------------
    # 2. Distance to nearest same-class training node
    # ------------------------------------------------------------------
    train_labels = data.y[data.train_mask]
    unique_classes = train_labels.unique().tolist()

    # BFS per class → dict: class_id -> per-node distances
    class_dist: dict[int, list[int]] = {}
    for cls in unique_classes:
        cls = int(cls)
        cls_sources = train_indices[train_labels == cls].tolist()
        class_dist[cls] = _multisource_bfs(cls_sources, adj, num_nodes)

    dist_to_same_class: list[int] = []
    for node, lbl in zip(test_nodes, test_labels):
        lbl = int(lbl)
        if lbl in class_dist:
            d = class_dist[lbl][node]
            dist_to_same_class.append(d if d != -1 else INF)
        else:
            dist_to_same_class.append(INF)

    dist_to_same_class_train = torch.tensor(dist_to_same_class, dtype=torch.long)

    return dist_to_train, dist_to_same_class_train


def get_distance_deg(
    deg: torch.Tensor,
    dist_to_train: torch.Tensor,
    dist_to_same_class: torch.Tensor,
    num_nodes: int | None = None,
) -> dict:
    """Group per-test-node hop distances by node degree.

    Parameters
    ----------
    deg : LongTensor, shape [num_test_nodes]
        Degree of each test node (e.g. from ``torch_geometric.utils.degree``).
    dist_to_train : LongTensor, shape [num_test_nodes]
        Min-hop distance to the nearest training node (any class).
    dist_to_same_class : LongTensor, shape [num_test_nodes]
        Min-hop distance to the nearest same-class training node.
    num_nodes : int, optional
        Total number of nodes in the graph.  When provided, entries equal to
        the INF sentinel (num_nodes + 1) are replaced with NaN so they are
        excluded from statistics / plots rather than inflating them.

    Returns
    -------
    dict mapping degree (int) -> {
        'dist_to_train'    : float32 numpy array (NaN = unreachable),
        'dist_to_same_class': float32 numpy array (NaN = unreachable),
        'count'            : int, number of test nodes with that degree,
    }
    """
    deg = deg.cpu()
    d_train = dist_to_train.float().cpu()
    d_same = dist_to_same_class.float().cpu()

    if num_nodes is not None:
        sentinel = num_nodes + 1
        d_train[d_train >= sentinel] = float("nan")
        d_same[d_same >= sentinel] = float("nan")

    result = {}
    for d in deg.unique():
        idx = (deg == d).nonzero(as_tuple=False).view(-1)
        result[d.item()] = {
            "dist_to_train":     d_train[idx].numpy(),
            "dist_to_same_class": d_same[idx].numpy(),
            "count":             idx.numel(),
        }
    return result
