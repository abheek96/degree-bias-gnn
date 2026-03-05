"""
utils.py – graph-structural features for studying degree bias in node classification.

Each function takes a PyG Data object (with train_mask, test_mask, y, edge_index)
and returns a tensor aligned to the *test* nodes (same order as data.test_mask.nonzero()).
"""

from collections import defaultdict

import numpy as np
import torch

def index_to_adj(
    x, edge_index, add_self_loop=False, remove_self_loop=False, sparse=False
):
    from torch_geometric.utils import to_dense_adj

    assert not (add_self_loop == True and remove_self_loop == True)
    num_nodes = len(x)
    adj = to_dense_adj(edge_index, max_num_nodes=num_nodes)[0].bool()
    if add_self_loop:
        adj.fill_diagonal_(True)
    if remove_self_loop:
        adj.fill_diagonal_(False)
    if sparse:
        adj = adj.to_sparse()
    return adj


def get_node_neighbor_het_rate(y, adj):
    if not torch.is_tensor(y):
        y = torch.tensor(y)
    y = y.to(adj.device)
    y_tile = torch.tile(y, (len(y), 1))
    ngb_label_mat = (adj * y_tile).float()
    ngb_label_mat[adj == 0] = torch.nan
    node_ngb_consis = (ngb_label_mat == y_tile.T).sum(axis=1) / adj.sum(axis=1)
    node_ngb_consis = node_ngb_consis.nan_to_num(0)  # handle 0 degree nodes
    node_ngb_het = 1 - node_ngb_consis
    return node_ngb_het

def get_node_amp(data, threshold=0.3, verbose=False):
    adj = index_to_adj(data.x, data.edge_index, add_self_loop=False)
    node_het = get_node_neighbor_het_rate(data.y, adj)
    node_amp = node_het > threshold
    if verbose:
        print(f"Avg Node Heterogeneity: {node_het.mean()}")
        print(f"Threrhold: {threshold}")
        print(f"Counts Node AMP: {torch.tensor(node_amp).unique(return_counts=True)}")
    return node_amp


def get_node_dmp(data, train_mask, verbose=False):
    from torch_geometric.utils import k_hop_subgraph, mask_to_index, index_to_mask

    y = data.y.cpu().numpy()
    label_idx = mask_to_index(train_mask).cpu().numpy()

    node_nearest_label = np.full(len(y), -1)
    node_nearest_label[label_idx] = y[label_idx]

    n_update = len(label_idx)

    for num_hop in range(1, 10):
        for node in label_idx:
            nbs, _, _, _ = k_hop_subgraph(
                node_idx=int(node),
                num_hops=num_hop,
                edge_index=data.edge_index,
                num_nodes=data.num_nodes,
            )
            nbs = nbs.cpu().numpy()
            nb_mask = (
                index_to_mask(torch.tensor(nbs), size=data.num_nodes).cpu().numpy()
            )

            unvisit_mask = node_nearest_label == -1
            node_nearest_label[unvisit_mask & nb_mask] = y[node]

            n_update += unvisit_mask.sum()

    node_dmp = node_nearest_label != y
    if verbose:
        print(torch.tensor(node_nearest_label).unique(return_counts=True))
        print(torch.tensor(node_dmp).unique(return_counts=True))

    return node_dmp


def get_node_dmp_dist(data, train_mask, verbose=False):
    from torch_geometric.utils import k_hop_subgraph, mask_to_index, index_to_mask

    y = data.y.cpu().numpy()
    label_idx = mask_to_index(train_mask).cpu().numpy()

    node_nearest_label = np.full(len(y), -1)
    node_nearest_label[label_idx] = y[label_idx]

    n_update = len(label_idx)

    for num_hop in range(1, 10):
        for node in label_idx:
            nbs, _, _, _ = k_hop_subgraph(
                node_idx=int(node),
                num_hops=num_hop,
                edge_index=data.edge_index,
                num_nodes=data.num_nodes,
            )
            nbs = nbs.cpu().numpy()
            nb_mask = (
                index_to_mask(torch.tensor(nbs), size=data.num_nodes).cpu().numpy()
            )

            unvisit_mask = node_nearest_label == -1
            node_nearest_label[unvisit_mask & nb_mask] = num_hop

            n_update += unvisit_mask.sum()

    node_dmp = node_nearest_label > 3
    if verbose:
        print(torch.tensor(node_nearest_label).unique(return_counts=True))
        print(torch.tensor(node_dmp).unique(return_counts=True))

    return node_dmp


def compute_hops_to_nearest_labeled_nodes(data, train_mask):
    from torch_geometric.utils import k_hop_subgraph, mask_to_index

    y = data.y.cpu().numpy()
    label_idx = mask_to_index(train_mask).cpu().numpy()
    num_hops = np.zeros(data.num_nodes).astype(int)
    for node in tqdm(
        range(data.num_nodes), desc="Computing hops to nearest labeled node"
    ):
        node_label = y[node]
        num_hop = 0
        while True:
            nbs, _, _, _ = k_hop_subgraph(
                node, num_hop, data.edge_index, num_nodes=data.num_nodes
            )
            labeled_nbs = set(nbs.cpu().numpy()).intersection(set(label_idx))
            if len(labeled_nbs) > 0 or num_hop >= 10:
                ngb_labels = y[list(labeled_nbs)]
                label_correct = node_label in ngb_labels
                if label_correct or num_hop >= 10:
                    # print (f'Node {node} label {node_label} hop {num_hop} n_ngb {len(nbs)} labeled_ngb {labeled_nbs} ngb_label {ngb_labels} isin {isin}')
                    break
            num_hop += 1
        num_hops[node] = num_hop
    return num_hops


def compute_distances_to_train(data) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute hop-distance features for every test node via k-hop expansion.

    Mirrors the logic of ``compute_hops_to_nearest_labeled_nodes``: for each
    test node the k-hop subgraph is expanded one hop at a time until both
    distances are resolved or ``MAX_HOP`` is reached.

    For each test node, two distances are recorded:

    dist_to_train
        The first hop ``k`` at which ``k_hop_subgraph(node, k)`` contains
        *any* training node — i.e. the shortest path to the nearest labeled
        node regardless of class.

    dist_to_same_class_train
        The first hop ``k`` at which ``k_hop_subgraph(node, k)`` contains a
        training node whose label matches the test node's true label — i.e.
        the shortest path to a correctly labeled training node.

    Both distances follow the same expansion strategy as
    ``compute_hops_to_nearest_labeled_nodes``: start at hop 0 (just the node
    itself), grow by one hop each iteration, and stop as soon as the target
    condition is satisfied.  Nodes for which no qualifying training node is
    found within MAX_HOP hops receive the sentinel value ``num_nodes + 1``.

    Parameters
    ----------
    data : torch_geometric.data.Data
        Graph with attributes: edge_index, train_mask, test_mask, y.

    Returns
    -------
    dist_to_train : LongTensor, shape [num_test_nodes]
    dist_to_same_class_train : LongTensor, shape [num_test_nodes]
    """
    from torch_geometric.utils import k_hop_subgraph, mask_to_index

    MAX_HOP = 10
    INF = data.num_nodes + 1  # sentinel for "not found within MAX_HOP hops"

    y            = data.y.cpu().numpy()
    train_idx    = mask_to_index(data.train_mask).cpu().numpy()
    label_set    = set(train_idx.tolist())

    # Pre-group training indices by class for O(1) same-class lookup
    class_sets: dict[int, set[int]] = defaultdict(set)
    for idx in train_idx:
        class_sets[int(y[idx])].add(int(idx))

    test_nodes = data.test_mask.nonzero(as_tuple=True)[0].tolist()

    dist_any_list  = []
    dist_same_list = []

    for node in test_nodes:
        node_label = int(y[node])
        found_any  = INF
        found_same = INF

        for num_hop in range(MAX_HOP + 1):
            nbs, _, _, _ = k_hop_subgraph(
                node_idx=int(node),
                num_hops=num_hop,
                edge_index=data.edge_index,
                num_nodes=data.num_nodes,
            )
            nbs_set     = set(nbs.cpu().numpy().tolist())
            labeled_nbs = nbs_set & label_set

            # First hop that reaches any training node
            if found_any == INF and labeled_nbs:
                found_any = num_hop

            # First hop that reaches a same-class training node
            if found_same == INF and (labeled_nbs & class_sets[node_label]):
                found_same = num_hop

            # Both resolved — no need to expand further
            if found_any != INF and found_same != INF:
                break

        dist_any_list.append(found_any)
        dist_same_list.append(found_same)

    return (
        torch.tensor(dist_any_list,  dtype=torch.long),
        torch.tensor(dist_same_list, dtype=torch.long),
    )


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


def get_amp_deg(deg: torch.Tensor, node_amp) -> dict:
    """Group per-test-node AMP flags by node degree.

    Parameters
    ----------
    deg : LongTensor, shape [num_test_nodes]
        Degree of each test node.
    node_amp : BoolTensor or numpy bool array, shape [num_test_nodes]
        AMP flag for each test node (True = ambivalent message passing node).

    Returns
    -------
    dict mapping degree (int) -> {
        'amp_rate' : float, fraction of AMP nodes at that degree,
        'count'    : int, number of test nodes with that degree,
    }
    """
    if not torch.is_tensor(node_amp):
        node_amp = torch.tensor(node_amp)
    node_amp = node_amp.bool().cpu()
    deg = deg.cpu()

    result = {}
    for d in deg.unique():
        idx = (deg == d).nonzero(as_tuple=False).view(-1)
        result[d.item()] = {
            "amp_rate": node_amp[idx].float().mean().item(),
            "count":    idx.numel(),
        }
    return result


def get_dmp_deg(deg: torch.Tensor, node_dmp) -> dict:
    """Group per-test-node DMP flags by node degree.

    Parameters
    ----------
    deg : LongTensor, shape [num_test_nodes]
        Degree of each test node.
    node_dmp : BoolTensor or numpy bool array, shape [num_test_nodes]
        DMP flag for each test node (True = distant message passing node).

    Returns
    -------
    dict mapping degree (int) -> {
        'dmp_rate' : float, fraction of DMP nodes at that degree,
        'count'    : int, number of test nodes with that degree,
    }
    """
    if not torch.is_tensor(node_dmp):
        node_dmp = torch.tensor(node_dmp)
    node_dmp = node_dmp.bool().cpu()
    deg = deg.cpu()

    result = {}
    for d in deg.unique():
        idx = (deg == d).nonzero(as_tuple=False).view(-1)
        result[d.item()] = {
            "dmp_rate": node_dmp[idx].float().mean().item(),
            "count":    idx.numel(),
        }
    return result


def get_node_amp(data, threshold=0.3):
    """
    Compute node amplification (AMP) based on neighbor heterogeneity.

    Parameters
    ----------
    data : torch_geometric.data.Data
        Graph with attributes: x, edge_index, y.
    threshold : float
        Heterogeneity threshold for identifying high-AMP nodes.

    Returns
    -------
    node_amp : BoolTensor
        Boolean tensor indicating which nodes have neighbor heterogeneity > threshold.

    Notes
    -----
    - index_to_adj() converts edge_index (COO format) into a dense boolean adjacency matrix.
    - get_node_neighbor_het_rate(y, adj) computes for each node the fraction of neighbors
      with a different label: heterogeneity = (# neighbors with label != y[i]) / (total neighbors).
    - A node is high-AMP if its neighbor heterogeneity > threshold.
    """
    adj = index_to_adj(data.x, data.edge_index, add_self_loop=False)
    node_het = get_node_neighbor_het_rate(data.y, adj)
    node_amp = node_het > threshold
    return node_amp