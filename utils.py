"""
utils.py – graph-structural features for studying degree bias in node classification.

Each function takes a PyG Data object (with train_mask, test_mask, y, edge_index)
and returns a tensor aligned to the *test* nodes (same order as data.test_mask.nonzero()).
"""

from collections import defaultdict

import numpy as np
import torch


# def get_node_neighbor_het_rate(y, adj):
#     if not torch.is_tensor(y):
#         y = torch.tensor(y)
#     y = y.to(adj.device)
#     y_tile = torch.tile(y, (len(y), 1))
#     ngb_label_mat = (adj * y_tile).float()
#     ngb_label_mat[adj == 0] = torch.nan
#     node_ngb_consis = (ngb_label_mat == y_tile.T).sum(axis=1) / adj.sum(axis=1)
#     node_ngb_consis = node_ngb_consis.nan_to_num(0)  # handle 0 degree nodes
#     node_ngb_het = 1 - node_ngb_consis
#     return node_ngb_het
#
# def get_node_amp(data, threshold=0.5, verbose=False):
#     adj = index_to_adj(data.x, data.edge_index, add_self_loop=False)
#     node_het = get_node_neighbor_het_rate(data.y, adj)
#     node_amp = node_het > threshold
#     if verbose:
#         print(f"Avg Node Heterogeneity: {node_het.mean()}")
#         print(f"Threrhold: {threshold}")
#         print(f"Counts Node AMP: {torch.tensor(node_amp).unique(return_counts=True)}")
#     return node_amp
#
#
# def get_node_dmp(data, train_mask, verbose=False):
#     from torch_geometric.utils import k_hop_subgraph, mask_to_index, index_to_mask
#
#     y = data.y.cpu().numpy()
#     label_idx = mask_to_index(train_mask).cpu().numpy()
#
#     node_nearest_label = np.full(len(y), -1)
#     node_nearest_label[label_idx] = y[label_idx]
#
#     n_update = len(label_idx)
#
#     for num_hop in range(1, 10):
#         for node in label_idx:
#             nbs, _, _, _ = k_hop_subgraph(
#                 node_idx=int(node),
#                 num_hops=num_hop,
#                 edge_index=data.edge_index,
#                 num_nodes=data.num_nodes,
#             )
#             nbs = nbs.cpu().numpy()
#             nb_mask = (
#                 index_to_mask(torch.tensor(nbs), size=data.num_nodes).cpu().numpy()
#             )
#
#             unvisit_mask = node_nearest_label == -1
#             node_nearest_label[unvisit_mask & nb_mask] = y[node]
#
#             n_update += unvisit_mask.sum()
#
#     node_dmp = node_nearest_label != y
#     if verbose:
#         print(torch.tensor(node_nearest_label).unique(return_counts=True))
#         print(torch.tensor(node_dmp).unique(return_counts=True))
#
#     return node_dmp
#
#
# def get_node_dmp_dist(data, train_mask, verbose=False):
#     from torch_geometric.utils import k_hop_subgraph, mask_to_index, index_to_mask
#
#     y = data.y.cpu().numpy()
#     label_idx = mask_to_index(train_mask).cpu().numpy()
#
#     node_nearest_label = np.full(len(y), -1)
#     node_nearest_label[label_idx] = y[label_idx]
#
#     n_update = len(label_idx)
#
#     for num_hop in range(1, 10):
#         for node in label_idx:
#             nbs, _, _, _ = k_hop_subgraph(
#                 node_idx=int(node),
#                 num_hops=num_hop,
#                 edge_index=data.edge_index,
#                 num_nodes=data.num_nodes,
#             )
#             nbs = nbs.cpu().numpy()
#             nb_mask = (
#                 index_to_mask(torch.tensor(nbs), size=data.num_nodes).cpu().numpy()
#             )
#
#             unvisit_mask = node_nearest_label == -1
#             node_nearest_label[unvisit_mask & nb_mask] = num_hop
#
#             n_update += unvisit_mask.sum()
#
#     node_dmp = node_nearest_label > 3
#     if verbose:
#         print(torch.tensor(node_nearest_label).unique(return_counts=True))
#         print(torch.tensor(node_dmp).unique(return_counts=True))
#
#     return node_dmp


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


def get_khop_cardinality(data, k: int) -> torch.Tensor:
    """Number of distinct nodes within k hops of each node (excluding self).

    For k=1 this equals the standard 1-hop degree.  Used to study how
    neighbourhood size grows with k and how that relates to degree bias.

    Uses boolean adjacency-matrix powers; feasible for Cora-scale graphs
    (~2,700 nodes).  Avoid on large graphs.

    Parameters
    ----------
    data : torch_geometric.data.Data
    k    : neighbourhood radius in hops

    Returns
    -------
    cardinality : LongTensor [num_nodes]
    """
    from torch_geometric.utils import to_dense_adj

    N   = data.num_nodes
    A   = to_dense_adj(data.edge_index, max_num_nodes=N).squeeze(0).bool()
    A.fill_diagonal_(False)

    reach = A.clone()
    power = A.float()
    A_f   = A.float()

    for _ in range(k - 1):
        power = power @ A_f
        reach = reach | power.bool()

    reach.fill_diagonal_(False)
    return reach.sum(dim=1).long()


def get_avg_spl_to_train(data) -> torch.Tensor:
    """Average shortest path length from every node to all training nodes.

    For each node v, runs BFS from every training node and averages the
    resulting distances.  Unreachable pairs are excluded from the average
    (returned as NaN for nodes with no reachable training node).

    Complexity: O(|train| × (V + E)).  Feasible for the citation graphs
    used here (~140 training nodes, ~2 700 nodes, ~5 400 edges for Cora).

    Parameters
    ----------
    data : torch_geometric.data.Data  (must have .edge_index and .train_mask)

    Returns
    -------
    avg_spl : FloatTensor, shape [num_nodes]
    """
    from collections import deque
    from torch_geometric.utils import mask_to_index

    N         = data.num_nodes
    train_idx = mask_to_index(data.train_mask).cpu().tolist()

    # Build undirected adjacency list
    src, dst = data.edge_index.cpu()
    adj = [[] for _ in range(N)]
    for u, v in zip(src.tolist(), dst.tolist()):
        adj[u].append(v)

    sum_dist = torch.zeros(N)
    count    = torch.zeros(N)

    for t in train_idx:
        dist = [-1] * N
        dist[t] = 0
        q = deque([t])
        while q:
            u = q.popleft()
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    q.append(v)
        for i, d in enumerate(dist):
            if d >= 0:
                sum_dist[i] += d
                count[i]    += 1

    avg_spl          = torch.full((N,), float("nan"))
    valid            = count > 0
    avg_spl[valid]   = sum_dist[valid] / count[valid]
    return avg_spl


def get_avg_spl_to_same_class_train(data) -> torch.Tensor:
    """Average shortest path length from every node to same-class training nodes.

    For each node v with label c, averages the BFS distances to all training
    nodes that also have label c.  Nodes with no reachable same-class training
    node get NaN.

    Complexity: O(|train| × (V + E)).

    Parameters
    ----------
    data : torch_geometric.data.Data  (must have .edge_index, .train_mask, .y)

    Returns
    -------
    avg_spl : FloatTensor, shape [num_nodes]
    """
    from collections import deque
    from torch_geometric.utils import mask_to_index

    N         = data.num_nodes
    train_idx = mask_to_index(data.train_mask).cpu().tolist()
    labels    = data.y.cpu()

    src, dst = data.edge_index.cpu()
    adj = [[] for _ in range(N)]
    for u, v in zip(src.tolist(), dst.tolist()):
        adj[u].append(v)

    sum_dist = torch.zeros(N)
    count    = torch.zeros(N)

    for t in train_idx:
        t_label = labels[t].item()
        dist = [-1] * N
        dist[t] = 0
        q = deque([t])
        while q:
            u = q.popleft()
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    q.append(v)
        for i, d in enumerate(dist):
            if d >= 0 and labels[i].item() == t_label:
                sum_dist[i] += d
                count[i]    += 1

    avg_spl        = torch.full((N,), float("nan"))
    valid          = count > 0
    avg_spl[valid] = sum_dist[valid] / count[valid]
    return avg_spl


def get_labelling_ratio(data) -> torch.Tensor:
    """For every node, indicate whether it has at least one labeled (training) neighbor.

    Returns a boolean tensor of shape [num_nodes] where True means the node
    has at least one immediate (1-hop) neighbor in the training set.

    Parameters
    ----------
    data : torch_geometric.data.Data  (must have .edge_index and .train_mask)

    Returns
    -------
    has_labeled_neighbor : BoolTensor, shape [num_nodes]
    """
    from torch_geometric.utils import to_dense_adj

    N     = data.num_nodes
    A     = to_dense_adj(data.edge_index, max_num_nodes=N).squeeze(0)  # (N, N)
    train = data.train_mask.cpu().float()                               # (N,)
    return (A.cpu() @ train) > 0                                        # (N,)


def get_class_labelling_ratio(data) -> tuple[torch.Tensor, torch.Tensor]:
    """For every node, indicate presence of same-class and diff-class training neighbors.

    Checks the immediate (1-hop) neighborhood for training nodes whose label
    matches vs. differs from the node's own label.

    Parameters
    ----------
    data : torch_geometric.data.Data  (must have .edge_index, .train_mask, .y)

    Returns
    -------
    has_same_class_train : BoolTensor, shape [num_nodes]
        True when the node has ≥1 same-class training neighbor.
    has_diff_class_train : BoolTensor, shape [num_nodes]
        True when the node has ≥1 different-class training neighbor.
    """
    from torch_geometric.utils import to_dense_adj

    N      = data.num_nodes
    A      = to_dense_adj(data.edge_index, max_num_nodes=N).squeeze(0).cpu()  # (N, N)
    labels = data.y.cpu()                                                       # (N,)
    train_mask = data.train_mask.cpu()                                          # (N,)

    # Vectorised per-class:
    #   same_train_c  = training nodes whose label == c
    #   diff_train_c  = training nodes whose label != c
    # For nodes with label c, check if any neighbor falls in each group.
    has_same = torch.zeros(N, dtype=torch.bool)
    has_diff = torch.zeros(N, dtype=torch.bool)

    for c in labels.unique():
        node_mask_c  = labels == c                             # nodes of class c
        same_train_c = (train_mask & (labels == c)).float()   # train nodes of class c
        diff_train_c = (train_mask & (labels != c)).float()   # train nodes of other classes
        has_same[node_mask_c] = ((A @ same_train_c) > 0)[node_mask_c]
        has_diff[node_mask_c] = ((A @ diff_train_c) > 0)[node_mask_c]

    return has_same, has_diff


def get_training_neighbor_degree_stats(data, k: int = 2) -> dict:
    """Degree statistics of same-class and diff-class training nodes in each
    test node's k-hop neighborhood.

    For each test node, finds all training nodes within k hops and records
    the mean degree of same-class vs diff-class training nodes.  This is used
    to investigate whether the relative degree of same-class vs diff-class
    training neighbors correlates with classification outcome.

    In GCN the aggregation weight for edge (u→v) is ``1/sqrt(deg_u * deg_v)``,
    so higher-degree training nodes contribute *less* per edge.  If same-class
    training nodes consistently have higher degree than diff-class ones, their
    per-edge signal is more diluted, which may contribute to misclassification.

    Parameters
    ----------
    data : torch_geometric.data.Data
    k    : hop radius for the receptive field (should match model's num_layers)

    Returns
    -------
    dict with numpy arrays aligned to test nodes (same order as test_mask):
        same_mean_deg : mean degree of same-class training neighbors (NaN if none)
        diff_mean_deg : mean degree of diff-class training neighbors (NaN if none)
        same_count    : number of same-class training nodes in k-hop neighborhood
        diff_count    : number of diff-class training nodes in k-hop neighborhood
    """
    from torch_geometric.utils import k_hop_subgraph
    from torch_geometric.utils import degree as pyg_degree

    N = data.num_nodes
    y = data.y.cpu()
    train_mask = data.train_mask.cpu()
    deg = pyg_degree(data.edge_index[1].cpu(), N).numpy()

    test_nodes = data.test_mask.nonzero(as_tuple=True)[0].tolist()

    same_mean_degs, diff_mean_degs = [], []
    same_counts, diff_counts = [], []

    for node in test_nodes:
        node_label = int(y[node])
        nbs, _, _, _ = k_hop_subgraph(
            node_idx=int(node), num_hops=k,
            edge_index=data.edge_index, num_nodes=N,
        )
        nbs_set = set(nbs.cpu().tolist()) - {node}

        same_train = [n for n in nbs_set if train_mask[n] and int(y[n]) == node_label]
        diff_train = [n for n in nbs_set if train_mask[n] and int(y[n]) != node_label]

        same_mean_degs.append(float(np.mean([deg[n] for n in same_train])) if same_train else np.nan)
        diff_mean_degs.append(float(np.mean([deg[n] for n in diff_train])) if diff_train else np.nan)
        same_counts.append(len(same_train))
        diff_counts.append(len(diff_train))

    return {
        "same_mean_deg": np.array(same_mean_degs, dtype=float),
        "diff_mean_deg": np.array(diff_mean_degs, dtype=float),
        "same_count":    np.array(same_counts, dtype=int),
        "diff_count":    np.array(diff_counts, dtype=int),
    }


def get_node_purity(data, k: int = 1, node_mask=None) -> torch.Tensor:
    """Neighborhood purity for selected nodes at receptive field radius k.

    purity(v) = |{u ∈ N_k(v) : label[u] == label[v]}| / |N_k(v)|

    where N_k(v) is the cumulative k-hop neighborhood of v (all nodes
    reachable within k hops, excluding v itself).  Nodes whose k-hop
    neighborhood is empty receive NaN.

    Parameters
    ----------
    data      : torch_geometric.data.Data  (must have .y labels)
    k         : neighbourhood radius in hops
    node_mask : optional BoolTensor of length num_nodes; when provided, purity
                is computed and returned only for nodes where mask is True
                (e.g. data.test_mask).  Returns shape [mask.sum()].
                When None, purity is computed for all nodes; returns shape
                [num_nodes].

    Returns
    -------
    purity : FloatTensor, shape [num_nodes] or [mask.sum()]
    """
    from torch_geometric.utils import to_dense_adj

    N   = data.num_nodes
    A   = to_dense_adj(data.edge_index, max_num_nodes=N).squeeze(0).bool()
    A.fill_diagonal_(False)

    reach = A.clone()
    power = A.float()
    A_f   = A.float()
    for _ in range(k - 1):
        power = power @ A_f
        reach = reach | power.bool()
    reach.fill_diagonal_(False)

    labels      = data.y.cpu()                                   # (N,)
    label_match = (labels.unsqueeze(0) == labels.unsqueeze(1))  # (N, N)

    # Restrict to requested rows
    reach_cpu = reach.cpu()
    if node_mask is not None:
        mask_cpu        = node_mask.cpu()
        reach_cpu       = reach_cpu[mask_cpu]        # (M, N)
        label_match     = label_match[mask_cpu]      # (M, N)
        out_size        = int(mask_cpu.sum())
    else:
        out_size        = N

    total_counts     = reach_cpu.sum(dim=1).float()              # (M,) or (N,)
    same_cls_counts  = (reach_cpu & label_match).sum(dim=1).float()

    purity        = torch.full((out_size,), float("nan"))
    valid         = total_counts > 0
    purity[valid] = same_cls_counts[valid] / total_counts[valid]

    return purity


def analyse_node_similarity(model, data, node_x: int, k_hops: int) -> dict | None:
    """Cosine similarity between node_x and each of its 1-hop neighbors at
    every representation level: raw input features, h^1, …, h^k_hops.

    Neighbors are classified as same_train / diff_train / non_train and sorted
    by raw-feature similarity (descending) so the ordering is consistent across
    all layer panels.

    Parameters
    ----------
    model   : trained GCN with get_intermediate()
    data    : PyG Data object
    node_x  : target node global index
    k_hops  : number of GCNConv layers (num_layers - 1)

    Returns
    -------
    dict with keys:
        node_idx, degree, true_label, pred_label, k_hops,
        neighbors: list of {node_idx, degree, type,
                             similarities: [sim_raw, sim_h1, …, sim_hk]}
    Returns None if node_x has no 1-hop neighbors.
    """
    import torch.nn.functional as F_fn
    from torch_geometric.utils import degree as graph_degree

    x          = data.x.cpu()
    y          = data.y.cpu()
    src        = data.edge_index[0].cpu()
    dst        = data.edge_index[1].cpu()
    train_mask = data.train_mask.cpu()
    all_deg    = graph_degree(data.edge_index[1], data.num_nodes).cpu()

    model.eval()
    with torch.no_grad():
        pred = model(data.x, data.edge_index).argmax(dim=1).cpu()

    # All representation levels: raw + one per GCNConv layer
    layer_reps = [x]
    for l in range(1, k_hops + 1):
        layer_reps.append(
            model.get_intermediate(data.x, data.edge_index, layer=l).cpu()
        )

    neighbors = dst[src == node_x].tolist()
    if not neighbors:
        log.warning("node_similarity: node %d has no 1-hop neighbors", node_x)
        return None

    true_lbl = int(y[node_x].item())
    log.info(
        "node_similarity: node %d  degree=%d  true=%d  pred=%d  n_neighbors=%d",
        node_x, int(all_deg[node_x].item()), true_lbl,
        int(pred[node_x].item()), len(neighbors),
    )

    neighbor_detail = []
    for nb in neighbors:
        nb_type = (
            ("same_train" if int(y[nb].item()) == true_lbl else "diff_train")
            if bool(train_mask[nb].item()) else "non_train"
        )
        sims = [
            float(F_fn.cosine_similarity(
                rep[node_x].unsqueeze(0), rep[nb].unsqueeze(0)
            ).item())
            for rep in layer_reps
        ]
        log.info(
            "  neighbor %d  type=%-12s  deg=%d  sim_raw=%.4f  sim_hk=%.4f  delta=%.4f",
            nb, nb_type, int(all_deg[nb].item()),
            sims[0], sims[-1], sims[-1] - sims[0],
        )
        neighbor_detail.append({
            "node_idx":     int(nb),
            "degree":       int(all_deg[nb].item()),
            "type":         nb_type,
            "similarities": sims,   # [sim_raw, sim_h1, …, sim_hk]
        })

    # Sort by raw feature similarity descending — fixed order across all panels
    neighbor_detail.sort(key=lambda d: d["similarities"][0], reverse=True)

    return {
        "node_idx":   node_x,
        "degree":     int(all_deg[node_x].item()),
        "true_label": true_lbl,
        "pred_label": int(pred[node_x].item()),
        "k_hops":     k_hops,
        "neighbors":  neighbor_detail,
    }


def compute_node_similarity_analysis(
    model, data, k_hops: int,
    target_nodes: list = None,
) -> list:
    """Run analyse_node_similarity for each node in target_nodes.

    Skips out-of-range indices and deduplicates.  Returns a list of result
    dicts (one per successfully analysed node).
    """
    N       = data.num_nodes
    results = []
    seen    = set()

    for node_x in (target_nodes or []):
        if node_x < 0 or node_x >= N:
            log.warning(
                "node_similarity: node_idx=%d out of range [0, %d) — skipping",
                node_x, N,
            )
            continue
        if node_x in seen:
            continue
        seen.add(node_x)
        result = analyse_node_similarity(model, data, node_x, k_hops)
        if result is not None:
            results.append(result)

    return results


def get_feature_similarity_delta(data, model, k_hops: int = 1) -> list:
    """Per-test-node cosine similarity to same-class training 1-hop neighbors,
    measured in raw feature space and after one step of message passing (h^(1)).

    For each test node v with at least one same-class training node as a direct
    (1-hop) neighbor:

        sim_raw(v)      = mean cosine_sim(x_v,        x_u)       for u in same-class train ∩ N_1(v)
        sim_hk(v)       = mean cosine_sim(h_v^k_hops, h_u^k_hops) for the same u
        delta(v)        = sim_hk(v) - sim_raw(v)

    delta > 0  : message passing brought v closer to its same-class training
                 neighbors in representation space (aggregation helped).
    delta < 0  : message passing pulled v away — diff-class neighbors in N_1(v)
                 introduced feature-space noise (analogous to low label purity).

    The post-aggregation representation is taken at layer=k_hops so the
    comparison is always against the representation the model actually uses
    before classification, regardless of network depth.

    Test nodes with no same-class training node in their 1-hop neighborhood
    are excluded.

    Parameters
    ----------
    data   : PyG Data object (must have .x, .y, .edge_index, .train_mask, .test_mask)
    model  : trained GCN instance with get_intermediate()
    k_hops : number of GCNConv layers = num_layers - 1 (the final Linear head
             does not count); defaults to 1 for a 2-layer GCN

    Returns
    -------
    list of dicts, one per qualifying test node:
        node_idx, degree, sim_raw, sim_h1, delta, n_same_1hop
    """
    import torch.nn.functional as F_fn
    from torch_geometric.utils import degree as graph_degree

    x         = data.x.cpu()
    y         = data.y.cpu()
    src       = data.edge_index[0].cpu()
    dst       = data.edge_index[1].cpu()
    train_mask = data.train_mask.cpu()
    test_mask  = data.test_mask.cpu()
    all_deg   = graph_degree(data.edge_index[1], data.num_nodes).cpu()

    h1 = model.get_intermediate(data.x, data.edge_index, layer=k_hops).cpu()

    test_nodes = test_mask.nonzero(as_tuple=False).view(-1).tolist()
    results = []

    for v in test_nodes:
        neighbors = dst[src == v]
        same_train = [int(u) for u in neighbors
                      if train_mask[u] and int(y[u]) == int(y[v])]
        if not same_train:
            continue

        x_v  = x[v].unsqueeze(0)           # (1, d_in)
        x_u  = x[same_train]               # (k, d_in)
        h_v  = h1[v].unsqueeze(0)          # (1, d_hidden)
        h_u  = h1[same_train]              # (k, d_hidden)

        sim_raw = float(F_fn.cosine_similarity(x_v, x_u).mean())
        sim_hk  = float(F_fn.cosine_similarity(h_v, h_u).mean())

        results.append({
            "node_idx":    v,
            "degree":      int(all_deg[v].item()),
            "sim_raw":     sim_raw,
            "sim_hk":      sim_hk,
            "delta":       sim_hk - sim_raw,
            "n_same_1hop": len(same_train),
        })

    return results


# def get_node_het(data, k: int = 1) -> torch.Tensor:
#     """Return the raw neighbor-heterogeneity ratio for every node.
#
#     For k=1 (default) this is the standard fraction of immediate neighbors
#     whose label differs.  For k>1 the k-hop neighborhood is used: all nodes
#     reachable in 1 to k steps (excluding the node itself) are treated as the
#     effective neighborhood, and the heterogeneity is the fraction of those
#     nodes whose label differs.
#
#     The k-hop reachability matrix is built as the boolean union of the 1-hop
#     through k-hop adjacency powers:  reach_k = adj | adj² | … | adj^k,
#     with the diagonal zeroed to exclude self-loops.
#
#     Parameters
#     ----------
#     data : torch_geometric.data.Data
#         Graph with attributes: x, edge_index, y.
#     k : int
#         Neighbourhood radius in hops.  k=1 reproduces the original behaviour.
#
#     Returns
#     -------
#     node_het : FloatTensor, shape [num_nodes]
#         Fraction of each node's k-hop neighbourhood whose label differs from
#         the node's own label.  Isolated nodes (degree 0) receive 0.
#     """
#     adj = index_to_adj(data.x, data.edge_index, add_self_loop=False)
#     if k > 1:
#         adj_float = adj.float()
#         reach = adj.clone()          # cumulative reachability (bool)
#         power = adj_float.clone()    # adj^hop (float, for matrix multiply)
#         for _ in range(k - 1):
#             power = power @ adj_float
#             reach = reach | power.bool()
#         reach.fill_diagonal_(False)  # exclude self from neighbourhood
#         adj = reach
#     return get_node_neighbor_het_rate(data.y, adj)


# def get_amp_dmp_groups(node_het, node_dmp_k, amp_threshold: float = 0.5):
#     """Assign each test node to one of four AMP × DMP groups.
#
#     Groups
#     ------
#     0 : Low AMP  + No DMP  (het ≤ threshold, dist_to_same_class ≤ dmp_coeff)
#     1 : Low AMP  + DMP
#     2 : High AMP + No DMP
#     3 : High AMP + DMP     (het > threshold, dist_to_same_class > dmp_coeff)
#
#     Parameters
#     ----------
#     node_het : FloatTensor or numpy array, shape [num_test_nodes]
#         Neighbor-heterogeneity ratio for each test node.
#     node_dmp_k : bool array, shape [num_test_nodes]
#         DMP flag for each test node (True = DMP node).
#     amp_threshold : float
#         Threshold that separates low / high AMP.
#
#     Returns
#     -------
#     group_labels : int numpy array, shape [num_test_nodes]
#         Values in {0, 1, 2, 3}.
#     group_names : list[str]
#         Human-readable name for each group index.
#     """
#     if torch.is_tensor(node_het):
#         node_het = node_het.float().cpu().numpy()
#     else:
#         node_het = np.array(node_het, dtype=np.float32)
#
#     if torch.is_tensor(node_dmp_k):
#         node_dmp_k = node_dmp_k.bool().cpu().numpy()
#     else:
#         node_dmp_k = np.asarray(node_dmp_k, dtype=bool)
#
#     high_amp = node_het > amp_threshold   # bool array
#     group_labels = (high_amp.astype(int) * 2) + node_dmp_k.astype(int)
#     # 0 = (0,0)=low+nodmp  1 = (0,1)=low+dmp  2 = (1,0)=high+nodmp  3 = (1,1)=high+dmp
#
#     group_names = [
#         f"Low AMP\nNo DMP",
#         f"Low AMP\nDMP",
#         f"High AMP\nNo DMP",
#         f"High AMP\nDMP",
#     ]
#     return group_labels, group_names
#
#
# def get_group_deg_counts(test_deg, group_labels) -> dict:
#     """Count test nodes per (degree, AMP×DMP group) cell.
#
#     Parameters
#     ----------
#     test_deg : LongTensor or int array, shape [num_test_nodes]
#     group_labels : int array, shape [num_test_nodes]  (values 0-3)
#
#     Returns
#     -------
#     dict mapping degree (int) -> {group (int): count (int)}
#     """
#     if torch.is_tensor(test_deg):
#         test_deg = test_deg.numpy()
#     test_deg = np.asarray(test_deg, dtype=int)
#     if torch.is_tensor(group_labels):
#         group_labels = group_labels.numpy()
#     group_labels = np.asarray(group_labels, dtype=int)
#
#     result = {}
#     for d in np.unique(test_deg):
#         d_mask = test_deg == d
#         result[int(d)] = {g: int((d_mask & (group_labels == g)).sum())
#                           for g in range(4)}
#     return result


# def get_amp_deg(deg: torch.Tensor, node_het) -> dict:
#     """Group per-test-node heterogeneity values by node degree.
#
#     Parameters
#     ----------
#     deg : LongTensor, shape [num_test_nodes]
#         Degree of each test node.
#     node_het : FloatTensor or numpy float array, shape [num_test_nodes]
#         Raw neighbor-heterogeneity ratio for each test node (output of
#         ``get_node_het`` indexed to test nodes).
#
#     Returns
#     -------
#     dict mapping degree (int) -> {
#         'het_values' : float32 numpy array of per-node het ratios,
#         'count'      : int, number of test nodes with that degree,
#     }
#     """
#     if torch.is_tensor(node_het):
#         node_het = node_het.float().cpu().numpy()
#     else:
#         node_het = np.array(node_het, dtype=np.float32)
#     deg = deg.cpu()
#
#     result = {}
#     for d in deg.unique():
#         idx = (deg == d).nonzero(as_tuple=False).view(-1).numpy()
#         result[d.item()] = {
#             "het_values": node_het[idx],
#             "count":      int(idx.size),
#         }
#     return result
#
#
# def get_dmp_deg(deg: torch.Tensor, node_dmp) -> dict:
#     """Group per-test-node DMP flags by node degree.
#
#     Parameters
#     ----------
#     deg : LongTensor, shape [num_test_nodes]
#         Degree of each test node.
#     node_dmp : BoolTensor or numpy bool array, shape [num_test_nodes]
#         DMP flag for each test node (True = distant message passing node).
#
#     Returns
#     -------
#     dict mapping degree (int) -> {
#         'count_0' : int, number of non-DMP nodes at that degree,
#         'count_1' : int, number of DMP nodes at that degree,
#         'count'   : int, total number of test nodes with that degree,
#     }
#     """
#     if not torch.is_tensor(node_dmp):
#         node_dmp = torch.tensor(node_dmp)
#     node_dmp = node_dmp.bool().cpu()
#     deg = deg.cpu()
#
#     result = {}
#     for d in deg.unique():
#         idx = (deg == d).nonzero(as_tuple=False).view(-1)
#         flags = node_dmp[idx]
#         result[d.item()] = {
#             "count_0": int((~flags).sum().item()),
#             "count_1": int(flags.sum().item()),
#             "count":   idx.numel(),
#         }
#     return result


# def get_totoro_neighborhood_groups(data, totoro_values, k: int = 2):
#     """Characterise each test node's k-hop training neighbourhood by comparing
#     same-class vs different-class training nodes on two dimensions:
#
#     1. **Count** — does the same-class have more training neighbours?
#     2. **Totoro** — do same-class training neighbours have a higher mean
#        Totoro score (i.e. do they carry a stronger, if more confused, signal)?
#
#     Groups
#     ------
#     0 : Same class wins both count and Totoro  → most advantaged
#     1 : Same class wins count only
#     2 : Same class wins Totoro only
#     3 : Diff class wins both / no same-class neighbours → most disadvantaged
#
#     Parameters
#     ----------
#     data : PyG Data
#         Graph with edge_index, y, train_mask, test_mask, num_nodes.
#     totoro_values : FloatTensor, shape [num_nodes]
#         Output of ``get_totoro_values``.
#     k : int
#         Neighbourhood radius in hops.
#
#     Returns
#     -------
#     group_labels : int numpy array, shape [num_test_nodes]
#     group_names  : list[str]
#     stats        : dict with per-test-node arrays:
#                    same_count, diff_count, same_totoro, diff_totoro
#     """
#     from torch_geometric.utils import k_hop_subgraph
#
#     y_cpu      = data.y.cpu()
#     train_cpu  = data.train_mask.cpu()
#     test_nodes = data.test_mask.nonzero(as_tuple=True)[0].tolist()
#     tot_cpu    = totoro_values.cpu()
#
#     group_labels                      = []
#     same_counts, diff_counts          = [], []
#     same_totoro_means, diff_totoro_means = [], []
#
#     for node in test_nodes:
#         node_label = int(y_cpu[node])
#
#         nbs, _, _, _ = k_hop_subgraph(
#             node_idx=int(node), num_hops=k,
#             edge_index=data.edge_index, num_nodes=data.num_nodes,
#         )
#         # Exclude the node itself; keep only training neighbours
#         nbs_set    = set(nbs.cpu().tolist()) - {node}
#         same_train = [n for n in nbs_set
#                       if train_cpu[n] and int(y_cpu[n]) == node_label]
#         diff_train = [n for n in nbs_set
#                       if train_cpu[n] and int(y_cpu[n]) != node_label]
#
#         same_count  = len(same_train)
#         diff_count  = len(diff_train)
#         same_totoro = float(tot_cpu[same_train].mean()) if same_count > 0 else 0.0
#         diff_totoro = float(tot_cpu[diff_train].mean()) if diff_count > 0 else 0.0
#
#         same_counts.append(same_count)
#         diff_counts.append(diff_count)
#         same_totoro_means.append(same_totoro)
#         diff_totoro_means.append(diff_totoro)
#
#         if same_count == 0:
#             g = 3   # no same-class training signal at all → most disadvantaged
#         elif diff_count == 0:
#             g = 0   # only same-class signal → most advantaged
#         else:
#             count_dom  = same_count  >= diff_count
#             totoro_dom = same_totoro >= diff_totoro
#             # 0: both, 1: count only, 2: totoro only, 3: neither
#             g = (1 - int(count_dom)) * 2 + (1 - int(totoro_dom))
#
#         group_labels.append(g)
#
#     group_names = [
#         "Same wins\nboth",
#         "Same wins\ncount only",
#         "Same wins\nTotoro only",
#         "Diff wins\nboth / no same",
#     ]
#     stats = {
#         "same_count":  np.array(same_counts,        dtype=float),
#         "diff_count":  np.array(diff_counts,        dtype=float),
#         "same_totoro": np.array(same_totoro_means,  dtype=float),
#         "diff_totoro": np.array(diff_totoro_means,  dtype=float),
#     }
#     return np.array(group_labels, dtype=int), group_names, stats
#
#
# def get_totoro_values(data, cfg) -> torch.Tensor:
#     """Compute per-node Totoro influence scores via Personalized PageRank (PPR).
#
#     Each node's Totoro score measures the total PPR-weighted influence it
#     receives from training nodes of a *different* class — a structural source
#     of misclassification that is distinct from node degree.
#
#     PPR is computed in closed form:
#         Pi = alpha * (I - (1 - alpha) * Â)^{-1}
#     where alpha = 1 - pagerank_prob is the restart probability and Â is the
#     symmetrically normalised adjacency with self-loops added.
#
#     The class-level influence (GPR) gpr[v, c] is the mean PPR contribution
#     from all training nodes of class c toward node v. The Totoro score for
#     node v is the total influence it receives from training nodes of classes
#     OTHER than its own true label — higher means more cross-class confusion.
#
#     Note: matrix inversion is O(N³). Feasible for the small citation graphs
#     (Cora ~2 700 nodes, CiteSeer ~3 300) used in this project; will be slow
#     for larger datasets.
#
#     Parameters
#     ----------
#     data : PyG Data
#         Graph with edge_index, y, train_mask, num_nodes. May be on any device;
#         computation is performed on CPU.
#     cfg : dict
#         cfg["dataset"]["pagerank_prob"] — probability of following an edge
#         (restart probability = 1 − pagerank_prob). Default 0.85.
#
#     Returns
#     -------
#     totoro_values : FloatTensor, shape [num_nodes]
#     """
#     import math as _math
#     import torch.nn.functional as F
#     from torch_geometric.utils import to_dense_adj
#
#     pagerank_prob = cfg["dataset"].get("pagerank_prob", 0.85)
#     alpha         = 1.0 - pagerank_prob   # restart probability
#
#     N           = data.num_nodes
#     num_classes = int(data.y.max().item()) + 1
#     y_cpu       = data.y.cpu()
#     train_cpu   = data.train_mask.cpu()
#
#     # Symmetrically normalised adjacency with self-loops (on CPU)
#     A     = to_dense_adj(data.edge_index, max_num_nodes=N).squeeze(0).cpu().float()
#     A_hat = A + torch.eye(N)
#     d_inv_sqrt = A_hat.sum(dim=1).pow(-0.5)
#     A_hat = d_inv_sqrt.unsqueeze(1) * A_hat * d_inv_sqrt.unsqueeze(0)
#
#     # Closed-form PPR matrix  [N, N]
#     Pi = alpha * torch.inverse(torch.eye(N) - (1.0 - alpha) * A_hat)
#
#     # Class-level GPR: mean PPR row over training nodes in each class  [N, num_classes]
#     gpr_cols = []
#     for c in range(num_classes):
#         idx = ((y_cpu == c) & train_cpu).nonzero(as_tuple=True)[0]
#         gpr_cols.append(Pi[idx].mean(dim=0) if len(idx) > 0 else torch.zeros(N))
#     gpr = torch.stack(gpr_cols, dim=1)   # [N, num_classes]
#
#     # Cross-class influence: total GPR minus same-class GPR  [N, num_classes]
#     gpr_other = gpr.sum(dim=1, keepdim=True) - gpr
#     influence  = Pi @ gpr_other                        # [N, num_classes]
#
#     # Extract each node's own-class component
#     label_oh      = F.one_hot(y_cpu, num_classes).float()   # [N, num_classes]
#     totoro_values = (influence * label_oh).sum(dim=1)        # [N]
#
#     return totoro_values
#
#
# def get_group2_signal_data(data, totoro_values, group_labels, test_deg, test_het, k: int = 2):
#     """For each Group 2 (High AMP, No DMP) test node, collect its degree,
#     heterogeneity score, and mean Totoro score of same-class and diff-class
#     k-hop training neighbours.
#
#     Parameters
#     ----------
#     data : PyG Data
#     totoro_values : FloatTensor [num_nodes]
#     group_labels : int numpy array [num_test_nodes]  (values 0–3)
#     test_deg : LongTensor [num_test_nodes]
#     test_het : FloatTensor or numpy array [num_test_nodes]
#         Per-test-node neighbour heterogeneity ratio (continuous AMP score).
#     k : int — neighbourhood radius in hops
#
#     Returns
#     -------
#     dict with float64 numpy arrays, one entry per Group 2 test node:
#         'degree'       : node degree
#         'het'          : heterogeneity ratio (continuous AMP score)
#         'same_totoro'  : mean Totoro of same-class training neighbours
#                          (NaN when no same-class training neighbour within k hops)
#         'diff_totoro'  : mean Totoro of diff-class training neighbours
#                          (NaN when no diff-class training neighbour within k hops)
#     """
#     from torch_geometric.utils import k_hop_subgraph
#
#     y_cpu      = data.y.cpu()
#     train_cpu  = data.train_mask.cpu()
#     tot_cpu    = totoro_values.cpu()
#     test_nodes = data.test_mask.nonzero(as_tuple=True)[0].tolist()
#     test_deg_np = test_deg.numpy() if torch.is_tensor(test_deg) else np.asarray(test_deg, dtype=int)
#     test_het_np = test_het.cpu().numpy() if torch.is_tensor(test_het) else np.asarray(test_het, dtype=float)
#
#     degrees, hets, same_tots, diff_tots = [], [], [], []
#
#     for i, node in enumerate(test_nodes):
#         if int(group_labels[i]) != 2:
#             continue
#
#         node_label = int(y_cpu[node])
#         nbs, _, _, _ = k_hop_subgraph(
#             node_idx=int(node), num_hops=k,
#             edge_index=data.edge_index, num_nodes=data.num_nodes,
#         )
#         nbs_set    = set(nbs.cpu().tolist()) - {node}
#         same_train = [n for n in nbs_set if train_cpu[n] and int(y_cpu[n]) == node_label]
#         diff_train = [n for n in nbs_set if train_cpu[n] and int(y_cpu[n]) != node_label]
#
#         degrees.append(int(test_deg_np[i]))
#         hets.append(float(test_het_np[i]))
#         same_tots.append(float(tot_cpu[same_train].mean()) if same_train else np.nan)
#         diff_tots.append(float(tot_cpu[diff_train].mean()) if diff_train else np.nan)
#
#     return {
#         'degree':      np.array(degrees,   dtype=float),
#         'het':         np.array(hets,      dtype=float),
#         'same_totoro': np.array(same_tots, dtype=float),
#         'diff_totoro': np.array(diff_tots, dtype=float),
#     }
#
#
# def get_renode_weight(data, cfg, totoro_values=None) -> torch.Tensor:
#     """Compute per-training-node ReNode loss weights from Totoro scores.
#
#     Training nodes are ranked by Totoro score (ascending — least cross-class
#     influence first). Weights are assigned via cosine annealing so that nodes
#     receiving cleaner label signal are up-weighted during training.
#
#     Based on ReNode (Liu et al., 2021).
#
#     Parameters
#     ----------
#     data : PyG Data
#     cfg : dict
#         cfg["dataset"]["rn_base_weight"]  — additive base (default 0.5)
#         cfg["dataset"]["rn_scale_weight"] — cosine amplitude (default 1.0)
#     totoro_values : FloatTensor [num_nodes] or None
#         Pre-computed Totoro scores; computed here if not provided.
#
#     Returns
#     -------
#     rn_weight : FloatTensor, shape [num_nodes]
#         Non-zero only for training nodes.
#     """
#     import math as _math
#
#     if totoro_values is None:
#         totoro_values = get_totoro_values(data, cfg)
#
#     base_w     = cfg["dataset"].get("rn_base_weight",  0.5)
#     scale_w    = cfg["dataset"].get("rn_scale_weight", 1.0)
#     N          = data.num_nodes
#     train_size = int(data.train_mask.sum().item())
#
#     # Rank all nodes by Totoro score ascending
#     sorted_nodes = sorted(range(N), key=lambda i: totoro_values[i].item())
#     id2rank      = {node_id: rank for rank, node_id in enumerate(sorted_nodes)}
#     totoro_rank  = [id2rank[i] for i in range(N)]
#
#     # Cosine annealing: lower rank (cleaner) → higher weight
#     rn_weight = torch.tensor(
#         [base_w + 0.5 * scale_w *
#          (1 + _math.cos(r * _math.pi / max(train_size - 1, 1)))
#          for r in totoro_rank],
#         dtype=torch.float,
#     )
#     rn_weight = rn_weight * data.train_mask.cpu().float()
#     return rn_weight



# ── Unused functions ──────────────────────────────────────────────────────────

# def index_to_adj(
#     x, edge_index, add_self_loop=False, remove_self_loop=False, sparse=False
# ):
#     from torch_geometric.utils import to_dense_adj
#
#     assert not (add_self_loop == True and remove_self_loop == True)
#     num_nodes = len(x)
#     adj = to_dense_adj(edge_index, max_num_nodes=num_nodes)[0].bool()
#     if add_self_loop:
#         adj.fill_diagonal_(True)
#     if remove_self_loop:
#         adj.fill_diagonal_(False)
#     if sparse:
#         adj = adj.to_sparse()
#     return adj
# def compute_hops_to_nearest_labeled_nodes(data, train_mask):
#     from torch_geometric.utils import k_hop_subgraph, mask_to_index
#
#     y = data.y.cpu().numpy()
#     label_idx = mask_to_index(train_mask).cpu().numpy()
#     num_hops = np.zeros(data.num_nodes).astype(int)
#     for node in tqdm(
#         range(data.num_nodes), desc="Computing hops to nearest labeled node"
#     ):
#         node_label = y[node]
#         num_hop = 0
#         while True:
#             nbs, _, _, _ = k_hop_subgraph(
#                 node, num_hop, data.edge_index, num_nodes=data.num_nodes
#             )
#             labeled_nbs = set(nbs.cpu().numpy()).intersection(set(label_idx))
#             if len(labeled_nbs) > 0 or num_hop >= 10:
#                 ngb_labels = y[list(labeled_nbs)]
#                 label_correct = node_label in ngb_labels
#                 if label_correct or num_hop >= 10:
#                     # print (f'Node {node} label {node_label} hop {num_hop} n_ngb {len(nbs)} labeled_ngb {labeled_nbs} ngb_label {ngb_labels} isin {isin}')
#                     break
#             num_hop += 1
#         num_hops[node] = num_hop
#     return num_hops
