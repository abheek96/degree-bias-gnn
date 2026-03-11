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


def get_khop_degree(data, k: int = 2, exclusive: bool = False) -> torch.Tensor:
    """Compute the k-hop degree of every node.

    The k-hop degree of node v is the number of distinct nodes reachable
    from v within at most k hops (shortest-path distance ≤ k), excluding v
    itself.  For k=1 this equals the standard degree.

    Parameters
    ----------
    data      : torch_geometric.data.Data
    k         : neighbourhood radius in hops (default 2)
    exclusive : if True, count only nodes at distance *exactly* k
                (the new shell added at hop k, not the full k-hop ball).
                Default False returns the cumulative k-hop neighbourhood.

    Returns
    -------
    khop_deg : LongTensor, shape [num_nodes]

    Notes
    -----
    Cumulative (exclusive=False) is the standard definition used in GNN
    analysis: it represents the full receptive field of a k-layer GNN.
    Exclusive (exclusive=True) isolates the shell at distance exactly k,
    which is useful for studying how much *new* information each extra layer
    contributes.

    Implementation uses boolean adjacency-matrix powers so it is exact and
    avoids walk/backtracking artefacts: reach_k = (A | A² | … | Aᵏ) with
    the diagonal zeroed out.  Feasible for the citation graphs used here
    (Cora ~2 700, CiteSeer ~3 300 nodes); avoid on large graphs.
    """
    from torch_geometric.utils import to_dense_adj

    N   = data.num_nodes
    A   = to_dense_adj(data.edge_index, max_num_nodes=N).squeeze(0).bool()
    A.fill_diagonal_(False)

    reach = A.clone()           # cumulative reachability up to current hop
    power = A.float()           # Aʰ as float for matrix multiply
    A_f   = A.float()

    for _ in range(k - 1):
        power = power @ A_f
        reach = reach | power.bool()

    reach.fill_diagonal_(False)

    if not exclusive:
        return reach.sum(dim=1).long()

    # Exclusive: nodes at distance exactly k = cumulative_k \ cumulative_(k-1)
    if k == 1:
        return reach.sum(dim=1).long()

    reach_prev = A.clone()
    power_prev = A.float()
    for _ in range(k - 2):
        power_prev = power_prev @ A_f
        reach_prev = reach_prev | power_prev.bool()
    reach_prev.fill_diagonal_(False)

    return (reach & ~reach_prev).sum(dim=1).long()


def get_node_purity(data, k: int = 1) -> torch.Tensor:
    """Neighborhood purity for every node at receptive field radius k.

    purity(v) = |{u ∈ N_k(v) : label[u] == label[v]}| / |N_k(v)|

    where N_k(v) is the cumulative k-hop neighborhood of v (all nodes
    reachable within k hops, excluding v itself).  Nodes whose k-hop
    neighborhood is empty receive NaN.

    Parameters
    ----------
    data : torch_geometric.data.Data  (must have .y labels)
    k    : neighbourhood radius in hops

    Returns
    -------
    purity : FloatTensor, shape [num_nodes]
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

    reach_cpu        = reach.cpu()
    total_counts     = reach_cpu.sum(dim=1).float()              # |N_k(v)|
    same_cls_counts  = (reach_cpu & label_match).sum(dim=1).float()

    purity          = torch.full((N,), float("nan"))
    valid           = total_counts > 0
    purity[valid]   = same_cls_counts[valid] / total_counts[valid]

    return purity


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
