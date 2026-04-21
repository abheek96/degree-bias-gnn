"""
influence.py — Influence analysis for GNN test nodes.

For a trained GNN, computes the exact influence distribution of every node
on a target node x (Definition 3.1), then compares the individual influence
scores of same-class vs. different-class training nodes within x's receptive
field.

Nodes can be selected either by degree (one node per requested degree value)
or by explicit node index.
"""

import logging

import torch
from torch import Tensor
from torch.autograd.functional import jacobian as torch_jacobian
from torch_geometric.utils import degree as graph_degree
from torch_geometric.utils import k_hop_subgraph
from tqdm import tqdm

log = logging.getLogger(__name__)


# ── core influence computation ─────────────────────────────────────────────────

def influence_distribution(model, data, node_x: int, k_hops: int,
                            *, vectorize: bool = True) -> Tensor:
    """Pair-wise Jacobian-based influence scores for node x over all nodes y.

    Computes the **L1 norm** of the Jacobian of node x's final-layer output
    with respect to each other node's input features, evaluated on the
    **k-hop induced sub-graph** centred at node x. Results are scattered
    back to the original global node index space so the returned tensor has
    length ``data.num_nodes``; nodes outside the k-hop subgraph have
    influence 0 (the true value — no message-passing path exists).

    I(x, y) = Σ_{i, f} |∂h_x^(k)[i] / ∂h_y^(0)[f]|

    Evaluating on the induced subgraph instead of the full graph makes this
    tractable for large graphs: the Jacobian is of shape [d_k, |V_sub|, d_0]
    rather than [d_k, N, d_0]. For GCN-style local aggregation the two
    formulations are mathematically equivalent at the root node.

    Compatibility with this repository
    ----------------------------------
    CC filtering is handled transparently — by the time this function is
    called, ``data`` has already been reindexed by ``LargestConnectedComponents``
    in ``dataset.py``, so ``node_x``, ``data.edge_index``, ``data.x`` and
    ``data.num_nodes`` are all in the filtered node index space.

    Parameters
    ----------
    model     : trained GNN, called as ``model(x, edge_index) → [N, d_k]``
    data      : PyG Data object
    node_x    : global (post-CC) index of the node to compute influence for
    k_hops    : receptive field radius (= number of GCN message-passing layers)
    vectorize : vectorise ``torch.autograd.functional.jacobian`` (faster)

    Returns
    -------
    influence_full : FloatTensor [data.num_nodes] on data.x's device.
        Raw L1-norm Jacobian sums; NOT normalised to sum to 1. Downstream
        code derives same-/diff-class fractions as ratios, so the absolute
        magnitude does not affect interpretation.
    """
    model.eval()
    edge_index = data.edge_index
    x          = data.x

    # Induced k-hop sub-graph centred at node_x, with local re-labelling
    k_hop_nodes, sub_edge_index, mapping, _ = k_hop_subgraph(
        node_x, k_hops, edge_index, relabel_nodes=True,
        num_nodes=data.num_nodes,
    )
    root_pos = int(mapping[0].item())

    device         = x.device
    sub_x          = x[k_hop_nodes].to(device)
    sub_edge_index = sub_edge_index.to(device)

    def _forward(X: Tensor) -> Tensor:
        return model(X, sub_edge_index)[root_pos]   # [d_k]

    # J[i, y_local, f] = ∂h_x^(k)[i] / ∂h_{y_local}^(0)[f]
    # shape: [d_k, |V_sub|, d_0]
    jac           = torch_jacobian(_forward, sub_x, vectorize=vectorize)
    influence_sub = jac.abs().sum(dim=(0, 2))   # [|V_sub|]

    influence_full = torch.zeros(
        data.num_nodes, dtype=influence_sub.dtype, device=device,
    )
    influence_full[k_hop_nodes] = influence_sub
    return influence_full


def _khop_neighbors(edge_index, node_x: int, k: int, num_nodes: int) -> set:
    """BFS to collect all nodes within k hops of node_x (node_x excluded)."""
    src, dst = edge_index.cpu()
    adj = [[] for _ in range(num_nodes)]
    for u, v in zip(src.tolist(), dst.tolist()):
        adj[v].append(u)   # reverse: BFS follows incoming edges, matching source_to_target aggregation

    visited  = {node_x}
    frontier = {node_x}
    for _ in range(k):
        nxt = set()
        for u in frontier:
            for v in adj[u]:
                if v not in visited:
                    visited.add(v)
                    nxt.add(v)
        frontier = nxt

    visited.discard(node_x)
    return visited


def _khop_distances(edge_index, node_x: int, k: int, num_nodes: int) -> dict:
    """BFS returning the exact hop distance to every node within k hops.

    Returns
    -------
    dist : dict {node_idx: hop_distance}  (node_x itself excluded)
    """
    src, dst = edge_index.cpu()
    adj = [[] for _ in range(num_nodes)]
    for u, v in zip(src.tolist(), dst.tolist()):
        adj[v].append(u)   # incoming edges, matching source_to_target aggregation

    dist     = {}
    visited  = {node_x}
    frontier = {node_x}
    for hop in range(1, k + 1):
        nxt = set()
        for u in frontier:
            for v in adj[u]:
                if v not in visited:
                    visited.add(v)
                    nxt.add(v)
                    dist[v] = hop
        frontier = nxt

    return dist


def k_hop_subsets_rough(node_idx: int, num_hops: int, edge_index: Tensor,
                         num_nodes: int) -> list:
    """Return *rough* (possibly overlapping) k-hop node subsets.

    Thin wrapper around the BFS used by
    :pyfunc:`torch_geometric.utils.k_hop_subgraph`, but returns **all**
    intermediate hop subsets rather than the full union only.

    Returns
    -------
    list[Tensor]
        ``[H_0, H_1, …, H_k]`` where ``H_0 = [node_idx]`` and ``H_i`` (i>0)
        contains all nodes exactly i hops away in the *expanded*
        neighbourhood (overlaps with earlier hops are **not** removed).
    """
    col, row = edge_index

    node_mask = row.new_empty(num_nodes, dtype=torch.bool)
    edge_mask = row.new_empty(row.size(0), dtype=torch.bool)

    node_idx_ = torch.tensor([node_idx], device=row.device)

    subsets = [node_idx_]
    for _ in range(num_hops):
        node_mask.zero_()
        node_mask[subsets[-1]] = True
        torch.index_select(node_mask, 0, row, out=edge_mask)
        subsets.append(col[edge_mask])

    return subsets


def k_hop_subsets_exact(node_idx: int, num_hops: int, edge_index: Tensor,
                         num_nodes: int, device) -> list:
    """Return **disjoint** k-hop subsets ``[S_0, S_1, …, S_k]``.

    Refines :pyfunc:`k_hop_subsets_rough` by removing nodes that have already
    appeared in previous hops, so each subset contains nodes *exactly* i hops
    away from the seed. Used by ``jacobian_l1_agg_per_hop`` to bucket
    influence by hop distance.
    """
    rough_subsets = k_hop_subsets_rough(node_idx, num_hops, edge_index,
                                        num_nodes)

    exact_subsets = [rough_subsets[0].tolist()]
    visited       = set(exact_subsets[0])

    for hop_subset in rough_subsets[1:]:
        fresh = set(hop_subset.tolist()) - visited
        visited |= fresh
        exact_subsets.append(list(fresh))

    return [
        torch.tensor(s, device=device, dtype=edge_index.dtype)
        for s in exact_subsets
    ]


def jacobian_l1_agg_per_hop(model, data, max_hops: int, node_idx: int,
                             *, vectorize: bool = True) -> Tensor:
    """Aggregate Jacobian L1 influence **per hop** for ``node_idx``.

    Returns a vector ``[I_0, I_1, …, I_k]`` where ``I_i`` is the *total*
    influence exerted on ``node_idx`` by nodes at exactly hop distance ``i``.
    ``I_0`` is the self-influence (jacobian w.r.t. node_idx's own features).

    Uses ``influence_distribution`` (the Jacobian-L1 implementation) for the
    pairwise scores, then buckets them by hop distance via
    ``k_hop_subsets_exact``.
    """
    num_nodes  = int(data.num_nodes)
    influence  = influence_distribution(
        model, data, node_idx, max_hops, vectorize=vectorize,
    )
    hop_subsets = k_hop_subsets_exact(
        node_idx, max_hops, data.edge_index, num_nodes, influence.device,
    )
    per_hop = [influence[s].sum() for s in hop_subsets]
    return torch.stack(per_hop)


def total_influence(
    model, data, max_hops: int,
    num_samples: int | None = None,
    normalize: bool = True,
    average: bool = True,
    device="cpu",
    vectorize: bool = True,
):
    """Jacobian-based influence aggregates across multiple seed nodes.

    Introduced in "Towards Quantifying Long-Range Interactions in Graph
    Machine Learning" (https://arxiv.org/abs/2503.09008). For every sampled
    node v, computes the per-hop Jacobian-L1 influence vector
    (I_0, I_1, …, I_k), then optionally averages over seeds and normalises
    by I_0.

    Parameters
    ----------
    model       : GNN with forward ``model(x, edge_index)``.
    data        : PyG Data with ``x`` and ``edge_index``.
    max_hops    : maximum hop distance k.
    num_samples : number of random seed nodes (default: all nodes).
    normalize   : divide per-hop influence by I_0 (self-influence).
    average     : if True, mean over seeds — shape [k+1]; else full matrix
                  of shape [num_samples, k+1].
    device      : forwarded to downstream calls (derived from data tensors
                  in this implementation).
    vectorize   : forwarded to ``torch.autograd.functional.jacobian``.

    Returns
    -------
    avg_influence : Tensor, shape [k+1] or [num_samples, k+1].
    R             : float, influence-weighted receptive-field breadth.
    """
    num_nodes   = int(data.num_nodes)
    num_samples = num_nodes if num_samples is None else num_samples
    nodes       = torch.randperm(num_nodes)[:num_samples].tolist()

    influence_all_nodes = [
        jacobian_l1_agg_per_hop(model, data, max_hops, n, vectorize=vectorize)
        for n in tqdm(nodes, desc="Influence")
    ]
    allnodes = torch.vstack(influence_all_nodes).detach().cpu()

    if average:
        avg_influence = avg_total_influence(allnodes, normalize=normalize)
    else:
        avg_influence = allnodes

    R = influence_weighted_receptive_field(allnodes)

    return avg_influence, R


# ── node selection + per-node analysis ────────────────────────────────────────

def _analyse_node(model, data, pred, node_x: int, k_hops: int,
                  train_set: set, y, all_deg) -> dict | None:
    """Run influence analysis for a single node.  Returns None if skipped.

    Shared logic called by both the degree-based and index-based entry points.
    """
    N        = data.num_nodes
    degree   = int(all_deg[node_x].item())
    true_lbl = int(y[node_x].item())
    pred_lbl = int(pred[node_x].item())

    hop_dist       = _khop_distances(data.edge_index, node_x, k_hops, N)
    train_in_field = [t for t in hop_dist if t in train_set]

    if not train_in_field:
        log.warning(
            "influence: node %d (degree=%d) has no training nodes within "
            "%d-hop receptive field — skipping",
            node_x, degree, k_hops,
        )
        return None

    same_class = [t for t in train_in_field if int(y[t].item()) == true_lbl]
    diff_class = [t for t in train_in_field if int(y[t].item()) != true_lbl]

    log.info("influence: node %d  degree=%d  same_train=%d  diff_train=%d",
             node_x, degree, len(same_class), len(diff_class))

    I_x = influence_distribution(model, data, node_x, k_hops)

    same_set        = set(same_class)
    diff_set        = set(diff_class)
    neighbor_detail = []
    for nb, hop in hop_dist.items():
        if nb in same_set:
            nb_type = "same_train"
        elif nb in diff_set:
            nb_type = "diff_train"
        else:
            nb_type = "non_train"
        neighbor_detail.append({
            "node_idx":    nb,
            "degree":      int(all_deg[nb].item()),
            "hop_distance": hop,
            "influence":   float(I_x[nb].item()),
            "type":        nb_type,
        })
    neighbor_detail.sort(key=lambda d: d["influence"], reverse=True)

    same_inf  = float(I_x[same_class].sum()) if same_class else 0.0
    diff_inf  = float(I_x[diff_class].sum()) if diff_class else 0.0
    total_inf = same_inf + diff_inf

    norm_same = same_inf / total_inf if total_inf > 0 else 0.0
    norm_diff = diff_inf / total_inf if total_inf > 0 else 0.0

    log.info("  raw:  same=%.4e  diff=%.4e  total_train=%.4e",
             same_inf, diff_inf, total_inf)
    log.info("  norm: same=%.4f  diff=%.4f  (fraction of training-node influence)",
             norm_same, norm_diff)
    for t in same_class:
        raw = float(I_x[t].item())
        log.info("    same_train node %-5d  deg=%-4d  hop=%-2d  raw=%.4e  norm=%.4f",
                 t, int(all_deg[t].item()), hop_dist[t], raw,
                 raw / total_inf if total_inf > 0 else 0.0)
    for t in diff_class:
        raw = float(I_x[t].item())
        log.info("    diff_train node %-5d  deg=%-4d  hop=%-2d  raw=%.4e  norm=%.4f",
                 t, int(all_deg[t].item()), hop_dist[t], raw,
                 raw / total_inf if total_inf > 0 else 0.0)

    for nb in neighbor_detail:
        nb["influence_norm"] = nb["influence"] / total_inf if total_inf > 0 else 0.0

    return {
        "node_idx":                  node_x,
        "degree":                    degree,
        "true_label":                true_lbl,
        "pred_label":                pred_lbl,
        "same_class_influence":      same_inf,
        "diff_class_influence":      diff_inf,
        "same_class_influence_norm": norm_same,
        "diff_class_influence_norm": norm_diff,
        "total_train_influence":     total_inf,
        "n_same_train":              len(same_class),
        "n_diff_train":              len(diff_class),
        "neighbors":                 neighbor_detail,
    }


def compute_influence_analysis(model, data, pred, k_hops: int,
                                target_degrees: list = None,
                                target_nodes: list = None) -> list:
    """Analyse training-node influence for selected test nodes.

    Nodes can be specified by degree, by explicit node index, or both.
    Results from both lists are merged and deduplicated (same node_idx appears
    at most once).

    Degree-based selection (``target_degrees``)
    -------------------------------------------
    Selects one test node per requested degree value:
      - 0 candidates  → log warning, skip.
      - 1 candidate   → use it directly.
      - >1 candidates → prefer misclassified; break ties by picking the first.

    Index-based selection (``target_nodes``)
    ----------------------------------------
    Analyses each specified global node index directly.  The node does not
    need to be in the test set — any node in the graph is accepted.  A warning
    is logged if the node is not a test node.

    Guardrail (both modes)
    ----------------------
    If the selected node has no training nodes within its k-hop receptive
    field, it is skipped with a warning.

    For each selected node x:
      1. Identify all nodes in the k-hop receptive field.
      2. Filter to training nodes; split into same-class / diff-class.
      3. Compute the exact influence distribution I_x.
      4. Return per-training-neighbor detail and aggregate scores.

    Parameters
    ----------
    model          : trained GNN
    data           : PyG Data object
    pred           : LongTensor [N], model predictions for all nodes
    k_hops         : receptive field radius (= number of GCNConv layers)
    target_degrees : list[int] — exact 1-hop degree values, e.g. [3, 22]
    target_nodes   : list[int] — global node indices, e.g. [1362, 42]

    Returns
    -------
    list of dicts, one per selected node, with keys:
        node_idx, degree, true_label, pred_label,
        same_class_influence, diff_class_influence,
        same_class_influence_norm, diff_class_influence_norm,
        total_train_influence, n_same_train, n_diff_train,
        neighbors : list of dicts {node_idx, influence, influence_norm, type}
            type ∈ {"same_train", "diff_train", "non_train"},
            sorted descending by influence score.
    """
    if not target_degrees and not target_nodes:
        log.warning("influence: neither target_degrees nor target_nodes provided — nothing to do")
        return []

    N         = data.num_nodes
    y         = data.y.cpu()
    pred      = pred.cpu()
    train_idx = data.train_mask.cpu().nonzero(as_tuple=False).view(-1).tolist()
    test_idx  = data.test_mask.cpu().nonzero(as_tuple=False).view(-1).tolist()

    all_deg  = graph_degree(data.edge_index[1], N).cpu()
    test_deg = all_deg[data.test_mask.cpu()]

    test_true = y[data.test_mask.cpu()]
    test_pred = pred[data.test_mask.cpu()]

    train_set = set(train_idx)
    test_set  = set(test_idx)
    model.eval()

    seen     = set()   # deduplicate by node_idx
    results  = []

    def _add(node_x):
        if node_x in seen:
            return
        seen.add(node_x)
        r = _analyse_node(model, data, pred, node_x, k_hops, train_set, y, all_deg)
        if r is not None:
            results.append(r)

    # ── degree-based selection ────────────────────────────────────────────────
    for deg_val in (target_degrees or []):
        candidates = [i for i in range(len(test_idx))
                      if test_deg[i].item() == deg_val]

        if not candidates:
            log.warning("influence: no test nodes found with degree=%d — skipping", deg_val)
            continue

        if len(candidates) == 1:
            local_i = candidates[0]
            log.info("influence: degree=%d — 1 candidate: node %d (%s)",
                     deg_val, test_idx[local_i],
                     "misclassified" if test_pred[local_i] != test_true[local_i] else "correct")
        else:
            log.info("influence: degree=%d — %d candidates:", deg_val, len(candidates))
            for i in candidates:
                status = "misclassified" if test_pred[i] != test_true[i] else "correct"
                log.info("  node_idx=%-5d  true=%-3d  pred=%-3d  %s",
                         test_idx[i], int(test_true[i].item()),
                         int(test_pred[i].item()), status)
            misclassified = [i for i in candidates if test_pred[i] != test_true[i]]
            local_i = misclassified[0] if misclassified else candidates[0]
            log.info("  → selected node %d", test_idx[local_i])

        _add(test_idx[local_i])

    # ── index-based selection ─────────────────────────────────────────────────
    for node_x in (target_nodes or []):
        if node_x < 0 or node_x >= N:
            log.warning("influence: node_idx=%d out of range [0, %d) — skipping",
                        node_x, N)
            continue
        if node_x not in test_set:
            log.warning("influence: node_idx=%d is not a test node (will analyse anyway)",
                        node_x)
        _add(node_x)

    return results


# ── influence disparity over all test nodes ────────────────────────────────────

def compute_influence_disparity_all(model, data, pred, k_hops: int) -> list:
    """Influence disparity for every test node that has ≥1 training neighbor.

    For each qualifying test node computes:
        disparity = same_class_influence_norm − diff_class_influence_norm

    where both scores are normalised by the total influence attributable to
    training nodes in the k-hop receptive field (so they sum to 1).

    Interpretation
    --------------
    +1  : all training-node influence comes from same-class nodes.
    -1  : all training-node influence comes from diff-class nodes.
     0  : equal balance between the two.

    Test nodes with no training nodes in their receptive field are excluded
    (no training signal reachable).

    Cost: one full Jacobian computation per qualifying test node.
    Progress is logged every 100 nodes.

    Returns
    -------
    list of dicts with keys:
        node_idx, degree, correct (bool), disparity,
        same_inf_norm, diff_inf_norm, n_same_train, n_diff_train
    """
    N         = data.num_nodes
    y         = data.y.cpu()
    pred_cpu  = pred.cpu()
    train_idx = data.train_mask.cpu().nonzero(as_tuple=False).view(-1).tolist()
    test_idx  = data.test_mask.cpu().nonzero(as_tuple=False).view(-1).tolist()
    all_deg   = graph_degree(data.edge_index[1], N).cpu()
    train_set = set(train_idx)

    model.eval()
    results  = []
    n_test   = len(test_idx)
    n_skip   = 0

    for i, node_x in enumerate(test_idx):
        if i % 100 == 0:
            log.info("influence disparity: %d / %d test nodes processed", i, n_test)

        neighbors      = _khop_neighbors(data.edge_index, node_x, k_hops, N)
        train_in_field = [t for t in neighbors if t in train_set]

        if not train_in_field:
            n_skip += 1
            continue

        true_lbl   = int(y[node_x].item())
        same_class = [t for t in train_in_field if int(y[t].item()) == true_lbl]
        diff_class = [t for t in train_in_field if int(y[t].item()) != true_lbl]

        I_x = influence_distribution(model, data, node_x, k_hops)

        same_inf  = float(I_x[same_class].sum()) if same_class else 0.0
        diff_inf  = float(I_x[diff_class].sum()) if diff_class else 0.0
        total_inf = same_inf + diff_inf

        if total_inf == 0:
            n_skip += 1
            continue

        norm_same = same_inf / total_inf
        norm_diff = diff_inf / total_inf

        results.append({
            "node_idx":      node_x,
            "degree":        int(all_deg[node_x].item()),
            "correct":       int(pred_cpu[node_x].item()) == true_lbl,
            "disparity":     norm_same - norm_diff,
            "same_inf_norm": norm_same,
            "diff_inf_norm": norm_diff,
            "n_same_train":  len(same_class),
            "n_diff_train":  len(diff_class),
        })

    log.info(
        "influence disparity: done — %d analysed, %d skipped (no training neighbors)",
        len(results), n_skip,
    )
    return results
