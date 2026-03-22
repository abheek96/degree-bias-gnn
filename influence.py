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
from torch.autograd.functional import jacobian as torch_jacobian
from torch_geometric.utils import degree as graph_degree

log = logging.getLogger(__name__)


# ── core influence computation ─────────────────────────────────────────────────

def influence_distribution(model, data, node_x: int) -> torch.Tensor:
    """Exact influence distribution I_x(y) for node x over all nodes y.

    I(x, y)  = Σ_{i,f} |∂h_x^(k)[i] / ∂h_y^(0)[f]|
    I_x(y)   = I(x, y) / Σ_z I(x, z)

    Parameters
    ----------
    model  : trained GNN, called as model(x, edge_index) → [N, d_k]
    data   : PyG Data object
    node_x : index of the node to compute influence for

    Returns
    -------
    I_x : FloatTensor [N], normalised influence scores summing to 1.
    """
    model.eval()
    edge_index = data.edge_index

    def forward_fn(X):
        return model(X, edge_index)[node_x]   # [d_k]

    # J[i, y, f] = ∂h_x^(k)[i] / ∂h_y^(0)[f],  shape [d_k, N, d_0]
    J    = torch_jacobian(forward_fn, data.x)
    I_xy = J.abs().sum(dim=(0, 2))             # [N]
    return I_xy / I_xy.sum()


def _khop_neighbors(edge_index, node_x: int, k: int, num_nodes: int) -> set:
    """BFS to collect all nodes within k hops of node_x (node_x excluded)."""
    src, dst = edge_index.cpu()
    adj = [[] for _ in range(num_nodes)]
    for u, v in zip(src.tolist(), dst.tolist()):
        adj[u].append(v)

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

    neighbors      = _khop_neighbors(data.edge_index, node_x, k_hops, N)
    train_in_field = [t for t in neighbors if t in train_set]

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

    I_x = influence_distribution(model, data, node_x)

    same_set        = set(same_class)
    diff_set        = set(diff_class)
    neighbor_detail = []
    for nb in neighbors:
        if nb in same_set:
            nb_type = "same_train"
        elif nb in diff_set:
            nb_type = "diff_train"
        else:
            nb_type = "non_train"
        neighbor_detail.append({
            "node_idx":  nb,
            "degree":    int(all_deg[nb].item()),
            "influence": float(I_x[nb].item()),
            "type":      nb_type,
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
        log.info("    same_train node %-5d  raw=%.4e  norm=%.4f",
                 t, raw, raw / total_inf if total_inf > 0 else 0.0)
    for t in diff_class:
        raw = float(I_x[t].item())
        log.info("    diff_train node %-5d  raw=%.4e  norm=%.4f",
                 t, raw, raw / total_inf if total_inf > 0 else 0.0)

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
