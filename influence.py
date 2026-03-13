"""
influence.py — Influence analysis for degree-selected test nodes.

For a trained GNN, computes the exact influence distribution of every node
on a target node x (Definition 3.1), then compares the individual influence
scores of same-class vs. different-class training nodes within x's receptive
field.
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

def compute_influence_analysis(model, data, pred, k_hops: int,
                                target_degrees: list) -> list:
    """Select one test node per requested degree and analyse training influence.

    Node selection per degree value:
      - 0 candidates  → log warning, skip.
      - 1 candidate   → use it directly.
      - >1 candidates → log all (node_idx, correct/misclassified), then select
                        the misclassified one; if multiple misclassified pick
                        the first; if all correct pick the first.

    Guardrail: if the selected node has no training nodes (same or different
    class) within its k-hop receptive field, log a warning and skip it.

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
    k_hops         : receptive field radius (= number of model layers)
    target_degrees : list of exact 1-hop degree values to investigate,
                     e.g. [3, 4, 5]

    Returns
    -------
    list of dicts, one per selected node, with keys:
        node_idx, degree, true_label, pred_label,
        same_class_influence, diff_class_influence,
        n_same_train, n_diff_train,
        neighbors : list of dicts {node_idx, influence, type}
            type ∈ {"same_train", "diff_train"} — only training nodes,
            sorted descending by influence score.
    """
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
    results   = []
    model.eval()

    for deg_val in target_degrees:
        # All test nodes with this exact degree (local indices into test_idx)
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
            # Log all candidates
            log.info("influence: degree=%d — %d candidates:", deg_val, len(candidates))
            for i in candidates:
                status = "misclassified" if test_pred[i] != test_true[i] else "correct"
                log.info("  node_idx=%-5d  true=%-3d  pred=%-3d  %s",
                         test_idx[i], int(test_true[i].item()),
                         int(test_pred[i].item()), status)
            # Select worst: misclassified first, else first correct
            misclassified = [i for i in candidates if test_pred[i] != test_true[i]]
            local_i = misclassified[0] if misclassified else candidates[0]
            log.info("  → selected node %d", test_idx[local_i])

        node_x   = test_idx[local_i]
        degree   = int(all_deg[node_x].item())
        true_lbl = int(y[node_x].item())
        pred_lbl = int(pred[node_x].item())

        # k-hop receptive field
        neighbors      = _khop_neighbors(data.edge_index, node_x, k_hops, N)
        train_in_field = [t for t in neighbors if t in train_set]

        if not train_in_field:
            log.warning(
                "influence: node %d (degree=%d) has no training nodes within "
                "%d-hop receptive field — skipping",
                node_x, degree, k_hops,
            )
            continue

        same_class = [t for t in train_in_field if int(y[t].item()) == true_lbl]
        diff_class = [t for t in train_in_field if int(y[t].item()) != true_lbl]

        log.info("influence: node %d  degree=%d  same_train=%d  diff_train=%d",
                 node_x, degree, len(same_class), len(diff_class))

        I_x = influence_distribution(model, data, node_x)

        # Per-neighbor detail for ALL nodes in the k-hop receptive field
        same_set  = set(same_class)
        diff_set  = set(diff_class)
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
                "influence": float(I_x[nb].item()),
                "type":      nb_type,
            })
        neighbor_detail.sort(key=lambda d: d["influence"], reverse=True)

        results.append({
            "node_idx":             node_x,
            "degree":               degree,
            "true_label":           true_lbl,
            "pred_label":           pred_lbl,
            "same_class_influence": float(I_x[same_class].sum()) if same_class else 0.0,
            "diff_class_influence": float(I_x[diff_class].sum()) if diff_class else 0.0,
            "n_same_train":         len(same_class),
            "n_diff_train":         len(diff_class),
            "neighbors":            neighbor_detail,
        })

    return results
