"""
influence.py — Influence analysis for high-degree misclassified test nodes.

For a trained GNN, computes the exact influence distribution of every node
on a target node x (Definition 3.1), then compares the aggregate influence
of same-class vs. different-class training nodes within x's receptive field.
"""

import torch
from collections import deque
from torch.autograd.functional import jacobian as torch_jacobian
from torch_geometric.utils import degree as graph_degree


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
                                degree_percentile: int = 75,
                                top_n: int = 20,
                                target_degrees: list = None) -> list:
    """Select test nodes and analyse same-class vs diff-class training influence.

    Node selection:
      - If ``target_degrees`` is provided, selects up to ``top_n`` test nodes
        per listed degree value, prioritising misclassified ones.
      - Otherwise, falls back to the top ``top_n`` misclassified test nodes
        whose degree is at or above ``degree_percentile``.

    For each selected node x:
      1. Identify training nodes within its k-hop receptive field.
      2. Split them into same-class and different-class w.r.t. x's true label.
      3. Compute the exact influence distribution I_x.
      4. Sum influence scores over each group.

    Parameters
    ----------
    model             : trained GNN
    data              : PyG Data object
    pred              : LongTensor [N], model predictions for all nodes
    k_hops            : receptive field radius (= number of model layers)
    degree_percentile : percentile threshold for fallback selection (default 75)
    top_n             : max nodes to analyse; when target_degrees is set this
                        is the cap *per degree value*
    target_degrees    : list of exact 1-hop degree values to investigate,
                        e.g. [3, 4, 5].  None → use percentile-based fallback.

    Returns
    -------
    list of dicts, one per selected node, with keys:
        node_idx, degree, true_label, pred_label,
        same_class_influence, diff_class_influence,
        n_same_train, n_diff_train
    """
    import numpy as np

    N         = data.num_nodes
    y         = data.y.cpu()
    pred      = pred.cpu()
    train_idx = data.train_mask.cpu().nonzero(as_tuple=False).view(-1).tolist()
    test_idx  = data.test_mask.cpu().nonzero(as_tuple=False).view(-1).tolist()

    all_deg  = graph_degree(data.edge_index[1], N).cpu()
    test_deg = all_deg[data.test_mask.cpu()]

    test_true = y[data.test_mask.cpu()]
    test_pred = pred[data.test_mask.cpu()]

    if target_degrees is not None:
        # Select up to top_n nodes per requested degree, misclassified first
        selected_local = []
        for deg_val in target_degrees:
            at_deg = [i for i in range(len(test_idx))
                      if test_deg[i].item() == deg_val]
            # misclassified first, then correctly classified
            at_deg.sort(key=lambda i: (test_pred[i] == test_true[i]).item())
            selected_local.extend(at_deg if top_n is None else at_deg[:top_n])
    else:
        # Fallback: high-degree misclassified nodes
        misc_local    = (test_pred != test_true).nonzero(as_tuple=False).view(-1).tolist()
        if not misc_local:
            return []
        deg_threshold = float(np.percentile(test_deg.numpy(), degree_percentile))
        candidates    = [i for i in misc_local if test_deg[i].item() >= deg_threshold]
        candidates.sort(key=lambda i: test_deg[i].item(), reverse=True)
        selected_local = candidates if top_n is None else candidates[:top_n]

    train_set = set(train_idx)
    results   = []
    model.eval()

    for local_i in selected_local:
        node_x   = test_idx[local_i]
        degree   = int(all_deg[node_x].item())
        true_lbl = int(y[node_x].item())
        pred_lbl = int(pred[node_x].item())

        neighbors       = _khop_neighbors(data.edge_index, node_x, k_hops, N)
        train_in_field  = [t for t in neighbors if t in train_set]
        same_class      = [t for t in train_in_field if int(y[t].item()) == true_lbl]
        diff_class      = [t for t in train_in_field if int(y[t].item()) != true_lbl]

        I_x = influence_distribution(model, data, node_x)

        results.append({
            "node_idx":             node_x,
            "degree":               degree,
            "true_label":           true_lbl,
            "pred_label":           pred_lbl,
            "same_class_influence": float(I_x[same_class].sum()) if same_class else 0.0,
            "diff_class_influence": float(I_x[diff_class].sum()) if diff_class else 0.0,
            "n_same_train":         len(same_class),
            "n_diff_train":         len(diff_class),
        })

    return results
