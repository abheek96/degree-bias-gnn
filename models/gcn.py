import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from torch_geometric.nn import GCNConv
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.utils import degree as pyg_degree


def inspect_node_aggregation(node_idx, edge_index, train_mask, y,
                              pred=None, edge_weight=None, deg=None):
    """Print a table of all neighbors that node_idx aggregates from in GCNConv.

    Computes normalized edge weights via gcn_norm (same as GCNConv with default
    settings: self-loops added, symmetric normalization) if not provided.

    Parameters
    ----------
    node_idx   : int — the target node to inspect.
    edge_index : LongTensor (2, E)
    train_mask : BoolTensor (N,)
    y          : LongTensor (N,) of class labels
    pred       : LongTensor (N,) of predicted labels or None — if provided,
                 adds a correct_pred column for each neighbor.
    edge_weight: FloatTensor (E,) or None — if None, gcn_norm is called.
    deg        : LongTensor (N,) or None — if None, computed from edge_index.

    Returns
    -------
    pandas.DataFrame with one row per neighbor (including self-loop).
    """
    edge_index = edge_index.cpu()
    train_mask = train_mask.cpu()
    y = y.cpu()
    num_nodes = y.size(0)

    if pred is not None:
        pred = pred.cpu()

    if deg is None:
        deg = pyg_degree(edge_index[0], num_nodes=num_nodes).long()
    else:
        deg = deg.cpu()

    if edge_weight is None:
        edge_index, edge_weight = gcn_norm(
            edge_index, edge_weight=None, num_nodes=num_nodes,
            improved=False, add_self_loops=True, flow="source_to_target",
        )
    edge_weight = edge_weight.cpu()

    col = edge_index[1]
    mask = col == node_idx
    src_nodes = edge_index[0][mask].tolist()
    weights   = edge_weight[mask].tolist()

    target_label = y[node_idx].item()
    rows = []
    for src, w in zip(src_nodes, weights):
        row = {
            "neighbor": src,
            "degree": deg[src].item(),
            "in_train_set": bool(train_mask[src].item()),
            "same_class": y[src].item() == target_label,
        }
        if pred is not None:
            row["correct_pred"] = bool((pred[src] == y[src]).item())
        row["edge_weight"] = round(w, 6)
        rows.append(row)

    df = (
        pd.DataFrame(rows)
        .sort_values(
            ["in_train_set", "same_class", "degree"],
            ascending=[False, False, True],
        )
        .reset_index(drop=True)
    )
    df.index += 1
    df.index.name = "#"

    print(f"\nAggregation neighborhood for node {node_idx}  "
          f"(class={target_label}, degree={deg[node_idx].item()}, "
          f"in_train={bool(train_mask[node_idx].item())})\n")
    print(df.to_string())
    return df


class GCN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers=2, dropout=0.5):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(GCNConv(in_dim, hidden_dim))
        for _ in range(num_layers - 2):
            self.layers.append(GCNConv(hidden_dim, hidden_dim))
        self.layers.append(nn.Linear(hidden_dim, out_dim))
        self.dropout = dropout

    def get_intermediate(self, x, edge_index, layer: int = 1):
        """Return node representations after ``layer`` GCNConv layers (no grad).

        ``layer=1`` gives h^(1) — after the first GCNConv + ReLU, before the
        linear head.  For the default num_layers=2 GCN this is the only
        message-passing layer, so layer=1 is the full aggregated representation
        before classification.

        Parameters
        ----------
        x          : input feature tensor (N, in_dim)
        edge_index : graph connectivity (2, E)
        layer      : which GCNConv output to return; must be in
                     [1, num_layers - 1]

        Returns
        -------
        FloatTensor of shape (N, hidden_dim), detached from the compute graph.
        """
        n_conv = len(self.layers) - 1   # number of GCNConv layers
        if layer < 1 or layer > n_conv:
            raise ValueError(
                f"layer must be in [1, {n_conv}] for this model, got {layer}"
            )
        self.eval()
        with torch.no_grad():
            for i, conv in enumerate(self.layers[:-1]):
                x = F.relu(conv(x, edge_index))
                if i + 1 == layer:
                    return x
        return x   # unreachable given bounds check

    def forward(self, x, edge_index):
        for layer in self.layers[:-1]:
            x = layer(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.layers[-1](x)
        return x
