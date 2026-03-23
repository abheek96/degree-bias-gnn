import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


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
