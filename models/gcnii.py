import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCN2Conv


class GCNII(nn.Module):
    """GCNII (Chen et al., 2020 — https://arxiv.org/abs/2007.02133).

    All GCN2Conv layers operate at ``hidden_dim``.  An input projection maps
    ``in_dim → hidden_dim`` before the first layer, and a linear head maps
    ``hidden_dim → out_dim`` at the end.
    """

    def __init__(self, in_dim, hidden_dim, out_dim, num_layers=2, dropout=0.5):
        super().__init__()
        self.input_proj = nn.Linear(in_dim, hidden_dim)
        self.convs = nn.ModuleList([
            GCN2Conv(hidden_dim, alpha=0.1) for _ in range(num_layers)
        ])
        self.output = nn.Linear(hidden_dim, out_dim)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = F.relu(self.input_proj(x))
        x_0 = x  # initial embedding kept for residual connections
        for conv in self.convs:
            x = conv(x, x_0, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.output(x)
