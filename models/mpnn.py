import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class MPNN(nn.Module):
    """MPNN with per-layer GCNConv + linear skip, BatchNorm, LayerNorm,
    an initial feature projection (lin_in), and a local prediction head.

    Architecture (num_layers=3, in_dim=1433, hidden_dim=512, out_dim=7):
        h_lins      : ModuleList()                                   # empty
        local_convs : [GCNConv(in,h), GCNConv(h,h), GCNConv(h,h)]
        lins        : [Linear(in,h),  Linear(h,h),  Linear(h,h)]
        lns         : [LayerNorm(h)] * num_layers
        bns         : [BatchNorm1d(h)] * num_layers
        lin_in      : Linear(in, h)
        pred_local  : Linear(h, out)
    """

    def __init__(self, in_dim, hidden_dim, out_dim, num_layers=3, dropout=0.0):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout

        self.h_lins = nn.ModuleList()

        self.local_convs = nn.ModuleList()
        self.local_convs.append(GCNConv(in_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.local_convs.append(GCNConv(hidden_dim, hidden_dim))

        self.lins = nn.ModuleList()
        self.lins.append(nn.Linear(in_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.lins.append(nn.Linear(hidden_dim, hidden_dim))

        self.lns = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
        self.bns = nn.ModuleList([nn.BatchNorm1d(hidden_dim) for _ in range(num_layers)])

        self.lin_in = nn.Linear(in_dim, hidden_dim)
        self.pred_local = nn.Linear(hidden_dim, out_dim)

    def forward(self, x, edge_index):
        h0 = F.relu(self.lin_in(x))
        h = x
        for i in range(self.num_layers):
            h = F.relu(self.lns[i](self.bns[i](
                self.local_convs[i](h, edge_index) + self.lins[i](h)
            )))
            h = F.dropout(h, p=self.dropout, training=self.training)
        return self.pred_local(h + h0)
