import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class MPNN(nn.Module):
    """MPNN with per-layer GCNConv + linear skip, optional BatchNorm, LayerNorm,
    and residual connections, plus an initial feature projection (lin_in) and a
    local prediction head (pred_local).

    Architecture (num_layers=3, in_dim=1433, hidden_dim=512, out_dim=7):
        h_lins      : ModuleList()                                   # empty
        local_convs : [GCNConv(in,h), GCNConv(h,h), GCNConv(h,h)]
        lins        : [Linear(in,h),  Linear(h,h),  Linear(h,h)]
        lns         : [LayerNorm(h)] * num_layers  (if layer_norm=True)
        bns         : [BatchNorm1d(h)] * num_layers  (if batch_norm=True)
        lin_in      : Linear(in, h)   — initial feature projection / residual base
        pred_local  : Linear(h, out)  — classification head
    """

    _STANDARD_KEYS = {"name", "hidden_dim", "num_layers", "dropout", "weights_path",
                      "layer_norm", "batch_norm", "residual"}

    def __init__(self, in_dim, hidden_dim, out_dim, num_layers=3, dropout=0.0,
                 layer_norm=True, batch_norm=True, residual=True):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.use_layer_norm = layer_norm
        self.use_batch_norm = batch_norm
        self.use_residual = residual

        self.h_lins = nn.ModuleList()

        self.local_convs = nn.ModuleList()
        self.local_convs.append(GCNConv(in_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.local_convs.append(GCNConv(hidden_dim, hidden_dim))

        self.lins = nn.ModuleList()
        self.lins.append(nn.Linear(in_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.lins.append(nn.Linear(hidden_dim, hidden_dim))

        self.lns = nn.ModuleList(
            [nn.LayerNorm(hidden_dim) for _ in range(num_layers)]
            if layer_norm else []
        )
        self.bns = nn.ModuleList(
            [nn.BatchNorm1d(hidden_dim) for _ in range(num_layers)]
            if batch_norm else []
        )

        self.lin_in = nn.Linear(in_dim, hidden_dim)
        self.pred_local = nn.Linear(hidden_dim, out_dim)

    def get_intermediate(self, x, edge_index, layer: int = 1):
        """Return node representations after ``layer`` message-passing steps
        (no gradient, eval mode).  Valid range: [1, num_layers].
        """
        if layer < 1 or layer > self.num_layers:
            raise ValueError(
                f"layer must be in [1, {self.num_layers}] for this model, got {layer}"
            )
        self.eval()
        with torch.no_grad():
            h0 = F.relu(self.lin_in(x)) if self.use_residual else None
            h = x
            for i in range(self.num_layers):
                h_new = self.local_convs[i](h, edge_index) + self.lins[i](h)
                if self.use_batch_norm:
                    h_new = self.bns[i](h_new)
                if self.use_layer_norm:
                    h_new = self.lns[i](h_new)
                h = F.relu(h_new)
                if i + 1 == layer:
                    return (h + h0) if self.use_residual else h
        return (h + h0) if self.use_residual else h

    def forward(self, x, edge_index):
        h0 = F.relu(self.lin_in(x)) if self.use_residual else None
        h = x
        for i in range(self.num_layers):
            h_new = self.local_convs[i](h, edge_index) + self.lins[i](h)
            if self.use_batch_norm:
                h_new = self.bns[i](h_new)
            if self.use_layer_norm:
                h_new = self.lns[i](h_new)
            h = F.relu(h_new)
            h = F.dropout(h, p=self.dropout, training=self.training)
        return self.pred_local((h + h0) if self.use_residual else h)
