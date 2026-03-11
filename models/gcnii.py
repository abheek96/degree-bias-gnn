import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCN2Conv



class GCN2ConvLayer(nn.Module):
    """GCNII Layer from https://arxiv.org/abs/2007.02133.
    """
    def __init__(self, dim_in, dim_out, num_layers, dropout):
        super().__init__()
        self.layers = nn.ModuleList()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.dropout = dropout
        self.num_layers = num_layers
        for _ in range(num_layers):
            self.layers.append(GCN2Conv(self.dim_in, alpha=0.1))
        # alpha value is set using results from the GCNII paper
        self.layers.append(nn.Linear(self.dim_in, self.dim_out))

    def forward(self, x, edge_index):
        x_in = x

        for layer in self.layers[:-1]:
            x = layer(x, x_in, edge_index)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = F.relu(x)

        x = self.layers[-1](x)

        return x