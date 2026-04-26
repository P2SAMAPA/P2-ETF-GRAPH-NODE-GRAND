"""
Static GCN with Layer Normalization for stable training.
"""
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv

class StableGCN(nn.Module):
    def __init__(self, node_dim, hidden_dim, num_layers, dropout=0.1):
        super().__init__()
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.convs.append(GCNConv(node_dim, hidden_dim))
        self.norms.append(nn.LayerNorm(hidden_dim))
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
            self.norms.append(nn.LayerNorm(hidden_dim))
        self.lin = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, edge_weight=None):
        for conv, norm in zip(self.convs, self.norms):
            x = conv(x, edge_index, edge_weight)
            x = norm(x)
            x = torch.relu(x)
            x = self.dropout(x)
        return self.lin(x).squeeze(-1)
