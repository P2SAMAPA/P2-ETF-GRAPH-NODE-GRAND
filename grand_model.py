"""
Static GCN without LayerNorm (allows more variation).
"""
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv

class StableGCN(nn.Module):
    def __init__(self, node_dim, hidden_dim, num_layers, dropout=0.1):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(node_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        self.lin = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, edge_weight=None):
        for conv in self.convs:
            x = conv(x, edge_index, edge_weight)
            x = torch.relu(x)
            x = self.dropout(x)
        return self.lin(x).squeeze(-1)
