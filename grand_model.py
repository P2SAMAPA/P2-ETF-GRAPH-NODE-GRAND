"""
Temporal Graph Network: GCN + GRU for ETF return prediction.
Replaces the unstable continuous-time ODE.
"""
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv
import config

class TemporalGNN(nn.Module):
    """GCN -> GRU -> GCN -> ... -> Linear"""
    def __init__(self, node_dim, hidden_dim, num_layers, dropout=0.1):
        super().__init__()
        self.node_dim = node_dim
        self.hidden_dim = hidden_dim
        self.rnn = nn.GRUCell(node_dim, hidden_dim)
        self.convs = nn.ModuleList()
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        self.lin = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, graph_sequence):
        """
        graph_sequence: list of (x, edge_index, edge_weight) tuples,
        each x: (n_nodes, node_dim)
        Returns: (seq_len, n_nodes, 1)
        """
        outputs = []
        h = torch.zeros(graph_sequence[0][0].size(0), self.hidden_dim, device=graph_sequence[0][0].device)
        for x, edge_index, edge_weight in graph_sequence:
            h = self.rnn(x, h)
            for conv in self.convs:
                h = conv(h, edge_index, edge_weight)
                h = torch.relu(h)
            h = self.dropout(h)
            out = self.lin(h)  # (n_nodes, 1)
            outputs.append(out.unsqueeze(0))
        return torch.cat(outputs, dim=0)  # (T, n_nodes, 1)
