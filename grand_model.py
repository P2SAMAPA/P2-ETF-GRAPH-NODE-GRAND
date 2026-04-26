"""
Temporal Graph Network: GCN + GRU with skip connection.
"""
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv


class TemporalGNN(nn.Module):
    """GCN → GRU → GCN → … → Linear (+ residual)"""
    def __init__(self, node_dim, hidden_dim, num_layers, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(node_dim, hidden_dim)   # for skip
        self.rnn = nn.GRUCell(hidden_dim, hidden_dim)
        self.convs = nn.ModuleList()
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        self.lin = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, graph_sequence):
        """
        graph_sequence: list of (x, edge_index, edge_weight)
        Returns: (seq_len, n_nodes, 1)
        """
        outputs = []
        h = None
        for x, edge_index, edge_weight in graph_sequence:
            # initial hidden state
            if h is None:
                h = torch.zeros(x.size(0), self.input_proj.out_features, device=x.device)

            # project input to hidden dimension
            x_proj = self.input_proj(x)

            # GRU step
            h = self.rnn(x_proj, h)

            # Graph convolutions with residual
            for conv in self.convs:
                h_res = conv(h, edge_index, edge_weight)
                h = torch.relu(h_res) + h                  # ← residual

            h = self.dropout(h)

            # prediction from current hidden state
            out = self.lin(h)                                # (n_nodes, 1)
            outputs.append(out.unsqueeze(0))

        return torch.cat(outputs, dim=0)                     # (T, n_nodes, 1)
