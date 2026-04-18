"""
GRAND (Graph Neural Diffusion) model for ETF return prediction.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchdiffeq import odeint
from torch_geometric.nn import GCNConv
import config


class DiffusionFunc(nn.Module):
    """The right-hand side function dX/dt = (A - I) * X * W."""
    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, hidden_dim, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.linear.weight)

    def forward(self, t, x):
        # x: (n_nodes, hidden_dim)
        return self.linear(x)


class GRAND(nn.Module):
    """
    Graph Neural Diffusion (GRAND) model.
    Adapted from: https://github.com/twitter-research/graph-neural-pde
    """
    def __init__(self, node_dim, edge_index, edge_weight, hidden_dim=32, ode_time=1.0, dropout=0.1):
        super().__init__()
        self.node_dim = node_dim
        self.hidden_dim = hidden_dim
        self.ode_time = ode_time
        self.dropout = dropout

        # Register graph structure
        self.register_buffer('edge_index', edge_index)
        self.register_buffer('edge_weight', edge_weight)

        # Input projection
        self.input_proj = nn.Linear(node_dim, hidden_dim)

        # Graph convolution for spatial diffusion
        self.conv = GCNConv(hidden_dim, hidden_dim, bias=False)

        # ODE function (temporal diffusion)
        self.odefunc = DiffusionFunc(hidden_dim, hidden_dim)

        # Output projection
        self.output_proj = nn.Linear(hidden_dim, 1)

    def forward(self, x, return_embeddings=False):
        # x: (n_nodes, node_dim)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.input_proj(x)
        x = F.relu(x)

        # Spatial diffusion via graph convolution
        x = self.conv(x, self.edge_index, self.edge_weight)
        x = F.relu(x)

        # Temporal diffusion via ODE
        integration_time = torch.tensor([0, self.ode_time], device=x.device)
        x = odeint(self.odefunc, x, integration_time, method='dopri5')[1]  # final state

        if return_embeddings:
            return x

        x = self.output_proj(x).squeeze(-1)  # (n_nodes,)
        return x
