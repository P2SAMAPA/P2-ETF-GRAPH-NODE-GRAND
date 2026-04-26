"""
Build daily graph snapshots using k‑nearest neighbour correlation.
"""
import numpy as np
import pandas as pd
import torch
import config
from sklearn.neighbors import kneighbors_graph

def build_knn_adjacency(corr_matrix: np.ndarray, k: int = 5) -> np.ndarray:
    """
    Build a directed k‑NN adjacency from the correlation matrix.
    Higher correlation = closer distance.
    """
    # Convert correlation to distance: 1 - |correlation|
    dist = 1.0 - np.abs(corr_matrix)
    np.fill_diagonal(dist, 0.0)               # ignore self

    # Build k‑NN graph (binary, undirected by symmetrising)
    adj = kneighbors_graph(dist, n_neighbors=k, mode='connectivity', include_self=False)
    adj = adj + adj.T                         # make undirected
    adj.data[:] = 1.0                         # binary
    adj = adj.toarray()
    return adj

def build_edge_index(adj: np.ndarray, device: str = 'cpu'):
    edges = np.where(adj != 0)
    edge_index = torch.tensor(np.array(edges), dtype=torch.long, device=device)
    # Use linear weights (correlation) for edges
    edge_weight = torch.ones(edge_index.size(1), dtype=torch.float32, device=device)
    return edge_index, edge_weight

def build_daily_graphs(returns: pd.DataFrame, macro: pd.DataFrame):
    """
    Build one graph snapshot per trading day.
    Each snapshot = (date, (x, edge_index, edge_weight, target))
    """
    common_idx = returns.index.intersection(macro.index)
    returns = returns.loc[common_idx]
    macro = macro.loc[common_idx]

    tickers = returns.columns.tolist()
    graphs = []

    for i in range(config.LOOKBACK_WINDOW, len(returns) - 1):
        window_ret = returns.iloc[i - config.LOOKBACK_WINDOW : i]
        corr = window_ret.corr().values
        adj = build_knn_adjacency(corr, k=config.KNN_K)
        edge_index, edge_weight = build_edge_index(adj)

        # Node features
        node_feats = []
        for tkr in tickers:
            ret_window = returns[tkr].iloc[max(0, i - config.FEATURE_WINDOW) : i].values
            if len(ret_window) < config.FEATURE_WINDOW:
                ret_window = np.pad(ret_window, (config.FEATURE_WINDOW - len(ret_window), 0), 'edge')
            macro_now = macro.iloc[i].values
            feat = np.concatenate([ret_window, macro_now])
            node_feats.append(feat)

        x = torch.tensor(np.stack(node_feats), dtype=torch.float32)
        y = torch.tensor(returns.iloc[i + 1].values, dtype=torch.float32)
        date = returns.index[i]
        graphs.append((date, (x, edge_index, edge_weight, y)))

    return graphs
