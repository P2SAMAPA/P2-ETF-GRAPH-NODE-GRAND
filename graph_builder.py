"""
Build daily graph snapshots using k‑nearest neighbour correlation.
"""
import numpy as np
import pandas as pd
import torch
import config
from sklearn.neighbors import kneighbors_graph

def build_knn_adjacency(corr_matrix: np.ndarray, k: int = 5) -> np.ndarray:
    """Build a k‑NN adjacency from the correlation matrix."""
    dist = 1.0 - np.abs(corr_matrix)
    np.fill_diagonal(dist, 0.0)
    adj = kneighbors_graph(dist, n_neighbors=k, mode='connectivity', include_self=False)
    adj = adj + adj.T
    adj.data[:] = 1.0
    return adj.toarray()

def build_edge_index(adj: np.ndarray, device: str = 'cpu'):
    edges = np.where(adj != 0)
    edge_index = torch.tensor(np.array(edges), dtype=torch.long, device=device)
    edge_weight = torch.ones(edge_index.size(1), dtype=torch.float32, device=device)
    return edge_index, edge_weight

def build_daily_graphs(returns: pd.DataFrame, macro: pd.DataFrame):
    """
    Build one graph snapshot per trading day.
    Node features: (FEATURE_WINDOW returns) + (current macro) + (one‑hot ETF identity)
    """
    common_idx = returns.index.intersection(macro.index)
    returns = returns.loc[common_idx]
    macro = macro.loc[common_idx]

    tickers = returns.columns.tolist()
    n_assets = len(tickers)
    # One‑hot vectors
    one_hot = np.eye(n_assets, dtype=np.float32)

    graphs = []
    for i in range(config.LOOKBACK_WINDOW, len(returns) - 1):
        window_ret = returns.iloc[i - config.LOOKBACK_WINDOW : i]
        corr = window_ret.corr().values
        adj = build_knn_adjacency(corr, k=config.KNN_K)
        edge_index, edge_weight = build_edge_index(adj)

        node_feats = []
        for j, tkr in enumerate(tickers):
            ret_window = returns[tkr].iloc[max(0, i - config.FEATURE_WINDOW) : i].values
            if len(ret_window) < config.FEATURE_WINDOW:
                ret_window = np.pad(ret_window, (config.FEATURE_WINDOW - len(ret_window), 0), 'edge')
            macro_now = macro.iloc[i].values
            feat = np.concatenate([ret_window, macro_now, one_hot[j]])
            node_feats.append(feat)

        x = torch.tensor(np.stack(node_feats), dtype=torch.float32)
        y = torch.tensor(returns.iloc[i + 1].values, dtype=torch.float32)
        date = returns.index[i]
        graphs.append((date, (x, edge_index, edge_weight, y)))

    return graphs
