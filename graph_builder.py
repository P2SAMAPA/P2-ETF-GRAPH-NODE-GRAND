"""
Build daily graph snapshots with ETF return features and macro conditioning.
"""
import numpy as np
import pandas as pd
import torch
import config

def correlation_to_adjacency(corr_matrix: np.ndarray) -> np.ndarray:
    """Convert correlation matrix to adjacency, using config.CORRELATION_THRESHOLD."""
    adj = corr_matrix.copy()
    adj[np.abs(adj) < config.CORRELATION_THRESHOLD] = 0
    np.fill_diagonal(adj, 0)
    adj = adj + np.eye(adj.shape[0])          # self‑loops
    return adj

def build_edge_index(adj: np.ndarray, device: str = 'cpu'):
    edges = np.where(adj != 0)
    edge_index = torch.tensor(np.array(edges), dtype=torch.long, device=device)
    edge_weight = torch.tensor(adj[edges], dtype=torch.float32, device=device)
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
        # Adjacency from rolling correlation
        window_ret = returns.iloc[i - config.LOOKBACK_WINDOW : i]
        corr = window_ret.corr().values
        adj = correlation_to_adjacency(corr)
        edge_index, edge_weight = build_edge_index(adj)

        # Node features: last FEATURE_WINDOW returns + current macro values
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
