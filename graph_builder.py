"""
Build daily graph snapshots with ETF return features and macro conditioning.
"""
import numpy as np
import pandas as pd
import torch
import config

def correlation_to_adjacency(corr_matrix: np.ndarray, threshold: float = 0.1) -> np.ndarray:
    """Convert correlation matrix to adjacency, keeping self‑loops for numerical stability."""
    adj = corr_matrix.copy()
    adj[np.abs(adj) < threshold] = 0
    np.fill_diagonal(adj, 0)
    # Add self‑loops so that every node has at least one connection
    adj = adj + np.eye(adj.shape[0])
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
    where target = next-day returns for all tickers.
    """
    common_idx = returns.index.intersection(macro.index)
    returns = returns.loc[common_idx]
    macro = macro.loc[common_idx]

    tickers = returns.columns.tolist()
    macro_cols = macro.columns.tolist()

    graphs = []
    for i in range(config.LOOKBACK_WINDOW, len(returns) - 1):
        # Adjacency from rolling correlation
        window_ret = returns.iloc[i - config.LOOKBACK_WINDOW : i]
        corr = window_ret.corr().values
        adj = correlation_to_adjacency(corr, threshold=0.1)
        edge_index, edge_weight = build_edge_index(adj)

        # Node features: last 5 returns + current macro values
        node_feats = []
        for tkr in tickers:
            ret_window = returns[tkr].iloc[max(0, i - 5) : i].values
            if len(ret_window) < 5:
                ret_window = np.pad(ret_window, (5 - len(ret_window), 0), 'edge')
            macro_now = macro.iloc[i].values
            feat = np.concatenate([ret_window, macro_now])
            node_feats.append(feat)

        x = torch.tensor(np.stack(node_feats), dtype=torch.float32)

        # Target: next-day returns per ETF
        y = torch.tensor(returns.iloc[i + 1].values, dtype=torch.float32)

        date = returns.index[i]
        graphs.append((date, (x, edge_index, edge_weight, y)))

    return graphs
