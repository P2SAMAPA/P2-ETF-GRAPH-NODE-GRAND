"""
Build daily graph snapshots with ETF and macro features.
"""
import numpy as np
import pandas as pd
import torch
import config

def build_edge_index(adj, device='cpu'):
    edges = np.where(adj != 0)
    edge_index = torch.tensor(np.array(edges), dtype=torch.long, device=device)
    edge_weight = torch.tensor(adj[edges], dtype=torch.float32, device=device)
    return edge_index, edge_weight

def build_daily_graphs(returns, macro):
    """
    returns: DataFrame of daily returns (wide), macro: DataFrame of macro features.
    Returns list of tuples: (date, (x, edge_index, edge_weight, target))
    target = next-day returns for all tickers.
    """
    common_idx = returns.index.intersection(macro.index)
    returns = returns.loc[common_idx]
    macro = macro.loc[common_idx]

    tickers = returns.columns.tolist()
    n_assets = len(tickers)
    n_macro = len(macro.columns)

    graphs = []
    for i in range(config.LOOKBACK_WINDOW, len(returns) - 1):
        window_ret = returns.iloc[i - config.LOOKBACK_WINDOW : i]
        corr = window_ret.corr().values
        adj = corr.copy()
        adj[np.abs(adj) < 0.1] = 0
        np.fill_diagonal(adj, 0)

        edge_index, edge_weight = build_edge_index(adj)

        # Node features: last 5 days of returns + current macro values
        node_feats = []
        for tkr in tickers:
            ret_window = returns[tkr].iloc[i-5 : i].values
            if len(ret_window) < 5:
                ret_window = np.pad(ret_window, (5 - len(ret_window), 0), 'edge')
            macro_vals = macro.iloc[i].values
            feat = np.concatenate([ret_window, macro_vals])
            node_feats.append(feat)
        x = torch.tensor(np.stack(node_feats), dtype=torch.float32)

        # Target: next-day returns for all ETFs
        target = torch.tensor(returns.iloc[i+1].values, dtype=torch.float32)

        date = returns.index[i]
        graphs.append((date, (x, edge_index, edge_weight, target)))
    return graphs
