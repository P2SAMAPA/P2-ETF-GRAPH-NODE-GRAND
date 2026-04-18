"""
Construct rolling correlation adjacency matrices.
"""
import numpy as np
import pandas as pd
import config


def correlation_to_adjacency(corr_matrix: np.ndarray, threshold: float = 0.1) -> np.ndarray:
    """Convert correlation matrix to adjacency (drop weak edges)."""
    adj = corr_matrix.copy()
    adj[np.abs(adj) < threshold] = 0
    np.fill_diagonal(adj, 0)
    return adj


def build_rolling_graphs(returns: pd.DataFrame):
    """Build a sequence of adjacency matrices using rolling windows."""
    lookback = config.LOOKBACK_WINDOW
    freq = config.REBALANCE_FREQ

    graphs = []
    for i in range(lookback, len(returns), freq):
        window_ret = returns.iloc[i - lookback : i]
        if len(window_ret) < lookback // 2:
            continue
        corr = window_ret.corr().values
        adj = correlation_to_adjacency(corr, threshold=0.1)
        date = returns.index[i]
        graphs.append((date, adj))
    return graphs


def get_latest_graph(graphs: list, date: pd.Timestamp = None):
    """Return the most recent graph on or before the given date."""
    if not graphs:
        return None
    if date is None:
        return graphs[-1][1]
    for g_date, adj in reversed(graphs):
        if g_date <= date:
            return adj
    return graphs[0][1]
