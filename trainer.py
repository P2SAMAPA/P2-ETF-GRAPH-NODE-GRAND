"""
Global and Adaptive Window training with GRAND.
"""
import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
from sklearn.preprocessing import StandardScaler

import config
from data_manager import load_master_data, prepare_data, get_universe_returns
from graph_builder import build_rolling_graphs, get_latest_graph
from grand_model import GRAND
from change_point_detector import universe_adaptive_start_date
from push_results import push_daily_result


def evaluate_etf(ticker: str, returns: pd.DataFrame) -> dict:
    col = f"{ticker}_ret"
    if col not in returns.columns:
        return {}
    ret_series = returns[col].dropna()
    if len(ret_series) < 5:
        return {}
    ann_return = ret_series.mean() * config.TRADING_DAYS_PER_YEAR
    ann_vol = ret_series.std() * np.sqrt(config.TRADING_DAYS_PER_YEAR)
    sharpe = ann_return / ann_vol if ann_vol > 0 else 0.0
    cum = (1 + ret_series).cumprod()
    rolling_max = cum.expanding().max()
    drawdown = (cum - rolling_max) / rolling_max
    max_dd = drawdown.min()
    hit_rate = (ret_series > 0).mean()
    cum_return = (1 + ret_series).prod() - 1
    return {
        "ann_return": ann_return, "ann_vol": ann_vol, "sharpe": sharpe,
        "max_dd": max_dd, "hit_rate": hit_rate, "cum_return": cum_return,
        "n_days": len(ret_series)
    }


def build_edge_index(adj: np.ndarray, device: str = 'cpu'):
    """Convert adjacency matrix to PyTorch Geometric edge_index and edge_weight."""
    edges = np.where(adj != 0)
    edge_index = torch.tensor(np.array(edges), dtype=torch.long, device=device)
    edge_weight = torch.tensor(adj[edges], dtype=torch.float32, device=device)
    return edge_index, edge_weight


def train_grand(model, x, y_train, y_val, epochs, lr, patience, device):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    best_val_loss = float('inf')
    patience_counter = 0
    best_state = None

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        pred = model(x)
        loss = criterion(pred, y_train)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_pred = model(x)
            val_loss = criterion(val_pred, y_val)

        if (epoch + 1) % 20 == 0:
            print(f"    Epoch {epoch+1:3d} | Train Loss: {loss.item():.6f} | Val Loss: {val_loss.item():.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"    Early stopping at epoch {epoch+1}")
                break

    model.load_state_dict(best_state)
    return model


def train_global(universe: str, returns: pd.DataFrame, graphs: list) -> dict:
    print(f"\n--- Global Training: {universe} ---")
    tickers = [col.replace("_ret", "") for col in returns.columns]
    total_days = len(returns)
    train_end = int(total_days * config.TRAIN_RATIO)
    val_end = train_end + int(total_days * config.VAL_RATIO)

    train_ret = returns.iloc[:train_end]
    val_ret = returns.iloc[train_end:val_end]
    test_ret = returns.iloc[val_end:]

    adj = get_latest_graph(graphs, train_ret.index[-1])
    if adj is None:
        print("  No graph available for training. Skipping.")
        return {"ticker": None, "metrics": {}}

    edge_index, edge_weight = build_edge_index(adj, config.DEVICE)

    # Node features: recent returns (last 20 days)
    window = 20
    recent = train_ret.iloc[-window:].values.T  # (n_nodes, window)
    scaler = StandardScaler()
    node_feats = scaler.fit_transform(recent)
    node_feats = torch.tensor(node_feats, dtype=torch.float32, device=config.DEVICE)

    # Targets: average forward return (simplified)
    y_train = torch.tensor(train_ret.shift(-1).mean().values, dtype=torch.float32, device=config.DEVICE)
    y_val = torch.tensor(val_ret.mean().values, dtype=torch.float32, device=config.DEVICE)

    model = GRAND(
        node_dim=window, edge_index=edge_index, edge_weight=edge_weight,
        hidden_dim=config.HIDDEN_DIM, ode_time=config.ODE_TIME, dropout=config.DROPOUT
    ).to(config.DEVICE)

    model = train_grand(model, node_feats, y_train, y_val,
                        config.EPOCHS, config.LEARNING_RATE, config.PATIENCE, config.DEVICE)

    # Predict on test set
    test_adj = get_latest_graph(graphs, test_ret.index[0])
    if test_adj is None:
        test_adj = adj
    test_edge_index, test_edge_weight = build_edge_index(test_adj, config.DEVICE)
    model.edge_index = test_edge_index
    model.edge_weight = test_edge_weight

    model.eval()
    with torch.no_grad():
        pred_returns = model(node_feats).cpu().numpy()

    best_idx = np.argmax(pred_returns)
    best_ticker = tickers[best_idx]
    pred_return = float(pred_returns[best_idx])
    all_pred_returns = {tickers[i]: float(pred_returns[i]) for i in range(len(tickers))}

    metrics = evaluate_etf(best_ticker, test_ret)
    print(f"  Selected ETF: {best_ticker}, Predicted Return: {pred_return*100:.2f}%")
    return {
        "ticker": best_ticker,
        "pred_return": pred_return,
        "all_pred_returns": all_pred_returns,
        "metrics": metrics,
        "test_start": test_ret.index[0].strftime("%Y-%m-%d"),
        "test_end": test_ret.index[-1].strftime("%Y-%m-%d"),
    }


def train_adaptive(universe: str, returns: pd.DataFrame, graphs: list) -> dict:
    print(f"\n--- Adaptive Training: {universe} ---")
    tickers = [col.replace("_ret", "") for col in returns.columns]
    cp_date = universe_adaptive_start_date(returns)
    print(f"  Adaptive window starts: {cp_date.date()}")

    end_date = returns.index[-1] - pd.Timedelta(days=config.MIN_TEST_DAYS)
    if end_date <= cp_date:
        end_date = returns.index[-1] - pd.Timedelta(days=10)
    train_mask = (returns.index >= cp_date) & (returns.index <= end_date)
    train_ret = returns.loc[train_mask]
    test_ret = returns.loc[returns.index > end_date]

    if len(train_ret) < config.MIN_TRAIN_DAYS:
        print("  Insufficient training days. Falling back to global.")
        return train_global(universe, returns, graphs)

    adj = get_latest_graph(graphs, train_ret.index[-1])
    if adj is None:
        print("  No graph available for adaptive training. Falling back to global.")
        return train_global(universe, returns, graphs)

    edge_index, edge_weight = build_edge_index(adj, config.DEVICE)

    window = 20
    recent = train_ret.iloc[-window:].values.T
    scaler = StandardScaler()
    node_feats = scaler.fit_transform(recent)
    node_feats = torch.tensor(node_feats, dtype=torch.float32, device=config.DEVICE)

    y_train = torch.tensor(train_ret.shift(-1).mean().values, dtype=torch.float32, device=config.DEVICE)
    y_val = torch.tensor(train_ret.iloc[-len(train_ret)//5:].mean().values, dtype=torch.float32, device=config.DEVICE)

    model = GRAND(
        node_dim=window, edge_index=edge_index, edge_weight=edge_weight,
        hidden_dim=config.HIDDEN_DIM, ode_time=config.ODE_TIME, dropout=config.DROPOUT
    ).to(config.DEVICE)

    model = train_grand(model, node_feats, y_train, y_val,
                        config.EPOCHS, config.LEARNING_RATE, config.PATIENCE, config.DEVICE)

    test_adj = get_latest_graph(graphs, test_ret.index[0] if len(test_ret) > 0 else returns.index[-1])
    if test_adj is None:
        test_adj = adj
    test_edge_index, test_edge_weight = build_edge_index(test_adj, config.DEVICE)
    model.edge_index = test_edge_index
    model.edge_weight = test_edge_weight

    model.eval()
    with torch.no_grad():
        pred_returns = model(node_feats).cpu().numpy()

    best_idx = np.argmax(pred_returns)
    best_ticker = tickers[best_idx]
    pred_return = float(pred_returns[best_idx])
    all_pred_returns = {tickers[i]: float(pred_returns[i]) for i in range(len(tickers))}

    metrics = evaluate_etf(best_ticker, test_ret) if len(test_ret) > 0 else {}
    lookback = (returns.index[-1] - cp_date).days
    print(f"  Selected ETF: {best_ticker}, Predicted Return: {pred_return*100:.2f}%")
    return {
        "ticker": best_ticker,
        "pred_return": pred_return,
        "all_pred_returns": all_pred_returns,
        "adaptive_window": lookback,
        "change_point_date": cp_date.strftime("%Y-%m-%d"),
        "metrics": metrics,
        "test_start": test_ret.index[0].strftime("%Y-%m-%d") if len(test_ret) else "",
        "test_end": test_ret.index[-1].strftime("%Y-%m-%d") if len(test_ret) else "",
    }


def run_training():
    print("Loading data...")
    df_raw = load_master_data()
    df = prepare_data(df_raw)

    all_results = {}
    for universe in ["fi", "equity", "combined"]:
        print(f"\n{'='*50}\nProcessing {universe.upper()}\n{'='*50}")
        returns = get_universe_returns(df, universe)
        if returns.empty:
            continue
        graphs = build_rolling_graphs(returns)
        print(f"  Built {len(graphs)} graphs.")
        global_res = train_global(universe, returns, graphs)
        adaptive_res = train_adaptive(universe, returns, graphs)
        all_results[universe] = {"global": global_res, "adaptive": adaptive_res}
    return all_results


if __name__ == "__main__":
    output = run_training()
    if config.HF_TOKEN:
        push_daily_result(output)
    else:
        print("HF_TOKEN not set.")
