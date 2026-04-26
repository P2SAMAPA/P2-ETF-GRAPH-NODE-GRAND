"""
Global, Adaptive, and Daily training with GCN+GRU (target scaling + residual).
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
from graph_builder import build_daily_graphs
from grand_model import TemporalGNN
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


def train_temporal(model, graphs, epochs, lr, patience, device):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    best_loss = float('inf')
    patience_counter = 0
    best_state = None

    x_seq = [g[1][0].to(device) for g in graphs]
    edge_seq = [g[1][1].to(device) for g in graphs]
    weight_seq = [g[1][2].to(device) for g in graphs]
    targets = torch.stack([g[1][3].to(device) for g in graphs], dim=0).nan_to_num()

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        preds = model(list(zip(x_seq, edge_seq, weight_seq)))
        loss = criterion(preds.squeeze(-1), targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if (epoch + 1) % 20 == 0:
            print(f"    Epoch {epoch+1:3d} | Loss: {loss.item():.6f}")

        if loss.item() < best_loss:
            best_loss = loss.item()
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"    Early stopping at epoch {epoch+1}")
                break

    if best_state:
        model.load_state_dict(best_state)
    return model


def train_mode(universe, returns, macro, mode='global'):
    print(f"\n--- {mode.upper()} Training: {universe} ---")
    tickers = [col.replace("_ret", "") for col in returns.columns]
    graphs = build_daily_graphs(returns, macro)

    if len(graphs) < config.MIN_TRAIN_DAYS:
        print(f"  Not enough daily graphs ({len(graphs)} < {config.MIN_TRAIN_DAYS}).")
        return None

    n = len(graphs)
    train_end = int(n * config.TRAIN_RATIO)
    val_end = train_end + int(n * config.VAL_RATIO)
    train_graphs = graphs[:train_end]
    test_graphs = graphs[val_end:]

    # ---- Scale targets to help GRU training ----
    target_scaler = StandardScaler()
    all_train_targets = np.concatenate([g[1][3].numpy() for g in train_graphs])
    target_scaler.fit(all_train_targets.reshape(-1, 1))

    train_graphs = [
        (date, (x, ei, ew, torch.tensor(target_scaler.transform(y.numpy().reshape(-1, 1)).flatten(), dtype=torch.float32)))
        for (date, (x, ei, ew, y)) in train_graphs
    ]

    node_dim = train_graphs[0][1][0].size(1)
    model = TemporalGNN(node_dim, config.HIDDEN_DIM, config.NUM_LAYERS, config.DROPOUT)
    model = train_temporal(model, train_graphs, config.EPOCHS, config.LEARNING_RATE,
                          config.PATIENCE, config.DEVICE)

    # Predict on latest graph
    model.eval()
    with torch.no_grad():
        if test_graphs:
            _, (latest_x, latest_edge, latest_weight, _) = test_graphs[-1]
        else:
            _, (latest_x, latest_edge, latest_weight, _) = train_graphs[-1]

        all_x = [g[1][0] for g in train_graphs + test_graphs]
        all_edge = [g[1][1] for g in train_graphs + test_graphs]
        all_w = [g[1][2] for g in train_graphs + test_graphs]
        preds = model(list(zip(all_x, all_edge, all_w)))[-1].squeeze(-1).cpu().numpy()
        # Inverse transform to original scale
        preds = target_scaler.inverse_transform(preds.reshape(-1, 1)).flatten()

    best_idx = np.argmax(preds)
    best_ticker = tickers[best_idx]
    pred_return = float(preds[best_idx])
    all_preds = {tickers[i]: float(preds[i]) for i in range(len(tickers))}

    metrics = evaluate_etf(best_ticker, returns.iloc[-config.MIN_TEST_DAYS:])
    print(f"  Selected ETF: {best_ticker}, Predicted Return: {pred_return*100:.2f}%")
    return {
        "ticker": best_ticker,
        "pred_return": pred_return,
        "all_pred_returns": all_preds,
        "metrics": metrics,
        "optimal_window": config.LOOKBACK_WINDOW,
        "test_start": str(returns.index[-config.MIN_TEST_DAYS].date()),
        "test_end": str(returns.index[-1].date()),
    }


def main():
    token = os.getenv("HF_TOKEN")
    if not token:
        print("HF_TOKEN not set")
        return

    df_raw = load_master_data()
    df = prepare_data(df_raw)

    results = {}
    for univ_id, univ_name in [("fi", "FI"), ("equity", "Equity"), ("combined", "Combined")]:
        returns = get_universe_returns(df, univ_id)
        if returns.empty:
            continue
        macro = df[config.MACRO_COLS].loc[returns.index].ffill().dropna()
        common = returns.index.intersection(macro.index)
        returns = returns.loc[common]
        macro = macro.loc[common]

        univ_res = {}

        global_out = train_mode(univ_name, returns, macro, 'global')
        if global_out:
            univ_res['global'] = global_out

        cp_date = universe_adaptive_start_date(returns)
        end_date = returns.index[-1] - pd.Timedelta(days=config.MIN_TEST_DAYS)
        if end_date <= cp_date:
            end_date = returns.index[-1] - pd.Timedelta(days=10)
        train_mask = (returns.index >= cp_date) & (returns.index <= end_date)
        adaptive_ret = returns.loc[train_mask]
        if len(adaptive_ret) >= config.MIN_TRAIN_DAYS:
            adaptive_macro = macro.loc[adaptive_ret.index]
            adaptive_out = train_mode(univ_name, adaptive_ret, adaptive_macro, 'adaptive')
            if adaptive_out:
                adaptive_out['adaptive_window'] = (returns.index[-1] - cp_date).days
                adaptive_out['change_point_date'] = cp_date.strftime("%Y-%m-%d")
                univ_res['adaptive'] = adaptive_out
        else:
            univ_res['adaptive'] = global_out

        daily_ret = returns.iloc[-config.DAILY_LOOKBACK:]
        daily_macro = macro.loc[daily_ret.index]
        daily_out = train_mode(univ_name, daily_ret, daily_macro, 'daily')
        if daily_out:
            univ_res['daily'] = daily_out

        results[univ_id] = univ_res

    push_daily_result(results)
    print("\n=== Run Complete ===")


if __name__ == "__main__":
    main()
