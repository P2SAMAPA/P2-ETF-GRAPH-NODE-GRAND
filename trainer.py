"""
Global, Adaptive, and Daily training with stable static GCN.
"""
import os, json
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
from grand_model import StableGCN
from change_point_detector import universe_adaptive_start_date
from push_results import push_daily_result


def evaluate_etf(ticker: str, returns: pd.DataFrame) -> dict:
    col = f"{ticker}_ret"
    if col not in returns.columns: return {}
    ret = returns[col].dropna()
    if len(ret) < 5: return {}
    ann_ret = ret.mean() * config.TRADING_DAYS_PER_YEAR
    ann_vol = ret.std() * np.sqrt(config.TRADING_DAYS_PER_YEAR)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0.0
    cum = (1 + ret).cumprod()
    dd = (cum - cum.expanding().max()) / cum.expanding().max()
    max_dd = dd.min()
    hit = (ret > 0).mean()
    cum_ret = (1 + ret).prod() - 1
    return {"ann_return": ann_ret, "ann_vol": ann_vol, "sharpe": sharpe,
            "max_dd": max_dd, "hit_rate": hit, "cum_return": cum_ret, "n_days": len(ret)}


def scale_node_features(graphs):
    """Fit a StandardScaler on all node features across all graphs and apply it."""
    all_x = np.concatenate([g[1][0].numpy() for g in graphs], axis=0)
    scaler = StandardScaler().fit(all_x)
    return [
        (date, (torch.tensor(scaler.transform(x.numpy()), dtype=torch.float32), ei, ew, y))
        for (date, (x, ei, ew, y)) in graphs
    ]


def train_mode(universe, returns, macro, mode='global'):
    print(f"\n--- {mode.upper()} Training: {universe} ---")
    tickers = [col.replace("_ret", "") for col in returns.columns]
    graphs = build_daily_graphs(returns, macro)

    if len(graphs) < config.MIN_TRAIN_DAYS:
        print(f"  Not enough daily graphs ({len(graphs)} < {config.MIN_TRAIN_DAYS}).")
        return None

    # Scale node features
    graphs = scale_node_features(graphs)

    # Train on the SINGLE most recent graph for simplicity and stability
    _, (x, edge_index, edge_weight, y) = graphs[-1]
    x, edge_index, edge_weight, y = x.to(config.DEVICE), edge_index.to(config.DEVICE), edge_weight.to(config.DEVICE), y.to(config.DEVICE)

    model = StableGCN(x.size(1), config.HIDDEN_DIM, config.NUM_LAYERS, config.DROPOUT).to(config.DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    criterion = nn.MSELoss()
    best_loss = float('inf')
    patience_counter = 0
    best_state = None

    for epoch in range(config.EPOCHS):
        model.train()
        optimizer.zero_grad()
        pred = model(x, edge_index, edge_weight)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 20 == 0:
            print(f"    Epoch {epoch+1:3d} | Loss: {loss.item():.6f}")

        if loss.item() < best_loss:
            best_loss = loss.item()
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config.PATIENCE:
                print(f"    Early stopping at epoch {epoch+1}")
                break

    if best_state:
        model.load_state_dict(best_state)

    # Predict on the same graph (in practice, we'd use the latest)
    model.eval()
    with torch.no_grad():
        preds = model(x, edge_index, edge_weight).cpu().numpy()

    best_idx = np.argmax(preds)
    best_ticker = tickers[best_idx]
    pred_return = float(preds[best_idx])
    all_preds = {tickers[i]: float(preds[i]) for i in range(len(tickers))}

    metrics = evaluate_etf(best_ticker, returns.iloc[-config.MIN_TEST_DAYS:])
    print(f"  Selected ETF: {best_ticker}, Predicted Return: {pred_return*100:.2f}%")
    return {
        "ticker": best_ticker, "pred_return": pred_return,
        "all_pred_returns": all_preds, "metrics": metrics,
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
        if returns.empty: continue
        macro = df[config.MACRO_COLS].loc[returns.index].ffill().dropna()
        common = returns.index.intersection(macro.index)
        returns = returns.loc[common]
        macro = macro.loc[common]

        univ_res = {}

        global_out = train_mode(univ_name, returns, macro, 'global')
        if global_out: univ_res['global'] = global_out

        cp_date = universe_adaptive_start_date(returns)
        end_date = returns.index[-1] - pd.Timedelta(days=config.MIN_TEST_DAYS)
        if end_date <= cp_date: end_date = returns.index[-1] - pd.Timedelta(days=10)
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
        if daily_out: univ_res['daily'] = daily_out

        results[univ_id] = univ_res

    push_daily_result(results)
    print("\n=== Run Complete ===")


if __name__ == "__main__":
    main()
