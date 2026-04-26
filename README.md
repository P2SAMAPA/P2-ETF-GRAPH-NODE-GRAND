# P2-ETF-GRAPH-NODE-GRAND

**Temporal Graph Neural Network (GCN + GRU) for dynamic ETF prediction.**

[![GitHub Actions](https://github.com/P2SAMAPA/P2-ETF-GRAPH-NODE-GRAND/actions/workflows/daily_run.yml/badge.svg)](https://github.com/P2SAMAPA/P2-ETF-GRAPH-NODE-GRAND/actions/workflows/daily_run.yml)

## Overview

This engine models ETF return dynamics as an evolving graph. Each trading day, a new graph is constructed where ETF nodes are connected if their rolling correlation exceeds a threshold. Node features combine recent returns with current macro conditions (VIX, DXY, spreads, etc.). A **GCN + GRU** temporal model processes this sequence of daily snapshots to predict next‑day returns for every ETF.

**Key Features:**
- **Dynamic Daily Graphs**: Adjacency rebuilt every trading day from 63‑day rolling correlations.
- **Macro Conditioning**: VIX, DXY, T10Y2Y, TBILL_3M, IG_SPREAD, HY_SPREAD concatenated to node features.
- **Temporal GNN (GCN + GRU)**: Graph convolution for spatial message‑passing, GRU for temporal memory across days.
- **Multi‑Output Prediction**: Predicts per‑ETF next‑day returns, not a market average.
- **Three Universes**: FI/Commodities, Equity Sectors, and Combined.
- **Three Training Modes**:
  - **Daily (504d)** — trained on the most recent 2 years.
  - **Global** — trained on the full 2008‑YTD history.
  - **Adaptive** — change‑point‑detected window for regime‑aware training.

## Data

- **Input**: `P2SAMAPA/fi-etf-macro-signal-master-data` (master_data.parquet)
- **Output**: `P2SAMAPA/p2-etf-graph-node-grand-results`

## Universe Coverage

| Universe | Tickers |
|----------|---------|
| **FI / Commodities** | TLT, VCIT, LQD, HYG, VNQ, GLD, SLV |
| **Equity Sectors** | SPY, QQQ, XLK, XLF, XLE, XLV, XLI, XLY, XLP, XLU, GDX, XME, IWF, XSD, XBI, IWM, XLB, XLRE |
| **Combined** | All tickers above |

## Methodology

1. **Graph Construction**: 63‑day rolling correlation → adjacency (edges where |corr| > 0.1).
2. **Node Features**: Last 5 daily returns + current macro values.
3. **Temporal Model**: GCN → GRU → GCN → … → Linear. The GRU carries hidden state across days.
4. **Training**: MSE loss on per‑ETF next‑day returns, early stopping.
5. **Inference**: Latest graph snapshot produces per‑ETF predictions.

## Architecture Change (v2)

The original GRAND model used a continuous‑time neural ODE (`torchdiffeq.odeint`) which suffered from NaN gradients and scalar‑target training. The engine was rewritten to use a **discrete GCN + GRU** architecture that is:

- Numerically stable (no ODE solver failures)
- Multi‑output (predicts per‑ETF returns)
- Faster to train (no adaptive‑step integration)

## Usage

```bash
pip install -r requirements.txt
python trainer.py           # Runs training and pushes to HF
streamlit run streamlit_app.py
Configuration
All parameters are in config.py:

Parameter	Default	Description
LOOKBACK_WINDOW	63	Rolling correlation window
REBALANCE_FREQ	1	Rebuild graph every N days
HIDDEN_DIM	64	GCN hidden dimension
NUM_LAYERS	3	Number of GCN layers
DROPOUT	0.1	Dropout rate
EPOCHS	100	Training epochs
DAILY_LOOKBACK	504	Days for Daily training tab
Dashboard
The Streamlit app (streamlit_app.py) displays three tabs per universe:

📅 Daily (504d) — trained on the most recent 2 years.

🌍 Global — trained on the full 2008‑YTD history.

🔄 Adaptive — change‑point‑detected window.

Each tab shows a hero card with the selected ETF, predicted return, backtest metrics, and a table of all ETF predictions.

Project Structure
text
P2-ETF-GRAPH-NODE-GRAND/
├── config.py                  # Paths, universes, model hyperparameters
├── data_manager.py            # Data loading and preprocessing
├── graph_builder.py           # Rolling correlation → daily graph snapshots
├── grand_model.py             # GCN + GRU temporal model
├── trainer.py                 # Global, Adaptive, and Daily training
├── change_point_detector.py   # Adaptive window detection
├── push_results.py            # Hugging Face upload/download
├── streamlit_app.py           # Dashboard UI
├── us_calendar.py             # NYSE trading calendar
├── requirements.txt
└── .github/workflows/
    └── daily_run.yml
License
MIT License
