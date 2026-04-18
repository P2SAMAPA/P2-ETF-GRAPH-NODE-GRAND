# P2 ETF GRAND Engine

**Graph Neural Diffusion (GRAND) for continuous-time ETF prediction.**

[![GitHub Actions](https://github.com/P2SAMAPA/P2-ETF-GRAPH-NODE-GRAND/actions/workflows/daily_run.yml/badge.svg)](https://github.com/P2SAMAPA/P2-ETF-GRAPH-NODE-GRAND/actions/workflows/daily_run.yml)

## Overview

This engine models ETF return dynamics as a continuous diffusion process on a graph. It uses the GRAND (Graph Neural Diffusion) architecture, which treats GNNs as discretizations of an underlying PDE and employs a neural ODE for temporal evolution. The result is a robust, depth-agnostic model that mitigates oversmoothing and captures complex spatiotemporal dependencies.

**Key Features:**
- **Continuous Diffusion**: Solves the PDE `dX/dt = (A - I) X W` using `torchdiffeq`.
- **Rolling Correlation Graphs**: Adjacency built from 60-day rolling correlations.
- **Three Universes**: FI/Commodities, Equity Sectors, and Combined.
- **Global & Adaptive Training**: Fixed 80/10/10 split and change‑point‑derived adaptive windows.

## Data

- **Input**: `P2SAMAPA/fi-etf-macro-signal-master-data` (master_data.parquet)
- **Output**: `P2SAMAPA/p2-etf-graph-node-grand-results`

## Usage

```bash
pip install -r requirements.txt
python trainer.py           # Runs training and pushes to HF
streamlit run streamlit_app.py
Configuration
All parameters are in config.py:

LOOKBACK_WINDOW: rolling correlation window (default 60)

HIDDEN_DIM: node feature dimension (default 32)

ODE_TIME: terminal integration time T (default 1.0)

EPOCHS: training epochs (default 100)
