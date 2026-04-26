"""
Configuration for P2-ETF-GRAPH-NODE-GRAND.
"""
import os

HF_INPUT_DATASET = "P2SAMAPA/fi-etf-macro-signal-master-data"
HF_INPUT_FILE = "master_data.parquet"
HF_OUTPUT_DATASET = "P2SAMAPA/p2-etf-graph-node-grand-results"
HF_TOKEN = os.environ.get("HF_TOKEN")

# Universes – SPY, IWF, XSD, XBI added; XLB, XLRE kept
FI_COMMODITY_TICKERS = ["TLT", "VCIT", "LQD", "HYG", "VNQ", "GLD", "SLV"]
EQUITY_TICKERS = [
    "SPY", "QQQ", "XLK", "XLF", "XLE", "XLV",
    "XLI", "XLY", "XLP", "XLU", "GDX", "XME",
    "IWF", "XSD", "XBI", "IWM",
    "XLB", "XLRE"
]
COMBINED_TICKERS = FI_COMMODITY_TICKERS + EQUITY_TICKERS
BENCHMARK_FI = "AGG"
BENCHMARK_EQ = "SPY"
MACRO_COLS = ["VIX", "DXY", "T10Y2Y", "TBILL_3M", "IG_SPREAD", "HY_SPREAD"]

# Training parameters
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1
MIN_TRAIN_DAYS = 400
MIN_TEST_DAYS = 63
TRADING_DAYS_PER_YEAR = 252
DAILY_LOOKBACK = 504

# Change Point Detection
CP_PENALTY = 3.0
CP_MODEL = "l2"
CP_MIN_DAYS_BETWEEN = 20
CP_CONSENSUS_FRACTION = 0.5
ADAPTIVE_MAX_LOOKBACK = 252

# Graph construction
LOOKBACK_WINDOW = 63
REBALANCE_FREQ = 1
CORRELATION_THRESHOLD = 0.3            # sparser graph (was 0.1)
FEATURE_WINDOW = 20                    # longer return history (was 5)

# GCN+GRU model
HIDDEN_DIM = 64
NUM_LAYERS = 3
DROPOUT = 0.1
LEARNING_RATE = 0.001
EPOCHS = 100
PATIENCE = 15
DEVICE = "cpu"
