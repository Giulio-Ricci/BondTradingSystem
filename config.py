"""Configuration for the Bond Trading System - RR Duration Strategy."""

import os

# ── Paths ────────────────────────────────────────────────────────────────────
DATA_DIR = r"C:\Users\rgiul\Desktop\PDFs\dati bond"
SAVE_DIR = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(SAVE_DIR, exist_ok=True)

# ── RR Column Mappings ──────────────────────────────────────────────────────
RR_TENORS = {
    "1W": ("TY1W25DP", "TY1W25DC"),
    "1M": ("TY1M25DP", "TY1M25DC"),
    "3M": ("TY3M25DP", "TY3M25DC"),
    "6M": ("TY6M25DP", "TY6M25DC"),
}

# ── ETF Tickers ─────────────────────────────────────────────────────────────
ETF_TICKERS = ["SHV", "SHY", "IEI", "IEF", "TLH", "TLT", "GOVT"]

# ── Default Strategy Parameters ─────────────────────────────────────────────
DEFAULT_PARAMS = {
    "tl_calm": -3.0,            # CALM SHORT threshold (z <= this -> SHORT)
    "ts_calm": 2.5,             # CALM LONG  threshold (z >= this -> LONG)
    "tl_stress": -4.0,          # STRESS SHORT threshold
    "ts_stress": 99,            # STRESS LONG threshold (disabled)
    "allow_long_stress": False,
    "long_etf": "TLT",
    "short_etf": "SHV",
    "cooldown": 5,              # Min days between signals
    "delay": 1,                 # Execution delay (days)
    "tcost_bps": 5,             # Transaction cost in basis points
    "w_long": 1.0,              # Weight when LONG  -> 100 % long_etf
    "w_short": 0.0,             # Weight when SHORT -> 100 % short_etf
    "w_initial": 0.5,           # Initial allocation (50/50)
}

# ── Z-Score Rolling Windows ─────────────────────────────────────────────────
ZSCORE_WINDOW = 250
ZSCORE_MIN_PERIODS = 63
MOVE_MEDIAN_WINDOW = 504
MOVE_RATIO_CLIP = (0.5, 2.5)

# ── HMM ─────────────────────────────────────────────────────────────────────
HMM_N_SEEDS = 10
HMM_N_ITER = 300

# ── Backtest ────────────────────────────────────────────────────────────────
BACKTEST_START = "2009-01-01"
RISK_FREE_RATE = 0.02

# ── Grid Search Ranges ──────────────────────────────────────────────────────
GRID_TL_CALM = (-4.0, -1.4, 0.5)     # arange args
GRID_TS_CALM = (1.5, 4.1, 0.5)
GRID_TL_STRESS = (-5.0, -2.9, 0.5)

# ── Walk-Forward ────────────────────────────────────────────────────────────
WF_FIRST_OOS_YEAR = 2014
WF_LAST_OOS_YEAR = 2025
WF_MIN_IS_OBS = 504
WF_MIN_OOS_OBS = 20

# ── Bootstrap ───────────────────────────────────────────────────────────────
BOOTSTRAP_N = 1000
BOOTSTRAP_BLOCK = 21
BOOTSTRAP_SEED = 42

# ── TC Sensitivity ──────────────────────────────────────────────────────────
TC_TEST_VALUES = [0, 2, 5, 10, 15, 20, 30, 50]

# ── Scorecard Thresholds ────────────────────────────────────────────────────
SC_IS_SHARPE_MIN = 0.0
SC_IS_EXCESS_MIN = 0.01
SC_WF_EXCESS_MIN = 0.0
SC_WF_POS_YEARS_MIN = 0.50
SC_DECAY_MAX = 0.5
SC_PARAM_STD_MAX = 1.0
SC_BOOT_P_MIN = 0.60
SC_REGIME_CONCORDANCE_MIN = 0.70
SC_MIN_EVENTS = 4
