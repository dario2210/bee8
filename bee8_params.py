"""
bee8_params.py
===============
Central configuration for the bee8 mean-reversion strategy.

Strategy summary (from logika.odt):
  ENTRY (LONG) - on 1h timeframe, confluence of oversold oscillators
    - Bollinger %B (20,2)  < bb_long_entry_max     (default 0.20)
    - Stochastic K (14,3,3) < stoch_long_entry_max (default 25)
    - MFI (14)             < mfi_long_entry_max    (default 30)
    - RSI (14)             < rsi_long_entry_max    (default 40)
    Need >= long_entry_min_confluence indicators below threshold.
  HTF FILTER (4h, at least one):
    - 4h RSI(14) < htf_rsi_max  (default 45)
    - 4h BB %B   < htf_bb_max   (default 0.25)
    - price below 4h VWAP (or 4h EMA fallback)
  CONFIRMATION (lookback 6 bars on 1h):
    - in last lookback bars min BB %B < bb_extreme   (default 0.15)
    - current BB %B > min seen in lookback (bouncing back)

  EXIT (LONG) - on 1h timeframe, confluence of overbought oscillators
    - Bollinger %B (20,2)  > bb_long_exit_min      (default 0.80)
    - Stochastic K (14,3,3) > stoch_long_exit_min  (default 80)
    - MFI (14)             > mfi_long_exit_min     (default 70)
    - RSI (14)             > rsi_long_exit_min     (default 65)
    - ROC (14)             > roc_long_exit_min_pct (default 3.0%)
    Need >= long_exit_min_confluence indicators above threshold.
  STRONG EXIT TRIGGERS (any of):
    - Supertrend (10,3) on 1h flips DOWN -> UP
    - Price above 4h Ichimoku Cloud (senkou_a/senkou_b max)
"""

from __future__ import annotations

import json
import os

# Input data
CSV_PATH = "ethusdt_1h.csv"
INITIAL_CAPITAL = 10_000.0

# Binance data source
BINANCE_SYMBOL = "ETHUSDT"
BINANCE_INTERVAL = "1h"
BINANCE_MARKET = "spot"
BINANCE_START_DATE = "2021-01-01"
BINANCE_CSV_CACHE = None

# Trade direction (bee8 is long-only mean reversion by design)
TRADE_DIRECTION = "long"
ALLOW_LONGS = True
ALLOW_SHORTS = False
SHORT_TRADING_ENABLED = False

# ── Oscillator core (1h) ───────────────────────────────────────────────────────
BB_LEN = 20
BB_STD = 2.0
STOCH_K = 14
STOCH_D = 3
STOCH_SMOOTH = 3
MFI_LEN = 14
RSI_LEN = 14
ROC_LEN = 14
SUPERTREND_LEN = 10
SUPERTREND_MULT = 3.0
ICHIMOKU_TENKAN = 9
ICHIMOKU_KIJUN = 26
ICHIMOKU_SENKOU = 52

# Higher timeframe for filter
HTF_FILTER_INTERVAL = "4h"

# ── Entry thresholds (1h) ─────────────────────────────────────────────────────
BB_LONG_ENTRY_MAX = 0.20
STOCH_LONG_ENTRY_MAX = 25.0
MFI_LONG_ENTRY_MAX = 30.0
RSI_LONG_ENTRY_MAX = 40.0
LONG_ENTRY_MIN_CONFLUENCE = 3   # need at least 3 of 4 oversold

# Confirmation lookback
BB_EXTREME_LEVEL = 0.15
LOOKBACK_BARS = 6
REQUIRE_BOUNCE = True

# ── HTF filter (4h) ───────────────────────────────────────────────────────────
USE_HTF_FILTER = True
HTF_RSI_MAX = 45.0
HTF_BB_MAX = 0.25
HTF_REQUIRE_BELOW_VWAP = False

# ── Exit thresholds (1h) ──────────────────────────────────────────────────────
BB_LONG_EXIT_MIN = 0.80
STOCH_LONG_EXIT_MIN = 80.0
MFI_LONG_EXIT_MIN = 70.0
RSI_LONG_EXIT_MIN = 65.0
ROC_LONG_EXIT_MIN_PCT = 3.0
LONG_EXIT_MIN_CONFLUENCE = 3

# Strong exit triggers
SUPERTREND_FLIP_EXIT = True
ICHIMOKU_ABOVE_CLOUD_EXIT = False  # off by default - too aggressive

# ── Risk management ───────────────────────────────────────────────────────────
ATR_LEN = 14
ATR_STOP_ENABLED = False
ATR_STOP_MULTIPLIER = 2.5
LONG_EMERGENCY_SL_ENABLED = True
LONG_EMERGENCY_SL_CAPITAL_PCT = 0.05  # 5% capital stop
LONG_TP1_ENABLED = False
LONG_TP1_PCT = 0.015
LONG_TP1_FRACTION = 1.0 / 3.0
LONG_TP2_ENABLED = False
LONG_TP2_PCT = 0.04
LONG_TP2_FRACTION = 1.0 / 3.0
MAX_BARS_IN_TRADE = 0  # 0 = unlimited, otherwise time stop

# Fees and execution friction
FEE_RATE = 0.00035
SLIPPAGE_BPS = 2.0
SPREAD_BPS = 1.0

# WFO windows
OPT_DAYS = 120
LIVE_DAYS = 14

# ── WFO grids ────────────────────────────────────────────────────────────────
BB_LONG_ENTRY_MAX_GRID = [0.15, 0.20, 0.25]
BB_LONG_ENTRY_MAX_OPTIONS = [0.10, 0.15, 0.20, 0.25, 0.30]
STOCH_LONG_ENTRY_MAX_GRID = [20.0, 25.0, 30.0]
STOCH_LONG_ENTRY_MAX_OPTIONS = [15.0, 20.0, 25.0, 30.0, 35.0]
MFI_LONG_ENTRY_MAX_GRID = [25.0, 30.0, 35.0]
MFI_LONG_ENTRY_MAX_OPTIONS = [20.0, 25.0, 30.0, 35.0, 40.0]
RSI_LONG_ENTRY_MAX_GRID = [35.0, 40.0, 45.0]
RSI_LONG_ENTRY_MAX_OPTIONS = [30.0, 35.0, 40.0, 45.0, 50.0]
LONG_ENTRY_MIN_CONFLUENCE_GRID = [2, 3, 4]
LONG_ENTRY_MIN_CONFLUENCE_OPTIONS = [1, 2, 3, 4]

BB_LONG_EXIT_MIN_GRID = [0.75, 0.80, 0.85]
BB_LONG_EXIT_MIN_OPTIONS = [0.70, 0.75, 0.80, 0.85, 0.90]
STOCH_LONG_EXIT_MIN_GRID = [75.0, 80.0, 85.0]
STOCH_LONG_EXIT_MIN_OPTIONS = [70.0, 75.0, 80.0, 85.0, 90.0]
MFI_LONG_EXIT_MIN_GRID = [65.0, 70.0, 75.0]
MFI_LONG_EXIT_MIN_OPTIONS = [60.0, 65.0, 70.0, 75.0, 80.0]
RSI_LONG_EXIT_MIN_GRID = [60.0, 65.0, 70.0]
RSI_LONG_EXIT_MIN_OPTIONS = [55.0, 60.0, 65.0, 70.0, 75.0]
ROC_LONG_EXIT_MIN_PCT_GRID = [2.0, 3.0, 4.0]
ROC_LONG_EXIT_MIN_PCT_OPTIONS = [1.0, 2.0, 3.0, 4.0, 5.0]
LONG_EXIT_MIN_CONFLUENCE_GRID = [2, 3, 4]
LONG_EXIT_MIN_CONFLUENCE_OPTIONS = [1, 2, 3, 4, 5]

LOOKBACK_BARS_GRID = [4, 6, 8]
LOOKBACK_BARS_OPTIONS = [3, 4, 6, 8, 12]
BB_EXTREME_LEVEL_GRID = [0.10, 0.15, 0.20]
BB_EXTREME_LEVEL_OPTIONS = [0.05, 0.10, 0.15, 0.20, 0.25]

USE_HTF_FILTER_GRID = [True, False]
HTF_RSI_MAX_GRID = [40.0, 45.0, 50.0]
HTF_RSI_MAX_OPTIONS = [35.0, 40.0, 45.0, 50.0, 55.0]
HTF_BB_MAX_GRID = [0.20, 0.25, 0.30]
HTF_BB_MAX_OPTIONS = [0.15, 0.20, 0.25, 0.30, 0.35]

SUPERTREND_FLIP_EXIT_GRID = [True, False]
ICHIMOKU_ABOVE_CLOUD_EXIT_GRID = [False, True]

LONG_EMERGENCY_SL_CAPITAL_PCT_GRID = [0.0, 0.03, 0.05, 0.08]
LONG_EMERGENCY_SL_CAPITAL_PCT_OPTIONS = [0.0, 0.02, 0.03, 0.05, 0.08, 0.10]
LONG_TP1_PCT_GRID = [0.01, 0.015, 0.02]
LONG_TP1_PCT_OPTIONS = [0.005, 0.01, 0.015, 0.02, 0.03]
LONG_TP1_FRACTION_GRID = [0.25, 1.0 / 3.0, 0.5]
LONG_TP1_FRACTION_OPTIONS = [0.0, 0.25, 1.0 / 3.0, 0.5]
LONG_TP2_PCT_GRID = [0.03, 0.04, 0.05]
LONG_TP2_PCT_OPTIONS = [0.02, 0.03, 0.04, 0.05, 0.06]
LONG_TP2_FRACTION_GRID = [0.25, 1.0 / 3.0, 0.5]
LONG_TP2_FRACTION_OPTIONS = [0.0, 0.25, 1.0 / 3.0, 0.5]

# Defaults pack
DEFAULT_PARAMS = {
    "trade_direction": TRADE_DIRECTION,
    "allow_longs": ALLOW_LONGS,
    "allow_shorts": ALLOW_SHORTS,
    "short_trading_enabled": SHORT_TRADING_ENABLED,
    "bb_len": BB_LEN,
    "bb_std": BB_STD,
    "stoch_k": STOCH_K,
    "stoch_d": STOCH_D,
    "stoch_smooth": STOCH_SMOOTH,
    "mfi_len": MFI_LEN,
    "rsi_len": RSI_LEN,
    "roc_len": ROC_LEN,
    "supertrend_len": SUPERTREND_LEN,
    "supertrend_mult": SUPERTREND_MULT,
    "ichimoku_tenkan": ICHIMOKU_TENKAN,
    "ichimoku_kijun": ICHIMOKU_KIJUN,
    "ichimoku_senkou": ICHIMOKU_SENKOU,
    "htf_filter_interval": HTF_FILTER_INTERVAL,
    "bb_long_entry_max": BB_LONG_ENTRY_MAX,
    "stoch_long_entry_max": STOCH_LONG_ENTRY_MAX,
    "mfi_long_entry_max": MFI_LONG_ENTRY_MAX,
    "rsi_long_entry_max": RSI_LONG_ENTRY_MAX,
    "long_entry_min_confluence": LONG_ENTRY_MIN_CONFLUENCE,
    "bb_extreme_level": BB_EXTREME_LEVEL,
    "lookback_bars": LOOKBACK_BARS,
    "require_bounce": REQUIRE_BOUNCE,
    "use_htf_filter": USE_HTF_FILTER,
    "htf_rsi_max": HTF_RSI_MAX,
    "htf_bb_max": HTF_BB_MAX,
    "htf_require_below_vwap": HTF_REQUIRE_BELOW_VWAP,
    "bb_long_exit_min": BB_LONG_EXIT_MIN,
    "stoch_long_exit_min": STOCH_LONG_EXIT_MIN,
    "mfi_long_exit_min": MFI_LONG_EXIT_MIN,
    "rsi_long_exit_min": RSI_LONG_EXIT_MIN,
    "roc_long_exit_min_pct": ROC_LONG_EXIT_MIN_PCT,
    "long_exit_min_confluence": LONG_EXIT_MIN_CONFLUENCE,
    "supertrend_flip_exit": SUPERTREND_FLIP_EXIT,
    "ichimoku_above_cloud_exit": ICHIMOKU_ABOVE_CLOUD_EXIT,
    "atr_len": ATR_LEN,
    "atr_stop_enabled": ATR_STOP_ENABLED,
    "atr_stop_multiplier": ATR_STOP_MULTIPLIER,
    "long_emergency_sl_enabled": LONG_EMERGENCY_SL_ENABLED,
    "long_emergency_sl_capital_pct": LONG_EMERGENCY_SL_CAPITAL_PCT,
    "long_tp1_enabled": LONG_TP1_ENABLED,
    "long_tp1_pct": LONG_TP1_PCT,
    "long_tp1_fraction": LONG_TP1_FRACTION,
    "long_tp2_enabled": LONG_TP2_ENABLED,
    "long_tp2_pct": LONG_TP2_PCT,
    "long_tp2_fraction": LONG_TP2_FRACTION,
    "max_bars_in_trade": MAX_BARS_IN_TRADE,
    "fee_rate": FEE_RATE,
    "slippage_bps": SLIPPAGE_BPS,
    "spread_bps": SPREAD_BPS,
}

WFO_BEST_PARAMS_PATH = "bee8_wfo_best_params.json"


def load_params() -> dict:
    """Return strategy params; override DEFAULT_PARAMS with WFO best params if present."""
    params = dict(DEFAULT_PARAMS)
    json_path = os.path.join(os.path.dirname(__file__), WFO_BEST_PARAMS_PATH)
    if os.path.exists(json_path):
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                for key in DEFAULT_PARAMS:
                    if key in data:
                        params[key] = data[key]
            print(f"[params] Loaded WFO params from {WFO_BEST_PARAMS_PATH}")
        except Exception as exc:
            print(f"[params] Could not load {WFO_BEST_PARAMS_PATH}: {exc!r}")
    else:
        print(f"[params] Missing {WFO_BEST_PARAMS_PATH} - using DEFAULT_PARAMS")

    # bee8 is long-only by design
    params["trade_direction"] = "long"
    params["allow_longs"] = True
    params["allow_shorts"] = False
    params["short_trading_enabled"] = False
    return params


def save_params(params: dict, path: str = WFO_BEST_PARAMS_PATH) -> None:
    """Persist strategy params to JSON."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(params, f, indent=2)
    print(f"[params] Saved params -> {path}")
