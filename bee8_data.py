"""
bee8_data.py
=============
OHLCV loading and indicator preparation for bee8 mean-reversion strategy.

Indicators computed (1h main timeframe + 4h higher timeframe):
  - Bollinger Bands and %B
  - Stochastic %K / %D
  - MFI (Money Flow Index)
  - RSI
  - ROC (Rate of Change, %)
  - ATR
  - Supertrend (direction +1 / -1, and prior direction)
  - Ichimoku (tenkan, kijun, senkou_a, senkou_b)
  - VWAP (anchored daily, simple cumulative)
  - Lookback minima of BB%B over N bars

All higher-timeframe series are aligned to the base timeframe with forward fill
and a one-bar shift so the strategy never sees the active HTF candle (no
look-ahead).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from bee8_params import (
    ATR_LEN,
    BB_LEN,
    BB_STD,
    HTF_FILTER_INTERVAL,
    ICHIMOKU_KIJUN,
    ICHIMOKU_SENKOU,
    ICHIMOKU_TENKAN,
    LOOKBACK_BARS,
    MFI_LEN,
    ROC_LEN,
    RSI_LEN,
    STOCH_D,
    STOCH_K,
    STOCH_SMOOTH,
    SUPERTREND_LEN,
    SUPERTREND_MULT,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _estimate_base_tf_minutes(df: pd.DataFrame) -> float:
    if "time" not in df.columns or len(df) < 2:
        return 0.0
    diffs = (
        pd.to_datetime(df["time"], utc=True, errors="coerce")
        .sort_values()
        .diff()
        .dropna()
        .dt.total_seconds()
        / 60.0
    )
    if diffs.empty:
        return 0.0
    return float(diffs.median())


def _align_htf_series_to_base(
    htf_series: pd.Series,
    base_times: pd.Series,
    target_interval: str,
    base_minutes: float,
    previous_bars: int = 0,
) -> pd.Series:
    """Forward-fill HTF values onto base timeline. Shift one HTF interval so the
    HTF candle becomes visible only after it has closed (no look-ahead)."""
    source = htf_series.shift(int(previous_bars))
    source_index = pd.to_datetime(source.index, utc=True, errors="coerce")
    target_minutes = pd.Timedelta(target_interval).total_seconds() / 60.0
    if base_minutes > 0 and base_minutes < target_minutes:
        source_index = source_index + pd.Timedelta(target_interval)
    aligned = pd.Series(source.to_numpy(), index=source_index, dtype="float64").reindex(
        base_times, method="ffill"
    )
    return pd.Series(aligned.to_numpy(), index=base_times.index, dtype="float64")


def _higher_timeframe_ohlcv(df: pd.DataFrame, target_interval: str) -> pd.DataFrame:
    times = pd.to_datetime(df["time"], utc=True, errors="coerce")
    base_minutes = _estimate_base_tf_minutes(df)
    target_minutes = pd.Timedelta(target_interval).total_seconds() / 60.0
    if base_minutes <= 0 or base_minutes >= target_minutes:
        out = df[["time", "open", "high", "low", "close", "volume"]].copy()
        out["time"] = times
        return out.dropna(subset=["time"]).reset_index(drop=True)
    ohlcv = (
        pd.DataFrame(
            {
                "time": times,
                "open": pd.to_numeric(df["open"], errors="coerce"),
                "high": pd.to_numeric(df["high"], errors="coerce"),
                "low": pd.to_numeric(df["low"], errors="coerce"),
                "close": pd.to_numeric(df["close"], errors="coerce"),
                "volume": pd.to_numeric(df["volume"], errors="coerce"),
            }
        )
        .dropna(subset=["time"])
        .set_index("time")
        .resample(target_interval)
        .agg(
            {
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
            }
        )
        .dropna(subset=["open", "high", "low", "close"])
        .reset_index()
    )
    return ohlcv


# ─────────────────────────────────────────────────────────────────────────────
# Core indicator calculators (vectorised pandas)
# ─────────────────────────────────────────────────────────────────────────────

def bollinger_pct_b(close: pd.Series, length: int = BB_LEN, std_mult: float = BB_STD) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    mid = close.rolling(int(length)).mean()
    std = close.rolling(int(length)).std(ddof=0)
    upper = mid + std_mult * std
    lower = mid - std_mult * std
    width = (upper - lower).replace(0.0, np.nan)
    pct_b = (close - lower) / width
    return mid, upper, lower, pct_b


def stochastic(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    k_len: int = STOCH_K,
    d_len: int = STOCH_D,
    smooth: int = STOCH_SMOOTH,
) -> tuple[pd.Series, pd.Series]:
    ll = low.rolling(int(k_len)).min()
    hh = high.rolling(int(k_len)).max()
    raw_k = (close - ll) / (hh - ll).replace(0.0, np.nan) * 100.0
    k = raw_k.rolling(int(smooth)).mean()
    d = k.rolling(int(d_len)).mean()
    return k, d


def mfi(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
    length: int = MFI_LEN,
) -> pd.Series:
    typical = (high + low + close) / 3.0
    raw_money = typical * volume
    delta = typical.diff()
    pos = raw_money.where(delta > 0, 0.0)
    neg = raw_money.where(delta < 0, 0.0)
    pos_sum = pos.rolling(int(length)).sum()
    neg_sum = neg.rolling(int(length)).sum().replace(0.0, np.nan)
    mfr = pos_sum / neg_sum
    return 100.0 - (100.0 / (1.0 + mfr))


def rsi(close: pd.Series, length: int = RSI_LEN) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0.0)
    down = -delta.clip(upper=0.0)
    roll_up = up.ewm(alpha=1.0 / max(int(length), 1), adjust=False).mean()
    roll_down = down.ewm(alpha=1.0 / max(int(length), 1), adjust=False).mean()
    rs = roll_up / roll_down.replace(0.0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))


def roc(close: pd.Series, length: int = ROC_LEN) -> pd.Series:
    prior = close.shift(int(length))
    return (close - prior) / prior * 100.0


def atr(df: pd.DataFrame, length: int = ATR_LEN) -> pd.Series:
    prev_close = df["close"].shift(1)
    true_range = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return true_range.ewm(alpha=1.0 / max(int(length), 1), adjust=False).mean()


def supertrend(
    df: pd.DataFrame,
    length: int = SUPERTREND_LEN,
    mult: float = SUPERTREND_MULT,
) -> tuple[pd.Series, pd.Series]:
    """Return (supertrend_line, direction) where direction is +1 (up) / -1 (down)."""
    hl2 = (df["high"] + df["low"]) / 2.0
    atr_s = atr(df, length)
    upper_basic = hl2 + mult * atr_s
    lower_basic = hl2 - mult * atr_s

    upper = upper_basic.copy()
    lower = lower_basic.copy()
    direction = pd.Series(np.ones(len(df)), index=df.index, dtype="float64")
    st_line = pd.Series(np.nan, index=df.index, dtype="float64")

    close = df["close"].to_numpy(copy=True)
    upper_arr = upper_basic.to_numpy(copy=True)
    lower_arr = lower_basic.to_numpy(copy=True)
    direction_arr = np.ones(len(df))
    st_arr = np.full(len(df), np.nan)

    for i in range(1, len(df)):
        if not np.isnan(upper_arr[i - 1]) and not np.isnan(upper_arr[i]):
            if upper_arr[i] < upper_arr[i - 1] or close[i - 1] > upper_arr[i - 1]:
                pass
            else:
                upper_arr[i] = upper_arr[i - 1]
        if not np.isnan(lower_arr[i - 1]) and not np.isnan(lower_arr[i]):
            if lower_arr[i] > lower_arr[i - 1] or close[i - 1] < lower_arr[i - 1]:
                pass
            else:
                lower_arr[i] = lower_arr[i - 1]

        prev_dir = direction_arr[i - 1]
        if np.isnan(upper_arr[i]) or np.isnan(lower_arr[i]):
            direction_arr[i] = prev_dir
            continue
        if prev_dir == 1 and close[i] < lower_arr[i]:
            direction_arr[i] = -1.0
        elif prev_dir == -1 and close[i] > upper_arr[i]:
            direction_arr[i] = 1.0
        else:
            direction_arr[i] = prev_dir
        st_arr[i] = lower_arr[i] if direction_arr[i] == 1 else upper_arr[i]

    direction = pd.Series(direction_arr, index=df.index, dtype="float64")
    st_line = pd.Series(st_arr, index=df.index, dtype="float64")
    return st_line, direction


def ichimoku(
    df: pd.DataFrame,
    tenkan: int = ICHIMOKU_TENKAN,
    kijun: int = ICHIMOKU_KIJUN,
    senkou: int = ICHIMOKU_SENKOU,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    high, low = df["high"], df["low"]
    tenkan_line = (high.rolling(int(tenkan)).max() + low.rolling(int(tenkan)).min()) / 2.0
    kijun_line = (high.rolling(int(kijun)).max() + low.rolling(int(kijun)).min()) / 2.0
    senkou_a = ((tenkan_line + kijun_line) / 2.0).shift(int(kijun))
    senkou_b = ((high.rolling(int(senkou)).max() + low.rolling(int(senkou)).min()) / 2.0).shift(int(kijun))
    return tenkan_line, kijun_line, senkou_a, senkou_b


def daily_vwap(df: pd.DataFrame) -> pd.Series:
    """VWAP anchored daily (resets every UTC day)."""
    times = pd.to_datetime(df["time"], utc=True, errors="coerce")
    typical = (df["high"] + df["low"] + df["close"]) / 3.0
    pv = typical * df["volume"]
    day = times.dt.floor("D")
    cum_pv = pv.groupby(day).cumsum()
    cum_v = df["volume"].groupby(day).cumsum().replace(0.0, np.nan)
    return cum_pv / cum_v


# ─────────────────────────────────────────────────────────────────────────────
# Indicator preparation pipeline
# ─────────────────────────────────────────────────────────────────────────────

def prepare_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Compute every indicator the bee8 engine and dashboard need."""
    df = df.copy()
    df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
    df = df.sort_values("time").reset_index(drop=True)

    # 1h indicators
    bb_mid, bb_up, bb_lo, bb_pct = bollinger_pct_b(df["close"], BB_LEN, BB_STD)
    df["bb_mid"] = bb_mid
    df["bb_upper"] = bb_up
    df["bb_lower"] = bb_lo
    df["bb_pct"] = bb_pct

    k, d = stochastic(df["high"], df["low"], df["close"], STOCH_K, STOCH_D, STOCH_SMOOTH)
    df["stoch_k"] = k
    df["stoch_d"] = d

    df["mfi"] = mfi(df["high"], df["low"], df["close"], df["volume"], MFI_LEN)
    df["rsi"] = rsi(df["close"], RSI_LEN)
    df["roc"] = roc(df["close"], ROC_LEN)
    df["atr"] = atr(df, ATR_LEN)

    st_line, st_dir = supertrend(df, SUPERTREND_LEN, SUPERTREND_MULT)
    df["supertrend"] = st_line
    df["supertrend_dir"] = st_dir
    df["supertrend_dir_prev"] = df["supertrend_dir"].shift(1)
    df["supertrend_flip_up"] = (df["supertrend_dir_prev"] < 0) & (df["supertrend_dir"] > 0)

    tenkan, kijun, senkou_a, senkou_b = ichimoku(df, ICHIMOKU_TENKAN, ICHIMOKU_KIJUN, ICHIMOKU_SENKOU)
    df["ichimoku_tenkan"] = tenkan
    df["ichimoku_kijun"] = kijun
    df["ichimoku_senkou_a"] = senkou_a
    df["ichimoku_senkou_b"] = senkou_b
    df["above_cloud"] = df["close"] > df[["ichimoku_senkou_a", "ichimoku_senkou_b"]].max(axis=1)

    if "volume" in df.columns and df["volume"].fillna(0).sum() > 0:
        df["vwap"] = daily_vwap(df)
    else:
        df["vwap"] = np.nan

    # Lookback minima for confirmation logic
    lb = int(LOOKBACK_BARS)
    df[f"bb_pct_min_{lb}"] = df["bb_pct"].rolling(lb, min_periods=1).min()
    df[f"rsi_min_{lb}"] = df["rsi"].rolling(lb, min_periods=1).min()
    df[f"stoch_min_{lb}"] = df["stoch_k"].rolling(lb, min_periods=1).min()
    df[f"mfi_min_{lb}"] = df["mfi"].rolling(lb, min_periods=1).min()

    # HTF (4h) indicators aligned to base
    htf_df = _higher_timeframe_ohlcv(df, HTF_FILTER_INTERVAL)
    base_minutes = _estimate_base_tf_minutes(df)
    base_times = df["time"]
    htf_index = pd.to_datetime(htf_df["time"], utc=True, errors="coerce")

    htf_rsi = rsi(htf_df["close"], RSI_LEN)
    _, _, _, htf_bb_pct = bollinger_pct_b(htf_df["close"], BB_LEN, BB_STD)
    htf_vwap_series = daily_vwap(htf_df) if "volume" in htf_df.columns and htf_df["volume"].fillna(0).sum() > 0 else pd.Series(np.nan, index=htf_df.index)
    tenkan_h, kijun_h, senkou_a_h, senkou_b_h = ichimoku(
        htf_df, ICHIMOKU_TENKAN, ICHIMOKU_KIJUN, ICHIMOKU_SENKOU
    )
    htf_close = htf_df["close"]
    htf_above_cloud = (htf_close > pd.concat([senkou_a_h, senkou_b_h], axis=1).max(axis=1)).astype(float)

    for series, col in [
        (htf_rsi, "htf_rsi"),
        (htf_bb_pct, "htf_bb_pct"),
        (htf_vwap_series, "htf_vwap"),
        (htf_close, "htf_close"),
        (htf_above_cloud, "htf_above_cloud"),
        (senkou_a_h, "htf_senkou_a"),
        (senkou_b_h, "htf_senkou_b"),
    ]:
        idx_series = pd.Series(series.to_numpy(), index=htf_index, dtype="float64")
        df[col] = _align_htf_series_to_base(
            idx_series, base_times, HTF_FILTER_INTERVAL, base_minutes
        ).to_numpy()

    df["below_htf_vwap"] = (df["close"] < df["htf_vwap"]).astype(float)

    return df


_TIME_COL_ALIASES = ("open_time", "open_time_ms", "open_time_utc", "time", "timestamp", "date")


def load_klines(csv_path: str) -> pd.DataFrame:
    """Load candle CSV.

    Accepts a wide range of common Binance dump column names:
      - open_time (ms or seconds, numeric)
      - open_time_ms (numeric ms)
      - open_time_utc (ISO string)
      - time / timestamp / date
    """
    df = pd.read_csv(csv_path)
    df.rename(columns={c: c.lower() for c in df.columns}, inplace=True)

    time_col = next((c for c in _TIME_COL_ALIASES if c in df.columns), None)
    if time_col is None:
        raise ValueError(
            "Missing time column in CSV - expected one of: "
            + ", ".join(_TIME_COL_ALIASES)
        )

    col = df[time_col]
    try:
        is_numeric = np.issubdtype(col.dtype, np.number)
    except TypeError:
        is_numeric = False
    if is_numeric:
        max_val = float(pd.to_numeric(col, errors="coerce").max())
        if max_val > 1e14:
            unit = "us"
        elif max_val > 1e11:
            unit = "ms"
        else:
            unit = "s"
        df["time"] = pd.to_datetime(col, unit=unit, utc=True)
    else:
        df["time"] = pd.to_datetime(col, utc=True, errors="coerce")

    for col_name in ["open", "high", "low", "close", "volume"]:
        if col_name not in df.columns:
            raise ValueError(f"Missing '{col_name}' column in CSV.")
        df[col_name] = pd.to_numeric(df[col_name], errors="coerce")

    df = df.dropna(subset=["time"])
    df = df.sort_values("time").reset_index(drop=True)
    return df


def format_ts(ts) -> str:
    if pd.isna(ts):
        return "NaT"
    return pd.Timestamp(ts).tz_convert("UTC").strftime("%Y-%m-%d %H:%M")
