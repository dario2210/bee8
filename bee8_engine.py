"""
bee8_engine.py
===============
Decision engine for the bee8 mean-reversion long strategy.

Reads pre-computed indicators (see bee8_data.prepare_indicators) and emits
entry/exit signals based on oscillator confluence + HTF filter + Supertrend
trigger.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional

import numpy as np

Side = Literal["long", "short"]
Action = Literal["none", "open_long", "open_short", "close_force", "close_partial"]


@dataclass
class BarData:
    time: object
    open: float
    high: float
    low: float
    close: float
    volume: float
    bb_pct: float
    bb_mid: float
    bb_upper: float
    bb_lower: float
    stoch_k: float
    stoch_d: float
    mfi: float
    rsi: float
    roc: float
    atr: float
    supertrend: float
    supertrend_dir: float
    supertrend_dir_prev: float
    supertrend_flip_up: bool
    above_cloud: bool
    vwap: float
    htf_rsi: float
    htf_bb_pct: float
    htf_vwap: float
    htf_close: float
    htf_above_cloud: float
    bb_pct_min_lb: float
    rsi_min_lb: float
    stoch_min_lb: float
    mfi_min_lb: float


@dataclass
class Signal:
    action: Action
    reason: str = ""
    exit_price: Optional[float] = None
    meta: dict = field(default_factory=dict)


@dataclass
class PositionState:
    side: Side
    entry_price: float
    entry_time: object
    bars_in_position: int = 0
    entry_meta: dict = field(default_factory=dict)
    stop_price: float = np.nan
    entry_atr: float = np.nan
    remaining_fraction: float = 1.0
    tp1_taken: bool = False
    tp2_taken: bool = False
    tp1_protection_after_bars: int = 0
    trade_id: int = 0


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _f(value, default=np.nan) -> float:
    try:
        if value is None:
            return float(default)
        v = float(value)
        return v if not np.isnan(v) else float(default)
    except (TypeError, ValueError):
        return float(default)


def _flag(value) -> bool:
    if value is None or value is False:
        return False
    if isinstance(value, float) and np.isnan(value):
        return False
    return bool(value)


def bar_from_row(row, params: dict | None = None) -> BarData:
    lb_key_bb = "bb_pct_min_lb"
    lb_key_rsi = "rsi_min_lb"
    lb_key_stoch = "stoch_min_lb"
    lb_key_mfi = "mfi_min_lb"

    lookback = int(params.get("lookback_bars", 6)) if params else 6
    bb_lb_col = f"bb_pct_min_{lookback}"
    rsi_lb_col = f"rsi_min_{lookback}"
    stoch_lb_col = f"stoch_min_{lookback}"
    mfi_lb_col = f"mfi_min_{lookback}"

    return BarData(
        time=row["time"],
        open=_f(row.get("open", row["close"]), 0.0),
        high=_f(row.get("high", row["close"]), 0.0),
        low=_f(row.get("low", row["close"]), 0.0),
        close=_f(row["close"], 0.0),
        volume=_f(row.get("volume", 0.0), 0.0),
        bb_pct=_f(row.get("bb_pct")),
        bb_mid=_f(row.get("bb_mid")),
        bb_upper=_f(row.get("bb_upper")),
        bb_lower=_f(row.get("bb_lower")),
        stoch_k=_f(row.get("stoch_k")),
        stoch_d=_f(row.get("stoch_d")),
        mfi=_f(row.get("mfi")),
        rsi=_f(row.get("rsi")),
        roc=_f(row.get("roc")),
        atr=_f(row.get("atr")),
        supertrend=_f(row.get("supertrend")),
        supertrend_dir=_f(row.get("supertrend_dir")),
        supertrend_dir_prev=_f(row.get("supertrend_dir_prev")),
        supertrend_flip_up=_flag(row.get("supertrend_flip_up")),
        above_cloud=_flag(row.get("above_cloud")),
        vwap=_f(row.get("vwap")),
        htf_rsi=_f(row.get("htf_rsi")),
        htf_bb_pct=_f(row.get("htf_bb_pct")),
        htf_vwap=_f(row.get("htf_vwap")),
        htf_close=_f(row.get("htf_close")),
        htf_above_cloud=_f(row.get("htf_above_cloud"), 0.0),
        bb_pct_min_lb=_f(row.get(bb_lb_col, row.get(lb_key_bb))),
        rsi_min_lb=_f(row.get(rsi_lb_col, row.get(lb_key_rsi))),
        stoch_min_lb=_f(row.get(stoch_lb_col, row.get(lb_key_stoch))),
        mfi_min_lb=_f(row.get(mfi_lb_col, row.get(lb_key_mfi))),
    )


def compute_trade_close(
    entry_price: float,
    exit_price: float,
    side: Side,
    fee_rate: float,
    capital_at_open: float,
) -> dict:
    if side == "long":
        gross_ret = (exit_price / entry_price) - 1.0
    else:
        gross_ret = (entry_price / exit_price) - 1.0
    net_ret = (1.0 + gross_ret) * (1.0 - fee_rate) ** 2 - 1.0
    fee_ret = net_ret - gross_ret
    fee_usd = abs(fee_ret) * capital_at_open
    pnl = capital_at_open * net_ret
    return {
        "gross_ret": gross_ret,
        "net_ret": net_ret,
        "fee_ret": fee_ret,
        "fee_usd": fee_usd,
        "pnl": pnl,
    }


def apply_slippage(
    price: float,
    side: Side,
    action: str,
    slippage_bps: float = 0.0,
    spread_bps: float = 0.0,
) -> float:
    if slippage_bps == 0.0 and spread_bps == 0.0:
        return price
    total_bps = slippage_bps + spread_bps / 2.0
    factor = total_bps / 10_000.0
    opening = action == "open"
    if (side == "long" and opening) or (side == "short" and not opening):
        return price * (1.0 + factor)
    return price * (1.0 - factor)


def build_position_state(
    side: Side,
    entry_price: float,
    entry_time: object,
    bar: BarData,
    params: dict,
    entry_meta: Optional[dict] = None,
) -> PositionState:
    stop_price = np.nan
    entry_atr = np.nan
    if bool(params.get("atr_stop_enabled", False)) and not np.isnan(bar.atr):
        atr_mult = float(params.get("atr_stop_multiplier", 2.5))
        offset = bar.atr * atr_mult
        stop_price = entry_price - offset if side == "long" else entry_price + offset
        entry_atr = float(bar.atr)
    elif side == "long" and bool(params.get("long_emergency_sl_enabled", True)):
        emergency_pct = float(params.get("long_emergency_sl_capital_pct", 0.05) or 0.0)
        if emergency_pct > 0.0:
            stop_price = entry_price * (1.0 - emergency_pct)
    return PositionState(
        side=side,
        entry_price=entry_price,
        entry_time=entry_time,
        entry_meta=entry_meta or {},
        stop_price=stop_price,
        entry_atr=entry_atr,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Signal logic
# ─────────────────────────────────────────────────────────────────────────────

def _entry_confluence_count(bar: BarData, params: dict) -> tuple[int, dict]:
    """Count how many of the 4 oversold indicators are below their thresholds."""
    flags = {
        "bb_oversold": (not np.isnan(bar.bb_pct))
        and bar.bb_pct < float(params.get("bb_long_entry_max", 0.20)),
        "stoch_oversold": (not np.isnan(bar.stoch_k))
        and bar.stoch_k < float(params.get("stoch_long_entry_max", 25.0)),
        "mfi_oversold": (not np.isnan(bar.mfi))
        and bar.mfi < float(params.get("mfi_long_entry_max", 30.0)),
        "rsi_oversold": (not np.isnan(bar.rsi))
        and bar.rsi < float(params.get("rsi_long_entry_max", 40.0)),
    }
    return sum(1 for v in flags.values() if v), flags


def _exit_confluence_count(bar: BarData, params: dict) -> tuple[int, dict]:
    flags = {
        "bb_overbought": (not np.isnan(bar.bb_pct))
        and bar.bb_pct > float(params.get("bb_long_exit_min", 0.80)),
        "stoch_overbought": (not np.isnan(bar.stoch_k))
        and bar.stoch_k > float(params.get("stoch_long_exit_min", 80.0)),
        "mfi_overbought": (not np.isnan(bar.mfi))
        and bar.mfi > float(params.get("mfi_long_exit_min", 70.0)),
        "rsi_overbought": (not np.isnan(bar.rsi))
        and bar.rsi > float(params.get("rsi_long_exit_min", 65.0)),
        "roc_high": (not np.isnan(bar.roc))
        and bar.roc > float(params.get("roc_long_exit_min_pct", 3.0)),
    }
    return sum(1 for v in flags.values() if v), flags


def _htf_filter_ok(bar: BarData, params: dict) -> tuple[bool, dict]:
    if not bool(params.get("use_htf_filter", True)):
        return True, {"htf_filter": "off"}
    rsi_max = float(params.get("htf_rsi_max", 45.0))
    bb_max = float(params.get("htf_bb_max", 0.25))
    require_vwap = bool(params.get("htf_require_below_vwap", False))

    rsi_ok = (not np.isnan(bar.htf_rsi)) and bar.htf_rsi < rsi_max
    bb_ok = (not np.isnan(bar.htf_bb_pct)) and bar.htf_bb_pct < bb_max
    vwap_ok = (
        (not np.isnan(bar.htf_vwap))
        and (not np.isnan(bar.close))
        and bar.close < bar.htf_vwap
    )
    flags = {"htf_rsi_ok": rsi_ok, "htf_bb_ok": bb_ok, "htf_vwap_ok": vwap_ok}
    if require_vwap:
        return (rsi_ok or bb_ok) and vwap_ok, flags
    return (rsi_ok or bb_ok or vwap_ok), flags


def _bounce_confirmation_ok(bar: BarData, params: dict) -> tuple[bool, dict]:
    if not bool(params.get("require_bounce", True)):
        return True, {"bounce": "off"}
    extreme = float(params.get("bb_extreme_level", 0.15))
    saw_extreme = (not np.isnan(bar.bb_pct_min_lb)) and bar.bb_pct_min_lb < extreme
    bouncing = (
        (not np.isnan(bar.bb_pct))
        and (not np.isnan(bar.bb_pct_min_lb))
        and bar.bb_pct > bar.bb_pct_min_lb
    )
    return saw_extreme and bouncing, {"saw_extreme": saw_extreme, "bouncing": bouncing}


def generate_entry_signal(
    bar: BarData,
    prev_bar: BarData,
    params: dict,
    position: Optional[PositionState],
) -> Signal:
    if position is not None:
        return Signal(action="none")

    if not bool(params.get("allow_longs", True)):
        return Signal(action="none")

    confluence, flags = _entry_confluence_count(bar, params)
    min_conf = int(params.get("long_entry_min_confluence", 3))
    if confluence < min_conf:
        return Signal(action="none")

    htf_ok, htf_flags = _htf_filter_ok(bar, params)
    if not htf_ok:
        return Signal(action="none")

    bounce_ok, bounce_flags = _bounce_confirmation_ok(bar, params)
    if not bounce_ok:
        return Signal(action="none")

    meta = {
        "entry_bb_pct": round(bar.bb_pct, 4) if not np.isnan(bar.bb_pct) else np.nan,
        "entry_stoch_k": round(bar.stoch_k, 2) if not np.isnan(bar.stoch_k) else np.nan,
        "entry_mfi": round(bar.mfi, 2) if not np.isnan(bar.mfi) else np.nan,
        "entry_rsi": round(bar.rsi, 2) if not np.isnan(bar.rsi) else np.nan,
        "entry_roc": round(bar.roc, 3) if not np.isnan(bar.roc) else np.nan,
        "entry_htf_rsi": round(bar.htf_rsi, 2) if not np.isnan(bar.htf_rsi) else np.nan,
        "entry_htf_bb_pct": round(bar.htf_bb_pct, 4) if not np.isnan(bar.htf_bb_pct) else np.nan,
        "entry_confluence": confluence,
        "entry_min_bb_lb": round(bar.bb_pct_min_lb, 4) if not np.isnan(bar.bb_pct_min_lb) else np.nan,
        "entry_atr": round(bar.atr, 4) if not np.isnan(bar.atr) else np.nan,
        "entry_supertrend_dir": bar.supertrend_dir,
        **flags,
        **htf_flags,
        **bounce_flags,
    }

    return Signal(
        action="open_long",
        reason="MR_LONG_CONFLUENCE_BOUNCE",
        meta=meta,
    )


def generate_emergency_exit_signal(
    bar: BarData,
    params: dict,
    position: PositionState,
) -> Signal:
    if position.side != "long":
        return Signal(action="none")

    enabled = bool(params.get("long_emergency_sl_enabled", True))
    capital_pct = float(params.get("long_emergency_sl_capital_pct", 0.05) or 0.0)
    remaining = max(float(position.remaining_fraction), 1e-9)
    if not enabled or capital_pct <= 0.0:
        return Signal(action="none")

    price_pct = capital_pct / remaining
    stop_price = position.entry_price * (1.0 - price_pct)
    if stop_price <= 0.0 or np.isnan(bar.low) or bar.low > stop_price:
        return Signal(action="none")

    return Signal(
        action="close_force",
        reason="LONG_EMERGENCY_SL_CAPITAL",
        exit_price=stop_price,
        meta={
            "bars_in_position": position.bars_in_position,
            "exit_trigger": "LONG_EMERGENCY_SL_CAPITAL",
            "stop_price": round(stop_price, 4),
            "emergency_sl_capital_pct": capital_pct,
            "remaining_fraction_before": position.remaining_fraction,
        },
    )


def generate_partial_exit_signal(
    bar: BarData,
    params: dict,
    position: PositionState,
) -> Signal:
    if position.side != "long" or position.remaining_fraction <= 0.0:
        return Signal(action="none")

    if not position.tp1_taken and bool(params.get("long_tp1_enabled", False)):
        tp_pct = float(params.get("long_tp1_pct", 0.015) or 0.0)
        fraction = float(params.get("long_tp1_fraction", 1.0 / 3.0) or 0.0)
        tp_name = "TP1"
    elif not position.tp2_taken and bool(params.get("long_tp2_enabled", False)):
        tp_pct = float(params.get("long_tp2_pct", 0.04) or 0.0)
        fraction = float(params.get("long_tp2_fraction", 1.0 / 3.0) or 0.0)
        tp_name = "TP2"
    else:
        return Signal(action="none")

    if tp_pct <= 0.0 or fraction <= 0.0:
        return Signal(action="none")
    target = position.entry_price * (1.0 + tp_pct)
    if np.isnan(bar.high) or bar.high < target:
        return Signal(action="none")

    fraction = min(fraction, position.remaining_fraction)
    return Signal(
        action="close_partial",
        reason=f"LONG_{tp_name}_PARTIAL",
        exit_price=target,
        meta={
            "bars_in_position": position.bars_in_position,
            "exit_trigger": f"LONG_{tp_name}_PARTIAL",
            "close_fraction": fraction,
            "remaining_fraction_before": position.remaining_fraction,
            "tp_level": tp_name,
            "tp_pct": tp_pct,
        },
    )


def generate_exit_signal(
    bar: BarData,
    prev_bar: BarData,
    params: dict,
    position: PositionState,
) -> Signal:
    if position.side != "long":
        return Signal(action="none")

    position.bars_in_position += 1
    max_bars = int(params.get("max_bars_in_trade", 0) or 0)

    def _base_meta(trigger: str) -> dict:
        return {
            "exit_bb_pct": round(bar.bb_pct, 4) if not np.isnan(bar.bb_pct) else np.nan,
            "exit_stoch_k": round(bar.stoch_k, 2) if not np.isnan(bar.stoch_k) else np.nan,
            "exit_mfi": round(bar.mfi, 2) if not np.isnan(bar.mfi) else np.nan,
            "exit_rsi": round(bar.rsi, 2) if not np.isnan(bar.rsi) else np.nan,
            "exit_roc": round(bar.roc, 3) if not np.isnan(bar.roc) else np.nan,
            "exit_supertrend_dir": bar.supertrend_dir,
            "exit_above_cloud": bool(bar.above_cloud),
            "bars_in_position": position.bars_in_position,
            "exit_trigger": trigger,
            "stop_price": round(position.stop_price, 4) if not np.isnan(position.stop_price) else np.nan,
        }

    if max_bars > 0 and position.bars_in_position >= max_bars:
        return Signal(action="close_force", reason="TIME_STOP", meta=_base_meta("TIME_STOP"))

    # Strong trigger 1: Supertrend flips DOWN -> UP. The analysis showed this is
    # present in ~93% of original exits.
    if bool(params.get("supertrend_flip_exit", True)) and bool(bar.supertrend_flip_up):
        meta = _base_meta("SUPERTREND_FLIP_UP")
        return Signal(action="close_force", reason="SUPERTREND_FLIP_UP", meta=meta)

    # Strong trigger 2 (optional): price above 4h Ichimoku cloud
    if bool(params.get("ichimoku_above_cloud_exit", False)) and float(bar.htf_above_cloud or 0.0) > 0.5:
        meta = _base_meta("HTF_ABOVE_CLOUD")
        return Signal(action="close_force", reason="HTF_ABOVE_CLOUD", meta=meta)

    # Confluence exit (oversold mirror)
    confluence, flags = _exit_confluence_count(bar, params)
    min_conf = int(params.get("long_exit_min_confluence", 3))
    if confluence >= min_conf:
        meta = _base_meta("OVERBOUGHT_CONFLUENCE_EXIT")
        meta["exit_confluence"] = confluence
        meta.update(flags)
        return Signal(action="close_force", reason="OVERBOUGHT_CONFLUENCE_EXIT", meta=meta)

    return Signal(action="none")
