"""
bee8_wfo.py
============
Walk-forward optimization for the bee8 mean-reversion long strategy.

The WFO grid sweeps the four entry oscillator thresholds, the confluence
counters, the HTF filter, and Supertrend / capital-stop choices. Each window
runs the backtest engine and the scoring function in bee8_wfo_scoring.
"""

from __future__ import annotations

from itertools import product
from typing import Optional

import numpy as np
import pandas as pd

from bee8_binance import wfo_bars
from bee8_params import (
    BB_EXTREME_LEVEL,
    BB_EXTREME_LEVEL_GRID,
    BB_LONG_ENTRY_MAX,
    BB_LONG_ENTRY_MAX_GRID,
    BB_LONG_EXIT_MIN,
    BB_LONG_EXIT_MIN_GRID,
    BINANCE_INTERVAL,
    FEE_RATE,
    HTF_BB_MAX,
    HTF_BB_MAX_GRID,
    HTF_RSI_MAX,
    HTF_RSI_MAX_GRID,
    ICHIMOKU_ABOVE_CLOUD_EXIT_GRID,
    INITIAL_CAPITAL,
    LIVE_DAYS,
    LONG_EMERGENCY_SL_CAPITAL_PCT,
    LONG_EMERGENCY_SL_CAPITAL_PCT_GRID,
    LONG_ENTRY_MIN_CONFLUENCE,
    LONG_ENTRY_MIN_CONFLUENCE_GRID,
    LONG_EXIT_MIN_CONFLUENCE,
    LONG_EXIT_MIN_CONFLUENCE_GRID,
    LOOKBACK_BARS,
    LOOKBACK_BARS_GRID,
    MFI_LONG_ENTRY_MAX,
    MFI_LONG_ENTRY_MAX_GRID,
    MFI_LONG_EXIT_MIN,
    MFI_LONG_EXIT_MIN_GRID,
    OPT_DAYS,
    ROC_LONG_EXIT_MIN_PCT,
    ROC_LONG_EXIT_MIN_PCT_GRID,
    RSI_LONG_ENTRY_MAX,
    RSI_LONG_ENTRY_MAX_GRID,
    RSI_LONG_EXIT_MIN,
    RSI_LONG_EXIT_MIN_GRID,
    STOCH_LONG_ENTRY_MAX,
    STOCH_LONG_ENTRY_MAX_GRID,
    STOCH_LONG_EXIT_MIN,
    STOCH_LONG_EXIT_MIN_GRID,
    SUPERTREND_FLIP_EXIT_GRID,
    USE_HTF_FILTER_GRID,
)
from bee8_strategy import Bee8Strategy
from bee8_wfo_scoring import score_params


PARAM_GRID_KEYS = [
    "bb_long_entry_max",
    "stoch_long_entry_max",
    "mfi_long_entry_max",
    "rsi_long_entry_max",
    "long_entry_min_confluence",
    "lookback_bars",
    "bb_extreme_level",
    "use_htf_filter",
    "htf_rsi_max",
    "htf_bb_max",
    "bb_long_exit_min",
    "stoch_long_exit_min",
    "mfi_long_exit_min",
    "rsi_long_exit_min",
    "roc_long_exit_min_pct",
    "long_exit_min_confluence",
    "supertrend_flip_exit",
    "ichimoku_above_cloud_exit",
    "long_emergency_sl_capital_pct",
]


def _clean_grid(values, fallback, caster):
    source = fallback if values is None or len(values) == 0 else values
    cleaned = []
    for value in source:
        casted = bool(value) if caster is bool else caster(value)
        if casted not in cleaned:
            cleaned.append(casted)
    return cleaned


def _snap_to_top_values(values: list, sample):
    if isinstance(sample, bool):
        return bool(pd.Series(values).mode(dropna=False).iloc[0])
    if isinstance(sample, str):
        return str(pd.Series(values).mode(dropna=False).iloc[0])
    unique = sorted(set(values))
    median = float(np.median([float(v) for v in values]))
    chosen = min(unique, key=lambda v: abs(float(v) - median))
    return int(chosen) if isinstance(sample, (int, np.integer)) else float(chosen)


def _select_robust_window_params(results: list[dict]) -> dict | None:
    if not results:
        return None
    ranked = sorted(results, key=lambda r: r["score"], reverse=True)
    top_n = max(1, int(np.ceil(len(ranked) * 0.1)))
    top = ranked[:top_n]

    aggregate_params = {}
    for name in PARAM_GRID_KEYS:
        values = [row["params"][name] for row in top]
        aggregate_params[name] = _snap_to_top_values(values, values[0])

    target = tuple(aggregate_params[name] for name in PARAM_GRID_KEYS)
    lookup = {
        tuple(row["params"][name] for name in PARAM_GRID_KEYS): row for row in ranked
    }
    chosen = lookup.get(target)
    if chosen is None:
        chosen = max(
            top,
            key=lambda row: (
                sum(row["params"][name] == aggregate_params[name] for name in PARAM_GRID_KEYS),
                row["score"],
            ),
        )

    return {"row": chosen, "aggregate_params": aggregate_params, "top_n": top_n}


def walk_forward_optimization(
    df: pd.DataFrame,
    interval: str = BINANCE_INTERVAL,
    score_mode: str = "balanced",
    verbose: bool = True,
    on_window_done=None,
    on_combo_progress=None,
    should_stop=None,
    fee_rate: float = FEE_RATE,
    opt_days: int = OPT_DAYS,
    live_days: int = LIVE_DAYS,
    initial_capital: float = INITIAL_CAPITAL,
    base_params: Optional[dict] = None,
    grid_overrides: Optional[dict] = None,
) -> tuple[pd.DataFrame, Optional[pd.DataFrame], pd.DataFrame, float, bool]:
    opt_bars, live_bars = wfo_bars(interval, opt_days, live_days)
    opt_capital = float(initial_capital)
    base_params = dict(base_params or {})
    grid_overrides = grid_overrides or {}

    bb_in_grid = _clean_grid(grid_overrides.get("bb_long_entry_max"), BB_LONG_ENTRY_MAX_GRID, float)
    stoch_in_grid = _clean_grid(grid_overrides.get("stoch_long_entry_max"), STOCH_LONG_ENTRY_MAX_GRID, float)
    mfi_in_grid = _clean_grid(grid_overrides.get("mfi_long_entry_max"), MFI_LONG_ENTRY_MAX_GRID, float)
    rsi_in_grid = _clean_grid(grid_overrides.get("rsi_long_entry_max"), RSI_LONG_ENTRY_MAX_GRID, float)
    conf_in_grid = _clean_grid(grid_overrides.get("long_entry_min_confluence"), LONG_ENTRY_MIN_CONFLUENCE_GRID, int)
    lookback_grid = _clean_grid(grid_overrides.get("lookback_bars"), LOOKBACK_BARS_GRID, int)
    bb_extreme_grid = _clean_grid(grid_overrides.get("bb_extreme_level"), BB_EXTREME_LEVEL_GRID, float)
    htf_use_grid = _clean_grid(grid_overrides.get("use_htf_filter"), USE_HTF_FILTER_GRID, bool)
    htf_rsi_grid = _clean_grid(grid_overrides.get("htf_rsi_max"), HTF_RSI_MAX_GRID, float)
    htf_bb_grid = _clean_grid(grid_overrides.get("htf_bb_max"), HTF_BB_MAX_GRID, float)
    bb_out_grid = _clean_grid(grid_overrides.get("bb_long_exit_min"), BB_LONG_EXIT_MIN_GRID, float)
    stoch_out_grid = _clean_grid(grid_overrides.get("stoch_long_exit_min"), STOCH_LONG_EXIT_MIN_GRID, float)
    mfi_out_grid = _clean_grid(grid_overrides.get("mfi_long_exit_min"), MFI_LONG_EXIT_MIN_GRID, float)
    rsi_out_grid = _clean_grid(grid_overrides.get("rsi_long_exit_min"), RSI_LONG_EXIT_MIN_GRID, float)
    roc_out_grid = _clean_grid(grid_overrides.get("roc_long_exit_min_pct"), ROC_LONG_EXIT_MIN_PCT_GRID, float)
    conf_out_grid = _clean_grid(grid_overrides.get("long_exit_min_confluence"), LONG_EXIT_MIN_CONFLUENCE_GRID, int)
    st_flip_grid = _clean_grid(grid_overrides.get("supertrend_flip_exit"), SUPERTREND_FLIP_EXIT_GRID, bool)
    cloud_exit_grid = _clean_grid(grid_overrides.get("ichimoku_above_cloud_exit"), ICHIMOKU_ABOVE_CLOUD_EXIT_GRID, bool)
    sl_grid = _clean_grid(grid_overrides.get("long_emergency_sl_capital_pct"), LONG_EMERGENCY_SL_CAPITAL_PCT_GRID, float)

    n = len(df)
    start = 0
    window_id = 0
    current_capital = float(initial_capital)
    carry_position = None
    carry_capital_at_open = None
    next_live_trade_id = 1

    all_live_trades: list[pd.DataFrame] = []
    global_equity: Optional[pd.DataFrame] = None
    window_stats: list[dict] = []
    stopped = False

    total_windows = max(0, (n - opt_bars) // live_bars)
    grids = [
        bb_in_grid, stoch_in_grid, mfi_in_grid, rsi_in_grid, conf_in_grid,
        lookback_grid, bb_extreme_grid, htf_use_grid, htf_rsi_grid, htf_bb_grid,
        bb_out_grid, stoch_out_grid, mfi_out_grid, rsi_out_grid, roc_out_grid,
        conf_out_grid, st_flip_grid, cloud_exit_grid, sl_grid,
    ]
    combo_total = 1
    for g in grids:
        combo_total *= max(1, len(g))
    combo_progress_step = max(1, combo_total // 20)

    if verbose:
        print(
            f"[WFO] candles={n} | windows~={total_windows} | "
            f"opt={opt_days}d ({opt_bars} bars) | live={live_days}d ({live_bars} bars) | "
            f"score_mode={score_mode} | combos/window={combo_total}"
        )
        print("-" * 70)

    while start + opt_bars + live_bars <= n:
        if should_stop is not None and should_stop():
            stopped = True
            break

        opt_slice = df.iloc[start : start + opt_bars]
        live_start_idx = start + opt_bars
        live_end_idx = live_start_idx + live_bars
        live_slice = df.iloc[live_start_idx:live_end_idx]
        live_previous_row = df.iloc[live_start_idx - 1] if live_start_idx > 0 else None

        best_score = -1e9
        best_params = None
        best_opt_trades = None
        best_opt_cap = opt_capital
        opt_results: list[dict] = []

        if on_combo_progress is not None:
            on_combo_progress(window_id, total_windows, 0, combo_total)

        combo_idx = 0
        for (
            bb_in, stoch_in, mfi_in, rsi_in, conf_in,
            lookback, bb_ext, htf_use, htf_rsi, htf_bb,
            bb_out, stoch_out, mfi_out, rsi_out, roc_out,
            conf_out, st_flip, cloud_exit, sl_pct,
        ) in product(
            bb_in_grid, stoch_in_grid, mfi_in_grid, rsi_in_grid, conf_in_grid,
            lookback_grid, bb_extreme_grid, htf_use_grid, htf_rsi_grid, htf_bb_grid,
            bb_out_grid, stoch_out_grid, mfi_out_grid, rsi_out_grid, roc_out_grid,
            conf_out_grid, st_flip_grid, cloud_exit_grid, sl_grid,
        ):
            if should_stop is not None and should_stop():
                stopped = True
                break

            combo_idx += 1
            if on_combo_progress is not None and (
                combo_idx == 1
                or combo_idx % combo_progress_step == 0
                or combo_idx == combo_total
            ):
                on_combo_progress(window_id, total_windows, combo_idx, combo_total)

            params = dict(base_params)
            params.update(
                {
                    "bb_long_entry_max": bb_in,
                    "stoch_long_entry_max": stoch_in,
                    "mfi_long_entry_max": mfi_in,
                    "rsi_long_entry_max": rsi_in,
                    "long_entry_min_confluence": conf_in,
                    "lookback_bars": lookback,
                    "bb_extreme_level": bb_ext,
                    "use_htf_filter": htf_use,
                    "htf_rsi_max": htf_rsi,
                    "htf_bb_max": htf_bb,
                    "bb_long_exit_min": bb_out,
                    "stoch_long_exit_min": stoch_out,
                    "mfi_long_exit_min": mfi_out,
                    "rsi_long_exit_min": rsi_out,
                    "roc_long_exit_min_pct": roc_out,
                    "long_exit_min_confluence": conf_out,
                    "supertrend_flip_exit": st_flip,
                    "ichimoku_above_cloud_exit": cloud_exit,
                    "long_emergency_sl_capital_pct": sl_pct,
                    "long_emergency_sl_enabled": sl_pct > 0.0,
                    "trade_direction": "long",
                    "allow_longs": True,
                    "allow_shorts": False,
                    "short_trading_enabled": False,
                }
            )
            strat = Bee8Strategy(params, fee_rate=fee_rate)
            trades_opt, _, final_cap_opt = strat.run(opt_slice, opt_capital)
            score = score_params(trades_opt, final_cap_opt, opt_capital, mode=score_mode)
            opt_results.append(
                {
                    "params": params,
                    "trades": trades_opt,
                    "final_capital": final_cap_opt,
                    "score": score,
                }
            )
            if score > best_score:
                best_score = score
                best_params = params
                best_opt_trades = trades_opt
                best_opt_cap = final_cap_opt

        if stopped:
            if verbose:
                print(f"[WFO] Stop requested during window {window_id + 1}/{total_windows}.")
            break

        if best_params is None:
            if verbose:
                print(f"[WFO] Window {window_id}: no usable params, stopping.")
            break

        robust = _select_robust_window_params(opt_results)
        if robust is None:
            break
        best_row = robust["row"]
        best_params = dict(best_row["params"])
        best_opt_trades = best_row["trades"]
        best_opt_cap = best_row["final_capital"]
        best_score = best_row["score"]

        opt_n_trades = 0 if best_opt_trades is None or best_opt_trades.empty else len(best_opt_trades)
        opt_ret_pct = (best_opt_cap / opt_capital - 1.0) * 100.0
        opt_pf = 0.0
        opt_max_dd = 0.0
        if best_opt_trades is not None and not best_opt_trades.empty:
            wins = best_opt_trades[best_opt_trades["pnl"] > 0]["pnl"].sum()
            losses = best_opt_trades[best_opt_trades["pnl"] <= 0]["pnl"].sum()
            opt_pf = wins / abs(losses) if losses < 0 else 0.0
            equity = np.array(
                [opt_capital] + list(opt_capital + best_opt_trades["pnl"].cumsum().values)
            )
            running_max = np.maximum.accumulate(equity)
            dd_arr = (equity - running_max) / running_max
            opt_max_dd = dd_arr.min() * 100.0

        strat = Bee8Strategy(best_params, fee_rate=fee_rate)
        (
            trades_live,
            equity_live,
            final_cap_live,
            carry_position,
            carry_capital_at_open,
            next_live_trade_id,
        ) = strat.run(
            live_slice,
            current_capital,
            initial_position=carry_position,
            initial_capital_at_open=carry_capital_at_open,
            initial_next_trade_id=next_live_trade_id,
            previous_row=live_previous_row,
            keep_open_position=True,
            return_state=True,
        )

        if not trades_live.empty:
            trades_live = trades_live.copy()
            trades_live["window_id"] = window_id
            for key in PARAM_GRID_KEYS:
                trades_live[key] = best_params.get(key)
            all_live_trades.append(trades_live)

        if equity_live is not None and not equity_live.empty:
            global_equity = (
                equity_live.copy()
                if global_equity is None
                else pd.concat([global_equity, equity_live.iloc[1:]], ignore_index=True)
            )

        live_ret_pct = (final_cap_live / current_capital - 1.0) * 100.0 if current_capital > 0 else 0.0
        n_trades_live = 0 if trades_live.empty else len(trades_live)

        window_stats.append(
            {
                "window_id": window_id,
                "live_start": live_slice["time"].iloc[0],
                "live_end": live_slice["time"].iloc[-1],
                **{f"best_{key}": best_params.get(key) for key in PARAM_GRID_KEYS},
                "opt_score": best_score,
                "opt_return_pct": opt_ret_pct,
                "opt_pf": opt_pf,
                "opt_max_dd_pct": opt_max_dd,
                "opt_n_trades": opt_n_trades,
                "live_return_pct": live_ret_pct,
                "live_final_cap": final_cap_live,
                "n_trades_live": n_trades_live,
                "open_position_carried": carry_position is not None,
                "open_position_trade_id": int(carry_position.trade_id) if carry_position is not None else 0,
                "selection_method": "top_decile_median",
                "selection_top_n": robust["top_n"],
            }
        )

        if verbose:
            print(
                f"[WFO] {window_id:3d} | "
                f"{live_slice['time'].iloc[0].strftime('%Y-%m-%d')} -> "
                f"{live_slice['time'].iloc[-1].strftime('%Y-%m-%d')} | "
                f"ret={live_ret_pct:+.2f}% tr={n_trades_live} "
                f"bb_in={best_params['bb_long_entry_max']:.2f} "
                f"stoch_in={best_params['stoch_long_entry_max']:.0f} "
                f"rsi_in={best_params['rsi_long_entry_max']:.0f} "
                f"conf_in={best_params['long_entry_min_confluence']} "
                f"htf={best_params['use_htf_filter']} "
                f"st_flip={best_params['supertrend_flip_exit']} "
                f"sl={best_params['long_emergency_sl_capital_pct'] * 100:.0f}%"
            )

        if on_window_done is not None:
            on_window_done(
                window_id,
                total_windows,
                list(window_stats),
                list(all_live_trades),
                global_equity.copy() if global_equity is not None else None,
                current_capital,
            )

        current_capital = final_cap_live
        start += live_bars
        window_id += 1

    all_trades_df = pd.concat(all_live_trades, ignore_index=True) if all_live_trades else pd.DataFrame()
    windows_df = pd.DataFrame(window_stats)
    return all_trades_df, global_equity, windows_df, current_capital, stopped


def get_latest_best_params(windows_df: pd.DataFrame) -> dict:
    """Return a stable parameter set from the last few WFO windows."""
    if windows_df is None or windows_df.empty:
        return {}

    recent = windows_df.tail(5).copy()
    if "n_trades_live" in recent.columns:
        active = recent[recent["n_trades_live"] >= 1]
        if len(active) >= 2:
            recent = active

    defaults_map = {
        "bb_long_entry_max": BB_LONG_ENTRY_MAX,
        "stoch_long_entry_max": STOCH_LONG_ENTRY_MAX,
        "mfi_long_entry_max": MFI_LONG_ENTRY_MAX,
        "rsi_long_entry_max": RSI_LONG_ENTRY_MAX,
        "long_entry_min_confluence": LONG_ENTRY_MIN_CONFLUENCE,
        "lookback_bars": LOOKBACK_BARS,
        "bb_extreme_level": BB_EXTREME_LEVEL,
        "use_htf_filter": True,
        "htf_rsi_max": HTF_RSI_MAX,
        "htf_bb_max": HTF_BB_MAX,
        "bb_long_exit_min": BB_LONG_EXIT_MIN,
        "stoch_long_exit_min": STOCH_LONG_EXIT_MIN,
        "mfi_long_exit_min": MFI_LONG_EXIT_MIN,
        "rsi_long_exit_min": RSI_LONG_EXIT_MIN,
        "roc_long_exit_min_pct": ROC_LONG_EXIT_MIN_PCT,
        "long_exit_min_confluence": LONG_EXIT_MIN_CONFLUENCE,
        "supertrend_flip_exit": True,
        "ichimoku_above_cloud_exit": False,
        "long_emergency_sl_capital_pct": LONG_EMERGENCY_SL_CAPITAL_PCT,
    }

    result = {}
    for key, default in defaults_map.items():
        col = f"best_{key}"
        if col in recent.columns:
            try:
                result[key] = recent[col].mode().iloc[0]
                if isinstance(default, bool):
                    result[key] = bool(result[key])
                elif isinstance(default, int):
                    result[key] = int(result[key])
                else:
                    result[key] = float(result[key])
            except Exception:
                result[key] = default
        else:
            result[key] = default

    result["long_emergency_sl_enabled"] = float(result["long_emergency_sl_capital_pct"]) > 0.0
    result["trade_direction"] = "long"
    result["allow_longs"] = True
    result["allow_shorts"] = False
    result["short_trading_enabled"] = False
    return result
