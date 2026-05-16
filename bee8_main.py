"""
bee8_main.py
=============
Entry point for the bee8 mean-reversion strategy.
"""

from __future__ import annotations

import argparse

import pandas as pd

from bee8_binance import update_csv_cache, wfo_bars
from bee8_data import load_klines, prepare_indicators
from bee8_params import (
    BINANCE_CSV_CACHE,
    BINANCE_INTERVAL,
    BINANCE_MARKET,
    BINANCE_START_DATE,
    BINANCE_SYMBOL,
    CSV_PATH,
    INITIAL_CAPITAL,
    LIVE_DAYS,
    OPT_DAYS,
    WFO_BEST_PARAMS_PATH,
    load_params,
    save_params,
)
from bee8_stats import compute_stats, fee_summary_by_period, print_wfo_windows
from bee8_strategy import Bee8Strategy
from bee8_wfo import get_latest_best_params, walk_forward_optimization


def run_backtest(
    df: pd.DataFrame,
    params: dict,
    start: str | None = None,
    end: str | None = None,
    freq: str = "ME",
    save: bool = False,
) -> None:
    if start:
        df = df[df["time"] >= pd.Timestamp(start, tz="UTC")]
    if end:
        df = df[df["time"] <= pd.Timestamp(end, tz="UTC")]
    df = df.reset_index(drop=True)

    if df.empty:
        print("\n[BACKTEST] No data left after applying the date filter.")
        return

    print(f"\n[BACKTEST] Data: {df['time'].iloc[0]} -> {df['time'].iloc[-1]} ({len(df)} candles)")
    print(f"[BACKTEST] Params: {params}\n")

    strat = Bee8Strategy(params)
    trades, equity, final_cap = strat.run(df, INITIAL_CAPITAL)

    compute_stats(trades, equity, INITIAL_CAPITAL, label="BACKTEST - Mean Reversion")

    if not trades.empty:
        fee_table = fee_summary_by_period(trades, INITIAL_CAPITAL, freq=freq)
        if not fee_table.empty:
            freq_label = {"ME": "monthly", "QE": "quarterly", "YE": "yearly"}.get(freq, freq)
            print(f"\n-- Fees ({freq_label}) --")
            print(fee_table.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    if save:
        trades.to_csv("bee8_backtest_trades.csv", index=False)
        equity.to_csv("bee8_backtest_equity.csv", index=False)
        print("\n[SAVE] bee8_backtest_trades.csv, bee8_backtest_equity.csv")


def run_wfo(
    df: pd.DataFrame,
    interval: str = BINANCE_INTERVAL,
    freq: str = "ME",
    save: bool = False,
    verbose: bool = True,
) -> None:
    base_params = load_params()
    all_trades, equity_wfo, windows_df, _final_cap, stopped = walk_forward_optimization(
        df,
        interval=interval,
        verbose=verbose,
        initial_capital=INITIAL_CAPITAL,
        base_params=base_params,
    )

    compute_stats(all_trades, equity_wfo, INITIAL_CAPITAL, label="WFO - Mean Reversion live windows")
    print_wfo_windows(windows_df)

    if all_trades is not None and not all_trades.empty:
        fee_table = fee_summary_by_period(all_trades, INITIAL_CAPITAL, freq=freq)
        if not fee_table.empty:
            freq_label = {"ME": "monthly", "QE": "quarterly", "YE": "yearly"}.get(freq, freq)
            print(f"\n-- Fees ({freq_label}) --")
            print(fee_table.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    if windows_df is not None and not windows_df.empty:
        best = get_latest_best_params(windows_df)
        save_params(best)
        print(f"\n[WFO] Stable params saved -> {WFO_BEST_PARAMS_PATH}")
        print(f"      {best}")

    if save:
        if all_trades is not None and not all_trades.empty:
            all_trades.to_csv("bee8_wfo_trades.csv", index=False)
        if equity_wfo is not None:
            equity_wfo.to_csv("bee8_wfo_equity.csv", index=False)
        if windows_df is not None and not windows_df.empty:
            windows_df.to_csv("bee8_wfo_windows.csv", index=False)
        print("[SAVE] bee8_wfo_trades.csv, bee8_wfo_equity.csv, bee8_wfo_windows.csv")

    if stopped:
        print("\n[WFO] Stopped early by user.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bee8 mean-reversion bot")
    parser.add_argument("--mode", choices=["backtest", "wfo"], default="backtest")
    parser.add_argument("--fetch", action="store_true", help="Download/update Binance data first")
    parser.add_argument("--symbol", default=BINANCE_SYMBOL, help="Pair, e.g. ETHUSDT")
    parser.add_argument("--interval", default=BINANCE_INTERVAL, help="Timeframe, e.g. 1h or 4h")
    parser.add_argument("--market", default=BINANCE_MARKET, choices=["spot", "futures"])
    parser.add_argument("--since", default=BINANCE_START_DATE, help="Binance start date YYYY-MM-DD")
    parser.add_argument("--start", default=None, help="Backtest start date YYYY-MM-DD")
    parser.add_argument("--end", default=None, help="Backtest end date YYYY-MM-DD")
    parser.add_argument("--freq", default="ME", help="Fee summary frequency: ME/QE/YE")
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--csv", default=None, help="Override CSV path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    interval = args.interval

    if args.csv:
        csv_path = args.csv
    elif BINANCE_CSV_CACHE:
        csv_path = BINANCE_CSV_CACHE
    elif args.fetch:
        csv_path = f"{args.symbol.lower()}_{interval}.csv"
    else:
        csv_path = CSV_PATH

    if args.fetch:
        print(f"[MAIN] Fetching Binance data: {args.symbol} {interval} ({args.market})")
        update_csv_cache(
            csv_path=csv_path,
            symbol=args.symbol,
            interval=interval,
            start_date=args.since,
            market=args.market,
            verbose=True,
        )
        print()

    print(f"[MAIN] Loading data: {csv_path}")
    df = load_klines(csv_path)
    df = prepare_indicators(df)
    print(f"[MAIN] Loaded {len(df)} candles ({df['time'].iloc[0]} -> {df['time'].iloc[-1]})")

    opt_bars, live_bars = wfo_bars(interval, OPT_DAYS, LIVE_DAYS)
    print(f"[MAIN] Timeframe: {interval} | WFO windows: opt={opt_bars} bars / live={live_bars} bars")
    print()

    if args.mode == "wfo":
        run_wfo(df, interval=interval, freq=args.freq, save=args.save, verbose=True)
    else:
        params = load_params()
        run_backtest(df, params, start=args.start, end=args.end, freq=args.freq, save=args.save)


if __name__ == "__main__":
    main()
