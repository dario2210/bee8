"""
bee8_stats.py
==============
Statystyki, raportowanie, rozbicie fee i PnL.

Zmiany (brief):
  [4] fee_total_usd sumowane z dokładnego fee_usd per trade (bez aproksymacji)
  [8] Dodatkowe metryki: expectancy, median return, avg winner/loser,
      longest losing streak, exposure time, breakdown long/short,
      breakdown by year/quarter/month
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional

from bee8_params import FEE_RATE, INITIAL_CAPITAL


def _fmt(val, decimals=2, pct=False):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "n/d"
    s = f"{val:,.{decimals}f}"
    return s + "%" if pct else s


# ──────────────────────────────────────────────────────────────────────────────
# GŁÓWNE STATYSTYKI
# ──────────────────────────────────────────────────────────────────────────────

def compute_stats(
    trades         : pd.DataFrame,
    equity         : pd.DataFrame,
    initial_capital: float = INITIAL_CAPITAL,
    label          : str   = "Bee8 Mean Reversion",
    print_output   : bool  = True,
) -> dict:
    stats = {}

    if trades is None or trades.empty or equity is None or equity.empty:
        if print_output:
            print("Brak wyników (zero transakcji lub equity).")
        return stats

    equity = equity.dropna(subset=["time"]).reset_index(drop=True)
    if equity.empty:
        return stats

    final_capital  = equity["equity"].iloc[-1]
    net_profit_usd = final_capital - initial_capital
    net_return_pct = (final_capital / initial_capital - 1.0) * 100.0
    n_trades       = len(trades)

    # ── opłaty: dokładne z fee_usd (punkt 4) ──────────────────────────────
    if "fee_usd" in trades.columns:
        fee_total_usd = trades["fee_usd"].sum()
    else:
        fee_total_usd = abs(trades["fee_ret"].sum()) * initial_capital \
                        if "fee_ret" in trades.columns \
                        else 2.0 * FEE_RATE * n_trades * initial_capital

    if "slippage_usd" in trades.columns:
        slippage_total_usd = trades["slippage_usd"].sum()
    else:
        slippage_total_usd = 0.0

    gross_profit_usd = net_profit_usd + fee_total_usd + slippage_total_usd

    # ── winrate / PF ───────────────────────────────────────────────────────
    wins_df   = trades[trades["pnl"] > 0]
    losses_df = trades[trades["pnl"] <= 0]
    winrate   = len(wins_df) / n_trades * 100.0 if n_trades > 0 else 0.0
    gp        = wins_df["pnl"].sum()
    gl        = losses_df["pnl"].sum()
    pf        = gp / abs(gl) if gl < 0 else float("nan")

    avg_pnl   = trades["pnl"].mean() if n_trades > 0 else 0.0
    avg_ret   = trades["net_ret"].mean() * 100.0 \
                if "net_ret" in trades.columns and n_trades > 0 else 0.0

    # ── expectancy (punkt 8) ───────────────────────────────────────────────
    avg_win  = wins_df["pnl"].mean()   if len(wins_df) > 0   else 0.0
    avg_loss = losses_df["pnl"].mean() if len(losses_df) > 0 else 0.0
    wr_frac  = winrate / 100.0
    expectancy = wr_frac * avg_win + (1.0 - wr_frac) * avg_loss   # USD per trade

    # ── median trade return (punkt 8) ─────────────────────────────────────
    median_ret = trades["net_ret"].median() * 100.0 \
                 if "net_ret" in trades.columns else float("nan")

    # ── longest losing streak (punkt 8) ───────────────────────────────────
    pnl_seq     = trades["pnl"].values
    max_streak  = 0
    cur_streak  = 0
    for p in pnl_seq:
        if p <= 0:
            cur_streak += 1
            max_streak  = max(max_streak, cur_streak)
        else:
            cur_streak  = 0

    # ── drawdown ───────────────────────────────────────────────────────────
    eq          = equity["equity"].values
    run_max     = np.maximum.accumulate(eq)
    dd_arr      = (eq - run_max) / run_max
    max_dd      = dd_arr.min() * 100.0

    # ── exposure time (punkt 8) ────────────────────────────────────────────
    if "entry_time" in trades.columns and "exit_time" in trades.columns:
        total_period = (equity["time"].iloc[-1] - equity["time"].iloc[0]).total_seconds()
        in_trade = sum(
            (pd.Timestamp(r.exit_time) - pd.Timestamp(r.entry_time)).total_seconds()
            for r in trades.itertuples()
        )
        exposure_pct = in_trade / total_period * 100.0 if total_period > 0 else 0.0
    else:
        exposure_pct = float("nan")

    # ── CAGR ───────────────────────────────────────────────────────────────
    t0   = equity["time"].iloc[0]
    t1   = equity["time"].iloc[-1]
    days = (t1 - t0).total_seconds() / 86400.0
    cagr = ((final_capital / initial_capital) ** (1.0 / (days / 365.0)) - 1.0) * 100.0 \
           if days > 0 else float("nan")

    return_drawdown_ratio = (
        net_return_pct / abs(max_dd)
        if max_dd < 0
        else float("nan")
    )

    trade_returns = (
        pd.to_numeric(trades["net_ret"], errors="coerce").dropna()
        if "net_ret" in trades.columns
        else pd.Series(dtype="float64")
    )
    if len(trade_returns) > 1:
        ret_std = trade_returns.std(ddof=1)
        sharpe_ratio = trade_returns.mean() / ret_std * np.sqrt(len(trade_returns)) if ret_std > 0 else float("nan")
        downside = trade_returns[trade_returns < 0]
        down_std = downside.std(ddof=1) if len(downside) > 1 else float("nan")
        sortino_ratio = trade_returns.mean() / down_std * np.sqrt(len(trade_returns)) if down_std and down_std > 0 else float("nan")
    else:
        sharpe_ratio = float("nan")
        sortino_ratio = float("nan")

    risk_reward_ratio = avg_win / abs(avg_loss) if avg_loss < 0 else float("nan")

    consistency_pct = float("nan")
    stability_score = float("nan")
    period_return_std_pct = float("nan")
    worst_period_return_pct = float("nan")
    if "exit_time" in trades.columns and "pnl" in trades.columns:
        period_df = trades.copy()
        period_df["exit_time"] = pd.to_datetime(period_df["exit_time"], utc=True, errors="coerce")
        period_df = period_df.dropna(subset=["exit_time"])
        if not period_df.empty:
            monthly_pnl = (
                period_df.set_index("exit_time")["pnl"]
                .resample("ME")
                .sum()
            )
            monthly_ret = monthly_pnl / initial_capital * 100.0
            monthly_ret = monthly_ret[monthly_ret != 0]
            if len(monthly_ret) > 0:
                consistency_pct = (monthly_ret > 0).mean() * 100.0
                worst_period_return_pct = monthly_ret.min()
            if len(monthly_ret) > 1:
                period_return_std_pct = monthly_ret.std(ddof=1)
                stability_score = monthly_ret.mean() / period_return_std_pct if period_return_std_pct > 0 else float("nan")

    stats = {
        "label"              : label,
        "n_trades"           : n_trades,
        "initial_capital"    : initial_capital,
        "final_capital"      : final_capital,
        "gross_profit_usd"   : gross_profit_usd,
        "fee_total_usd"      : fee_total_usd,
        "fee_total_pct"      : fee_total_usd / initial_capital * 100.0,
        "slippage_total_usd" : slippage_total_usd,
        "net_profit_usd"     : net_profit_usd,
        "net_return_pct"     : net_return_pct,
        "winrate_pct"        : winrate,
        "profit_factor"      : pf,
        "avg_pnl"            : avg_pnl,
        "avg_ret_pct"        : avg_ret,
        "avg_winner_usd"     : avg_win,
        "avg_loser_usd"      : avg_loss,
        "expectancy_usd"     : expectancy,
        "return_drawdown_ratio": return_drawdown_ratio,
        "sharpe_ratio"       : sharpe_ratio,
        "sortino_ratio"      : sortino_ratio,
        "risk_reward_ratio"  : risk_reward_ratio,
        "consistency_pct"    : consistency_pct,
        "stability_score"    : stability_score,
        "period_return_std_pct": period_return_std_pct,
        "worst_period_return_pct": worst_period_return_pct,
        "median_ret_pct"     : median_ret,
        "longest_losing_streak": max_streak,
        "exposure_pct"       : exposure_pct,
        "max_drawdown_pct"   : max_dd,
        "cagr_pct"           : cagr,
        "period_start"       : t0,
        "period_end"         : t1,
        "period_days"        : days,
    }

    if print_output:
        _print_stats(stats, trades)

    return stats


# ──────────────────────────────────────────────────────────────────────────────
# DRUKOWANIE
# ──────────────────────────────────────────────────────────────────────────────

def _print_stats(stats: dict, trades: pd.DataFrame) -> None:
    w = 62
    print()
    print("=" * w)
    print(f"  {stats['label']}")
    print("=" * w)

    print(f"\nOKRES")
    print(f"  Od:                      {stats['period_start']}")
    print(f"  Do:                      {stats['period_end']}")
    print(f"  Dni:                     {stats['period_days']:.0f}")

    print(f"\nWYNIKI")
    print(f"  Transakcji:              {stats['n_trades']}")
    print(f"  Kapitał startowy:        {_fmt(stats['initial_capital'])} USD")
    print(f"  Kapitał końcowy:         {_fmt(stats['final_capital'])} USD")

    print(f"\nOPŁATY (dokładne)")
    print(f"  Gross zysk (bez opłat):  {_fmt(stats['gross_profit_usd'])} USD")
    print(f"  Łączne opłaty:           {_fmt(stats['fee_total_usd'])} USD"
          f"  ({_fmt(stats['fee_total_pct'], pct=True)})")
    if stats["slippage_total_usd"] > 0:
        print(f"  Slippage / spread:       {_fmt(stats['slippage_total_usd'])} USD")
    print(f"  Net zysk:                {_fmt(stats['net_profit_usd'])} USD"
          f"  ({_fmt(stats['net_return_pct'], pct=True)})")

    print(f"\nJAKOŚĆ")
    print(f"  Winrate:                 {_fmt(stats['winrate_pct'], pct=True)}")
    print(f"  Profit Factor:           {_fmt(stats['profit_factor'])}")
    print(f"  Expectancy / trade:      {_fmt(stats['expectancy_usd'])} USD")
    print(f"  Return / Drawdown:       {_fmt(stats['return_drawdown_ratio'])}")
    print(f"  Sharpe Ratio:            {_fmt(stats['sharpe_ratio'])}")
    print(f"  Sortino Ratio:           {_fmt(stats['sortino_ratio'])}")
    print(f"  Risk/Reward (avg):       {_fmt(stats['risk_reward_ratio'])}")
    print(f"  Consistency:             {_fmt(stats['consistency_pct'], pct=True)}")
    print(f"  Stability score:         {_fmt(stats['stability_score'])}")
    print(f"  Śr. winner:              {_fmt(stats['avg_winner_usd'])} USD")
    print(f"  Śr. loser:               {_fmt(stats['avg_loser_usd'])} USD")
    print(f"  Median zwrot / trade:    {_fmt(stats['median_ret_pct'], 3, pct=True)}")
    print(f"  Najdłuższa seria strat:  {stats['longest_losing_streak']}")
    print(f"  Czas w pozycji:          {_fmt(stats['exposure_pct'], pct=True)}")
    print(f"  Max Drawdown:            {_fmt(stats['max_drawdown_pct'], pct=True)}")
    print(f"  CAGR:                    {_fmt(stats['cagr_pct'], pct=True)}")

    print(f"\nSTRONY")
    print(trades["side"].value_counts().to_string())

    print(f"\nPOWODY WYJŚCIA")
    print(trades["reason"].value_counts().to_string())
    print("=" * w)


def print_wfo_windows(windows_df: pd.DataFrame) -> None:
    if windows_df is None or windows_df.empty:
        print("\nBrak okien WFO.")
        return

    n          = len(windows_df)
    profitable = (windows_df["live_return_pct"] > 0).sum()
    avg_ret    = windows_df["live_return_pct"].mean()
    med_ret    = windows_df["live_return_pct"].median()

    print(f"\nWFO – okna live")
    print("=" * 62)
    print(f"  Liczba okien:            {n}")
    print(f"  Zyskownych:              {profitable}/{n}  ({profitable/n*100:.1f}%)")
    print(f"  Śr. zwrot / okno:        {avg_ret:.3f}%")
    print(f"  Mediana zwrotu:          {med_ret:.3f}%")

    param_labels = [
        ("best_bb_long_entry_max", "BB %B in"),
        ("best_stoch_long_entry_max", "Stoch in"),
        ("best_mfi_long_entry_max", "MFI in"),
        ("best_rsi_long_entry_max", "RSI in"),
        ("best_long_entry_min_confluence", "Min conf in"),
        ("best_bb_long_exit_min", "BB %B out"),
        ("best_stoch_long_exit_min", "Stoch out"),
        ("best_mfi_long_exit_min", "MFI out"),
        ("best_rsi_long_exit_min", "RSI out"),
        ("best_roc_long_exit_min_pct", "ROC out"),
        ("best_long_exit_min_confluence", "Min conf out"),
        ("best_use_htf_filter", "HTF filter"),
        ("best_supertrend_flip_exit", "Supertrend flip"),
        ("best_long_emergency_sl_capital_pct", "SL %"),
    ]

    print("\n  Rozkład parametrów:")
    for col, lbl in param_labels:
        if col in windows_df.columns:
            vals = windows_df[col].value_counts().sort_index()
            print(f"    {lbl:14s}: " + "  ".join(f"{k}→{v}" for k, v in vals.items()))

    print("\n  Śr. zwrot wg parametrów:")
    for col, lbl in param_labels:
        if col in windows_df.columns:
            avg = windows_df.groupby(col)["live_return_pct"].mean()
            print(f"    {lbl:14s}: " + "  ".join(f"{k}={v:.3f}%" for k, v in avg.items()))

    print("\n  Ostatnie 10 okien:")
    cols = [c for c in [
        "window_id",
        "live_start",
        "live_end",
        "best_bb_long_entry_max",
        "best_stoch_long_entry_max",
        "best_mfi_long_entry_max",
        "best_rsi_long_entry_max",
        "best_long_entry_min_confluence",
        "best_bb_long_exit_min",
        "best_long_exit_min_confluence",
        "best_use_htf_filter",
        "best_supertrend_flip_exit",
        "best_long_emergency_sl_capital_pct",
        "live_return_pct",
        "n_trades_live",
    ]
            if c in windows_df.columns]
    print(windows_df[cols].tail(10).to_string(index=False))
    print("=" * 62)


# ──────────────────────────────────────────────────────────────────────────────
# OPŁATY W CZASIE
# ──────────────────────────────────────────────────────────────────────────────

def fee_summary_by_period(
    trades         : pd.DataFrame,
    initial_capital: float = INITIAL_CAPITAL,
    freq           : str   = "ME",
) -> pd.DataFrame:
    if trades is None or trades.empty:
        return pd.DataFrame()

    df = trades.copy()
    df["exit_time"] = pd.to_datetime(df["exit_time"])
    df = df.set_index("exit_time").sort_index()

    grp = df.resample(freq)

    # dokładne fee_usd jeśli dostępne, inaczej przybliżenie
    if "fee_usd" in df.columns:
        fee_series = grp["fee_usd"].sum()
    else:
        fee_series = grp["fee_ret"].sum().abs() * initial_capital \
                     if "fee_ret" in df.columns \
                     else grp["pnl"].count() * 2 * FEE_RATE * initial_capital

    slip_series = grp["slippage_usd"].sum() if "slippage_usd" in df.columns \
                  else pd.Series(0.0, index=fee_series.index)

    result = pd.DataFrame({
        "n_trades"        : grp["pnl"].count(),
        "gross_pnl_usd"   : grp["gross_ret"].sum() * initial_capital \
                            if "gross_ret" in df.columns else float("nan"),
        "fee_usd"         : fee_series,
        "slippage_usd"    : slip_series,
        "net_pnl_usd"     : grp["pnl"].sum(),
        "fee_pct_capital" : fee_series / initial_capital * 100.0,
    })
    return result.reset_index().rename(columns={"exit_time": "period"})


# ──────────────────────────────────────────────────────────────────────────────
# BREAKDOWN LONG/SHORT + CZASOWY (punkt 8)
# ──────────────────────────────────────────────────────────────────────────────

def breakdown_by_side(trades: pd.DataFrame) -> pd.DataFrame:
    """Porównanie Long vs Short."""
    if trades is None or trades.empty:
        return pd.DataFrame()
    grp = trades.groupby("side")
    return pd.DataFrame({
        "n_trades"   : grp["pnl"].count(),
        "win_rate"   : grp.apply(lambda g: (g["pnl"] > 0).mean() * 100),
        "avg_pnl"    : grp["pnl"].mean(),
        "total_pnl"  : grp["pnl"].sum(),
        "total_fee"  : grp["fee_usd"].sum() if "fee_usd" in trades.columns
                       else grp["fee_ret"].sum().abs(),
    }).reset_index()


def breakdown_by_exit_trigger(trades: pd.DataFrame) -> pd.DataFrame:
    """Entry performance breakdown by exit trigger label."""
    if trades is None or trades.empty or "exit_trigger" not in trades.columns:
        return pd.DataFrame()
    grp = trades.groupby("exit_trigger")
    return pd.DataFrame({
        "n_trades": grp["pnl"].count(),
        "win_rate": grp.apply(lambda g: (g["pnl"] > 0).mean() * 100),
        "avg_pnl": grp["pnl"].mean(),
        "total_pnl": grp["pnl"].sum(),
        "avg_ret_pct": grp["net_ret"].mean() * 100 if "net_ret" in trades.columns else float("nan"),
    }).reset_index()


def breakdown_by_period(
    trades : pd.DataFrame,
    freq   : str = "YE",   # "YE"=rok, "QE"=kwartał, "ME"=miesiąc
) -> pd.DataFrame:
    """PnL i winrate w rozbiciu czasowym."""
    if trades is None or trades.empty:
        return pd.DataFrame()
    df = trades.copy()
    df["exit_time"] = pd.to_datetime(df["exit_time"])
    df = df.set_index("exit_time").sort_index()
    grp = df.resample(freq)
    res = pd.DataFrame({
        "n_trades" : grp["pnl"].count(),
        "win_rate" : grp.apply(lambda g: (g["pnl"] > 0).mean() * 100 if len(g) > 0 else float("nan")),
        "total_pnl": grp["pnl"].sum(),
        "avg_pnl"  : grp["pnl"].mean(),
        "total_fee": grp["fee_usd"].sum() if "fee_usd" in df.columns
                     else grp["fee_ret"].sum().abs(),
    })
    return res[res["n_trades"] > 0].reset_index().rename(columns={"exit_time": "period"})


def print_extended_report(
    trades         : pd.DataFrame,
    initial_capital: float = INITIAL_CAPITAL,
) -> None:
    """Drukuje rozszerzony raport (punkt 8)."""
    if trades is None or trades.empty:
        print("Brak transakcji do raportu.")
        return

    print("\n── Breakdown Long / Short ──────────────────────────────────")
    print(breakdown_by_side(trades).to_string(index=False))

    print("\n── Exit Trigger Breakdown ──────────────────────────────────")
    et_df = breakdown_by_exit_trigger(trades)
    if not et_df.empty:
        print(et_df.to_string(index=False))

    print("\n── Breakdown roczny ────────────────────────────────────────")
    print(breakdown_by_period(trades, "YE").to_string(index=False))

    print("\n── Breakdown kwartalny ─────────────────────────────────────")
    print(breakdown_by_period(trades, "QE").to_string(index=False))

    print("\n── Opłaty miesięczne ───────────────────────────────────────")
    ft = fee_summary_by_period(trades, initial_capital, "ME")
    if not ft.empty:
        print(ft.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

