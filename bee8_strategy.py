"""
bee8_strategy.py
=================
Backtest layer for the bee8 mean-reversion long strategy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from bee8_engine import (
    PositionState,
    Signal,
    apply_slippage,
    bar_from_row,
    build_position_state,
    compute_trade_close,
    generate_emergency_exit_signal,
    generate_entry_signal,
    generate_exit_signal,
    generate_partial_exit_signal,
)
from bee8_params import FEE_RATE


@dataclass
class TradeRecord:
    side: str
    entry_time: object
    exit_time: object
    entry_price: float
    exit_price: float
    gross_ret: float
    fee_ret: float
    net_ret: float
    pnl: float
    reason: str
    capital_before: float
    capital_after: float
    position_notional: float
    fee_usd: float
    slippage_usd: float
    entry_bb_pct: float = 0.0
    entry_stoch_k: float = 0.0
    entry_mfi: float = 0.0
    entry_rsi: float = 0.0
    entry_roc: float = 0.0
    entry_htf_rsi: float = 0.0
    entry_htf_bb_pct: float = 0.0
    entry_confluence: int = 0
    entry_min_bb_lb: float = 0.0
    entry_atr: float = 0.0
    entry_stop_price: float = 0.0
    exit_bb_pct: float = 0.0
    exit_stoch_k: float = 0.0
    exit_mfi: float = 0.0
    exit_rsi: float = 0.0
    exit_roc: float = 0.0
    exit_supertrend_dir: float = 0.0
    exit_above_cloud: bool = False
    exit_confluence: int = 0
    exit_bars: int = 0
    holding_hours: float = np.nan
    exit_trigger: str = ""
    close_fraction: float = 1.0
    remaining_fraction_after: float = 0.0
    logical_trade_no: int = 0
    trade_event: str = ""
    trade_label: str = ""


class Bee8Strategy:
    """Mean-reversion long strategy used by backtest, WFO and live runner."""

    def __init__(self, params: dict, fee_rate: float = FEE_RATE):
        self.params = params
        self.fee_rate = params.get("fee_rate", fee_rate)
        self.slippage_bps = params.get("slippage_bps", 0.0)
        self.spread_bps = params.get("spread_bps", 0.0)
        self.position: Optional[PositionState] = None
        self.next_trade_id = 1

    @staticmethod
    def _trade_event(signal: Signal) -> str:
        if signal.action == "close_partial":
            if "TP2" in signal.reason:
                return "TP2"
            if "TP1" in signal.reason:
                return "TP1"
            return "TP"
        return "EXIT"

    @staticmethod
    def _holding_hours(entry_time, exit_time) -> float:
        try:
            start = pd.to_datetime(entry_time, utc=True)
            end = pd.to_datetime(exit_time, utc=True)
            if pd.isna(start) or pd.isna(end):
                return np.nan
            return max(0.0, (end - start).total_seconds() / 3600.0)
        except Exception:
            return np.nan

    def _close_position(self, capital, bar, signal, capital_at_open, entry_meta=None):
        pos = self.position
        raw_exit = signal.exit_price if signal.exit_price is not None else bar.close
        exit_price = apply_slippage(
            raw_exit,
            pos.side,
            "close",
            self.slippage_bps,
            self.spread_bps,
        )
        requested_fraction = float((signal.meta or {}).get("close_fraction", pos.remaining_fraction))
        close_fraction = min(max(requested_fraction, 0.0), pos.remaining_fraction)
        close_notional = capital_at_open * close_fraction

        result = compute_trade_close(
            entry_price=pos.entry_price,
            exit_price=exit_price,
            side=pos.side,
            fee_rate=self.fee_rate,
            capital_at_open=close_notional,
        )
        slip_delta = abs(exit_price - raw_exit)
        slippage_usd = (slip_delta / pos.entry_price) * close_notional if pos.entry_price else 0.0
        new_capital = capital + result["pnl"]

        em = entry_meta or pos.entry_meta or {}
        xm = signal.meta or {}
        remaining_after = max(0.0, pos.remaining_fraction - close_fraction)
        trade_id = int(pos.trade_id or 0)
        trade_event = self._trade_event(signal)
        trade_label = f"{trade_id} {trade_event}".strip()
        holding_hours = self._holding_hours(pos.entry_time, bar.time)

        rec = TradeRecord(
            side=pos.side,
            entry_time=pos.entry_time,
            exit_time=bar.time,
            entry_price=pos.entry_price,
            exit_price=exit_price,
            gross_ret=result["gross_ret"],
            fee_ret=result["fee_ret"],
            net_ret=result["net_ret"],
            pnl=result["pnl"],
            reason=signal.reason,
            capital_before=capital,
            capital_after=new_capital,
            position_notional=close_notional,
            fee_usd=result["fee_usd"],
            slippage_usd=slippage_usd,
            entry_bb_pct=em.get("entry_bb_pct", 0.0),
            entry_stoch_k=em.get("entry_stoch_k", 0.0),
            entry_mfi=em.get("entry_mfi", 0.0),
            entry_rsi=em.get("entry_rsi", 0.0),
            entry_roc=em.get("entry_roc", 0.0),
            entry_htf_rsi=em.get("entry_htf_rsi", 0.0),
            entry_htf_bb_pct=em.get("entry_htf_bb_pct", 0.0),
            entry_confluence=em.get("entry_confluence", 0),
            entry_min_bb_lb=em.get("entry_min_bb_lb", 0.0),
            entry_atr=em.get("entry_atr", 0.0),
            entry_stop_price=em.get("entry_stop_price", pos.stop_price),
            exit_bb_pct=xm.get("exit_bb_pct", 0.0),
            exit_stoch_k=xm.get("exit_stoch_k", 0.0),
            exit_mfi=xm.get("exit_mfi", 0.0),
            exit_rsi=xm.get("exit_rsi", 0.0),
            exit_roc=xm.get("exit_roc", 0.0),
            exit_supertrend_dir=xm.get("exit_supertrend_dir", 0.0),
            exit_above_cloud=bool(xm.get("exit_above_cloud", False)),
            exit_confluence=xm.get("exit_confluence", 0),
            exit_bars=xm.get("bars_in_position", 0),
            holding_hours=holding_hours,
            exit_trigger=xm.get("exit_trigger", signal.reason),
            close_fraction=close_fraction,
            remaining_fraction_after=remaining_after,
            logical_trade_no=trade_id,
            trade_event=trade_event,
            trade_label=trade_label,
        )

        if signal.action == "close_partial" and remaining_after > 1e-9:
            pos.remaining_fraction = remaining_after
            if "TP2" in signal.reason:
                pos.tp2_taken = True
                pos.tp1_protection_after_bars = pos.bars_in_position + 1
            elif "TP1" in signal.reason:
                pos.tp1_taken = True
                pos.tp1_protection_after_bars = pos.bars_in_position + 1
            self.position = pos
        else:
            self.position = None
        return rec, new_capital

    def run(
        self,
        df,
        initial_capital,
        *,
        initial_position: Optional[PositionState] = None,
        initial_capital_at_open: Optional[float] = None,
        initial_next_trade_id: int = 1,
        previous_row: Optional[pd.Series] = None,
        keep_open_position: bool = False,
        return_state: bool = False,
    ):
        capital = initial_capital
        capital_at_open = (
            float(initial_capital_at_open)
            if initial_capital_at_open is not None
            else float(initial_capital)
        )
        trades = []
        equity_curve = []
        self.position = initial_position
        self.next_trade_id = int(initial_next_trade_id or 1)
        if self.position is not None and int(self.position.trade_id or 0) >= self.next_trade_id:
            self.next_trade_id = int(self.position.trade_id) + 1

        if len(df) == 0:
            equity_df = pd.DataFrame(columns=["time", "equity"])
            if return_state:
                open_capital_at_open = capital_at_open if self.position is not None else None
                return (
                    pd.DataFrame(),
                    equity_df,
                    capital,
                    self.position,
                    open_capital_at_open,
                    self.next_trade_id,
                )
            return pd.DataFrame(), equity_df, capital

        equity_curve.append((df["time"].iloc[0], capital))

        start_idx = 0 if previous_row is not None else 1
        for i in range(start_idx, len(df)):
            bar = bar_from_row(df.iloc[i], self.params)
            prev_source = previous_row if i == 0 else df.iloc[i - 1]
            prev = bar_from_row(prev_source, self.params)

            if np.isnan(bar.bb_pct) or np.isnan(bar.rsi):
                continue

            if self.position is not None:
                sig = generate_emergency_exit_signal(bar, self.params, self.position)
                if sig.action != "none":
                    rec, capital = self._close_position(capital, bar, sig, capital_at_open)
                    trades.append(rec)
                    equity_curve.append((bar.time, capital))

            if self.position is not None:
                guard = 0
                sig = generate_partial_exit_signal(bar, self.params, self.position)
                while sig.action != "none" and self.position is not None and guard < 3:
                    rec, capital = self._close_position(capital, bar, sig, capital_at_open)
                    trades.append(rec)
                    equity_curve.append((bar.time, capital))
                    guard += 1
                    if self.position is None:
                        break
                    sig = generate_partial_exit_signal(bar, self.params, self.position)

            if self.position is not None:
                sig = generate_exit_signal(bar, prev, self.params, self.position)
                if sig.action != "none":
                    rec, capital = self._close_position(capital, bar, sig, capital_at_open)
                    trades.append(rec)
                    equity_curve.append((bar.time, capital))

            if self.position is None:
                sig = generate_entry_signal(bar, prev, self.params, self.position)
                if sig.action == "open_long":
                    entry_price = apply_slippage(
                        bar.close,
                        "long",
                        "open",
                        self.slippage_bps,
                        self.spread_bps,
                    )
                    entry_meta = dict(sig.meta or {})
                    self.position = build_position_state(
                        side="long",
                        entry_price=entry_price,
                        entry_time=bar.time,
                        bar=bar,
                        params=self.params,
                        entry_meta=entry_meta,
                    )
                    self.position.trade_id = self.next_trade_id
                    self.next_trade_id += 1
                    self.position.entry_meta["entry_stop_price"] = (
                        round(self.position.stop_price, 4)
                        if not np.isnan(self.position.stop_price)
                        else np.nan
                    )
                    capital_at_open = capital

        if self.position is not None:
            last_time = df["time"].iloc[-1]
            if not equity_curve or equity_curve[-1][0] != last_time:
                equity_curve.append((last_time, capital))
            if not keep_open_position:
                self.position = None

        cols = [
            "side",
            "entry_time",
            "exit_time",
            "entry_price",
            "exit_price",
            "gross_ret",
            "fee_ret",
            "net_ret",
            "pnl",
            "reason",
            "capital_before",
            "capital_after",
            "position_notional",
            "fee_usd",
            "slippage_usd",
            "entry_bb_pct",
            "entry_stoch_k",
            "entry_mfi",
            "entry_rsi",
            "entry_roc",
            "entry_htf_rsi",
            "entry_htf_bb_pct",
            "entry_confluence",
            "entry_min_bb_lb",
            "entry_atr",
            "entry_stop_price",
            "exit_bb_pct",
            "exit_stoch_k",
            "exit_mfi",
            "exit_rsi",
            "exit_roc",
            "exit_supertrend_dir",
            "exit_above_cloud",
            "exit_confluence",
            "exit_bars",
            "holding_hours",
            "exit_trigger",
            "close_fraction",
            "remaining_fraction_after",
            "logical_trade_no",
            "trade_event",
            "trade_label",
        ]

        if trades:
            trades_df = pd.DataFrame([{col: getattr(t, col) for col in cols} for t in trades])
        else:
            trades_df = pd.DataFrame(columns=cols)

        equity_df = pd.DataFrame(equity_curve, columns=["time", "equity"])
        equity_df = equity_df.dropna(subset=["time"]).reset_index(drop=True)
        if return_state:
            open_capital_at_open = capital_at_open if self.position is not None else None
            return (
                trades_df,
                equity_df,
                capital,
                self.position,
                open_capital_at_open,
                self.next_trade_id,
            )
        return trades_df, equity_df, capital
