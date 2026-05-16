"""
bee8_live_runner.py
====================
Paper/live runner for the bee8 mean-reversion strategy. Uses the same engine
functions as the backtest so a closed bar produces the same signal in either
context.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from bee8_binance import update_csv_cache
from bee8_data import prepare_indicators
from bee8_engine import (
    PositionState,
    apply_slippage,
    bar_from_row as _bar_from_row,
    build_position_state,
    compute_trade_close,
    generate_emergency_exit_signal,
    generate_entry_signal,
    generate_exit_signal,
    generate_partial_exit_signal,
)
from bee8_params import (
    BINANCE_INTERVAL,
    BINANCE_MARKET,
    BINANCE_SYMBOL,
    FEE_RATE,
    INITIAL_CAPITAL,
    load_params,
)

STATE_FILE = "bee8_live_state.json"
TRADES_LOG = "bee8_live_trades.jsonl"
LOG_FILE = "bee8_live_runner.log"
LOOP_SLEEP_SEC = 60

MAX_DAILY_LOSS_PCT = 3.0
MAX_POSITIONS = 1

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("bee8_live")


def load_state() -> dict:
    if not Path(STATE_FILE).exists():
        return {
            "position": None,
            "entry_price": None,
            "entry_time": None,
            "bars_in_position": 0,
            "capital": INITIAL_CAPITAL,
            "capital_at_open": None,
            "stop_price": None,
            "entry_atr": None,
            "remaining_fraction": 1.0,
            "tp1_taken": False,
            "tp2_taken": False,
            "tp1_protection_after_bars": 0,
            "trade_id": 0,
            "next_trade_id": 1,
            "last_bar_time": None,
            "daily_loss_usd": 0.0,
            "daily_date": None,
            "paused": False,
            "mode": "paper",
        }
    with open(STATE_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def save_state(state: dict) -> None:
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, default=str)


def log_trade(trade: dict) -> None:
    with open(TRADES_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(trade, default=str) + "\n")


def holding_hours(entry_time, exit_time) -> float:
    try:
        start = datetime.fromisoformat(str(entry_time).replace("Z", "+00:00"))
        end = datetime.fromisoformat(str(exit_time).replace("Z", "+00:00"))
        if start.tzinfo is None:
            start = start.replace(tzinfo=timezone.utc)
        if end.tzinfo is None:
            end = end.replace(tzinfo=timezone.utc)
        return max(0.0, (end - start).total_seconds() / 3600.0)
    except Exception:
        return float("nan")


def check_daily_loss(state: dict, capital: float) -> bool:
    today = datetime.now(timezone.utc).date().isoformat()
    if state.get("daily_date") != today:
        state["daily_date"] = today
        state["daily_loss_usd"] = 0.0

    base_capital = max(capital, 1.0)
    loss_pct = state["daily_loss_usd"] / base_capital * 100.0
    if loss_pct >= MAX_DAILY_LOSS_PCT:
        log.warning(
            f"[SAFETY] Daily loss limit reached: {loss_pct:.2f}% "
            f"(limit {MAX_DAILY_LOSS_PCT}%). Runner paused."
        )
        return True
    return False


def is_duplicate_bar(state: dict, bar_time: str) -> bool:
    return state.get("last_bar_time") == bar_time


def sync_position_with_exchange(state: dict, mode: str) -> None:
    if mode != "live":
        return
    log.warning(
        "[LIVE-SAFETY] sync_position_with_exchange() is a stub. "
        "Wire in the exchange client before going real."
    )


def position_from_state(state: dict) -> Optional[PositionState]:
    if not state.get("position"):
        return None
    return PositionState(
        side=state["position"],
        entry_price=float(state["entry_price"]),
        entry_time=state["entry_time"],
        bars_in_position=int(state.get("bars_in_position", 0)),
        stop_price=float(state["stop_price"]) if state.get("stop_price") is not None else float("nan"),
        entry_atr=float(state["entry_atr"]) if state.get("entry_atr") is not None else float("nan"),
        remaining_fraction=float(state.get("remaining_fraction", 1.0)),
        tp1_taken=bool(state.get("tp1_taken", False)),
        tp2_taken=bool(state.get("tp2_taken", False)),
        tp1_protection_after_bars=int(state.get("tp1_protection_after_bars", 0)),
        trade_id=int(state.get("trade_id", 0)),
    )


def position_to_state(state: dict, pos: Optional[PositionState]) -> None:
    if pos is None:
        state["position"] = None
        state["entry_price"] = None
        state["entry_time"] = None
        state["bars_in_position"] = 0
        state["capital_at_open"] = None
        state["stop_price"] = None
        state["entry_atr"] = None
        state["remaining_fraction"] = 1.0
        state["tp1_taken"] = False
        state["tp2_taken"] = False
        state["tp1_protection_after_bars"] = 0
        state["trade_id"] = 0
    else:
        state["position"] = pos.side
        state["entry_price"] = pos.entry_price
        state["entry_time"] = str(pos.entry_time)
        state["bars_in_position"] = pos.bars_in_position
        state["stop_price"] = None if np.isnan(pos.stop_price) else pos.stop_price
        state["entry_atr"] = None if np.isnan(pos.entry_atr) else pos.entry_atr
        state["remaining_fraction"] = pos.remaining_fraction
        state["tp1_taken"] = pos.tp1_taken
        state["tp2_taken"] = pos.tp2_taken
        state["tp1_protection_after_bars"] = pos.tp1_protection_after_bars
        state["trade_id"] = pos.trade_id


def execute_order(
    side: str,
    action: str,
    price: float,
    capital: float,
    mode: str,
    reason: str = "",
) -> float:
    if mode == "paper":
        log.info(
            f"[PAPER] {action.upper()} {side.upper()} @ {price:.2f} "
            f"capital={capital:.2f} reason={reason}"
        )
        return price
    log.warning("[LIVE] Execution stub - exchange client not wired in.")
    raise NotImplementedError("Live mode requires exchange integration.")


def process_bar(bar, prev, params: dict, state: dict, mode: str) -> None:
    bar_time_str = str(bar.time)
    capital = float(state.get("capital", INITIAL_CAPITAL))
    fee_rate = float(params.get("fee_rate", FEE_RATE))
    slip_bps = float(params.get("slippage_bps", 0.0))
    spread_bps = float(params.get("spread_bps", 0.0))

    if is_duplicate_bar(state, bar_time_str):
        log.debug(f"[SKIP] Duplicate bar: {bar_time_str}")
        return

    if check_daily_loss(state, capital):
        state["paused"] = True
        save_state(state)
        return

    if state.get("paused"):
        log.info("[PAUSED] Runner paused. Clear `paused` in state to continue.")
        return

    position = position_from_state(state)

    def _close(position: PositionState, sig, capital: float, final_close: bool) -> float:
        raw_exit = sig.exit_price if sig.exit_price is not None else bar.close
        close_fraction = min(
            max(float((sig.meta or {}).get("close_fraction", position.remaining_fraction)), 0.0),
            position.remaining_fraction,
        )
        capital_at_open = float(state.get("capital_at_open") or capital)
        close_notional = capital_at_open * close_fraction
        exec_price = apply_slippage(raw_exit, position.side, "close", slip_bps, spread_bps)
        exec_price = execute_order(position.side, "close", exec_price, close_notional, mode, sig.reason)

        result = compute_trade_close(
            entry_price=position.entry_price,
            exit_price=exec_price,
            side=position.side,
            fee_rate=fee_rate,
            capital_at_open=close_notional,
        )

        pnl = result["pnl"]
        if pnl < 0:
            state["daily_loss_usd"] = state.get("daily_loss_usd", 0.0) + abs(pnl)

        capital += pnl
        state["capital"] = capital

        remaining_after = max(0.0, position.remaining_fraction - close_fraction)
        if sig.action == "close_partial":
            trade_event = "TP2" if "TP2" in sig.reason else "TP1" if "TP1" in sig.reason else "TP"
        else:
            trade_event = "EXIT"
        trade_id = int(position.trade_id or 0)
        hold_hours = holding_hours(position.entry_time, bar_time_str)
        log_trade(
            {
                "ts": bar_time_str,
                "side": position.side,
                "entry_price": position.entry_price,
                "exit_price": exec_price,
                "gross_ret": result["gross_ret"],
                "net_ret": result["net_ret"],
                "pnl_usd": pnl,
                "fee_usd": result["fee_usd"],
                "reason": sig.reason,
                "capital": capital,
                "capital_at_open": capital_at_open,
                "position_notional": close_notional,
                "close_fraction": close_fraction,
                "remaining_fraction_after": remaining_after,
                "holding_hours": hold_hours,
                "logical_trade_no": trade_id,
                "trade_event": trade_event,
                "trade_label": f"{trade_id} {trade_event}".strip(),
                "mode": mode,
            }
        )
        log.info(
            f"[CLOSE] {position.side.upper()} entry={position.entry_price:.2f} "
            f"exit={exec_price:.2f} net={result['net_ret'] * 100:.3f}% pnl={pnl:.2f} USD | {sig.reason}"
        )

        if final_close or remaining_after <= 1e-9:
            position_to_state(state, None)
        else:
            position.remaining_fraction = remaining_after
            if "TP2" in sig.reason:
                position.tp2_taken = True
                position.tp1_protection_after_bars = position.bars_in_position + 1
            elif "TP1" in sig.reason:
                position.tp1_taken = True
                position.tp1_protection_after_bars = position.bars_in_position + 1
            position_to_state(state, position)
        return capital

    if position is not None:
        sig = generate_emergency_exit_signal(bar, params, position)
        if sig.action != "none":
            capital = _close(position, sig, capital, final_close=True)
            position = position_from_state(state)

    if position is not None:
        guard = 0
        sig = generate_partial_exit_signal(bar, params, position)
        while sig.action != "none" and position is not None and guard < 3:
            capital = _close(position, sig, capital, final_close=False)
            position = position_from_state(state)
            guard += 1
            if position is None:
                break
            sig = generate_partial_exit_signal(bar, params, position)

    if position is not None:
        sig = generate_exit_signal(bar, prev, params, position)
        if sig.action != "none":
            capital = _close(position, sig, capital, final_close=True)
        else:
            position_to_state(state, position)

    if state.get("position") is None:
        sig = generate_entry_signal(bar, prev, params, None)
        if sig.action == "open_long":
            if MAX_POSITIONS <= 0:
                raise RuntimeError("MAX_POSITIONS must be at least 1")
            side = "long"
            raw_entry = bar.close
            exec_price = apply_slippage(raw_entry, side, "open", slip_bps, spread_bps)
            exec_price = execute_order(side, "open", exec_price, capital, mode, sig.reason)
            entry_meta = dict(sig.meta or {})
            new_pos = build_position_state(
                side=side,
                entry_price=exec_price,
                entry_time=bar_time_str,
                bar=bar,
                params=params,
                entry_meta=entry_meta,
            )
            trade_id = int(state.get("next_trade_id", 1) or 1)
            new_pos.trade_id = trade_id
            state["next_trade_id"] = trade_id + 1
            new_pos.entry_meta["entry_stop_price"] = (
                round(new_pos.stop_price, 4) if not np.isnan(new_pos.stop_price) else np.nan
            )
            position_to_state(state, new_pos)
            state["capital_at_open"] = capital
            log.info(
                f"[OPEN] LONG @ {exec_price:.2f} | "
                f"bb%={bar.bb_pct:.2f} stoch={bar.stoch_k:.0f} mfi={bar.mfi:.0f} rsi={bar.rsi:.0f} | {sig.reason}"
            )

    state["last_bar_time"] = bar_time_str
    save_state(state)


def main_loop(mode: str = "paper") -> None:
    log.info(
        f"=== bee8 live runner START | mode={mode} | "
        f"symbol={BINANCE_SYMBOL} interval={BINANCE_INTERVAL} ==="
    )
    params = load_params()
    state = load_state()
    state["mode"] = mode
    save_state(state)

    sync_position_with_exchange(state, mode)
    csv_path = f"{BINANCE_SYMBOL.lower()}_{BINANCE_INTERVAL}.csv"

    while True:
        try:
            df_raw = update_csv_cache(
                csv_path=csv_path,
                symbol=BINANCE_SYMBOL,
                interval=BINANCE_INTERVAL,
                market=BINANCE_MARKET,
                verbose=False,
            )
            df_raw = df_raw.rename(columns={"open_time": "time"})
            df_raw["time"] = pd.to_datetime(df_raw["time"], unit="ms", utc=True)
            df = prepare_indicators(df_raw)
            df = df.dropna(subset=["bb_pct", "rsi"]).reset_index(drop=True)

            if len(df) < 3:
                log.warning("Not enough rows after indicator preparation.")
                time.sleep(LOOP_SLEEP_SEC)
                continue

            # closed candle only:
            # df.iloc[-1] is still forming; -2 is the latest closed; -3 is the previous one
            bar = _bar_from_row(df.iloc[-2], params)
            prev = _bar_from_row(df.iloc[-3], params)

            log.info(
                f"Bar: {bar.time} close={bar.close:.2f} "
                f"bb%={bar.bb_pct:.2f} stoch={bar.stoch_k:.0f} mfi={bar.mfi:.0f} rsi={bar.rsi:.0f}"
            )
            process_bar(bar, prev, params, state, mode)

        except KeyboardInterrupt:
            log.info("Runner stopped by user.")
            break
        except Exception as exc:
            log.exception(f"Main loop error: {exc!r}")

        time.sleep(LOOP_SLEEP_SEC)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bee8 mean-reversion live/paper runner")
    parser.add_argument("--mode", default="paper", choices=["paper", "live"])
    parser.add_argument("--pause", action="store_true", help="Set paused=true in state and exit")
    parser.add_argument("--resume", action="store_true", help="Clear paused flag in state and exit")
    args = parser.parse_args()

    if args.pause:
        state = load_state()
        state["paused"] = True
        save_state(state)
        print("[PAUSE] Runner paused.")
        sys.exit(0)
    if args.resume:
        state = load_state()
        state["paused"] = False
        save_state(state)
        print("[RESUME] Runner resumed.")
        sys.exit(0)

    main_loop(mode=args.mode)
