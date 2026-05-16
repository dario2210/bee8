"""
bee8_dashboard.py
==================
Dash dashboard for the bee8 mean-reversion strategy.
Inspired by the bee4_4 dashboard but trimmed down to the controls that matter
for the new strategy: oscillator thresholds, HTF filter, Supertrend trigger.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import io
import json
import os
import threading
import time as _time
from pathlib import Path

import numpy as np
import pandas as pd
import dash
from dash import dcc, html, dash_table, Input, Output, State, ctx
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from bee8_binance import update_csv_cache, wfo_bars
from bee8_data import load_klines, prepare_indicators
from bee8_params import (
    BB_EXTREME_LEVEL_OPTIONS,
    BB_LONG_ENTRY_MAX_OPTIONS,
    BB_LONG_EXIT_MIN_OPTIONS,
    BINANCE_INTERVAL,
    BINANCE_MARKET,
    BINANCE_START_DATE,
    BINANCE_SYMBOL,
    DEFAULT_PARAMS,
    FEE_RATE,
    HTF_BB_MAX_OPTIONS,
    HTF_RSI_MAX_OPTIONS,
    INITIAL_CAPITAL,
    LIVE_DAYS,
    LONG_EMERGENCY_SL_CAPITAL_PCT_OPTIONS,
    LONG_ENTRY_MIN_CONFLUENCE_OPTIONS,
    LONG_EXIT_MIN_CONFLUENCE_OPTIONS,
    LOOKBACK_BARS_OPTIONS,
    MFI_LONG_ENTRY_MAX_OPTIONS,
    MFI_LONG_EXIT_MIN_OPTIONS,
    OPT_DAYS,
    ROC_LONG_EXIT_MIN_PCT_OPTIONS,
    RSI_LONG_ENTRY_MAX_OPTIONS,
    RSI_LONG_EXIT_MIN_OPTIONS,
    STOCH_LONG_ENTRY_MAX_OPTIONS,
    STOCH_LONG_EXIT_MIN_OPTIONS,
    load_params,
    save_params,
)
from bee8_stats import (
    breakdown_by_period,
    breakdown_by_side,
    breakdown_by_exit_trigger,
    compute_stats,
    fee_summary_by_period,
)
from bee8_strategy import Bee8Strategy
from bee8_wfo import get_latest_best_params, walk_forward_optimization


# ─── Colours ──────────────────────────────────────────────────────────────────
C = {
    "bg": "#09111b",
    "surface": "rgba(11, 21, 36, 0.86)",
    "surf2": "rgba(18, 31, 52, 0.9)",
    "border": "rgba(131, 153, 179, 0.18)",
    "text": "#eef6ff",
    "muted": "#95a6bc",
    "green": "#22d3aa",
    "red": "#ff5d73",
    "blue": "#69b7ff",
    "amber": "#ffb454",
    "purple": "#8b5cf6",
    "coral": "#ff7a59",
}

_SERVER_TOKEN = str(int(_time.time()))
_lock = threading.Lock()
_state = {
    "running": False,
    "stop": False,
    "status": "",
    "progress": "",
    "result": None,
    "result_version": 0,
}
_chart_df_cache: dict = {}
_APP_DIR = Path(__file__).resolve().parent


def gs():
    with _lock:
        return dict(_state)


def ss(**kw):
    with _lock:
        if "result" in kw:
            _state["result_version"] += 1
        _state.update(kw)


# ─── UI helpers ───────────────────────────────────────────────────────────────
def lbl(t):
    return html.Div(
        t,
        style={
            "fontSize": "11px",
            "color": C["muted"],
            "marginBottom": "4px",
            "fontWeight": "500",
            "letterSpacing": "0.05em",
            "textTransform": "uppercase",
        },
    )


def inp(id_, val, **kw):
    debounce = kw.pop("debounce", True)
    return dcc.Input(
        id=id_,
        value=val,
        debounce=debounce,
        style={
            "width": "100%",
            "background": C["surf2"],
            "border": f"1px solid {C['border']}",
            "borderRadius": "14px",
            "color": C["text"],
            "padding": "11px 14px",
            "fontSize": "13px",
            "boxSizing": "border-box",
        },
        **kw,
    )


def drp(id_, opts, val):
    return dcc.Dropdown(
        id=id_,
        options=opts,
        value=val,
        clearable=False,
        className="bee8-drp",
        style={"color": "#0b1220"},
    )


card_s = {
    "background": C["surface"],
    "border": f"1px solid {C['border']}",
    "borderRadius": "24px",
    "padding": "18px 20px",
    "marginBottom": "14px",
    "boxShadow": "0 20px 60px rgba(0, 0, 0, 0.28)",
    "backdropFilter": "blur(20px)",
}


def sec(t):
    return html.Div(
        t,
        style={
            "fontSize": "11px",
            "fontWeight": "600",
            "color": C["muted"],
            "textTransform": "uppercase",
            "letterSpacing": "0.06em",
            "marginBottom": "8px",
            "marginTop": "4px",
        },
    )


def field(label, ctrl):
    return html.Div([lbl(label), ctrl], style={"marginBottom": "10px"})


def btn(id_, label, color=C["blue"], text="#fff"):
    return html.Button(
        label,
        id=id_,
        n_clicks=0,
        style={
            "width": "100%",
            "background": color,
            "border": "none",
            "borderRadius": "16px",
            "color": text,
            "padding": "12px 14px",
            "fontSize": "13px",
            "fontWeight": "700",
            "cursor": "pointer",
            "marginBottom": "8px",
        },
    )


def _json_safe(value):
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, (_dt.datetime, _dt.date, _dt.time)):
        return value.isoformat()
    if isinstance(value, np.ndarray):
        return [_json_safe(v) for v in value.tolist()]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    return value


# ─── Indicator data loading ───────────────────────────────────────────────────
def _load_indicator_df(symbol: str, interval: str, market: str = BINANCE_MARKET) -> pd.DataFrame:
    key = (symbol.upper(), interval)
    if key in _chart_df_cache:
        return _chart_df_cache[key]

    csv_path = _APP_DIR / f"{symbol.lower()}_{interval}.csv"
    df_raw = None
    if csv_path.exists():
        try:
            df_raw = load_klines(str(csv_path))
        except Exception as exc:
            print(f"[chart] Cannot read {csv_path}: {exc!r}")

    if df_raw is None:
        df_full = update_csv_cache(
            csv_path=str(csv_path),
            symbol=symbol,
            interval=interval,
            start_date=BINANCE_START_DATE,
            market=market,
            verbose=False,
        )
        df_raw = df_full.rename(columns={"open_time": "time"})
        df_raw["time"] = pd.to_datetime(df_raw["time"], unit="ms", utc=True)

    df = prepare_indicators(df_raw)
    _chart_df_cache[key] = df
    return df


# ─── Backtest / WFO runners (background threads) ──────────────────────────────
def _bt_thread(params, start, end, save):
    try:
        ss(status="Wczytywanie danych...", progress="")
        symbol = params.get("_symbol", BINANCE_SYMBOL)
        interval = params.get("_interval", BINANCE_INTERVAL)
        df = _load_indicator_df(symbol, interval)
        if start:
            df = df[df["time"] >= pd.Timestamp(start, tz="UTC")]
        if end:
            df = df[df["time"] <= pd.Timestamp(end, tz="UTC")]
        df = df.reset_index(drop=True)
        if df.empty:
            ss(running=False, status="Brak danych po filtrze.", result=None)
            return

        ss(status=f"Backtest na {len(df)} świecach...")
        clean_params = {k: v for k, v in params.items() if not k.startswith("_")}
        strat = Bee8Strategy(clean_params)
        trades, equity, final_cap = strat.run(df, INITIAL_CAPITAL)
        stats = compute_stats(trades, equity, INITIAL_CAPITAL, label="Backtest", print_output=False)

        result = {
            "kind": "backtest",
            "symbol": symbol,
            "interval": interval,
            "params": _json_safe(clean_params),
            "stats": _json_safe(stats),
            "trades": _json_safe(trades.to_dict(orient="records")),
            "equity": _json_safe(equity.to_dict(orient="records")),
            "final_capital": float(final_cap),
            "candles": len(df),
            "first_time": df["time"].iloc[0].isoformat(),
            "last_time": df["time"].iloc[-1].isoformat(),
        }
        if save and not trades.empty:
            trades.to_csv(_APP_DIR / "bee8_backtest_trades.csv", index=False)
            equity.to_csv(_APP_DIR / "bee8_backtest_equity.csv", index=False)
        ss(running=False, status="Backtest zakończony.", result=result)
    except Exception as exc:
        ss(running=False, status=f"Błąd backtestu: {exc!r}", result=None)


def _wfo_thread(base_params, grid_overrides, score_mode):
    try:
        ss(status="Wczytywanie danych do WFO...", progress="")
        symbol = base_params.get("_symbol", BINANCE_SYMBOL)
        interval = base_params.get("_interval", BINANCE_INTERVAL)
        df = _load_indicator_df(symbol, interval)
        ss(status=f"WFO na {len(df)} świecach...")

        clean = {k: v for k, v in base_params.items() if not k.startswith("_")}

        def _stop():
            return gs().get("stop", False)

        def _progress(window_id, total_windows, combo_idx, combo_total):
            ss(progress=f"Okno {window_id + 1}/{total_windows or '?'} | combo {combo_idx}/{combo_total}")

        all_trades, equity_wfo, windows_df, _final_cap, stopped = walk_forward_optimization(
            df,
            interval=interval,
            score_mode=score_mode,
            verbose=False,
            initial_capital=INITIAL_CAPITAL,
            base_params=clean,
            grid_overrides=grid_overrides or None,
            on_combo_progress=_progress,
            should_stop=_stop,
        )
        stats = compute_stats(all_trades, equity_wfo, INITIAL_CAPITAL, label="WFO live", print_output=False)
        best = get_latest_best_params(windows_df) if (windows_df is not None and not windows_df.empty) else {}

        result = {
            "kind": "wfo",
            "symbol": symbol,
            "interval": interval,
            "stats": _json_safe(stats),
            "trades": _json_safe(all_trades.to_dict(orient="records")) if all_trades is not None else [],
            "equity": _json_safe(equity_wfo.to_dict(orient="records")) if equity_wfo is not None else [],
            "windows": _json_safe(windows_df.to_dict(orient="records")) if windows_df is not None else [],
            "best_params": _json_safe(best),
            "stopped": stopped,
        }
        ss(running=False, status="WFO zakończone.", result=result)
    except Exception as exc:
        ss(running=False, status=f"Błąd WFO: {exc!r}", result=None)


# ─── Plot helpers ─────────────────────────────────────────────────────────────
def _make_chart(df: pd.DataFrame, trades: list[dict]) -> go.Figure:
    fig = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.5, 0.18, 0.16, 0.16],
        vertical_spacing=0.02,
        subplot_titles=("Cena + BB", "RSI / Stoch K", "MFI / BB %B", "Supertrend dir + ROC"),
    )
    fig.add_trace(
        go.Candlestick(
            x=df["time"],
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
            name="ETHUSDT",
            increasing_line_color=C["green"],
            decreasing_line_color=C["red"],
        ),
        row=1,
        col=1,
    )
    fig.add_trace(go.Scatter(x=df["time"], y=df["bb_upper"], name="BB upper", line=dict(color=C["amber"], width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df["time"], y=df["bb_mid"], name="BB mid", line=dict(color=C["muted"], width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df["time"], y=df["bb_lower"], name="BB lower", line=dict(color=C["amber"], width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df["time"], y=df["supertrend"], name="Supertrend", line=dict(color=C["purple"], width=1)), row=1, col=1)

    if trades:
        t_df = pd.DataFrame(trades)
        if "entry_time" in t_df.columns:
            t_df["entry_time"] = pd.to_datetime(t_df["entry_time"], errors="coerce")
            t_df["exit_time"] = pd.to_datetime(t_df["exit_time"], errors="coerce")
            fig.add_trace(
                go.Scatter(
                    x=t_df["entry_time"],
                    y=t_df["entry_price"],
                    mode="markers",
                    marker=dict(symbol="triangle-up", color=C["green"], size=10, line=dict(color="#0b1220", width=1)),
                    name="Wejścia",
                ),
                row=1,
                col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=t_df["exit_time"],
                    y=t_df["exit_price"],
                    mode="markers",
                    marker=dict(symbol="triangle-down", color=C["red"], size=10, line=dict(color="#0b1220", width=1)),
                    name="Wyjścia",
                ),
                row=1,
                col=1,
            )

    fig.add_trace(go.Scatter(x=df["time"], y=df["rsi"], name="RSI", line=dict(color=C["blue"], width=1)), row=2, col=1)
    fig.add_trace(go.Scatter(x=df["time"], y=df["stoch_k"], name="Stoch %K", line=dict(color=C["purple"], width=1)), row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color=C["green"], opacity=0.4, row=2, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color=C["red"], opacity=0.4, row=2, col=1)

    fig.add_trace(go.Scatter(x=df["time"], y=df["mfi"], name="MFI", line=dict(color=C["amber"], width=1)), row=3, col=1)
    fig.add_trace(go.Scatter(x=df["time"], y=df["bb_pct"] * 100.0, name="BB %B x100", line=dict(color=C["coral"], width=1)), row=3, col=1)
    fig.add_hline(y=20, line_dash="dash", line_color=C["green"], opacity=0.4, row=3, col=1)
    fig.add_hline(y=80, line_dash="dash", line_color=C["red"], opacity=0.4, row=3, col=1)

    fig.add_trace(go.Scatter(x=df["time"], y=df["supertrend_dir"], name="Supertrend dir", line=dict(color=C["green"], width=1)), row=4, col=1)
    fig.add_trace(go.Scatter(x=df["time"], y=df["roc"], name="ROC %", line=dict(color=C["coral"], width=1)), row=4, col=1)
    fig.add_hline(y=0, line_color=C["muted"], opacity=0.4, row=4, col=1)

    fig.update_layout(
        template="plotly_dark",
        plot_bgcolor=C["bg"],
        paper_bgcolor=C["bg"],
        font=dict(color=C["text"]),
        showlegend=True,
        legend=dict(orientation="h", y=1.05, x=0.0),
        margin=dict(l=20, r=20, t=40, b=20),
        height=820,
        xaxis_rangeslider_visible=False,
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=True, gridcolor=C["border"])
    return fig


# ─── App ──────────────────────────────────────────────────────────────────────
app = dash.Dash(__name__, update_title=None)
app.title = "bee8 - Mean Reversion"
app.layout = html.Div(
    style={
        "background": C["bg"],
        "minHeight": "100vh",
        "color": C["text"],
        "fontFamily": "Inter, sans-serif",
        "padding": "16px 28px",
    },
    children=[
        dcc.Store(id="bt-result-store"),
        dcc.Interval(id="poll", interval=1500, n_intervals=0),
        html.Div(
            style={"display": "flex", "justifyContent": "space-between", "alignItems": "center", "marginBottom": "16px"},
            children=[
                html.Div(
                    [
                        html.H1(
                            "bee8 - Mean Reversion (BB / Stoch / MFI / RSI + Supertrend)",
                            style={"margin": 0, "fontSize": "22px", "letterSpacing": "0.02em"},
                        ),
                        html.Div(
                            f"server token: {_SERVER_TOKEN}",
                            style={"color": C["muted"], "fontSize": "11px"},
                        ),
                    ]
                ),
                html.Div(
                    [
                        html.A(
                            "GitHub",
                            href="https://github.com/dario2210/bee8",
                            target="_blank",
                            style={"color": C["blue"], "textDecoration": "none", "fontSize": "13px"},
                        )
                    ]
                ),
            ],
        ),
        html.Div(
            style={"display": "grid", "gridTemplateColumns": "360px 1fr", "gap": "16px"},
            children=[
                html.Div(
                    [
                        html.Div(
                            card_s_wrap(
                                [
                                    sec("Dane"),
                                    field("Symbol", inp("symbol", BINANCE_SYMBOL)),
                                    field("Interval", drp("interval", [{"label": x, "value": x} for x in ["15m", "30m", "1h", "2h", "4h"]], BINANCE_INTERVAL)),
                                    field("Market", drp("market", [{"label": x, "value": x} for x in ["spot", "futures"]], BINANCE_MARKET)),
                                    field("Start (YYYY-MM-DD)", inp("bt-start", "")),
                                    field("End (YYYY-MM-DD)", inp("bt-end", "")),
                                ]
                            )
                        ),
                        html.Div(
                            card_s_wrap(
                                [
                                    sec("Entry (1h, mean reversion)"),
                                    field("BB %B max", drp("bb-in", _opts(BB_LONG_ENTRY_MAX_OPTIONS), DEFAULT_PARAMS["bb_long_entry_max"])),
                                    field("Stoch K max", drp("stoch-in", _opts(STOCH_LONG_ENTRY_MAX_OPTIONS), DEFAULT_PARAMS["stoch_long_entry_max"])),
                                    field("MFI max", drp("mfi-in", _opts(MFI_LONG_ENTRY_MAX_OPTIONS), DEFAULT_PARAMS["mfi_long_entry_max"])),
                                    field("RSI max", drp("rsi-in", _opts(RSI_LONG_ENTRY_MAX_OPTIONS), DEFAULT_PARAMS["rsi_long_entry_max"])),
                                    field("Min confluence", drp("conf-in", _opts(LONG_ENTRY_MIN_CONFLUENCE_OPTIONS), DEFAULT_PARAMS["long_entry_min_confluence"])),
                                    field("Lookback bars", drp("lookback", _opts(LOOKBACK_BARS_OPTIONS), DEFAULT_PARAMS["lookback_bars"])),
                                    field("BB extreme level", drp("bb-extreme", _opts(BB_EXTREME_LEVEL_OPTIONS), DEFAULT_PARAMS["bb_extreme_level"])),
                                ]
                            )
                        ),
                        html.Div(
                            card_s_wrap(
                                [
                                    sec("Filtr 4h"),
                                    field("Użyj filtra HTF", drp("htf-use", [{"label": "Tak", "value": True}, {"label": "Nie", "value": False}], DEFAULT_PARAMS["use_htf_filter"])),
                                    field("HTF RSI max", drp("htf-rsi", _opts(HTF_RSI_MAX_OPTIONS), DEFAULT_PARAMS["htf_rsi_max"])),
                                    field("HTF BB %B max", drp("htf-bb", _opts(HTF_BB_MAX_OPTIONS), DEFAULT_PARAMS["htf_bb_max"])),
                                ]
                            )
                        ),
                        html.Div(
                            card_s_wrap(
                                [
                                    sec("Exit"),
                                    field("BB %B min", drp("bb-out", _opts(BB_LONG_EXIT_MIN_OPTIONS), DEFAULT_PARAMS["bb_long_exit_min"])),
                                    field("Stoch K min", drp("stoch-out", _opts(STOCH_LONG_EXIT_MIN_OPTIONS), DEFAULT_PARAMS["stoch_long_exit_min"])),
                                    field("MFI min", drp("mfi-out", _opts(MFI_LONG_EXIT_MIN_OPTIONS), DEFAULT_PARAMS["mfi_long_exit_min"])),
                                    field("RSI min", drp("rsi-out", _opts(RSI_LONG_EXIT_MIN_OPTIONS), DEFAULT_PARAMS["rsi_long_exit_min"])),
                                    field("ROC % min", drp("roc-out", _opts(ROC_LONG_EXIT_MIN_PCT_OPTIONS), DEFAULT_PARAMS["roc_long_exit_min_pct"])),
                                    field("Min confluence out", drp("conf-out", _opts(LONG_EXIT_MIN_CONFLUENCE_OPTIONS), DEFAULT_PARAMS["long_exit_min_confluence"])),
                                    field("Supertrend flip exit", drp("st-flip", [{"label": "Tak", "value": True}, {"label": "Nie", "value": False}], DEFAULT_PARAMS["supertrend_flip_exit"])),
                                    field("Exit nad chmurą 4h", drp("cloud-exit", [{"label": "Tak", "value": True}, {"label": "Nie", "value": False}], DEFAULT_PARAMS["ichimoku_above_cloud_exit"])),
                                ]
                            )
                        ),
                        html.Div(
                            card_s_wrap(
                                [
                                    sec("Ryzyko"),
                                    field("Emergency SL (% kapitału)", drp("sl-pct", _opts(LONG_EMERGENCY_SL_CAPITAL_PCT_OPTIONS), DEFAULT_PARAMS["long_emergency_sl_capital_pct"])),
                                ]
                            )
                        ),
                        html.Div(
                            card_s_wrap(
                                [
                                    sec("Akcje"),
                                    btn("run-bt", "Uruchom backtest", C["blue"]),
                                    btn("run-wfo", "Uruchom WFO", C["purple"]),
                                    btn("stop-wfo", "Stop WFO", C["red"]),
                                    btn("save-params", "Zapisz parametry jako WFO best", C["green"]),
                                ]
                            )
                        ),
                    ],
                    style={"display": "flex", "flexDirection": "column"},
                ),
                html.Div(
                    [
                        html.Div(
                            card_s_wrap(
                                [
                                    html.Div(id="status-line", style={"fontSize": "13px"}),
                                    html.Div(id="progress-line", style={"fontSize": "12px", "color": C["muted"]}),
                                ]
                            )
                        ),
                        html.Div(card_s_wrap([dcc.Graph(id="chart", config={"displaylogo": False})])),
                        html.Div(card_s_wrap([html.Div(id="stats-pane")])),
                        html.Div(card_s_wrap([html.Div(id="trades-pane")])),
                    ],
                ),
            ],
        ),
    ],
)


def _opts(values):
    return [{"label": str(v), "value": v} for v in values]


def card_s_wrap(children):
    return html.Div(children, style=card_s)


# ─── Build params from controls ───────────────────────────────────────────────
def _gather_params(ctrl_values, symbol, interval, market):
    params = dict(DEFAULT_PARAMS)
    keys = [
        "bb_long_entry_max", "stoch_long_entry_max", "mfi_long_entry_max", "rsi_long_entry_max",
        "long_entry_min_confluence", "lookback_bars", "bb_extreme_level",
        "use_htf_filter", "htf_rsi_max", "htf_bb_max",
        "bb_long_exit_min", "stoch_long_exit_min", "mfi_long_exit_min", "rsi_long_exit_min",
        "roc_long_exit_min_pct", "long_exit_min_confluence",
        "supertrend_flip_exit", "ichimoku_above_cloud_exit",
        "long_emergency_sl_capital_pct",
    ]
    for k, v in zip(keys, ctrl_values):
        params[k] = v
    params["long_emergency_sl_enabled"] = float(params["long_emergency_sl_capital_pct"]) > 0.0
    params["_symbol"] = symbol or BINANCE_SYMBOL
    params["_interval"] = interval or BINANCE_INTERVAL
    params["_market"] = market or BINANCE_MARKET
    return params


# ─── Callbacks ────────────────────────────────────────────────────────────────
@app.callback(
    Output("status-line", "children"),
    Output("progress-line", "children"),
    Output("bt-result-store", "data"),
    Input("poll", "n_intervals"),
)
def _poll(_):
    st = gs()
    status_txt = st["status"] or ""
    progress_txt = st["progress"] or ""
    result_payload = {"v": st["result_version"], "result": st["result"]}
    return status_txt, progress_txt, result_payload


@app.callback(
    Output("status-line", "children", allow_duplicate=True),
    Input("run-bt", "n_clicks"),
    State("symbol", "value"),
    State("interval", "value"),
    State("market", "value"),
    State("bt-start", "value"),
    State("bt-end", "value"),
    State("bb-in", "value"),
    State("stoch-in", "value"),
    State("mfi-in", "value"),
    State("rsi-in", "value"),
    State("conf-in", "value"),
    State("lookback", "value"),
    State("bb-extreme", "value"),
    State("htf-use", "value"),
    State("htf-rsi", "value"),
    State("htf-bb", "value"),
    State("bb-out", "value"),
    State("stoch-out", "value"),
    State("mfi-out", "value"),
    State("rsi-out", "value"),
    State("roc-out", "value"),
    State("conf-out", "value"),
    State("st-flip", "value"),
    State("cloud-exit", "value"),
    State("sl-pct", "value"),
    prevent_initial_call=True,
)
def _on_run_bt(n, symbol, interval, market, start, end, *ctrl):
    if not n:
        raise dash.exceptions.PreventUpdate
    if gs().get("running"):
        return "Już coś biega - poczekaj."
    params = _gather_params(ctrl, symbol, interval, market)
    ss(running=True, stop=False, status="Start backtestu...", progress="", result=None)
    threading.Thread(target=_bt_thread, args=(params, start, end, True), daemon=True).start()
    return "Start backtestu..."


@app.callback(
    Output("status-line", "children", allow_duplicate=True),
    Input("run-wfo", "n_clicks"),
    State("symbol", "value"),
    State("interval", "value"),
    State("market", "value"),
    State("bb-in", "value"),
    State("stoch-in", "value"),
    State("mfi-in", "value"),
    State("rsi-in", "value"),
    State("conf-in", "value"),
    State("lookback", "value"),
    State("bb-extreme", "value"),
    State("htf-use", "value"),
    State("htf-rsi", "value"),
    State("htf-bb", "value"),
    State("bb-out", "value"),
    State("stoch-out", "value"),
    State("mfi-out", "value"),
    State("rsi-out", "value"),
    State("roc-out", "value"),
    State("conf-out", "value"),
    State("st-flip", "value"),
    State("cloud-exit", "value"),
    State("sl-pct", "value"),
    prevent_initial_call=True,
)
def _on_run_wfo(n, symbol, interval, market, *ctrl):
    if not n:
        raise dash.exceptions.PreventUpdate
    if gs().get("running"):
        return "Już coś biega - poczekaj."
    params = _gather_params(ctrl, symbol, interval, market)
    ss(running=True, stop=False, status="Start WFO...", progress="", result=None)
    threading.Thread(target=_wfo_thread, args=(params, None, "balanced"), daemon=True).start()
    return "Start WFO..."


@app.callback(
    Output("status-line", "children", allow_duplicate=True),
    Input("stop-wfo", "n_clicks"),
    prevent_initial_call=True,
)
def _on_stop(n):
    if not n:
        raise dash.exceptions.PreventUpdate
    ss(stop=True)
    return "Stop zażądany - dokończę bieżące combo."


@app.callback(
    Output("status-line", "children", allow_duplicate=True),
    Input("save-params", "n_clicks"),
    State("bb-in", "value"),
    State("stoch-in", "value"),
    State("mfi-in", "value"),
    State("rsi-in", "value"),
    State("conf-in", "value"),
    State("lookback", "value"),
    State("bb-extreme", "value"),
    State("htf-use", "value"),
    State("htf-rsi", "value"),
    State("htf-bb", "value"),
    State("bb-out", "value"),
    State("stoch-out", "value"),
    State("mfi-out", "value"),
    State("rsi-out", "value"),
    State("roc-out", "value"),
    State("conf-out", "value"),
    State("st-flip", "value"),
    State("cloud-exit", "value"),
    State("sl-pct", "value"),
    prevent_initial_call=True,
)
def _on_save(n, *ctrl):
    if not n:
        raise dash.exceptions.PreventUpdate
    params = _gather_params(ctrl, BINANCE_SYMBOL, BINANCE_INTERVAL, BINANCE_MARKET)
    clean = {k: v for k, v in params.items() if not k.startswith("_")}
    save_params(clean)
    return f"Zapisano parametry do {Path(_APP_DIR / 'bee8_wfo_best_params.json').name}"


@app.callback(
    Output("chart", "figure"),
    Output("stats-pane", "children"),
    Output("trades-pane", "children"),
    Input("bt-result-store", "data"),
)
def _render_result(store):
    result = (store or {}).get("result")
    if not result:
        empty = go.Figure(layout=dict(template="plotly_dark", paper_bgcolor=C["bg"], plot_bgcolor=C["bg"], height=820))
        return empty, html.Div("Brak wyników. Uruchom backtest lub WFO.", style={"color": C["muted"]}), html.Div()

    symbol = result.get("symbol", BINANCE_SYMBOL)
    interval = result.get("interval", BINANCE_INTERVAL)
    try:
        df = _load_indicator_df(symbol, interval)
    except Exception as exc:
        df = pd.DataFrame()
        print(f"[chart] load failed: {exc!r}")

    trades = result.get("trades") or []
    if not df.empty and trades:
        first_entry = pd.to_datetime(trades[0].get("entry_time"), errors="coerce", utc=True)
        last_exit = pd.to_datetime(trades[-1].get("exit_time"), errors="coerce", utc=True)
        if pd.notna(first_entry) and pd.notna(last_exit):
            mask = (df["time"] >= first_entry - pd.Timedelta(days=2)) & (df["time"] <= last_exit + pd.Timedelta(days=2))
            df = df[mask]
    fig = _make_chart(df, trades) if not df.empty else go.Figure(layout=dict(template="plotly_dark"))

    stats = result.get("stats") or {}
    stats_table = _stats_table(stats)
    trades_table = _trades_table(trades)
    return fig, stats_table, trades_table


def _stats_table(stats: dict):
    if not stats:
        return html.Div("Brak statystyk.", style={"color": C["muted"]})
    rows = []
    keys = [
        ("n_trades", "Transakcji"),
        ("final_capital", "Kapitał końcowy"),
        ("net_return_pct", "Zwrot netto %"),
        ("winrate_pct", "Winrate %"),
        ("profit_factor", "Profit factor"),
        ("expectancy_usd", "Expectancy USD/trade"),
        ("sharpe_ratio", "Sharpe"),
        ("sortino_ratio", "Sortino"),
        ("max_drawdown_pct", "Max drawdown %"),
        ("cagr_pct", "CAGR %"),
        ("exposure_pct", "Exposure %"),
        ("avg_winner_usd", "Avg winner USD"),
        ("avg_loser_usd", "Avg loser USD"),
        ("longest_losing_streak", "Max losing streak"),
    ]
    for k, label in keys:
        v = stats.get(k)
        if v is None or (isinstance(v, float) and not np.isfinite(v)):
            txt = "n/d"
        elif isinstance(v, float):
            txt = f"{v:,.3f}"
        else:
            txt = str(v)
        rows.append(html.Tr([html.Td(label, style={"padding": "4px 12px"}), html.Td(txt, style={"padding": "4px 12px", "fontWeight": 600})]))
    return html.Table(rows, style={"borderCollapse": "collapse", "width": "100%", "fontSize": "13px"})


def _trades_table(trades: list[dict]):
    if not trades:
        return html.Div("Brak transakcji.", style={"color": C["muted"]})
    df = pd.DataFrame(trades)
    cols = [
        "logical_trade_no", "side", "entry_time", "exit_time", "entry_price",
        "exit_price", "net_ret", "pnl", "exit_trigger", "entry_confluence",
        "exit_confluence", "holding_hours",
    ]
    cols = [c for c in cols if c in df.columns]
    df = df[cols].copy()
    if "net_ret" in df.columns:
        df["net_ret"] = (df["net_ret"] * 100.0).round(3)
    if "pnl" in df.columns:
        df["pnl"] = df["pnl"].round(2)
    if "holding_hours" in df.columns:
        df["holding_hours"] = df["holding_hours"].round(1)
    return dash_table.DataTable(
        data=df.to_dict(orient="records"),
        columns=[{"name": c, "id": c} for c in df.columns],
        page_size=15,
        style_header={"backgroundColor": C["surf2"], "color": C["text"], "fontWeight": "600"},
        style_cell={"backgroundColor": C["surface"], "color": C["text"], "border": f"1px solid {C['border']}", "fontSize": "12px", "padding": "6px"},
    )


# ─── Entrypoint ──────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="bee8 dashboard")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8068)
    p.add_argument("--debug", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    app.run(host=args.host, port=args.port, debug=args.debug)
