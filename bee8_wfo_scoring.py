"""
bee8_wfo_scoring.py
====================
Funkcje scoringu dla WFO (punkt 5 z briefu).

Tryby:
  return_only  – tylko zwrot % (jak wcześniej, szybki)
  balanced     – zwrot + PF + DD + kara za mało transakcji
  defensive    – nacisk na drawdown i stabilność, penalty za wysokie DD
  growth       – większy nacisk na zwrot, ale z nadal aktywną karą za DD/PF

Użycie:
  from bee8_wfo_scoring import score_params
  score = score_params(trades, final_cap, initial_cap, mode="balanced")
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def score_params(
    trades         : pd.DataFrame,
    final_capital  : float,
    initial_capital: float,
    mode           : str = "balanced",
) -> float:
    """
    Ocenia wyniki fazy optymalizacji.

    Parametry:
      trades          – DataFrame z transakcjami (musi mieć kolumnę 'pnl')
      final_capital   – kapitał końcowy okna opt
      initial_capital – kapitał startowy okna opt
      mode            – "return_only" | "balanced" | "defensive" | "growth"

    Zwraca float – wyższy = lepszy.
    """
    if trades is None or trades.empty:
        return -9999.0

    n = len(trades)

    # ── zwrot ─────────────────────────────────────────────────────────────
    ret_pct = (final_capital / initial_capital - 1.0) * 100.0

    if mode == "return_only":
        wins   = trades[trades["pnl"] > 0]["pnl"].sum()
        losses = trades[trades["pnl"] <= 0]["pnl"].sum()
        pf     = wins / abs(losses) if losses < 0 else 0.0
        return ret_pct - (10.0 if pf < 1.0 else 0.0)

    # ── wspólne metryki ───────────────────────────────────────────────────
    wins_df   = trades[trades["pnl"] > 0]
    losses_df = trades[trades["pnl"] <= 0]

    gross_profit = wins_df["pnl"].sum()
    gross_loss   = losses_df["pnl"].sum()
    pf           = gross_profit / abs(gross_loss) if gross_loss < 0 else 0.0
    winrate      = len(wins_df) / n if n > 0 else 0.0

    # max drawdown z equity (jeśli brak equity, szacujemy z pnl)
    equity = np.array([initial_capital] + list(
        initial_capital + trades["pnl"].cumsum().values
    ))
    running_max = np.maximum.accumulate(equity)
    dd_arr      = (equity - running_max) / running_max
    max_dd_pct  = dd_arr.min() * 100.0   # ujemna wartość

    # kara za zbyt małą aktywność (< 3 transakcje → wyniki niereprezentatywne)
    activity_penalty = 0.0
    if n < 3:
        activity_penalty = 20.0
    elif n < 5:
        activity_penalty = 8.0

    # ── kara za koncentrację wyniku (2 największe trade'y > 60% sumy) ────────
    if trades["pnl"].sum() > 1e-9:
        top_share = trades["pnl"].nlargest(min(2, n)).sum() / trades["pnl"].sum()
        concentration_penalty = max(0.0, top_share - 0.6) * 20.0
    else:
        concentration_penalty = 0.0

    if mode == "balanced":
        # składowe:
        #   zwrot           waga 1.0
        #   profit factor   waga 3.0  (skalowany do %)
        #   drawdown kara   odliczana wprost
        pf_score  = min(pf, 3.0) * 3.0   # cap na 3, maks +9
        dd_penalty= max(0.0, -max_dd_pct - 5.0) * 0.5  # kara gdy DD > 5%
        pf_penalty= 10.0 if pf < 1.0 else 0.0

        score = ret_pct + pf_score - dd_penalty - pf_penalty - activity_penalty - concentration_penalty
        return score

    if mode == "defensive":
        # priorytet: unikanie obsunięcia
        #   zwrot           waga 0.5
        #   PF              waga 2.0
        #   DD kara         mocna: każdy % DD powyżej 3% kosztuje 1.5 pkt
        pf_score   = min(pf, 3.0) * 2.0
        dd_penalty = max(0.0, -max_dd_pct - 3.0) * 1.5
        pf_penalty = 15.0 if pf < 1.0 else 0.0

        score = ret_pct * 0.5 + pf_score - dd_penalty - pf_penalty - activity_penalty - concentration_penalty
        return score

    if mode == "growth":
        # Priorytet: większy zwrot, ale bez ignorowania jakości wyniku.
        # DD powyżej 8% jest karany łagodnie, a powyżej 15% mocniej.
        pf_score = min(pf, 3.0) * 1.5
        dd_penalty = max(0.0, -max_dd_pct - 8.0) * 0.8
        severe_dd_penalty = max(0.0, -max_dd_pct - 15.0) * 2.0
        pf_penalty = 12.0 if pf < 1.0 else 0.0

        score = (
            ret_pct * 1.5
            + pf_score
            + winrate * 2.0
            - dd_penalty
            - severe_dd_penalty
            - pf_penalty
            - activity_penalty
            - concentration_penalty * 0.5
        )
        return score

    # fallback → balanced
    return ret_pct

