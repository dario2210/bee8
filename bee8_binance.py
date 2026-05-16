"""
bee4_binance.py
================
Pobieranie świec OHLCV z publicznego API Binance (spot + futures).
Nie wymaga klucza API – dane świec są publiczne.

Funkcje:
  fetch_klines_binance()   – pobiera kawałek danych z Binance
  update_csv_cache()       – tworzy lub dociąga lokalny CSV cache
  get_bars_per_day()       – przelicza timeframe na liczbę barów dziennie
  bars_for_days()          – pomocnicze: N dni → N barów dla danego TF

Obsługiwane timeframes (Binance interval strings):
  1m, 3m, 5m, 15m, 30m
  1h, 2h, 4h, 6h, 8h, 12h
  1d, 3d, 1w
"""

from __future__ import annotations

import time
import numpy as np
import pandas as pd
import requests

# ──────────────────────────────────────────────────────────────────────────────
# KONFIGURACJA
# ──────────────────────────────────────────────────────────────────────────────

# Publiczne endpointy – bez klucza API, realne dane giełdowe (nie testnet)
BINANCE_SPOT_URL    = "https://api.binance.com/api/v3/klines"
BINANCE_FUTURES_URL = "https://fapi.binance.com/fapi/v1/klines"

BINANCE_LIMIT = 1000   # max świec na jedno zapytanie (limit Binance)
REQUEST_DELAY = 0.25   # s przerwy między requestami (rate-limit safety)

DEFAULT_SYMBOL    = "ETHUSDT"
DEFAULT_INTERVAL  = "1h"
DEFAULT_START_DATE = "2021-01-01"

# Mapa: interval string → minuty
TF_MINUTES: dict[str, float] = {
    "1m"  :    1,
    "3m"  :    3,
    "5m"  :    5,
    "15m" :   15,
    "30m" :   30,
    "1h"  :   60,
    "2h"  :  120,
    "4h"  :  240,
    "6h"  :  360,
    "8h"  :  480,
    "12h" :  720,
    "1d"  : 1440,
    "3d"  : 4320,
    "1w"  : 10080,
}


# ──────────────────────────────────────────────────────────────────────────────
# POMOCNICZE
# ──────────────────────────────────────────────────────────────────────────────

def get_bars_per_day(interval: str) -> float:
    """Liczba barów na dobę dla danego timeframe."""
    minutes = TF_MINUTES.get(interval)
    if minutes is None:
        raise ValueError(
            f"Nieznany interval '{interval}'. "
            f"Dostępne: {list(TF_MINUTES.keys())}"
        )
    return 1440.0 / minutes


def bars_for_days(days: int, interval: str) -> int:
    """Ilu barów potrzeba na N dni trading (cały dobę, crypto 24/7)."""
    return int(round(days * get_bars_per_day(interval)))


def interval_to_ms(interval: str) -> int:
    """Długość jednego baru w milisekundach."""
    return int(TF_MINUTES[interval] * 60 * 1000)


# ──────────────────────────────────────────────────────────────────────────────
# POBIERANIE Z BINANCE
# ──────────────────────────────────────────────────────────────────────────────

def fetch_klines_binance(
    symbol         : str  = DEFAULT_SYMBOL,
    interval       : str  = DEFAULT_INTERVAL,
    start_time_ms  : int  = None,
    end_time_ms    : int  = None,
    market         : str  = "spot",   # "spot" lub "futures"
    verbose        : bool = True,
) -> pd.DataFrame:
    """
    Pobiera świece OHLCV z publicznego API Binance.

    Parametry:
      symbol       – np. "ETHUSDT", "BTCUSDT"
      interval     – timeframe: "1h", "4h", "1d" itd.
      start_time_ms– czas startu w ms (unix epoch). None = brak filtra od
      end_time_ms  – czas końca w ms. None = do teraz
      market       – "spot" lub "futures"
      verbose      – drukuj postęp

    Zwraca DataFrame z kolumnami:
      open_time (int, ms), open, high, low, close, volume (float)
    """
    base_url = BINANCE_FUTURES_URL if market == "futures" else BINANCE_SPOT_URL

    if interval not in TF_MINUTES:
        raise ValueError(f"Nieznany interval '{interval}'. Dostępne: {list(TF_MINUTES.keys())}")

    all_rows: list = []
    current_start = int(start_time_ms) if start_time_ms is not None else None

    bar_ms = interval_to_ms(interval)

    if verbose:
        start_str = pd.Timestamp(current_start, unit="ms", tz="UTC").strftime("%Y-%m-%d") if current_start else "początek"
        end_str   = pd.Timestamp(end_time_ms,   unit="ms", tz="UTC").strftime("%Y-%m-%d") if end_time_ms   else "teraz"
        print(f"[Binance] Pobieranie {symbol} {interval}  |  {start_str} → {end_str}  |  rynek: {market}")

    batch = 0
    while True:
        params: dict = {
            "symbol"  : symbol,
            "interval": interval,
            "limit"   : BINANCE_LIMIT,
        }
        if current_start is not None:
            params["startTime"] = current_start
        if end_time_ms is not None:
            params["endTime"] = end_time_ms

        try:
            resp = requests.get(base_url, params=params, timeout=15)
            resp.raise_for_status()
            data = resp.json()
        except requests.RequestException as e:
            raise RuntimeError(f"[Binance] Błąd HTTP: {e!r}") from e

        if not data:
            break

        all_rows.extend(data)
        batch += 1

        last_open_time_ms = int(data[-1][0])

        if verbose and batch % 5 == 0:
            ts = pd.Timestamp(last_open_time_ms, unit="ms", tz="UTC").strftime("%Y-%m-%d %H:%M")
            print(f"  batch {batch:3d} | do: {ts} | razem: {len(all_rows)} świec")

        # koniec: ostatni batch był niepełny LUB doszliśmy do end_time
        if len(data) < BINANCE_LIMIT:
            break
        if end_time_ms is not None and last_open_time_ms >= end_time_ms:
            break

        # przesuń start na następną świecę po ostatniej pobranej
        current_start = last_open_time_ms + bar_ms
        time.sleep(REQUEST_DELAY)

    if not all_rows:
        return pd.DataFrame(columns=["open_time", "open", "high", "low", "close", "volume"])

    cols = [
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "number_of_trades",
        "taker_buy_base", "taker_buy_quote", "ignore",
    ]
    df = pd.DataFrame(all_rows, columns=cols)
    df["open_time"] = df["open_time"].astype("int64")
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = df[c].astype(float)

    if verbose:
        print(f"[Binance] Pobrano {len(df)} świec  "
              f"({pd.Timestamp(df['open_time'].iloc[0],  unit='ms', tz='UTC').strftime('%Y-%m-%d')} → "
              f"{pd.Timestamp(df['open_time'].iloc[-1], unit='ms', tz='UTC').strftime('%Y-%m-%d')})")

    return df[["open_time", "open", "high", "low", "close", "volume"]]


# ──────────────────────────────────────────────────────────────────────────────
# CACHE CSV
# ──────────────────────────────────────────────────────────────────────────────

def update_csv_cache(
    csv_path   : str,
    symbol     : str  = DEFAULT_SYMBOL,
    interval   : str  = DEFAULT_INTERVAL,
    start_date : str  = DEFAULT_START_DATE,
    market     : str  = "spot",
    verbose    : bool = True,
) -> pd.DataFrame:
    """
    Tworzy lub aktualizuje lokalny plik CSV ze świecami.

    Logika:
      - Jeśli CSV nie istnieje → pobiera dane od start_date do teraz
      - Jeśli CSV istnieje → uzupełnia starsze świece, jeśli start_date jest
        wcześniejsze niż początek cache, oraz dociąga nowe świece od końca cache

    Zwraca pełny DataFrame gotowy do użycia w strategii (kolumna 'open_time' jako int ms).
    """
    import os

    requested_start_ms = int(pd.Timestamp(start_date, tz="UTC").timestamp() * 1000)
    bar_ms = interval_to_ms(interval)
    keep = ["open_time", "open", "high", "low", "close", "volume"]

    if os.path.exists(csv_path):
        if verbose:
            print(f"[Cache] Wczytuję istniejący CSV: {csv_path}")
        df_old = pd.read_csv(csv_path)

        # normalizacja nazwy kolumny czasu
        if "open_time" not in df_old.columns:
            if "time" in df_old.columns:
                df_old = df_old.rename(columns={"time": "open_time"})
            else:
                raise ValueError("CSV musi mieć kolumnę 'open_time' lub 'time'.")

        # konwersja open_time na int ms (może być zapisany jako string daty)
        if not np.issubdtype(df_old["open_time"].dtype, np.number):
            if verbose:
                print("[Cache] Kolumna open_time to string – konwertuję na ms.")
            df_old["open_time"] = (
                pd.to_datetime(df_old["open_time"], utc=True).astype("int64") // 10**6
            ).astype("int64")

        df_old = (
            df_old[keep]
            .drop_duplicates(subset=["open_time"])
            .sort_values("open_time")
            .reset_index(drop=True)
        )

        first_ms = int(df_old["open_time"].iloc[0])
        last_ms = int(df_old["open_time"].iloc[-1])
        next_start_ms = last_ms + bar_ms   # pierwsza brakująca świeca

        now_ms = int(pd.Timestamp.now(tz="UTC").timestamp() * 1000)
        parts = []

        if requested_start_ms < first_ms:
            backfill_end_ms = first_ms - bar_ms
            if verbose:
                print(
                    "[Cache] Uzupełniam starsze świece od "
                    f"{pd.Timestamp(requested_start_ms, unit='ms', tz='UTC').strftime('%Y-%m-%d %H:%M')} "
                    "do "
                    f"{pd.Timestamp(backfill_end_ms, unit='ms', tz='UTC').strftime('%Y-%m-%d %H:%M')}..."
                )
            df_backfill = fetch_klines_binance(
                symbol=symbol,
                interval=interval,
                start_time_ms=requested_start_ms,
                end_time_ms=backfill_end_ms,
                market=market,
                verbose=verbose,
            )
            if not df_backfill.empty:
                parts.append(df_backfill[keep])

        if next_start_ms < now_ms:
            if verbose:
                print(f"[Cache] Dociągam nowe świece od {pd.Timestamp(next_start_ms, unit='ms', tz='UTC').strftime('%Y-%m-%d %H:%M')}...")

            df_new = fetch_klines_binance(
                symbol       = symbol,
                interval     = interval,
                start_time_ms= next_start_ms,
                market       = market,
                verbose      = verbose,
            )
            if not df_new.empty:
                parts.append(df_new[keep])
        elif verbose:
            print("[Cache] CSV jest aktualny – brak nowych świec do pobrania.")

        if not parts:
            if verbose:
                print("[Cache] Brak brakujących świec do dopisania.")
            return df_old[["open_time", "open", "high", "low", "close", "volume"]].copy()

        df_all = pd.concat([*parts, df_old[keep]], ignore_index=True)
        df_all = (
            df_all
            .drop_duplicates(subset=["open_time"])
            .sort_values("open_time")
            .reset_index(drop=True)
        )

    else:
        if verbose:
            print(f"[Cache] CSV nie istnieje – pobieram od {start_date}.")
        df_all = fetch_klines_binance(
            symbol        = symbol,
            interval      = interval,
            start_time_ms = requested_start_ms,
            market        = market,
            verbose       = verbose,
        )
        if df_all.empty:
            raise RuntimeError(f"Nie udało się pobrać żadnych danych dla {symbol} {interval}.")

    df_all.to_csv(csv_path, index=False)
    if verbose:
        print(f"[Cache] Zapisano → {csv_path}  ({len(df_all)} świec)")

    return df_all


# ──────────────────────────────────────────────────────────────────────────────
# PRZELICZNIK WFO – okna w barach dla danego TF
# ──────────────────────────────────────────────────────────────────────────────

def wfo_bars(interval: str, opt_days: int, live_days: int) -> tuple[int, int]:
    """
    Przelicza dni okna WFO na liczbę barów dla danego timeframe.

    Przykład:
      wfo_bars("4h", 90, 14)  →  (540, 84)
      wfo_bars("1h", 90, 14)  →  (2160, 336)
      wfo_bars("15m",90, 14)  →  (8640, 1344)

    Zwraca (opt_bars, live_bars).
    """
    bpd = get_bars_per_day(interval)
    return int(round(opt_days * bpd)), int(round(live_days * bpd))


# ──────────────────────────────────────────────────────────────────────────────
# CLI – uruchom bezpośrednio żeby pobrać dane
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Pobierz dane świec z Binance i zapisz do CSV")
    p.add_argument("--symbol",   default="ETHUSDT",      help="Symbol np. ETHUSDT, BTCUSDT")
    p.add_argument("--interval", default="1h",           help="Timeframe: 1m,5m,15m,30m,1h,4h,1d itd.")
    p.add_argument("--start",    default="2021-01-01",   help="Data startowa YYYY-MM-DD")
    p.add_argument("--market",   default="spot",         choices=["spot","futures"])
    p.add_argument("--out",      default=None,           help="Ścieżka CSV (domyślnie: {symbol}_{interval}.csv)")
    args = p.parse_args()

    out = args.out or f"{args.symbol.lower()}_{args.interval}.csv"

    df = update_csv_cache(
        csv_path   = out,
        symbol     = args.symbol,
        interval   = args.interval,
        start_date = args.start,
        market     = args.market,
        verbose    = True,
    )

    print(f"\nZakres danych: {pd.Timestamp(df['open_time'].iloc[0],  unit='ms', tz='UTC').strftime('%Y-%m-%d')} → "
          f"{pd.Timestamp(df['open_time'].iloc[-1], unit='ms', tz='UTC').strftime('%Y-%m-%d')}")
    print(f"Liczba świec:  {len(df)}")
    print(f"Zapisano:      {out}")

    # przykładowe przeliczenie WFO
    for tf in ["15m", "30m", "1h", "4h", "1d"]:
        ob, lb = wfo_bars(tf, 90, 14)
        print(f"  WFO bary dla {tf:4s}: opt={ob:5d}  live={lb:4d}")

