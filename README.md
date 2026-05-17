# bee8 - Mean Reversion (BB / Stoch / MFI / RSI + Supertrend)

`bee8` to bot tradingowy ETH/USDT zbudowany na tej samej architekturze co
`bee4_4` (backtest + WFO + paper/live runner + dashboard Dash), ale z całkiem
nową logiką: klasyczny **mean-reversion long** oparty na konfluencji
wyprzedanych oscylatorów na 1h z filtrem trendu na 4h i silnym triggerem
wyjścia opartym o Supertrend.

Logika została wyprowadzona z analizy 107 zyskownych transakcji historycznych
opisanej w `logika.odt` (rdzeń: BB %B, Stoch K, MFI, RSI; potwierdzenie:
lookback 6 świec; trigger wyjścia: Supertrend(10,3) zmiana DOWN→UP, dający 93%
recall na oryginalnych wyjściach).

## Co robi projekt

- Pracuje wyłącznie po stronie **long** (mean reversion na ETH/USDT spot 1h).
- Wejście: konfluencja co najmniej `N` z 4 oscylatorów wyprzedanych
  (BB %B < 0.20, Stoch K < 25, MFI < 30, RSI < 40), filtr 4h
  (RSI < 45 / BB %B < 0.25 / cena < 4h VWAP) i potwierdzenie odbicia
  (lookback 6 świec: minimum BB %B < 0.15 + aktualne BB %B powyżej minimum).
- Wyjście: konfluencja oscylatorów wykupionych (BB %B > 0.80, Stoch K > 80,
  MFI > 70, RSI > 65, ROC > 3%) lub Supertrend (10,3) flip DOWN→UP, opcjonalnie
  cena powyżej chmury Ichimoku 4h.
- Risk: emergency stop kapitałowy (domyślnie 5%), opcjonalne TP1/TP2.
- Backtest, walk-forward optimization (WFO) i paper/live runner używają tego
  samego silnika sygnałów.
- Dashboard Dash (port 8072) pozwala uruchomić backtest i WFO, podejrzeć
  wykres ze świecami i markerami transakcji, statystyki i listę transakcji.

## Pliki

- [bee8_main.py](bee8_main.py) — entrypoint CLI (backtest / WFO).
- [bee8_dashboard.py](bee8_dashboard.py) — dashboard Dash.
- [bee8_live_runner.py](bee8_live_runner.py) — paper / live runner.
- [bee8_strategy.py](bee8_strategy.py) — pętla backtest.
- [bee8_engine.py](bee8_engine.py) — logika sygnałów (entry/exit).
- [bee8_data.py](bee8_data.py) — ładowanie świec + wskaźniki (BB, Stoch, MFI,
  RSI, ROC, ATR, Supertrend, Ichimoku, VWAP, HTF 4h aligned).
- [bee8_wfo.py](bee8_wfo.py), [bee8_wfo_scoring.py](bee8_wfo_scoring.py) — WFO.
- [bee8_stats.py](bee8_stats.py) — raporty.
- [bee8_binance.py](bee8_binance.py) — pobieranie OHLCV z publicznego API
  Binance.
- [bee8_params.py](bee8_params.py) — defaultowe parametry + siatki WFO.

## Uruchomienie

### Dashboard

```bash
docker compose up -d bee8-bot
# http://127.0.0.1:8072
```

albo lokalnie:

```bash
pip install -r requirements.txt
python bee8_dashboard.py --host 0.0.0.0 --port 8072
```

Porty zajęte na hoście (lokalna konwencja):
- 8067 bee7, 8068 bee4_3, 8069 bee4_4, 8070 chart_inspector, 8071 bee6 →
  bee8 dostaje **8072**.

### Backtest / WFO z linii poleceń

```bash
python bee8_main.py --mode backtest --csv ethusdt_1h.csv --save
python bee8_main.py --mode wfo      --csv ethusdt_1h.csv --save
```

`--fetch` dociągnie świeże świece z Binance przed uruchomieniem.

### Paper runner

```bash
python bee8_live_runner.py --mode paper
```

Stan jest zapisywany w `bee8_live_state.json`, transakcje w
`bee8_live_trades.jsonl`, log w `bee8_live_runner.log`.

## Parametry WFO

Domyślna siatka WFO testuje:

- `bb_long_entry_max` (BB %B próg wejścia, długie): 0.15 / 0.20 / 0.25
- `stoch_long_entry_max`: 20 / 25 / 30
- `mfi_long_entry_max`: 25 / 30 / 35
- `rsi_long_entry_max`: 35 / 40 / 45
- `long_entry_min_confluence`: 2 / 3 / 4
- `lookback_bars`: 4 / 6 / 8
- `bb_extreme_level`: 0.10 / 0.15 / 0.20
- `use_htf_filter`: True / False
- `htf_rsi_max`: 40 / 45 / 50
- `htf_bb_max`: 0.20 / 0.25 / 0.30
- `bb_long_exit_min`: 0.75 / 0.80 / 0.85
- `stoch_long_exit_min`: 75 / 80 / 85
- `mfi_long_exit_min`: 65 / 70 / 75
- `rsi_long_exit_min`: 60 / 65 / 70
- `roc_long_exit_min_pct`: 2 / 3 / 4
- `long_exit_min_confluence`: 2 / 3 / 4
- `supertrend_flip_exit`: True / False
- `ichimoku_above_cloud_exit`: False / True
- `long_emergency_sl_capital_pct`: 0 / 3% / 5% / 8%

Najlepsze stabilne ustawienia są wybierane medianą / modą z top-10% wyników
optymalizacji per okno (`top_decile_median`) i serializowane do
`bee8_wfo_best_params.json`.

## Dane

W repo są przykładowe dane ETH/USDT z Binance spot (`ethusdt_15m.csv`,
`ethusdt_1h.csv`, `ethusdt_4h.csv`) oraz materiały źródłowe analizy:
- `logika.odt` — finalna specyfikacja wskaźników i progów,
- `rozmowa.odt` — pełen log analizy 107 transakcji.

## Origin

Wytyczne projektu są w `logika.odt`. Architektura (dashboard, flow
backtest/WFO/runner, layout plików) skopiowana z `bee4_4` i odchudzona — bee8
nie używa WaveTrend, więc wszystko związane z WT zostało zastąpione
oscylatorami / Supertrendem.
