# Cointegration

Root `analysis_cointegration.ipynb` displays only this method. Pairs were formed and trained during 2022-2025; 288 frozen pairs were deployed from 2026-01-02 to 2026-10-22.

`data/pool.csv` is the frozen input. `result/cointegration/frozen2026/` contains only the full Python preset and result JSON. The old record has aggregates without daily NAV. Default notebook execution reads it without downloads or retraining. Explicit replay uses the frozen pool and may download missing prices; local histories can be passed to `run.run(..., histories=...)`.

`settings.py` supplies routing and execution defaults. `execution.py` handles beta sizing and integer shares or FX lots. Shared trading, borrow and financing calculations live in root `backtest.py`.

The old zero-RF Sharpe was withdrawn from the main table because daily returns were not saved. It remains unverified until an explicit historical DGS1MO replay. The detailed cointegration narrative is pending; the HTML currently completes AQR market neutral first.
