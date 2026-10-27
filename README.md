# Long / Short Tax Saving

The strategy is named **Market Neutral**. The three AQR reports cited in the HTML are research references; the model implementation, parameter selection and results belong to this project.

- [Research report](report.html): Market Neutral methodology, data, settings, results and the proposed tax-aware extension, with AQR reports cited as references. Use the fixed English/Chinese control to switch languages; the report works offline.
- [Market Neutral analysis](analysis_market_neutral.ipynb): current monthly model, daily RF and annual fee breakdown.
- [Cointegration analysis](analysis_cointegration.ipynb): this method's frozen historical record.
- [Historical universe](aqr_research/universe.html): dated IWB membership proxy.

## Current Market Neutral result

2022-01-03 to 2026-10-22; monthly rebalance; initial NAV $1,000,000.

| Metric | Result |
|---|---:|
| Sharpe, historical daily RF | 1.6364 |
| Cumulative net return | 91.66% |
| Maximum drawdown | -4.79% |
| Beta vs SPY | -0.0320 |
| 2022 / 2023 / 2024 / 2025 | 14.62% / 8.30% / 9.49% / 21.77% |
| 2026 YTD, through October 22 | 15.81% |
| Trading / borrow costs | $23,405 / $32,484 |

This is an in-sample selected model, not independent out-of-sample validation. The latest search evaluated 5,473 computable cases, including sector component controls; the selected account separately passed full-ledger and constraint checks. All years were positive, but returns differ. Earlier seed models also used inspected history.

Cash-flow yield is required in every underlying small factor combination. Sixteen top-level component allocations remain fixed across years; each date's targets are rebuilt and projected to the account constraints. Blend coefficients are not static realized sector or factor exposures.

## Results: two files per candidate

```text
result/
  aqr/monthly_sharpe158/
    parameters.py
    result.json
  cointegration/frozen2026/
    parameters.py
    result.json
```

The Market Neutral Python preset contains all 30 shared model nodes, allocations, factor weights, construction, limits, dates, fees and RF settings. It does not depend on search outputs. The JSON contains daily NAV, annual attribution and verification.

Both analysis notebooks default to `RERUN=True`, so Run All recalculates the current parameters. Market Neutral uses the restored local SQLite database and dated inputs without searching or downloading. Cointegration replays the frozen pool and may request missing historical prices. Use `RERUN=False` only to inspect a saved record. Market Neutral displays average long/short stock counts beside the performance summary, separately for rebalance targets and daily executed holdings.

The self-contained Market Neutral preset can also be run directly: `.venv/Scripts/python.exe result/aqr/monthly_sharpe158/parameters.py`.

## Database after cloning

The repository includes a lossless SQLite snapshot in `aqr_research/data/aqr.sqlite3.gz`, stored with Git LFS, and a checksum manifest. The active uncompressed database, 3,611 raw quarterly statement CSV exports, local backups, recovery evidence and generated search caches stay local. The historical RF file, dated replay inputs and cointegration pool are retained because the saved runs use them in addition to SQLite.

After creating the Python environment and installing `requirements.txt`, run this from the project root on a fresh clone:

```powershell
git lfs pull
.venv/Scripts/python.exe -m methods.aqr_factors.database
```

Restoration streams the database to its relative path, verifies the complete byte count and SHA-256, and refuses to overwrite an existing database. The snapshot passed SQLite `quick_check` and a complete compression/decompression checksum comparison before publication. Local-only source evidence referenced elsewhere in this README is not included in the Git release.

## Settings precedence

`PARAMETER_FILE=None` in the Market Neutral notebook uses `AQRSettings.selected_parameter_file`. An explicit path overrides it; a missing explicit path raises. Preset execution and performance fields override method defaults; omitted fields in those two sections use defaults. The separate quarterly three-factor baseline does not override the selected monthly preset. `RERUN=True` is the default; a preset edit takes effect on the next Run All. Optional `RERUN=False` requires a matching saved-result fingerprint.

[Selected preset](result/aqr/monthly_sharpe158/parameters.py) | [Method defaults](methods/aqr_factors/settings.py) | [Shared routing](settings.py)

### Weighting choices

Set `PARAMETERS["weighting_method"]` in the selected preset (it overrides the method default):

1. **Supplied alpha, demean / L1**: subtract the configured group's mean alpha and divide by the absolute sum across the date. `alpha_column` chooses a dated input column; the bundled `"score"` is the frozen original value/quality/momentum baseline score, not the selected blend. Set `portfolio.group_by=None` for whole-pool demeaning, as in the five-stock example. This branch bypasses the model, tails, volatility sizing, beta controls, smoothing and single-name cap. Dollar neutrality alone does not guarantee beta neutrality.
2. **Configured factor model**: rank/standardize factors and apply the selected model's construction, blending and limits. This remains selected; the saved Sharpe 1.6364 belongs to this branch.
3. **AQR reference method**: reserved, not implemented; selecting it raises before any data loading or trading. The legacy low-level `aqr_style` approximation is not this reference implementation.

Both simple examples in the report use the same final normalization, `(alpha - mean(alpha)) / sum(abs(alpha - mean(alpha)))`. They differ in how alpha is supplied or generated. The direct API is `PortfolioConstructor(ConstructionConfig(method="demean_l1", group_by=None)).build(frame, score_column="alpha")`, with dated `as_of`, `ticker` and `alpha` columns. This reproduces `[-5, -15, -66, -85, -25]` without ranking those supplied alpha values. Changing the weighting choice invalidates the saved-result fingerprint; rerun to evaluate it. No new simple-method performance result is claimed.

## Execution, annual fees and RF

Targets: gross=1, long=+0.5, short=-0.5, single-name weight <=5%, sector net exposure=0, estimated beta=0. Integer shares and price drift can change actual exposures. Completed month-end signals execute at the following session Open. Trading costs are 5 bps of actual traded notional, including trims and sign flips. Short borrow is 1% annualized, ACT/360, including weekends. Dividends, short dividend compensation, cash interest and tax are excluded.

The notebook and HTML split annual trading and borrow costs in USD and as contributions divided by each year's opening NAV. Fees are already deducted from net returns; do not deduct them again.

RF uses cached [FRED DGS1MO](https://fred.stlouisfed.org/series/DGS1MO), daily observations of an annual percent one-month Treasury yield. The valid quote mean for 2022-01-01 to 2026-10-22 is **4.0349% per year**; mean aligned NAV-interval RF return is **0.01609%**. These averages describe the data; the selected Sharpe uses historical daily matching rather than a fixed average yield.

For each NAV interval, use the latest valid quote strictly before its start and accrue `(1 + annual_percent / 100 / 365) ** calendar_days - 1`. The first interval uses a one-day approximation. This is a yield-accrual proxy, not a realized Treasury-index return. Missing quotes or quotes older than seven days raise. Sharpe is `sqrt(252) * mean(daily net return - RF) / sample_std(daily net return - RF)`, ddof=1. RF is a benchmark, not an interest credit to NAV.

## Files and provenance

- Root notebooks and shared market data, risk, reporting and cost modules.
- `methods/aqr_factors/`: Market Neutral settings, scoring, construction, ledger, replay and tests.
- `methods/cointegration/`: pair training and execution; frozen pool in `data/pool.csv`.
- `aqr_research/data/aqr.sqlite3`: sole active AQR database, with raw CSVs and backups nearby.
- `aqr_research/data/backtest_inputs/`: frozen dated inputs; replays recompute ranks, targets and ledger.
- `aqr_research/data/DGS1MO.csv`: raw RF quotes; companion metadata stores provenance and checksum.
- `methods/aqr_factors/research/rf_search/`: latest search audit, separate from presentation results.

Financials are latest vendor versions, potentially restated. Known availability dates take priority; missing dates use the explicit period-end +90-day assumption. IWB membership release times are not independently verified. These are not a fully verified point-in-time archive. The old cointegration record lacks daily NAV, so historical-RF Sharpe is unverified.

Shared cost calculations live in root `backtest.py`; pair-specific sizing stays in the cointegration execution module. HTML is a dated saved-result report, not a live update. Authored project files are English; original documents, raw data and historical evidence remain verbatim.

## Continuing fundamental recovery

Source evidence, identity mappings and checkpoints remain in `methods/aqr_factors/research/results/` and SQLite. Earlier worker instructions were superseded by member-site quarterly CSV downloads. Do not restart recovery workers or recreate per-ticker official parsers during normal analysis. Preserve source documents, units, currency, periods and availability evidence. Do not fill missing historical facts using acquirer financials or current ratios.
