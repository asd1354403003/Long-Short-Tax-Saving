# Market Neutral

Presentation: root `analysis_market_neutral.ipynb` and `report.html`. Replay with `python result/aqr/monthly_sharpe158/parameters.py` from the project root.

`run.py` loads a preset and local frozen dated inputs. `portfolio.py` recomputes ranks, constructs and blends targets. `backtest.py` executes integer shares and the cash/cost ledger. `performance.py` reconciles annual long/short attribution.

`load_parameters(None)` uses `AQRSettings.selected_parameter_file`; an explicit path takes precedence. Preset execution and performance fields override defaults; omitted fields in those two sections use defaults. The separate original three-factor quarterly baseline is not the selected monthly model. A parameter fingerprint prevents showing an old result after an edit.

The sole active source database is `aqr_research/data/aqr.sqlite3`. Frozen inputs live in `aqr_research/data/backtest_inputs/`. Changing source, universe, missing-value or base ranking semantics, or expanding dates, requires a new input snapshot. Incompatible settings raise instead of silently reusing different inputs.

Sharpe uses daily historical DGS1MO. Its annual-quote mean is descriptive, not a fixed RF applied to every period. `performance.risk_free` records the source and accrual convention. No cash interest is credited to NAV. Annual trading and borrow fees appear separately in USD and relative to the year's opening NAV.

## Continuing fundamental recovery

Original sources, identity mappings and checkpoints are preserved in `research/results/` and SQLite. Worker recovery instructions were superseded by member-site quarterly CSV downloads. Do not restart workers or create per-ticker official parsers during normal replay. Preserve source documents and historical gaps.

Known availability dates take priority; otherwise period-end +90 days is an explicit assumption. Retain currency, units and provenance; never replace missing historical facts with acquirer or current financials. Latest vendor versions are not an as-filed archive.
