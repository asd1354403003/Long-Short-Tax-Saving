# Long-Short Tax Saving

V2 studies a large pair pool, z-score entry/exit and realized losses. It reports price P&L before tax; it does not calculate deductible losses, tax payable or after-tax wealth. The old A/B/C version is preserved in GitHub tag [v1.0.0](https://github.com/asd1354403003/Long-Short-Tax-Saving/tree/v1.0.0). Its entry notebook, rotation code and two CSVs have been removed from local V2 after extracting shared utilities. Development stays on local branch `research/pair-pool-v2`; it is not pushed.

## Run

Use the project Python environment and relative paths. Install `requirements.txt`, then restart the notebook kernel after changing Python modules.

1. `pair_pool.ipynb`: discover candidates, train fixed-scale signals, validate in 2025 and save `data/pair_pool.csv`.
2. `analysis.ipynb`: load that frozen pool, download test-period Open/Close in memory and run the 2026 strategy.

Analysis saves no additional CSV or price cache. The pair-pool notebook only overwrites its final CSV.

## Files

| File | Purpose |
| --- | --- |
| `cointegration.py` | Explicit same-date Close OLS, Engle-Granger and ADF |
| `market_data.py` | Shared formation configuration, pair analysis and Yahoo downloads |
| `pair_pool.py` | Industry/exposure discovery and size/liquidity/statistical screening |
| `pair_training.py` | Fixed mean_square scale, single-Z training and independent validation |
| `pair_backtest.py` | Ranked multi-pair execution, integer sizing and cash/P&L accounting |
| `risk.py` | Market beta, rolling beta and dollar-P&L VaR/CVaR |
| `industry_relationships.json` | Curated related-industry graph; not verified company contracts |
| `etf_universe.json` | Commodity and other ETF/ETP catalog and issuer references |
| `data/pair_pool.csv` | Frozen candidates, parameters, provenance and validation results |

Chinese notes: [readme2.md](readme2.md).

## Formation and validation

- Equities target 300 pairs among at most 1,000 US-listed USD equities, with current market cap >= USD 10 billion. ETFs target 50 commodity and 50 other pairs, with current AUM >= USD 100 million. No equity market-cap substitution for missing ETF AUM.
- The catalog covers 45 commodity products plus broad/style/sector/international equity, REIT and bond ETFs. Physical bullion, commodity futures and mining/energy-company equities are distinguished. It excludes leveraged/inverse products, ETNs and closed-end funds. Same or explicitly related industry/exposure groups can pair; no stock/ETF cross-pairs. Each ETF exposure-group combination has a five-pair cap. `tracking_pair` flags configured same-index/metal candidates, without establishing a trading edge or tax eligibility.
- Formation uses **2022-2024 only**, at least 750 observations, Close correlation > 0.90, EG 95%, residual ADF -2.86 and fixed lag 1. ETF levels additionally must support I(1); the existing equity I(1) assumption remains. Final-60-observation mean Close*Volume must be >= USD 20 million, formation-end Close >= USD 5, and the illustrative order <= 0.5% of historical ADV in either direction. Targets are not filled by weakening filters.
- Market cap, AUM, classifications and current catalog membership are snapshots, not historical constituents. They introduce survivorship/selection bias even though price-based formation stops in 2024.

## Original fixed-scale single-Z signal

There is **no rolling z-score window**. Fit `Close_y = alpha + beta*Close_x`, then the residual AR(1), `e_t=C+B*e_(t-1)+innovation`, with `0<B<1`. Preserve the original mean_square formula:

```text
tau = 1/252
theta = -log(B)/tau
mu_e = C/(1-B)
sigma_eq = sqrt(mean(e**2)*tau/(1-exp(-2*theta*tau)))
z = (e-mu_e)/sigma_eq
```

The numerator deliberately matches the original spread second moment rather than innovation variance. Test entry Z=0.25 to 3.50 in steps of 0.25. Default exit is Z/2. Choose training cumulative budget return first and Sharpe second, requiring five completed trades; training performance is in-sample. Freeze training coefficients/scale/Z and evaluate once in **2025**. `validation_passed` requires positive training P&L/Sharpe, at least two validation trades and positive validation P&L/Sharpe. Rejected rows retain their results/reasons. Final CSV order is descending `validation_sharpe`, with missing values last.

Separate `refit_*` regression/scale estimates use **2022-2025** and are computed after validation. They do not choose Z or change validation results. The 2026 strategy uses these pre-2026 estimates, fixed throughout the test, with selected `entry_z`, `exit_z` and `stop_loss`. **2026 is never used for fitting or parameter selection.**

## 2026 ranked strategy and capital

Default `max_pairs=15`, `long_amount=1_000_000`. Each slot has a nominal long budget of **USD 1,000,000/15 = USD 66,666.67** and the same short-leg cap. With frozen positive beta, one common `strict_one_leg` scale limits both legs at entry before rounding down to whole shares. Integer rounding and the common hedge scale can leave either leg below its cap. Shares stay fixed until exit; limits are entry limits, not daily trims, so exposures can drift as prices move. The fixed reference account budget is **USD 2 million**, including unused cash; interest is zero.

Only validated pairs with `refit_status="ok"` are eligible. Scan descending 2025 Sharpe at each Close. High z shorts Y/longs X; low z does the reverse. Close signals fill at the next common trading Open, so the final 2025 Close can generate the first 2026 entry. January 1 is a requested start boundary, not a forced fill date. `end_date=None` includes today's data only after 17:00 New York time, allowing a publication buffer; otherwise it ends before today.

Exit both legs when z returns to or crosses its signed exit threshold, including overshoots, or when Close-marked trade loss reaches 12% of entry gross exposure. Execute at next Open; gaps can exceed the stop. Process exits before ranked new entries, maintaining at most 15 active pairs. Every Close scan restarts at the highest Sharpe. Currently held pairs are skipped. **Closed pairs can be traded again** when their signals qualify. Preserve the saved `prevent_immediate_reentry` rule: when true, suppress that pair's new entry signal on its execution-exit day, then resume normal ranking the following session. This prevents immediate stop/re-entry churn without imposing a lifetime ban. Tickers may appear in later unused pairs, but by default cannot overlap between simultaneous pairs (`allow_shared_tickers=False`). This rule is configurable.

Missing entry Open prices skip that candidate. A missing exit Open delays its pending exit until valid prices are available. Missing Close marks use the last already-known mark, or entry Open for a new holding, and increment `stale_price_legs`; they are never backfilled from future prices. Data gaps do not remove pairs by inspecting their entire future test history.

Unfilled slots stay in cash when no eligible signal survives position/ticker/cooldown checks. `active_pairs` counts current positions, `entry_count` counts all entries, and `traded_pairs` counts distinct pairs ever traded. Past use does not deplete eligibility. The backtest does not force 15 holdings or liquidate everything at the last available date. End holdings retain unrealized P&L. This execution rule differs from training's scheduled period-end liquidation, which exists to compare complete training/validation P&L and is reported by `forced_exits`.

## Outputs and accounting

`PairPoolBacktester.run()` returns daily long/short dollar-position matrices, pair-level daily holdings, events, closed pairs, end holdings, z-scores and `daily_summary`. Positive shares/amounts are long; negatives are short. Events show the triggering `signal_date` and actual `execution_date`. The trade table records both legs' realized P&L. Each round trip has a unique `trade_id`, so repeated entries in the same pair remain separate. `pair_performance` reconciles all realized round trips and remaining unrealized P&L by pair; current holdings include calendar `holding_days`.

- Daily cash plus signed market values determines NAV. Trades exchange cash and positions; trading notional is not profit. Old positions' overnight gaps and new positions' Open-to-Close moves are included.
- `daily_return = daily_pnl / (2*long_amount)` and `cumulative_return = cumulative_pnl / (2*long_amount)`. These fixed-budget returns include idle cash; cumulative return is not the product of the daily budget returns.
- `cumulative_pnl = cumulative_realized_pnl + unrealized_pnl` at every date. Realized losses sum negative **leg** P&L, even in a profitable pair; realized gains sum positive leg P&L. Neither amount is automatically tax deductible/taxable as recorded.
- SPY uses its first test-session Open as the benchmark base and later Close valuations. Its return excludes distributions. Market beta compares the same daily strategy/SPY series after excluding the first Open-to-Close day. Market rolling beta remains a separate 60-observation statistic.
- Historical/normal VaR and CVaR use daily dollar P&L directly.

All prices use the vendor Open/Close with `auto_adjust=False`, not Adj Close; vendor history may reflect splits. Ignore dividend/distribution receipts, short compensation payments, financing, margin, borrow fees and transaction costs. These are price returns, not distribution-inclusive returns. Nonzero fee settings are rejected. Trading P&L is not a historical tax-lot ledger, and commodity ETP structures need separate tax modelling.

## Tax research scope

The long-term objective is to compare investment returns and realized losses under tax constraints. Cointegration or changing tickers does not establish wash-sale or straddle eligibility. V2 currently evaluates the signal strategy and economic leg-level gains/losses; a tax-aware strategy and final tax liability remain future work.

## Wash-sale rules

Section 1091(a) refers to **"substantially identical stock or securities"** and states **"no deduction shall be allowed under section 165"** for the specified loss transactions. The window runs from 30 calendar days before through 30 days after the loss disposition, including the disposition date. Section 1091(e) provides related rules for losses on closing short sales. See [26 U.S.C. section 1091](https://uscode.house.gov/us/usc/119-118/t26/s1091/).

For ordinary taxable-account replacement purchases, the IRS instructs taxpayers to **"add the disallowed loss to the cost of the new stock or securities"**; the holding period generally carries over. IRA/Roth IRA replacement acquisitions have different consequences. See [IRS Publication 550, Wash Sales](https://www.irs.gov/publications/p550).

Renaming a basket does not avoid a security-level restriction: selling X at a loss in B and immediately buying X in C still requires review. Buying shares to cover an existing short is a closing transaction; separately evaluate whether the surrounding transactions trigger the short-sale rules.

## Straddle rules and continued deferral

Section 1092(c)(1) states: **"The term 'straddle' means offsetting positions with respect to personal property."** The definition concerns reduced loss risk, not merely a strategy name. Stock eligibility also depends on section 1092(d)(3). See [26 U.S.C. section 1092](https://uscode.house.gov/us/usc/119-118/t26/s1092/).

Under the general rule, a loss is recognized only to the extent it exceeds relevant offsetting positions' unrecognized gains. Year-end valuations matter. Section 1092(a)(1)(B) carries a restricted loss into the succeeding year **subject to the same restriction**; a new year does not automatically release it.

A simplified example, assuming the general rule and no other adjustments:

| Year | State of the related positions | Illustrative treatment |
| --- | --- | --- |
| 1 | Realize a $10,000 loss; year-end unrecognized gain is $10,000 | Defer the $10,000 loss |
| 2 | Keep the gain position; cumulative unrecognized gain reaches $12,000 | The $10,000 can remain deferred |
| 3 | Realize the $12,000 gain; dispose of all related positions, with no relevant successor positions | If the $10,000 loss is fully released, the net gain is $2,000 |

Replacement and successor-position coordination also matters. See [IRS Publication 550, Straddles and Coordination of Loss Deferral Rules and Wash Sale Rules](https://www.irs.gov/publications/p550). Identified straddles and other special cases require separate treatment.
