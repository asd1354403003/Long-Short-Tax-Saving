# Long-Short Tax Saving

## Local V2: equity, commodity and diversified ETF pair pools

Development remains on local branch `research/pair-pool-v2`. GitHub tag `v1.0.0` preserves the A/B/C baseline at `9b60b46`; local tag `abc-baseline-2026-10-22` points to the same code. The new research branch is not pushed. The original A/B/C code, notebooks and their two CSVs remain available unchanged.

Open `pair_pool.ipynb`. Its screening classes in `pair_pool.py` reuse the existing Yahoo downloader and Close-based EG/ADF implementation. `PairSignalTrainer` and `SignalTrainingConfig` in `pair_training.py` handle parameter training and validation. The workflow writes only `data/pair_pool.csv`; it saves no intermediate CSV or price cache. A candidate requires two assets, without a third C substitute.

### Equity and ETF coverage

- **Equities:** target up to 300 pairs among at most 1,000 US-listed USD equities. Both assets require current market cap >= USD 10 billion. Same-industry and configured related-industry pairs are tested. Known configured share classes, including PBR/PBR-A and LEN/LEN-B, are reduced to the more liquid formation-period class. The exclusion list is not a complete issuer-identity or tax-eligibility check.
- **Commodity ETF/ETPs:** target up to 50 additional pairs. `etf_universe.json` includes 45 products spanning physical gold/silver/platinum/palladium, precious-metal baskets, crude oil/Brent, natural gas, gasoline, copper, industrial/battery metals, corn/wheat/soybeans/sugar, agricultural baskets and broad commodity strategies. Physical bullion and commodity futures are explicitly distinguished.
- **Other ETFs:** target up to 50 additional pairs. The catalog also covers US/international indices, styles and sectors, mining/resource-company equities, real estate, Treasuries, inflation-linked, investment-grade, high-yield, municipal and international bonds. Mining and energy-company ETFs are equity exposures, not direct commodity exposure.

The curated catalog contains 189 US-listed USD products in 58 exposure groups. It excludes leveraged/inverse products, ETNs and closed-end funds. Same or explicitly related ETF groups can pair; stock/ETF and commodity/bond cross-pairs are not formed. Each ETF exposure-group combination is capped at five accepted pairs so near-identical index products do not consume all additions. `tracking_pair` flags catalogued same-index/physical-metal candidates; it does not establish tax eligibility or an economic trading edge.

### Instrument-specific size and common liquidity filters

ETFs use **AUM >= USD 100 million**, rather than the equity market-cap filter. Missing AUM does not get replaced by market cap. Both stocks and ETFs require a formation-end Close >= USD 5, one whole share within the reference budget, and **USD 20 million mean daily Close * Volume over the final 60 formation observations**.

For a USD 100,000 reference long slot, each implied hedge order must be <= **0.5% of historical average daily dollar volume** in both directions at formation-end prices. `reference_participation_y/x` report the worse direction. These are screening estimates, not guaranteed fills or slippage estimates; execution Open prices, contemporary liquidity, spreads, costs and borrow availability need separate modelling.

### Statistics, provenance and coverage diagnostics

Correlation must exceed 0.90, EG uses 95% critical values, and residual ADF uses fixed -2.86. Both tests use lag 1 and autolag=None. ETF price levels must additionally support I(1): ADF does not reject at the level and rejects after first differencing. This avoids treating stationary cash-like ETF Close prices as price-level cointegration. The existing equity I(1) policy is unchanged. No threshold is relaxed to meet a family target.

Candidate screening, correlation, regressions, liquidity, I(1) checks and spread mean/std use **2022-2024 only**. Signal parameters are trained in that same period and validated in **2025**. This stage downloads no 2026 prices. Market caps, fund AUM, classifications and catalog availability are current snapshots; their timestamps record metadata retrieval, not reconstructed 2025 constituents. Survivorship/selection bias remains. SPYM is the current ticker for former SPLG, effective 2025-10-31.

The final pair CSV includes instrument type, family, exposure, implementation, issuer-reference URLs, size/AUM provenance, liquidity and OLS/EG/ADF diagnostics. The notebook separately displays every catalog group, including zero-acceptance groups, at metadata, size and formation-eligibility stages. Catalog coverage is not the same as having tradable accepted pairs in every category.

Close-only modelling and omitted dividend cash flows are retained. ETF distributions and futures rolls can affect spreads. Physical trusts, futures pools and investment-company ETFs can have different tax treatment; the pipeline does not calculate tax or certify wash-sale/straddle treatment. Individual pair signal validation is implemented below; combined active-position/ticker/exposure limits remain future work.

### Original single-Z method: training and independent validation

`PairSignalTrainer` and `SignalTrainingConfig` live in `pair_training.py`. `run()` rebuilds candidates through 2024 before training. `train()` rejects input pairs formed after that cutoff. **No rolling window is fitted or used.**

Match the original `TS_implement.ipynb` and `TS_lib.PairTraderSingleZ(sigma_mode="mean_square")` scaling. Fit the training residual AR(1), `e_t = C + B*e_(t-1) + innovation`, requiring `0 < B < 1`. With `tau=1/252`, set `theta=-log(B)/tau`, `mu_e=C/(1-B)`, and `sigma_eq=sqrt(mean(e**2)*tau/(1-exp(-2*theta*tau)))`. The numerator intentionally preserves the legacy spread second moment rather than replacing it with innovation variance. The fixed standardized signal is `(e-mu_e)/sigma_eq`.

Search 14 entry thresholds, Z=0.25 to 3.50 in steps of 0.25. The default `exit_on_cross_mean=False` couples `exit_z=entry_z/2`; it is not an independent exit search. Rank eligible choices by **training cumulative budget return first, Sharpe second**, as in the original auto-report, with at least five completed trades. Training performance is in-sample. Neither validation data nor the later model refit choose the threshold.

Evaluate the frozen training alpha/beta, mu_e, sigma_eq and selected Z once during 2025, starting flat. The last training Close may provide the first validation signal. High z shorts Y and longs X; low z reverses the direction. Close signals trade at the next common trading day's Open. Exit on returning to or crossing the signed exit threshold, including jumps across zero. The default 12% stop is monitored at Close on trade P&L divided by its entry gross exposure and executed at next Open; gaps can exceed the threshold. A trade exit suppresses a new entry signal on that execution day. The legacy signal rules are retained, while accounting uses integer shares and actual Open/Close fills rather than copying the old unit-return ledger.

Each pair is independently simulated with a fixed USD 200,000 budget (`2 * long_amount`). A common entry scale caps both legs at USD 100,000 before whole-share rounding; shares stay fixed until exit. Cash plus signed positions determines NAV/P&L, including overnight moves. Daily returns are P&L divided by the fixed budget. Cumulative return is total P&L/budget, not the product of those daily budget returns. Sharpe includes idle days, annualizes with 252 observations and assumes zero risk-free rate; drawdown includes unused cash. Close remaining trades at the final Close of each evaluation period. Trade counts include scheduled exits, separately identified by `forced_exits`; `stop_exits` reports stop-driven closures. Costs, borrow, financing and dividend cash flows remain excluded. Overlapping independent pair results are not one combined account.

`validation_passed` requires positive training P&L/Sharpe, at least two validation round trips and positive validation P&L/Sharpe. Rejected rows and reasons remain in the same CSV. This is a selection rule, not a profitability guarantee. The CSV saves `entry_z`, `exit_z`, training OU estimates, training/validation P&L, budget return, Sharpe, drawdown, trade count, win rate, exit counts, observations and status. There is no `window` column in this version. The final CSV and notebook preview sort by 2025 `validation_sharpe` descending, with missing values last; this ordering does not change training parameter selection.

The original pool `alpha`/`beta` remain pre-validation formation estimates; `train_alpha`/`train_beta` explicitly identify the simulation model. Separate `refit_*` regression and scale estimates use 2022-2025 for possible future 2026 deployment; check `refit_status` before using them. They are calculated after validation and do not alter validation results or the selected Z. No 2026 observations enter fitting or selection. Only `data/pair_pool.csv` is written.

Issuer references: [USCF commodity funds](https://www.uscfinvestments.com/), [Invesco commodity products](https://www.invesco.com/us/en/solutions/invesco-etfs/commodity-investing.html), [Teucrium agriculture](https://teucrium.com/TAGS), [Aberdeen commodities](https://www.aberdeeninvestments.com/en-us/institutional/investment-solutions/commodities). The catalog includes further issuer references for each group.

Chinese project notes are preserved in [readme2.md](readme2.md).

## V1 A/B/C baseline: purpose and scope

Study whether realizing losses and rotating cointegrated long/short positions can improve after-tax outcomes while maintaining related investment exposure. First construct three candidate groups, A, B and C, using 2022-2025 formation data. Start long A / short B in 2026 and use the currently unused asset in each triplet as a replacement.

The current implementation screens assets, constructs integer-share positions, rotates losing legs or baskets, and measures price P&L, realized losses, exposure and market beta. It does **not** calculate tax payable, legally deductible losses or after-tax wealth. A reduction in current tax may represent deferral rather than permanent savings; later work must include remaining tax liabilities or terminal liquidation.

## Research assumptions

### Tax assumption

For the initial concept study, assume the simulated A/B/C trades do not trigger wash-sale loss restrictions or straddle loss deferral, and that sufficient eligible gains are available to absorb the losses. This is an external research assumption, **not a conclusion of cointegration**. The code records economic closing losses, not approved tax deductions.

### Prices, distributions and costs

- Correlation, cointegration, ADF and formation OLS use **Close** from 2022-2025. No adjusted-price series is used.
- Initial entry and rebalance fills use **Open**; daily valuation uses **Close**, retaining the vendor fields downloaded with `auto_adjust=False`. Vendor historical quotes may already reflect stock splits; these files are not a historical tax-lot ledger.
- Ignore long dividend/distribution receipts, short dividend/distribution compensation payments, reinvestment and associated taxes. SPY uses the same dividend exclusion. This does not assume the two legs' distributions cancel.
- Ex-dividend price movements remain in Close prices. All reported portfolio and benchmark returns are **price returns, not total returns including distributions**. No dividend cash flows are added by the model.
- Borrow fees and transaction costs are fixed at zero. Nonzero values are not implemented and are rejected by the notebook configuration. Financing, margin constraints, interest and dividend cash flows are excluded.
- Shares are integers; scale share magnitudes down and round toward zero. Shorts have negative shares and market values.
- This is a calendar-entry holding/rotation study. Entry does not require a spread z-score signal. Cointegration alone does not select a profitable trading direction.
- The `beta` option freezes formation slopes; `quote_beta` refits them through the preceding trading Close. `check_i1=False` treats the required integration order as a research assumption rather than verifying it for every asset.

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

## Why three candidate groups?

Three is a design choice, not a tax-law requirement. Each original triplet contains three distinct assets, with one held long and one short. The unused asset provides an immediate replacement while preserving two directions of exposure.

For example, long A / short B can become long A / short C. If A subsequently has the larger loss, it can become long B / short C. Original A/B/C membership does not fix whether an asset must always be long or short. Three groups do not guarantee wash-sale or straddle eligibility; more substitutes, waiting or reducing exposure may be needed in a tax-constrained model.

## Asset discovery and formation

1. Read the seed's current Yahoo industry label and query peers with `EquityQuery` and `yf.screen`.
2. Expand the pool using `industry_relationships.json`. The configured graph covers related industries, including some upstream/downstream categories. These are industry-level candidate relationships, not verified company-specific supplier/customer contracts. Sector fallback is disabled by default.
3. Download candidates in batches. Use only 2022-2025 Close, with at least 750 common observations.
4. Keep training correlation **strictly above 0.90**, rank descending, then test in that order until two substitutes pass. Otherwise skip the seed.
5. Estimate same-date OLS with an intercept: `y_t = alpha + beta * x_t + residual_t`. Do not shift x in this regression. Engle-Granger uses fixed lag 1 and `autolag=None`; the lag applies to the residual unit-root test.
6. AB and AC use the EG 5% critical value. BC is a post-selection diagnostic using the 10% critical value, retaining a marginal 10% result. BC failure does not automatically restart the candidate search. The ordinary residual ADF diagnostic uses the requested fixed threshold **-2.86**; EG uses its own cointegration critical values.
7. Exclude reused tickers and configured equivalent share classes. Retrieve 2026 data for selected assets only after formation. SPY is a benchmark, not a selected portfolio member.

`SEARCH_AGAIN=False` uses the existing ten A tickers. The optional search scans current large-cap seeds in descending market-cap order and keeps search results in memory. Only the final build writes the two requested CSVs.

The universe uses current classifications and market-cap snapshots, so it is not a fully point-in-time 2025 universe. Historical data may be revised. The industry universe is not restricted to operating-company common stocks and can include ADRs and funds.

## Saved files and modules

Only these two data files are written:

| File | Contents |
| --- | --- |
| `data/portfolios_open_close.csv` | Date index and two column levels: ticker, then Open/Close; A/B/C constituents and SPY |
| `data/pair_betas.csv` | 30 rows: AB, AC and BC for each of ten triplets, with coefficients and diagnostics |

The price file currently contains 1,204 dates from 2022-01-03 through 2026-10-20 and 62 price columns. Analysis keeps the 201 available 2026 dates. Assets with shorter formation histories, such as NYAX, must still meet the minimum observation requirement.

| File | Responsibility |
| --- | --- |
| `cointegration.py` | `Cointegration`: same-date OLS, fixed-lag Engle-Granger and ADF |
| `portfolio.py` | `YahooFinanceSource`, configuration classes and `PortfolioBuilder` |
| `positions.py` | `construct_portfolios`, `PortfolioRebalancer`, daily positions and P&L |
| `risk.py` | `BetaAnalysis` and P&L-based `RiskAnalysis` |
| `data.ipynb` | Configure discovery/formation and export the two CSVs |
| `analysis.ipynb` | Read the files, show A/B/C membership, construct/rotate holdings and analyze results |

Diagnostics, event history and trade ledgers stay in memory. There are no extra search/cache/output data exports. All code paths are relative to the project directory.

```python
prices = pd.read_csv(
    "data/portfolios_open_close.csv", header=[0, 1], index_col=0, parse_dates=True,
)
prices["NVDA"]  # Open and Close for one constituent
```

## Selecting the hedge coefficient

All equations use same-date Close with an intercept: AB is A on B, AC is A on C, and BC is B on C. `inherited_beta` for BC is a separate diagnostic; it is not the directly fitted BC coefficient.

```python
beta_column = "quote_beta"  # Or "beta"
```

- `beta`: use the fixed 2022-2025 Close OLS slope in the CSV.
- `quote_beta`: refit Close OLS over an expanding window from the pair's `formation_start` through the last trading Close before execution. January 2 entry uses observations through December 31; an April 1 rebalance uses observations through March 31. No execution-day or later Close enters the regression. Cointegration screening is not repeated during rebalancing.

Here `quote_beta` is explicitly a dynamic Close regression coefficient. It is no longer an adjusted-price conversion. The CSV's `quote_beta`/`quote_date` are the formation-end snapshot; dynamic estimates live in the analysis event table. Pass the full historical Close DataFrame as `close_history` to both construction and rebalancing. `beta_start`, `beta_date` and `beta_nobs` show the actual fit window.

Use the same `beta_column` for initial construction and rebalancing. Initial construction selects AB; replacements use the relevant current pair and the reciprocal for a reversed direction. `hedge_beta` shows the current estimate, while `held_ratio` records actual shares. A retained pair is not automatically resized merely because its dynamic coefficient changes.

## Integer-share sizing and exposure modes

`long_amount = 1_000_000` is the target long-side budget, equally allocated among initial A assets. Initial long shares are `floor(per_asset_budget / Open_A)`; initial short shares are `-floor(selected_beta_AB * long_shares)`.

| `mode` | Constraint at sizing time |
| --- | --- |
| `beta` | Use the coefficient-based quantities without additional scaling |
| `strict` | After replacements, cap long plus absolute short exposure at `2 * long_amount`; also enforce this cap at initial entry |
| `strict_one_leg` | At entry and every scheduled execution Open, cap the total long side and absolute total short side separately at `long_amount` |

Both constrained modes scale **all positions by the same factor**, then round share magnitudes down. For `strict_one_leg`, the scale is `min(1, long_amount / max(long_exposure, abs(short_exposure)))`. Price moves can exceed the limits between events. `mode="beta"` is an exposure-sizing rule; it is independent of `beta_column="beta"`.

## Rebalancing and realized gains

`rebalance_frequency=3` uses March, June and September month-end Close cutoffs for hedge estimation and selects/trades at the next available trading day's Open. If no regular check can execute in the sample, use the penultimate Close and final Open. Thus a 12-month setting can run on this incomplete-year sample; this fallback deliberately depends on the sample endpoint.

- `selection_mode="pair"`: replace only the losing leg in each currently held pair; if both lose, replace the larger dollar loss. Do nothing if neither loses.
- `selection_mode="portfolio"`: compare aggregate long and short unrealized P&L, then replace the whole losing side with the larger loss. This includes individual winners on that side.

Use the currently unused asset within each original triplet, retaining the replaced leg's long/short direction. Size it against the retained asset with the selected fixed or dynamically refitted Close coefficient. Integer rounding and subsequent common scaling can make the held ratio differ slightly from the model ratio.

`entry_date` is the actual initial/rebalance execution date for all event rows. Pre-trade prices and unrealized P&L use `check_date` Close. `basket` identifies the actual pre-trade ticker's original A/B/C membership; `basket_after` identifies the resulting asset's membership. `shares_after` and `replace_with` describe the post-trade position. Full fills appear in `rebalance_result["trades"]`.

At every scheduled execution Open, recalculate signed P&L as `shares * (Open - cost_price)` and select again from both current legs/sides. In pair mode, replace the asset with the larger dollar loss in each pair; if neither asset loses, keep both. In whole-portfolio mode, compare aggregate Open P&L of the entire long and short sides and replace the side with the larger loss; if neither side loses, keep both. The choice can switch from the prior Close's worse leg/side to the other one. A tie between losses selects long. Selection uses no execution-day Close. Dynamic hedge estimation still stops at the preceding Close.

The history retains check-date Close prices/P&L and adds `open_price` and `open_unrealized_pnl` to show the actual selection basis. Schedule `long_pnl` and `short_pnl` also use execution Open. For example, with `rebalance_frequency=6`, BRK-B is profitable at the July 1 Open, so it is not replaced with FNF; if its paired short has a loss then, that short can be replaced instead.

Exposure limits still apply independently. Scaling can trim profitable retained positions; these trades have `reason="strict_cap"`, and their gains remain in realized P&L. Whole-portfolio replacement can still close individual winners when the selected basket has an aggregate loss at Open.

## Daily P&L and the realized/unrealized identity

Long shares and market values are positive; shorts are negative. Trade quantities are positive for buys/covers and negative for sells/short sales. Gross exposure is long market value plus absolute short market value.

On a trade date, remove net trading cash flows from each side's change in signed market value:

```text
daily_pnl = market_value_today - market_value_previous - net_buying_at_Open
```

This includes old holdings' overnight P&L through Open and new holdings' Open-to-Close P&L. Initial-day P&L uses entry Open as the starting value. Returns divide P&L by actual initial gross exposure; cumulative return is cumulative P&L divided by that fixed denominator, not an account-equity return or a compounded daily-return series.

Classify each individual closing trade before aggregation: losses are negative and gains positive. The daily summary reports their separate cumulative totals and net realized P&L:

```text
cumulative_realized_pnl = cumulative_realized_loss + cumulative_realized_gain
cumulative_pnl = cumulative_realized_pnl + unrealized_pnl
```

The notebook displays the final realized/unrealized breakdown. Total cumulative P&L includes both components; it is not the gross harvested loss or a tax-saving amount. Totals are calculated before display rounding.

Daily position matrices retain all 30 tickers in triplet order (`NVDA, ANET, NVMI, ...`). Unheld assets are zero. Displayed long/short tails hide columns that are zero throughout the selected window. The event pair table shows actual tickers, signed shares, selected coefficient and held ratio.

## Market beta, VaR and CVaR

`BetaAnalysis(x, y)` regresses aligned daily returns as `y = alpha + beta*x + error`. The shared notebook setting `beta_y` selects the same return column for baseline and rebalanced analyses, defaulting to combined long/short `daily_return`. Both use identical SPY dates, initial-day exclusion and rolling windows. The final comparison table reports full-sample and latest rolling estimates side by side; it does not compare long-only returns with combined long/short returns. With SPY as x, this estimates market exposure; it is different from a formation hedge coefficient. Rolling beta uses trailing covariance(y, x) divided by variance(x), with 60 trading days in the notebook. The initial Open-to-Close day is excluded when compared with Close-to-Close market returns.

`RiskAnalysis(pnl, confidence=0.95)` uses monetary P&L directly. Historical VaR is the empirical loss quantile; CVaR averages the worst probability mass, including partial boundary weights. The parametric estimates assume normal P&L and use sample mean and sample standard deviation. No exposure multiplier or multi-day scaling is applied.

The current VaR cell uses **the fixed-holdings baseline's 200 Close-to-Close observations**, not the rebalanced portfolio. Rebalanced holdings and market-beta analyses appear separately later in the notebook.

## Current original A/B/C groups

| A | B | C |
| --- | --- | --- |
| NVDA | ANET | NVMI |
| AAPL | DAKT | BELFA |
| MSFT | UBER | CDNS |
| TSM | AVGO | FLEX |
| BRK-B | AIG | FNF |
| JPM | BLX | MFG |
| V | MUFG | SEIC |
| MA | HLI | BAM |
| PLTR | NYAX | NET |
| ORCL | INOD | GWRE |

These Close-only triplets replace the earlier adjusted-price selection. ABBV did not find two qualifying Close candidates and was replaced by ORCL. All thirty AB/AC/BC rows pass their configured EG plus residual-ADF checks. Original membership is not a claim that these remain the long/short holdings after rotation.

## Running the notebooks

Run from this project directory, using the local virtual environment and the `Long Short Tax Saving (.venv)` notebook kernel. Run `data.ipynb` when rebuilding the saved universe/data is intended; run `analysis.ipynb` to analyze the existing two CSVs.

```powershell
uv venv --python 3.10 .venv
uv pip install --python .venv\Scripts\python.exe -r requirements.txt
.venv\Scripts\python.exe -m ipykernel install --user --name long-short-tax-saving --display-name "Long Short Tax Saving (.venv)"
```

Source code, comments and notebook explanations are in English. The Chinese documentation is retained separately in `readme2.md`.
