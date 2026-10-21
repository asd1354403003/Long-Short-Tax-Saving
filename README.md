# Long-Short Tax Saving

Chinese project notes are preserved in [readme2.md](readme2.md).

## Purpose and scope

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

`rebalance_frequency=3` checks March, June and September month-end Closes and executes at the next available trading day's Open. If no regular check can execute in the sample, use the penultimate Close and final Open. Thus a 12-month setting can run on this incomplete-year sample; this fallback deliberately depends on the sample endpoint.

- `selection_mode="pair"`: replace only the losing leg in each currently held pair; if both lose, replace the larger dollar loss. Do nothing if neither loses.
- `selection_mode="portfolio"`: compare aggregate long and short unrealized P&L, then replace the whole losing side with the larger loss. This includes individual winners on that side.

Use the currently unused asset within each original triplet, retaining the replaced leg's long/short direction. Size it against the retained asset with the selected fixed or dynamically refitted Close coefficient. Integer rounding and subsequent common scaling can make the held ratio differ slightly from the model ratio.

`entry_date` is the actual initial/rebalance execution date for all event rows. Pre-trade prices and unrealized P&L use `check_date` Close. `basket` identifies the actual pre-trade ticker's original A/B/C membership; `basket_after` identifies the resulting asset's membership. `shares_after` and `replace_with` describe the post-trade position. Full fills appear in `rebalance_result["trades"]`.

Before executing a Close-selected replacement, confirm the loss at execution-day Open using signed shares times `(Open - cost_price)`. In pair mode, cancel that replacement if its Open P&L is zero or positive; retain the asset and do not select the other leg instead. In whole-portfolio mode, cancel the entire selected side if its aggregate Open P&L is zero or positive, preserving complete baskets. Open confirmation uses no execution-day Close. For example, BRK-B's June 30 Close loss becomes a gain at the July 1 Open with `rebalance_frequency=6`, so BRK-B is no longer replaced with FNF.

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
