"""Monthly factors from cached Yahoo inputs and dated universe classifications."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from pathlib import Path
import json

import numpy as np
import pandas as pd
import yfinance as yf

from .operators import FACTORS, OPERATORS, combine, transform
from .factors import FACTOR_DEFINITIONS, balance_ratios
from .universe import Universe
from .database import AQRDatabase
from .data import CachedYahooData
from .fundamentals import FundamentalStore


@dataclass(frozen=True)
class SignalConfig:
    universe: str = "russell1000_iwb_proxy"
    tickers: tuple[str, ...] = ()
    membership_file: str | None = None
    group_by: str | None = "sector"
    factor_operators: dict[str, tuple[str, ...]] = field(default_factory=lambda: {
        name: ("group_rank", "group_zscore") for name in FACTORS
    })
    combine_operator: str = "vec_avg"
    database_path: str = "aqr_research/data/aqr.sqlite3"
    start: str = "2022-01-01"
    end_exclusive: str = "2026-10-24"
    min_stocks: int = 10
    financial_source: str = "website_quarterly"
    financial_availability: str = "existing_dates_only"
    financial_reporting_lag_days: int = 90
    missing_value_method: str = "drop"
    missing_value_scope: str = "score_group"
    require_positive_book_equity: bool = False

    def __post_init__(self):
        if self.group_by not in {None, "industry", "sector"}:
            raise ValueError("group_by must be industry, sector or None")
        if not self.factor_operators or set(self.factor_operators) - set(FACTOR_DEFINITIONS):
            raise ValueError("Select one or more supported factors: " + ", ".join(FACTOR_DEFINITIONS))
        pipelines = {}
        for name, operators in self.factor_operators.items():
            pipeline = (operators,) if isinstance(operators, str) else tuple(operators)
            if not pipeline or any(operator not in OPERATORS for operator in pipeline):
                raise ValueError("Invalid operator pipeline for " + name)
            pipelines[name] = pipeline
        object.__setattr__(self, "factor_operators", pipelines)
        if self.combine_operator not in {"vec_avg", "vec_sum"}:
            raise ValueError("combine_operator must be vec_avg or vec_sum")
        if not isinstance(self.min_stocks, int) or self.min_stocks < 2:
            raise ValueError("min_stocks must be an integer of at least 2")
        if pd.Timestamp(self.start) >= pd.Timestamp(self.end_exclusive):
            raise ValueError("start must be earlier than end_exclusive")
        if self.financial_source not in {"website_quarterly", "legacy_annual"}:
            raise ValueError("financial_source must be website_quarterly or legacy_annual")
        if self.financial_availability not in {"existing_dates_only", "period_lag"}:
            raise ValueError("financial_availability must be existing_dates_only or period_lag")
        if not isinstance(self.financial_reporting_lag_days, int) or self.financial_reporting_lag_days < 1:
            raise ValueError("financial_reporting_lag_days must be a positive integer")
        if self.missing_value_method not in {"drop", "mean", "median"}:
            raise ValueError("missing_value_method must be drop, mean or median")
        if self.missing_value_scope not in {"score_group", "universe"}:
            raise ValueError("missing_value_scope must be score_group or universe")
        if not isinstance(self.require_positive_book_equity, bool):
            raise ValueError("require_positive_book_equity must be a boolean")


class FactorSignals:
    """Calculate characteristics and ranks; never create positions or returns.

    Yahoo financials are the versions available at download time, not archived
    original filings. Existing earnings-date references are availability proxies.
    Missing early statements stay missing. IWB sector groups use dated holdings;
    optional Yahoo classifications are current metadata, not historical labels.
    """

    def __init__(self, config: SignalConfig | None = None):
        self.config = config or SignalConfig()
        self.database = AQRDatabase(self.config.database_path)
        self.universe = self._universe(self.config)
        historical_sector = self.universe.proxy is not None and self.config.group_by == "sector"
        self.data_source = CachedYahooData(
            self.database, required_group=None if historical_sector else self.config.group_by,
        )
        self.signals = pd.DataFrame()
        self.financials = pd.DataFrame()
        self.companies = pd.DataFrame()
        self.coverage = pd.DataFrame()
        self.errors: list[str] = []
        self.metadata: dict = {}

    def _universe(self, config):
        return Universe(config.universe, config.tickers, config.membership_file,
                        database=self.database, start_date=config.start,
                        end_date=str((pd.Timestamp(config.end_exclusive) - pd.Timedelta(days=1)).date()))

    def _download_one(self, symbol):
        warmup = str((pd.Timestamp(self.config.start) - pd.DateOffset(years=1, months=1)).date())
        if self.config.financial_source == "website_quarterly":
            data = self.data_source.load_cached(symbol, warmup, self.config.end_exclusive,
                                               include_legacy_financials=False)
            data["annual"] = self.website_events.get(symbol, [])
            currencies = self.website_currencies.get(symbol, set())
            if currencies and currencies != {"USD"}:
                raise ValueError("Website statement currency mismatch")
            if (data["country"] != "United States" or data["currency"] != "USD"
                    or data["financial_currency"] != "USD"):
                raise ValueError("Domicile or USD currency mismatch")
            return data
        loader = self.data_source.load if getattr(self,"fetch_missing",True) else self.data_source.load_cached
        try:
            data = loader(symbol, warmup, self.config.end_exclusive)
        except Exception as exc:
            if not getattr(self,"fetch_missing",True):
                raise
            # A failed refresh must not discard usable historical records that
            # were already saved. Missing dates/fields remain missing.
            data = self.data_source.load_cached(symbol, warmup, self.config.end_exclusive)
            self.errors.append("{} source refresh failed; using saved raw records: {}".format(symbol,exc))
        FundamentalStore(self.database).build_monthly(self.config.start,
            str((pd.Timestamp(self.config.end_exclusive) - pd.Timedelta(days=1)).date()), [symbol])
        if (data["country"] != "United States" or data["currency"] != "USD"
                or data["financial_currency"] != "USD"):
            raise ValueError("Industry, domicile or USD financial currency mismatch")
        return data

    def run(self, *, force_refresh=False, workers=6, progress=None, fetch_missing=True):
        """Reuse raw sources, fetch missing intervals, and recalculate all scores."""
        self.errors = []
        self.data_source.stats.clear()
        self.data_source.fundamental_errors = []
        self.data_source.force_refresh = force_refresh
        self.fetch_missing = fetch_missing
        datasets = {}
        website_metadata = {}
        if self.config.financial_source == "website_quarterly":
            self.website_events, self.website_currencies, website_metadata = FundamentalStore(
                self.database).website_quarterly_events(
                    availability=self.config.financial_availability,
                    lag_days=self.config.financial_reporting_lag_days)
        warmup = str((pd.Timestamp(self.config.start) - pd.DateOffset(years=1, months=1)).date())
        if fetch_missing:
            self.data_source._calendar(warmup, str(self.data_source.completed_end.date()))
        with ThreadPoolExecutor(max_workers=workers) as executor:
            jobs = {executor.submit(self._download_one, symbol): symbol for symbol in self.universe.tickers}
            for count, future in enumerate(as_completed(jobs), 1):
                symbol = jobs[future]
                error = None
                try:
                    datasets[symbol] = future.result()
                except Exception as exc:
                    error = "{}: {}: {}".format(symbol, type(exc).__name__, exc)
                    self.errors.append(error)
                if progress is not None:
                    progress(count, len(jobs), symbol, error)
        if not datasets:
            raise RuntimeError("No usable source data: " + "; ".join(self.errors))
        first = max((data["prices"].index for data in datasets.values()), key=len)
        last_month = min(pd.Timestamp(self.config.end_exclusive).to_period("M") - 1,
                         self.data_source.completed_end.to_period("M") - 1)
        month_ends = first.to_series().groupby(first.to_period("M")).max()
        month_ends = month_ends.loc[
            (month_ends >= (pd.Timestamp(self.config.start).to_period("M")-1).start_time) & (month_ends.index <= last_month)
        ]
        rows = []
        companies = []
        annual_rows = []
        for symbol in self.universe.tickers:
            if symbol not in datasets:
                for date in month_ends:
                    rows.append({"as_of": date, "ticker": symbol, "status": "download_failed"})
                continue
            data = datasets[symbol]
            companies.append({key: data[key] for key in (
                "ticker", "name", "industry", "sector", "country", "currency", "financial_currency"
            )})
            annual_rows.extend(data["annual"])
            prices, shares = data["prices"], data["shares"]
            splits = data["splits"]
            annual = pd.DataFrame(data["annual"])
            if not annual.empty:
                for field in ("book_equity", "total_assets", "gross_profit"):
                    annual[field] = pd.to_numeric(annual[field], errors="coerce")
            for date in month_ends:
                row = {"as_of": date, "ticker": symbol, "status": "missing_financials"}
                if date not in prices.index:
                    row["status"] = "missing_month_end_price"
                    rows.append(row)
                    continue
                i = prices.index.get_loc(date)
                row["close"] = float(prices.at[date, "Close"])
                if not np.isfinite(row["close"]) or row["close"] <= 0:
                    row["status"] = "missing_month_end_price"
                    rows.append(row)
                    continue
                if i >= 252:
                    row["momentum"] = float(prices["Close"].iloc[i - 21] / prices["Close"].iloc[i - 252] - 1)
                    row["momentum_start"] = prices.index[i - 252]
                    row["momentum_end"] = prices.index[i - 21]
                    row["momentum_start_close"] = float(prices["Close"].iloc[i - 252])
                    row["momentum_end_close"] = float(prices["Close"].iloc[i - 21])
                if i >= 126:
                    daily_returns = prices["Close"].iloc[i-126:i+1].pct_change(fill_method=None).iloc[1:]
                    if len(daily_returns) == 126 and np.isfinite(daily_returns).all():
                        row["low_volatility"] = -float(daily_returns.std(ddof=1) * np.sqrt(252))
                historical_shares = shares.loc[shares.index <= date]
                if not historical_shares.empty:
                    share_day = historical_shares.index[-1]
                    # Yahoo Close already reflects subsequent splits. Convert
                    # historical share counts to the same quote units. This is
                    # unit conversion, not a dividend adjustment or a signal.
                    factor = float(splits.loc[splits.index > share_day].prod())
                    row.update({
                        "shares_date": share_day,
                        "shares_reported": float(historical_shares.iloc[-1]),
                        "split_factor_to_quote": factor,
                        "shares_quote_basis": float(historical_shares.iloc[-1]) * factor,
                    })
                    row["market_cap"] = row["close"] * row["shares_quote_basis"]
                usable = annual.loc[annual["available_from"].notna() & annual["available_from"].le(date)] if not annual.empty else annual
                if not usable.empty:
                    latest = usable.sort_values(["period_end", "available_from"]).iloc[-1]
                    for field in ("period_end", "earnings_proxy_at", "available_from", "book_equity", "total_assets", "gross_profit",
                                  "balance_period_end", "income_period_end", "cashflow_period_end",
                                  "net_income_ttm", "common_net_income_ttm", "operating_cashflow_ttm",
                                  "earnings_ttm_start", "earnings_ttm_end", "cashflow_ttm_start", "cashflow_ttm_end",
                                  "balance_date_assumed", "income_date_assumed", "cashflow_date_assumed"):
                        if field not in latest:
                            continue
                        row[field] = latest[field]
                    if "market_cap" not in row:
                        row["status"] = "missing_historical_shares"
                    else:
                        # Raw value/quality are derived independently in rescore().
                        row["status"] = "eligible" if "momentum" in row else "insufficient_price_history"
                    if row.get("market_cap", 0) > 0:
                        for factor, field in (("earnings_yield", "common_net_income_ttm"),
                                              ("cashflow_yield", "operating_cashflow_ttm")):
                            if pd.notna(row.get(field)):
                                row[factor] = row[field] / row["market_cap"]
                        # PE is a presentation field, not the ranking input.
                        if row.get("earnings_yield", 0) > 0:
                            row["pe_ttm"] = 1 / row["earnings_yield"]
                    if row.get("total_assets", 0) > 0 and pd.notna(row.get("net_income_ttm")):
                        row["roa"] = row["net_income_ttm"] / row["total_assets"]
                rows.append(row)
        self.signals = pd.DataFrame(rows)
        if rows:
            self.signals["input_status"] = self.signals["status"]
        self.financials = pd.DataFrame(annual_rows, columns=[
            "ticker", "period_end", "earnings_proxy_at", "available_from",
            "availability_method", "book_equity", "total_assets", "gross_profit",
        ]).sort_values(["ticker", "period_end"])
        self.companies = pd.DataFrame(companies)
        self.metadata = {
            "downloaded_at": self.database.read("""SELECT MAX(observed_at) latest FROM (
                SELECT observed_at FROM prices UNION ALL SELECT observed_at FROM shares
                UNION ALL SELECT observed_at FROM snapshots WHERE dataset!='monthly_fundamental_state')""").iloc[0, 0],
            "loaded_at": pd.Timestamp.now(tz="UTC").isoformat(),
            "requested_end_exclusive": self.config.end_exclusive,
            "start": self.config.start, "yfinance_version": yf.__version__,
            "financial_frequency": "quarterly" if self.config.financial_source == "website_quarterly" else "yearly",
            "availability": website_metadata or "First reported earnings event within 120 days after fiscal period end; usable next session. Proxy, not verified filing time.",
            "financial_versions": "Latest Yahoo versions, potentially restated; not a point-in-time archive",
            "shares": "Historical Yahoo shares timestamps, last duplicate kept; publication times not independently verified",
            "price_basis": "Yahoo Close, no Adj Close or dividends; share counts matched to split-adjusted quote units",
            "filters": "US domicile, USD prices and financials; finite selected factors; quality/ROA require positive total assets; signed B/M is allowed unless configured otherwise; no size/liquidity filters",
            "factor_definitions": FACTOR_DEFINITIONS,
            "partial_month": "Omitted; only completed calendar months are ranked",
            "backtest_run": False,
            "database_path": self.config.database_path,
            "grouping_source": "historical_iwb_holdings" if self.universe.proxy is not None and self.config.group_by == "sector" else "yahoo_company_metadata" if self.config.group_by else "universe",
            "source_requests": dict(self.data_source.stats),
            "fundamental_source_errors": list(self.data_source.fundamental_errors),
            "score_storage": "recalculated_in_memory_from_cached_sources",
            "calculated_months": len(month_ends),
        }
        if rows:
            self.rescore()
        pieces = [self.signals] if rows else []
        if not pieces:
            raise ValueError("No completed month-end signals in the requested range")
        records = [record for piece in pieces for record in piece.to_dict(orient="records")]
        self.signals = pd.DataFrame(records).sort_values(["as_of", "rank", "ticker"], na_position="last").reset_index(drop=True)
        if "input_status" not in self.signals:
            self.signals["input_status"] = self.signals["status"].replace({"fewer_than_min_stocks": "eligible"})
        # rescore() owns coverage, including missing-value and imputation counts.
        self.metadata.update({"config": asdict(self.config), "min_stocks": self.config.min_stocks,
                              "universe": self.config.universe, "industry": ", ".join(sorted(self.companies["industry"].dropna().unique()))})
        return self

    def rescore(self, config: SignalConfig | None = None, *, eligibility_mask=None):
        """Recheck monthly membership/eligibility and apply the active settings.

        This rescoring helper only reads the exported presentation snapshot.
        Use run() for automatic SQLite coverage checks and missing-source fetches.
        """
        config = config or self.config
        universe = self._universe(config)
        missing = set(universe.tickers) - set(self.signals["ticker"])
        if missing:
            raise ValueError("Tickers missing from this export: {}. Use IndustrySignals(config).run().".format(
                ", ".join(sorted(missing))))
        if (config.start != self.metadata["start"]
                or config.end_exclusive != self.metadata["requested_end_exclusive"]):
            raise ValueError("Export dates differ. Use IndustrySignals(config).run() to load the requested history.")
        self.config, self.universe = config, universe
        factors = tuple(config.factor_operators)
        frame = self.signals.copy()
        if eligibility_mask is not None:
            if not eligibility_mask.index.equals(frame.index) or eligibility_mask.isna().any():
                raise ValueError("eligibility_mask must align exactly to the current signal rows")
            eligibility_mask = eligibility_mask.astype(bool)
        if "input_status" not in frame:
            frame["input_status"] = frame["status"].replace({"fewer_than_min_stocks": "eligible"})
        legacy_equity_gate = frame["input_status"].eq("nonpositive_equity_or_assets")
        if legacy_equity_gate.any() and {"book_equity", "market_cap", "gross_profit", "total_assets"}.issubset(frame):
            restored_status = np.where(pd.to_numeric(frame["momentum"], errors="coerce").notna(),
                                       "eligible", "insufficient_price_history")
            frame.loc[legacy_equity_gate, "input_status"] = restored_status[legacy_equity_gate]
        frame["status"] = frame["input_status"]
        # Re-derive from retained raw financial fields, including old exports
        # where the former global negative-equity gate suppressed both ratios.
        for name, values in balance_ratios(frame).items():
            frame[name] = values
            frame[name + "_raw"] = values.replace([np.inf, -np.inf], np.nan)
        for name in factors:
            if name not in frame:
                frame[name] = np.nan
            if name + "_raw" not in frame:
                frame[name + "_raw"] = pd.to_numeric(frame[name], errors="coerce")
            frame[name] = pd.to_numeric(frame[name + "_raw"], errors="coerce").replace([np.inf, -np.inf], np.nan)
            frame[name + "_imputed"] = False
            frame[name + "_rank"] = np.nan
            frame[name + "_score"] = np.nan
        if "value" in factors and config.require_positive_book_equity:
            # Preserve the signed raw characteristic for reversible rescoring.
            frame.loc[frame["value"].le(0), "value"] = np.nan
        frame["score"], frame["rank"] = np.nan, np.nan
        frame["in_universe"] = False
        frame["score_group"] = "universe"
        use_groups = config.group_by is not None and any(
            operator.startswith("group_")
            for pipeline in config.factor_operators.values() for operator in pipeline
        )
        historical_sector = universe.proxy is not None and config.group_by == "sector"
        if use_groups and not historical_sector:
            if config.group_by not in self.companies:
                raise ValueError("Export lacks grouping field {}. Use IndustrySignals(config).run().".format(config.group_by))
            labels = self.companies.set_index("ticker")[config.group_by].replace("", np.nan)
            frame["score_group"] = frame["ticker"].map(labels)
        coverage = []
        for date, month in frame.groupby("as_of", sort=True):
            if use_groups and historical_sector:
                labels = universe.group_labels(date, config.group_by)
                frame.loc[month.index, "score_group"] = month["ticker"].map(labels)
                month = frame.loc[month.index]
            active = month["ticker"].isin(universe.members(date))
            frame.loc[month.index, "in_universe"] = active
            frame.loc[month.index[~active], "status"] = "not_in_universe"
            # Impute characteristics only. Never impute prices, dates,
            # membership, currencies or invalid ratio denominators.
            candidates = active & ~month["input_status"].isin([
                "download_failed", "missing_month_end_price"])
            if eligibility_mask is not None:
                candidates &= eligibility_mask.loc[month.index]
            if "value" in factors and config.require_positive_book_equity:
                book_value = pd.to_numeric(month["book_equity"], errors="coerce") if "book_equity" in month else month["value_raw"]
                excluded_book = active & book_value.le(0)
                frame.loc[month.index[excluded_book], "status"] = "nonpositive_book_equity_filter"
                candidates &= ~excluded_book
            if {"quality", "roa"} & set(factors) and "total_assets" in month:
                invalid_assets = active & pd.to_numeric(month["total_assets"], errors="coerce").le(0)
                frame.loc[month.index[invalid_assets], "status"] = "invalid_total_assets"
                candidates &= ~invalid_assets
            if {"value", "earnings_yield", "cashflow_yield"} & set(factors) and "market_cap" in month:
                invalid_cap = active & pd.to_numeric(month["market_cap"], errors="coerce").le(0)
                frame.loc[month.index[invalid_cap], "status"] = "invalid_market_cap"
                candidates &= ~invalid_cap
            if "close" in month:
                candidates &= pd.to_numeric(month["close"], errors="coerce").gt(0)
            raw_complete = candidates & np.isfinite(month[list(factors)]).all(axis=1)
            complete_count = int(raw_complete.sum())
            if config.missing_value_method != "drop":
                donors = month.loc[candidates]
                for name in factors:
                    if config.missing_value_scope == "score_group":
                        fill = donors.groupby("score_group")[name].transform(config.missing_value_method)
                    else:
                        fill = pd.Series(getattr(donors[name], config.missing_value_method)(), index=donors.index)
                    needed = donors[name].isna() & fill.notna()
                    chosen = donors.index[needed]
                    frame.loc[chosen, name] = fill.loc[chosen]
                    frame.loc[chosen, name + "_imputed"] = True
            month = frame.loc[month.index]
            complete = candidates & np.isfinite(month[list(factors)]).all(axis=1)
            frame.loc[month.index[candidates & ~complete], "status"] = "missing_factor_inputs"
            imputed = month[[name + "_imputed" for name in factors]].any(axis=1)
            frame.loc[month.index[complete], "status"] = "eligible"
            frame.loc[month.index[complete & imputed], "status"] = "eligible_imputed"
            valid_group = month["score_group"].notna()
            frame.loc[month.index[complete & ~valid_group], "status"] = "missing_group"
            eligible = month.index[complete & valid_group]
            sizes = frame.loc[eligible].groupby("score_group")["ticker"].transform("size")
            usable = eligible[sizes.ge(config.min_stocks)]
            frame.loc[eligible[sizes.lt(config.min_stocks)], "status"] = "fewer_than_min_stocks"
            if len(usable):
                groups = frame.loc[usable, "score_group"]
                for name in factors:
                    values = frame.loc[usable, name]
                    pipeline = config.factor_operators[name]
                    if config.group_by is not None and any(op.startswith("group_") for op in pipeline):
                        ranks = values.groupby(groups).rank(ascending=False, method="average")
                    else:
                        ranks = values.rank(ascending=False, method="average")
                    frame.loc[usable, name + "_rank"] = ranks
                    frame.loc[usable, name + "_score"] = transform(values, pipeline, groups)
                score = combine(frame.loc[usable, [name + "_score" for name in factors]], config.combine_operator)
                frame.loc[usable, "score"] = score
                frame.loc[usable, "rank"] = score.rank(ascending=False, method="min")
            coverage.append({
                "as_of": date, "sample_size": int(active.sum()), "complete_inputs": complete_count,
                "ranked_stocks": len(usable),
                "missing_financials": int((active & month["input_status"].isin(["missing_financials", "missing_annual_financials"])).sum()),
                "dropped_missing_factors": int((candidates & ~complete).sum()),
                "imputed_stocks": int(month.loc[usable, [name + "_imputed" for name in factors]].any(axis=1).sum()),
                "imputed_factor_values": int(month.loc[usable, [name + "_imputed" for name in factors]].sum().sum()),
            })
        self.signals = frame.sort_values(["as_of", "rank", "ticker"], na_position="last").reset_index(drop=True)
        self.coverage = pd.DataFrame(coverage)
        self.metadata.update({
            "config": asdict(config), "min_stocks": config.min_stocks,
            "universe": config.universe,
            "grouping_source": "historical_iwb_holdings" if historical_sector else "yahoo_company_metadata" if use_groups else "universe",
            "industry": ", ".join(sorted(self.companies["industry"].dropna().unique())),
            "ranking": "Configured per-factor operator pipelines, {}; global final rank 1=best".format(config.combine_operator),
            "calculated_at": pd.Timestamp.now(tz="UTC").isoformat(),
            "missing_value_policy": dict(method=config.missing_value_method,
                scope=config.missing_value_scope, stage="raw_factors_before_ranking",
                donor_pool="same monthly active eligible universe; no future observations",
                empty_group="remains missing and is dropped"),
            "book_equity_policy": "positive-only for value" if config.require_positive_book_equity else "finite signed B/M retained; negative equity is not a global exclusion",
            "factor_definitions": FACTOR_DEFINITIONS,
            "filters": "US domicile, USD prices and financials; finite selected factors; quality/ROA require positive total assets; book-equity policy applies to selected value only; no size/liquidity filters",
        })
        return self
    def construct(self, config=None, *, as_of=None, **inputs):
        """Build target weights using dated classifications and existing scores."""
        from .construction import ConstructionConfig, PortfolioConstructor
        config = config or ConstructionConfig(group_by=self.config.group_by)
        if self.signals.empty:
            raise ValueError("Calculate or load factor scores before constructing weights")
        frame = self.signals.copy()
        if as_of is not None:
            frame = frame.loc[frame["as_of"].eq(pd.Timestamp(as_of).normalize())].copy()
            if frame.empty:
                raise ValueError("No scores for the requested construction date")
        if config.group_by is not None:
            if self.universe.proxy is not None and config.group_by == "sector":
                for date, month in frame.groupby("as_of"):
                    labels = self.universe.group_labels(date, config.group_by)
                    frame.loc[month.index, config.group_by] = month["ticker"].map(labels)
            else:
                if config.group_by not in self.companies:
                    raise ValueError("Missing construction labels: " + config.group_by)
                frame[config.group_by] = frame["ticker"].map(self.companies.set_index("ticker")[config.group_by])
        if config.method == "aqr_style" and inputs.get("prices") is None and inputs.get("risk_model") is None:
            start = str((frame["as_of"].min() - pd.DateOffset(years=2)).date())
            end = str((frame["as_of"].max() + pd.Timedelta(days=1)).date())
            inputs["prices"] = PortfolioConstructor.cached_prices(self.database, frame["ticker"].unique(), start, end)
        return PortfolioConstructor(config).build(frame, **inputs)

    def snapshot(self, month: str | None = None):
        date = self.signals["as_of"].max() if month is None else pd.Timestamp(month)
        if month is not None and len(month) == 7:
            mask = self.signals["as_of"].dt.to_period("M").astype(str).eq(month)
        else:
            mask = self.signals["as_of"].eq(date)
        mask &= self.signals["in_universe"]
        return self.signals.loc[mask].copy().sort_values(["rank", "ticker"], na_position="last")

    def _payload(self):
        def records(frame):
            return json.loads(frame.to_json(orient="records", date_format="iso"))
        payload = {"metadata": self.metadata, "errors": self.errors}
        for name in ("signals", "financials", "companies", "coverage"):
            payload[name] = records(getattr(self, name))
        return payload

    def save(self, path):
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self._payload(), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


    @classmethod
    def load(cls, path, config: SignalConfig | None = None):
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        saved = payload["metadata"].get("config", {
            "start": payload["metadata"]["start"],
            "end_exclusive": payload["metadata"]["requested_end_exclusive"],
        })
        if "require_positive_book_equity" not in saved and "positive equity" in payload["metadata"].get("filters", ""):
            saved = {**saved, "require_positive_book_equity": True}
        result = cls(config or SignalConfig(**saved))
        result.metadata, result.errors = payload["metadata"], payload["errors"]
        date_columns = {"as_of", "period_end", "available_from", "shares_date", "momentum_start", "momentum_end"}
        for name in ("signals", "financials", "companies", "coverage"):
            frame = pd.DataFrame(payload[name])
            for column in date_columns & set(frame.columns):
                frame[column] = pd.to_datetime(frame[column])
            setattr(result, name, frame)
        return result.rescore()
