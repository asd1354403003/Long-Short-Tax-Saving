"""Incremental yfinance downloads backed by the AQR method's SQLite database."""
from __future__ import annotations

from collections import Counter
import json
import threading

import numpy as np
import pandas as pd
import yfinance as yf

from .database import AQRDatabase
from .fundamentals import FundamentalStore, validate_yahoo_info


def normalize_index(frame):
    frame = frame.copy()
    frame.index = pd.DatetimeIndex(frame.index).tz_localize(None).normalize()
    return frame.loc[~frame.index.duplicated(keep="last")].sort_index()


def split_factors(dates, splits):
    return pd.Series([float(splits.loc[splits.index > day].prod()) for day in dates], index=dates)


class CachedYahooData:
    def __init__(self, database: AQRDatabase, *, force_refresh=False, required_group="industry"):
        self.database = database
        self.force_refresh = force_refresh
        self.required_group = required_group
        self.stats = Counter()
        self.fundamental_errors = []
        self._calendar_lock = threading.Lock()
        self._calendar_cache = []
        # Never mark an unfinished/current day or a future date as fully cached.
        self.completed_end = pd.Timestamp.now(tz="America/New_York").normalize().tz_localize(None)

    def _calendar(self, start, end_exclusive):
        start, end_exclusive = str(pd.Timestamp(start).date()), str(pd.Timestamp(end_exclusive).date())
        key = "__NYSE_CALENDAR__"
        with self._calendar_lock:
            for lower, upper, index in self._calendar_cache:
                if lower <= start and upper >= end_exclusive:
                    return index[(index >= start) & (index < end_exclusive)]
            for lower, upper in self.database.missing_ranges("prices", key, start, end_exclusive):
                # A small left overlap gives a valid response for weekend-only gaps.
                query_start = str((pd.Timestamp(lower) - pd.Timedelta(days=7)).date())
                raw = yf.Ticker("SPY").history(start=query_start, end=upper, auto_adjust=False, actions=True, raise_errors=True)
                self.stats["calendar_requests"] += 1
                if raw.empty:
                    raise ValueError("Unable to verify the exchange calendar")
                raw = normalize_index(raw)
                selected = raw.loc[(raw.index >= lower) & (raw.index < upper)]
                self.database.store_range("prices", key, lower, upper, selected)
            result = self.database.price_frame(key, start, end_exclusive)
            index = pd.DatetimeIndex([]) if result.empty else result.index
            self._calendar_cache.append((start, end_exclusive, index))
            return index

    def _snapshot_needed(self, dataset, symbol, through):
        saved = self.database.latest_snapshot(dataset, symbol)
        needed = self.force_refresh or saved is None or saved["verified_through"] < through
        return saved, needed

    def load(self, symbol, start, end_exclusive):
        end = min(pd.Timestamp(end_exclusive), self.completed_end)
        if pd.Timestamp(start) >= end:
            raise ValueError("No completed source dates in the requested interval")
        upper = str(end.date())
        through = str((end - pd.Timedelta(days=1)).date())
        observed_through = str(self.completed_end.date())
        ticker = yf.Ticker(symbol)
        company, needed = self._snapshot_needed("company", symbol, through)
        required = ("currency", "financialCurrency", "country")
        if self.required_group:
            required += (self.required_group,)
        if company and any(not company["payload"].get(name) for name in required):
            needed = True
        if needed:
            full_info = self.database.latest_snapshot("yahoo_info", symbol)
            valid_saved_info = full_info and all(full_info["payload"].get(name) for name in required)
            if not self.force_refresh and valid_saved_info and full_info["verified_through"] >= through:
                info = full_info["payload"]
            else:
                info = ticker.get_info()
                self.stats["company_requests"] += 1
                info = validate_yahoo_info(info, required)
                self.database.put_snapshot("yahoo_info", symbol, info, observed_through)
            company_data = {name: info.get(name) for name in (
                "shortName", "industry", "sector", "country", "currency", "financialCurrency"
            )}
            self.database.put_snapshot("company", symbol, company_data, observed_through)
        else:
            company_data = company["payload"]
        _, needed = self._snapshot_needed("split_state", symbol, through)
        if needed:
            full_splits = ticker.get_splits(period="max")
            self.stats["split_requests"] += 1
            if full_splits is None:
                raise ValueError("No corporate-action response")
            full_splits = normalize_index(full_splits)
            self.database.replace_splits(symbol, full_splits)
            self.database.put_snapshot("split_state", symbol, {}, observed_through)
        splits = self.database.split_series(symbol)
        sessions = None
        price_ranges = [(start, upper)] if self.force_refresh else self.database.missing_ranges("prices", symbol, start, upper)
        for lower, higher in price_ranges:
            if sessions is None:
                sessions = self._calendar(start, upper)
            expected = sessions[(sessions >= lower) & (sessions < higher)]
            if not len(expected):
                self.database.store_range("prices", symbol, lower, higher, pd.DataFrame())
                continue
            raw = ticker.history(start=lower, end=higher, auto_adjust=False, actions=True, raise_errors=True)
            self.stats["price_requests"] += 1
            if raw.empty:
                raise ValueError("Empty prices for an interval with exchange sessions")
            raw = normalize_index(raw)
            factors = split_factors(raw.index, splits)
            # Canonical market-unit OHLC avoids incompatible old/new quote bases
            # when a later split is encountered on a right-hand extension.
            for column in ("Open", "High", "Low", "Close", "Dividends"):
                if column in raw:
                    raw[column] = raw[column] * factors
            self.database.store_range("prices", symbol, lower, higher, raw)
        share_ranges = [(start, upper)] if self.force_refresh else self.database.missing_ranges("shares", symbol, start, upper)
        for lower, higher in share_ranges:
            anchor = self.database.share_series(symbol, lower, higher)
            query_start = min(lower, str(anchor.index[0].date())) if not anchor.empty else lower
            raw = ticker.get_shares_full(start=query_start, end=higher)
            self.stats["shares_requests"] += 1
            if raw is None:
                raise ValueError("Historical shares request failed")
            raw = normalize_index(raw).loc[lambda series: series.gt(0)]
            raw = raw.loc[(raw.index >= lower) & (raw.index < higher)]
            self.database.store_range("shares", symbol, lower, higher, raw)
        _, needed = self._snapshot_needed("financials", symbol, through)
        fundamentals = FundamentalStore(self.database)
        if needed or not fundamentals.has_sources(symbol, through):
            statements = fundamentals.refresh(symbol, through, ticker=ticker, calendar=self._calendar,
                                               force=self.force_refresh)
            self.stats.update(fundamentals.stats)
            self.fundamental_errors.extend(fundamentals.errors)
            balance = statements.get(("balance", "yearly"), pd.DataFrame())
            income = statements.get(("income", "yearly"), pd.DataFrame())
            earnings = statements["earnings"]
            if balance.empty or income.empty or earnings is None or earnings.empty:
                raise ValueError("Incomplete annual financial response")
            earnings = earnings.dropna(subset=["Reported EPS"]).sort_index()
            days = earnings.index.tz_localize(None).normalize()
            annual = []
            for period in sorted(set(balance.columns) & set(income.columns)):
                def number(frame, name):
                    value = frame.loc[name, period] if name in frame.index else np.nan
                    return float(value) if pd.notna(value) else np.nan
                equity, assets = number(balance, "CommonStockEquity"), number(balance, "TotalAssets")
                profit = number(income, "GrossProfit")
                if not np.isfinite(profit):
                    profit = number(income, "TotalRevenue") - number(income, "CostOfRevenue")
                if not all(np.isfinite([equity, assets, profit])):
                    continue
                dates = earnings.index[(days > period) & (days <= period + pd.Timedelta(days=120))]
                release = dates[0] if len(dates) else None
                available = None
                if release is not None:
                    release_day = release.tz_localize(None).normalize()
                    calendar = self._calendar(str(release_day.date()), str(self.completed_end.date()))
                    later = calendar[calendar > release_day]
                    available = str(later[0].date()) if len(later) else None
                annual.append({
                    "ticker": symbol, "period_end": str(pd.Timestamp(period).date()),
                    "earnings_proxy_at": release.isoformat() if release is not None else None,
                    "available_from": available,
                    "availability_method": "next_session_after_annual_earnings_proxy",
                    "book_equity": equity, "total_assets": assets, "gross_profit": profit,
                })
            self.database.put_snapshot("financials", symbol, annual, observed_through)
        return self.load_cached(symbol, start, upper)

    def load_cached(self, symbol, start, end_exclusive, *, include_legacy_financials=True):
        """Read source records only, with no network or cached score results."""
        company = self.database.latest_snapshot("company", symbol)
        if company is None:
            raise ValueError("No cached company metadata")
        company_data = company["payload"]
        splits = self.database.split_series(symbol)
        prices = self.database.price_frame(symbol, start, end_exclusive)
        if prices.empty:
            raise ValueError("No cached prices in the requested interval")
        factors = split_factors(prices.index, splits)
        for column in ("Open", "High", "Low", "Close", "Dividends"):
            prices[column] = prices[column] / factors
        return {
            "ticker": symbol, "name": company_data.get("shortName", symbol),
            "industry": company_data.get("industry"), "sector": company_data.get("sector"),
            "country": company_data.get("country"), "currency": company_data.get("currency"),
            "financial_currency": company_data.get("financialCurrency"),
            "prices": prices, "shares": self.database.share_series(symbol, start, end_exclusive),
            "splits": splits, "annual": self.database.all_financials(symbol) if include_legacy_financials else [],
        }

    def import_legacy(self, path):
        """One-time reuse of already downloaded company/annual source inputs."""
        if not path.is_file():
            return
        payload = json.loads(path.read_text(encoding="utf-8"))
        observed = payload["metadata"]["downloaded_at"]
        through = observed[:10]
        for company in payload["companies"]:
            symbol = company["ticker"]
            if self.database.latest_snapshot("company", symbol) is None:
                data = {"shortName": company["name"], "industry": company.get("industry"),
                        "sector": company.get("sector"), "country": company.get("country"),
                        "currency": company.get("currency"), "financialCurrency": company.get("financial_currency")}
                self.database.put_snapshot("company", symbol, data, through, observed_at=observed)
            if self.database.latest_snapshot("financials", symbol) is None:
                facts = [row for row in payload["financials"] if row["ticker"] == symbol]
                self.database.put_snapshot("financials", symbol, facts, through, observed_at=observed)
