"""Discover industry peers with yfinance, form A/B/C triplets, and save data.

Each input ticker is one A leg. B and C are individual replacement assets.
This is a screening pipeline, not a trading backtest or tax eligibility test.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import hashlib
import json
from importlib.metadata import version
import logging
from pathlib import Path
import re
import time
import unicodedata

import numpy as np
import pandas as pd
import yfinance as yf

from cointegration import Cointegration

LOGGER = logging.getLogger(__name__)
MATCH_COLUMNS = ["A", "B", "C", "industry", "sector"]


@dataclass(frozen=True)
class CointegrationConfig:
    train_start: str = "2022-01-01"
    train_end: str = "2025-12-31"  # inclusive
    min_observations: int = 750
    max_lag: int = 1
    check_i1: bool = False

    def __post_init__(self):
        if pd.Timestamp(self.train_start) >= pd.Timestamp(self.train_end):
            raise ValueError("train_start must be earlier than train_end")
        if self.min_observations < 20 or self.max_lag < 0:
            raise ValueError("min_observations >= 20 and max_lag >= 0 required")


@dataclass(frozen=True)
class PortfolioConfig:
    cointegration: CointegrationConfig = field(default_factory=CointegrationConfig)
    data_end: str = "2026-12-31"  # inclusive; Yahoo's end parameter is exclusive
    as_of_date: str | None = None
    benchmark_ticker: str | None = None
    max_candidates: int | None = None
    batch_size: int = 10
    min_correlation: float = 0.90
    correlation_basis: str = "price"  # Close price levels; optionally "returns"
    ab_confidence_level: float = 0.95
    ac_confidence_level: float = 0.95
    bc_confidence_level: float = 0.90
    allow_sector_fallback: bool = False
    related_industries: dict[str, list[str]] = field(default_factory=dict)
    related_tickers: dict[str, list[str]] = field(default_factory=dict)
    disjoint_portfolios: bool = True
    equivalent_ticker_groups: tuple[tuple[str, ...], ...] = (
        ("GOOG", "GOOGL"), ("BRK-A", "BRK-B"), ("FOX", "FOXA"),
        ("NWS", "NWSA"), ("UA", "UAA"), ("HEI", "HEI-A"), ("BF-A", "BF-B"),
    )
    output_dir: Path = Path("data")

    @classmethod
    def from_relationship_file(cls, path: Path, **settings):
        """Load the curated industry graph; discovery of its tickers is automatic."""
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        related = {}
        for group in payload["groups"]:
            for industry in group["industries"]:
                peers = related.setdefault(industry, [])
                peers.extend(name for name in group["industries"] if name != industry and name not in peers)
        return cls(related_industries=related, related_tickers=payload.get("related_tickers", {}), **settings)

    def __post_init__(self):
        for name in ("ab_confidence_level", "ac_confidence_level", "bc_confidence_level"):
            if getattr(self, name) not in (0.95, 0.90):
                raise ValueError(name + " must be 0.95 or 0.90")
        if self.max_candidates is not None and self.max_candidates < 2:
            raise ValueError("max_candidates must be None or at least 2")
        if not -1 < self.min_correlation < 1 or self.correlation_basis not in ("price", "returns"):
            raise ValueError("Use a correlation threshold in (-1, 1) and basis price or returns")
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive")
        if pd.Timestamp(self.data_end) <= pd.Timestamp(self.cointegration.train_end):
            raise ValueError("data_end must be after the formation period")
        if self.as_of_date and pd.Timestamp(self.as_of_date) < pd.Timestamp(self.cointegration.train_end):
            raise ValueError("as_of_date must not precede the formation period end")


@dataclass
class Portfolio:
    name: str
    members: pd.DataFrame  # ticker, anchor_A, ratio, intercept, correlation
    data: pd.DataFrame  # daily vendor data joined to the frozen member ratios

    @property
    def tickers(self) -> list[str]:
        return self.members["ticker"].tolist()

    @property
    def prices(self) -> pd.DataFrame:
        if self.data.empty:
            return pd.DataFrame(index=pd.DatetimeIndex([], name="Date"))
        return self.data.drop_duplicates(["Date", "ticker"]).pivot(index="Date", columns="ticker", values="Close")


def _normalize_tickers(tickers) -> list[str]:
    if isinstance(tickers, str):
        raise TypeError("Pass a list of tickers, not a single string")
    result = []
    for ticker in tickers:
        if not isinstance(ticker, str) or not ticker.strip():
            raise ValueError("Each ticker must be a nonempty string")
        ticker = ticker.strip().upper().replace(".", "-")
        if not re.fullmatch(r"[A-Z0-9^=\-]{1,24}", ticker):
            raise ValueError("Unsupported ticker: {}".format(ticker))
        if ticker not in result:
            result.append(ticker)
    return result


def _key(value):
    return re.sub(r"[^a-z0-9]", "", unicodedata.normalize("NFKD", str(value)).lower())


class YahooFinanceSource:
    """Paginated industry discovery and small downloads with in-memory caching.

    Yahoo classification and screener ordering are CURRENT snapshots, not
    point-in-time 2025 constituents. Saved metadata makes this explicit.
    """

    def __init__(self, config: PortfolioConfig):
        self.config = config
        self.events = []
        self._history_memo = {}
        self._screen_memo = {}
        self._metadata_memo = {}
        self._universe_memo = {}
        if not hasattr(yf, "screen") or not hasattr(yf, "EquityQuery"):
            raise ImportError("Update yfinance using this project's requirements.txt")

    def _request(self, call, context):
        for attempt in range(2):
            try:
                return call()
            except Exception as exc:
                if attempt == 0:
                    time.sleep(0.5)
                else:
                    message = "{}: {}: {}".format(context, type(exc).__name__, exc)
                    self.events.append(message)
                    raise RuntimeError(message) from exc

    def get_industry(self, ticker):
        """Return Yahoo's current industry/sector labels and source metadata.

        This reads a provider classification; it does not infer a company's
        business or identify its suppliers/customers.
        """
        if ticker in self._metadata_memo:
            return self._metadata_memo[ticker]
        def fetch_metadata():
            info = yf.Ticker(ticker).get_info()
            if not info.get("industry") and not info.get("sector"):
                raise ValueError("No Yahoo industry/sector metadata for " + ticker)
            return info
        info = self._request(fetch_metadata, "metadata " + ticker)
        result = {key: info.get(key) for key in (
            "symbol", "shortName", "industry", "sector", "industryKey", "sectorKey", "currency", "quoteType"
        )}
        result["retrieved_at_utc"] = datetime.now(timezone.utc).isoformat()
        # A failed/empty response must not be treated as a valid classification.
        if not result["industry"] and not result["sector"]:
            raise RuntimeError("No Yahoo industry/sector metadata for " + ticker)
        self._metadata_memo[ticker] = result
        return result

    def get_large_cap_universe(self, limit: int = 300) -> pd.DataFrame:
        """Current US exchange listings tagged EQUITY, market cap descending."""
        query = yf.EquityQuery("and", [
            yf.EquityQuery("eq", ["region", "us"]),
            yf.EquityQuery("is-in", ["exchange", "NYQ", "NMS", "NGM", "NCM", "ASE"]),
            yf.EquityQuery("gt", ["intradaymarketcap", 1_000_000_000]),
        ])
        rows, seen, offset = [], set(), 0
        while len(rows) < limit:
            response = self._request(lambda: yf.screen(
                query, offset=offset, size=250, sortField="intradaymarketcap", sortAsc=False,
            ), "large-cap seed universe")
            page = response.get("quotes", [])
            if not page:
                break
            added = 0
            for quote in page:
                symbol = str(quote.get("symbol", "")).upper()
                if (symbol in seen or not re.fullmatch(r"[A-Z0-9\-]{1,24}", symbol)
                        or re.search(r"-P[A-Z0-9]+$", symbol)
                        or symbol.endswith(("-WT", "-WS"))
                        or quote.get("quoteType", "EQUITY") != "EQUITY"):
                    continue
                seen.add(symbol)
                added += 1
                rows.append({"ticker": symbol, "name": quote.get("longName", quote.get("shortName")),
                             "market_cap": quote.get("marketCap", 0)})
                if len(rows) >= limit:
                    break
            offset += len(page)
            if not added or len(page) < 250 or offset >= response.get("total", float("inf")):
                break
        return pd.DataFrame(rows).sort_values("market_cap", ascending=False, kind="stable").reset_index(drop=True)

    def _screen(self, field_name, label, limit=None):
        """Page through a single industry/sector's metadata, not all-market prices."""
        query_class = yf.EquityQuery
        sample = query_class("eq", ["region", "us"])
        allowed = sample.valid_values.get(field_name, [])
        if isinstance(allowed, dict):
            allowed = [item for items in allowed.values() for item in items]
        canonical = next((item for item in allowed if _key(item) == _key(label)), label)
        memo_key = (field_name, canonical, limit)
        if memo_key in self._screen_memo:
            return self._screen_memo[memo_key]
        query = query_class("and", [
            query_class("eq", ["region", "us"]),
            query_class("is-in", ["exchange", "NYQ", "NMS", "NGM", "NCM", "ASE"]),
            query_class("eq", [field_name, canonical]),
        ])
        quotes, seen, offset = [], set(), 0
        while limit is None or len(quotes) < limit:
            size = 250 if limit is None else min(250, limit - len(quotes))
            response = self._request(
                lambda: yf.screen(query, offset=offset, size=size,
                                  sortField="intradaymarketcap", sortAsc=False),
                "{} screen {}".format(field_name, label),
            )
            page = response.get("quotes")
            if not isinstance(page, list):
                raise RuntimeError("Yahoo returned no quotes list for " + str(label))
            fresh = [row for row in page if row.get("symbol") not in seen]
            if not fresh:
                if page:
                    self.events.append("repeated_screener_page: " + str(label))
                break
            quotes.extend(fresh)
            seen.update(row.get("symbol") for row in fresh)
            offset += len(page)
            if len(page) < size or (response.get("total") is not None and offset >= response["total"]):
                break
        result = [item for item in quotes if item.get("quoteType", "EQUITY") == "EQUITY"]
        self._screen_memo[memo_key] = result
        return result

    def get_candidates(self, ticker, excluded):
        """Query the same industry and explicitly configured related pools.

        industry: Yahoo industry label equality.
        related_industry / related_ticker: user-configured relationships,
        not independently verified upstream/downstream company links.
        Each returned row keeps scope and label for auditing.
        """
        info = self.get_industry(ticker)
        if info.get("quoteType") not in (None, "EQUITY"):
            raise RuntimeError("Automatic industry screening currently supports equities only")
        if info.get("currency") not in (None, "USD"):
            raise RuntimeError("Automatic US peer screening requires a USD-denominated seed")
        industry = info.get("industry")
        # Yahoo metadata uses "Banks - Diversified", while EquityQuery uses
        # "Banks—Diversified". Match normalized labels so both expand the graph.
        related = list(dict.fromkeys(
            name for label, names in self.config.related_industries.items()
            if label == ticker or (industry and _key(label) == _key(industry))
            for name in names
        ))
        explicit = _normalize_tickers(self.config.related_tickers.get(ticker, []))
        settings = {"industry": industry, "related_industries": related,
                    "related_tickers": explicit, "limit": self.config.max_candidates,
                    "sector": info.get("sector") if self.config.allow_sector_fallback else None}
        digest = hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()[:12]
        memo_key = (ticker, digest)
        if memo_key in self._universe_memo:
            return info, [row for row in self._universe_memo[memo_key] if row["ticker"] not in excluded]
        rows = [{"ticker": name, "scope": "related_ticker", "label": "configured_related_tickers"}
                for name in explicit if name != ticker]
        seen = {row["ticker"] for row in rows}
        scopes = [("industry", industry, "industry")]
        scopes.extend(("industry", label, "related_industry") for label in related if label != industry)
        if self.config.allow_sector_fallback:
            scopes.append(("sector", info.get("sector"), "sector"))
        successful_screens = 0
        for query_field, label, scope in scopes:
            if not label:
                continue
            try:
                limit = None if self.config.max_candidates is None else self.config.max_candidates * 3
                quotes = self._screen(query_field, label, limit)
                successful_screens += 1
            except Exception as exc:
                LOGGER.warning("%s", exc)
                continue
            for quote in quotes:
                symbol = str(quote.get("symbol", "")).upper()
                if (not re.fullmatch(r"[A-Z0-9\-]{1,24}", symbol)
                        or re.search(r"-P[A-Z0-9]+$", symbol)
                        or symbol == ticker or symbol in seen):
                    continue
                if quote.get("currency") not in (None, "USD"):
                    continue
                rows.append({"ticker": symbol, "scope": scope, "label": label})
                seen.add(symbol)
        if not successful_screens and not rows:
            raise RuntimeError("All industry/sector screens failed for " + ticker)
        if self.config.max_candidates is not None:
            rows = rows[:self.config.max_candidates]
        self._universe_memo[memo_key] = rows
        return info, [row for row in rows if row["ticker"] not in excluded]

    def history(self, tickers, start, end_exclusive):
        tickers = _normalize_tickers(tickers)
        histories, pending = {}, []
        for ticker in tickers:
            memo_key = (ticker, start, end_exclusive)
            if memo_key in self._history_memo:
                if self._history_memo[memo_key] is not None:
                    histories[ticker] = self._history_memo[memo_key]
                continue
            pending.append(ticker)
        for offset in range(0, len(pending), self.config.batch_size):
            batch = pending[offset:offset + self.config.batch_size]
            try:
                raw = self._request(lambda: yf.download(
                    batch, start=start, end=end_exclusive, auto_adjust=False, actions=True,
                    group_by="ticker", progress=False, threads=min(5, len(batch)), timeout=20,
                ), "prices " + ",".join(batch))
            except RuntimeError:
                continue
            if raw is None or raw.empty:
                self.events.append("empty_history: " + ",".join(batch))
                continue
            for ticker in batch:
                try:
                    if isinstance(raw.columns, pd.MultiIndex):
                        level = next(i for i in range(raw.columns.nlevels) if ticker in raw.columns.get_level_values(i))
                        frame = raw.xs(ticker, axis=1, level=level).copy()
                    elif len(batch) == 1:
                        frame = raw.copy()
                    else:
                        raise ValueError("Unexpected multi-ticker column format")
                    frame.index = pd.DatetimeIndex(pd.to_datetime(frame.index)).tz_localize(None).normalize()
                    frame = frame.loc[~frame.index.duplicated(keep="last")].sort_index()
                    frame = frame.loc[(frame.index >= pd.Timestamp(start)) & (frame.index < pd.Timestamp(end_exclusive))]
                    frame = frame.dropna(subset=["Close"])
                    if frame.empty:
                        raise ValueError("No closing prices")
                    frame = frame.drop(columns=["Adj Close"], errors="ignore")
                    frame.index.name = "Date"
                    histories[ticker] = frame
                    self._history_memo[(ticker, start, end_exclusive)] = frame
                except (KeyError, ValueError, StopIteration) as exc:
                    self.events.append("missing_history {}: {}".format(ticker, exc))
        for ticker in tickers:
            self._history_memo.setdefault((ticker, start, end_exclusive), histories.get(ticker))
        return histories


class PortfolioBuilder:
    """Build A/B/C portfolios using a source and a Cointegration instance."""

    def __init__(self, config: PortfolioConfig | None = None, *, source=None):
        self.config = config or PortfolioConfig()
        self.source = source
        self.cointegration = Cointegration(max_lag=self.config.cointegration.max_lag)
        self.result = None

    def training_prices(self, series: pd.Series) -> pd.Series:
        """Isolate training dates before calling either statistical test."""
        config = self.config.cointegration
        values = pd.to_numeric(series.copy(), errors="coerce")
        index = pd.DatetimeIndex(pd.to_datetime(values.index))
        if index.tz is not None:
            index = index.tz_localize(None)
        values.index = index.normalize()
        values = values.loc[~values.index.duplicated(keep="last")].sort_index()
        values = values.loc[config.train_start:config.train_end]
        return values.replace([np.inf, -np.inf], np.nan).where(values > 0).dropna()

    def analyze_pair(self, y: pd.Series, x: pd.Series, *, confidence_level: float | None = None) -> dict:
        """OLS hedge ratio, residual ADF and EG on training prices only."""
        config = self.config.cointegration
        confidence_level = self.config.ab_confidence_level if confidence_level is None else confidence_level
        result = {
            "y": str(y.name), "x": str(x.name), "nobs": 0,
            "test_statistic": np.nan, "critical_value_5pct": np.nan,
            "pvalue": np.nan, "intercept": np.nan, "hedge_ratio": np.nan, "ols_r_squared": np.nan,
            "cointegrated_95": False, "start": None, "end": None,
            "confidence_level": confidence_level, "critical_value": np.nan, "cointegrated": False,
            "passed": False, "reason": "",
        }
        try:
            paired = pd.concat([
                self.training_prices(y).rename("y"),
                self.training_prices(x).rename("x"),
            ], axis=1).dropna()
            result["nobs"] = len(paired)
            if len(paired) < config.min_observations:
                result["reason"] = "insufficient_training_observations"
                return result
            result.update(start=str(paired.index.min().date()), end=str(paired.index.max().date()))
            eg = self.cointegration.engle_granger(paired["y"], paired["x"], confidence_level=confidence_level)
            result.update({key: eg[key] for key in (
                "test_statistic", "critical_value_5pct", "pvalue", "intercept", "hedge_ratio", "ols_r_squared", "cointegrated_95", "used_lag",
                "confidence_level", "critical_value", "cointegrated",
            )})
            for key in ("test_statistic", "critical_value_5pct", "stationary_95", "used_lag"):
                result["residual_adf_" + key] = eg["residual_adf"][key]
            series_to_check = {}
            if config.check_i1:
                series_to_check.update({
                    "y_level": paired["y"], "x_level": paired["x"],
                    "y_diff": paired["y"].diff(), "x_diff": paired["x"].diff(),
                })
            for name, series in series_to_check.items():
                adf = self.cointegration.adf_test(series)
                for key in ("test_statistic", "critical_value_5pct", "stationary_95"):
                    result[name + "_adf_" + key] = adf[key]
            reasons = []
            if config.check_i1:
                if result["y_level_adf_stationary_95"] or result["x_level_adf_stationary_95"]:
                    reasons.append("level_stationary_i1_not_supported")
                if not (result["y_diff_adf_stationary_95"] and result["x_diff_adf_stationary_95"]):
                    reasons.append("difference_nonstationary_i1_not_supported")
            if not eg["cointegrated"]:
                reasons.append("cointegration_not_significant")
            if not result["residual_adf_stationary_95"]:
                reasons.append("residual_adf_not_stationary")
            result["passed"] = not reasons
            result["reason"] = ";".join(reasons) if reasons else "passed"
        except (ValueError, np.linalg.LinAlgError, FloatingPointError) as exc:
            result["reason"] = "test_error: {}".format(exc)
        return result

    @staticmethod
    def _prices(histories, tickers):
        series = [histories[ticker]["Close"].rename(ticker) for ticker in tickers if ticker in histories]
        if not series:
            return pd.DataFrame(index=pd.DatetimeIndex([], name="Date"))
        frame = pd.concat(series, axis=1).sort_index()
        frame.index.name = "Date"
        return frame

    def pair_beta_table(self, triplets: pd.DataFrame, train_prices: pd.DataFrame) -> pd.DataFrame:
        """Export Close-only formation OLS coefficients and EG/ADF diagnostics.

        quote_beta is the formation-end snapshot. Positions refit it through
        the preceding Close when dynamic sizing is selected.
        """
        columns = [
            "anchor_A", "pair", "y_ticker", "x_ticker", "alpha", "beta", "inherited_beta",
            "price_basis", "formation_start", "formation_end", "nobs", "used_lag", "correlation",
            "eg_statistic", "eg_pvalue", "eg_critical_5pct", "eg_critical_10pct",
            "confidence_level", "eg_critical_value", "eg_passed", "eg_passed_5pct", "eg_passed_10pct",
            "residual_adf_statistic", "residual_adf_critical_5pct", "residual_adf_passed_5pct",
            "cointegration_status", "accepted_for_research", "error", "quote_beta", "quote_date",
        ]
        rows = []
        for match in triplets.to_dict("records"):
            for pair, y_ticker, x_ticker, confidence in (
                ("AB", match["A"], match["B"], self.config.ab_confidence_level),
                ("AC", match["A"], match["C"], self.config.ac_confidence_level),
                ("BC", match["B"], match["C"], self.config.bc_confidence_level),
            ):
                row = dict.fromkeys(columns, np.nan)
                row.update(anchor_A=match["A"], pair=pair, y_ticker=y_ticker, x_ticker=x_ticker,
                           price_basis="Close", used_lag=self.cointegration.max_lag,
                           eg_passed=False, eg_passed_5pct=False, eg_passed_10pct=False,
                           residual_adf_passed_5pct=False, confidence_level=confidence,
                           cointegration_status="failed", accepted_for_research=False, error="")
                if pair == "BC" and match["AB_hedge_ratio"] != 0:
                    row["inherited_beta"] = match["AC_hedge_ratio"] / match["AB_hedge_ratio"]
                try:
                    paired = pd.concat([
                        self.training_prices(train_prices[y_ticker]).rename("y"),
                        self.training_prices(train_prices[x_ticker]).rename("x"),
                    ], axis=1).dropna()
                    if len(paired) < self.config.cointegration.min_observations:
                        raise ValueError("insufficient_training_observations")
                    test = self.cointegration.engle_granger(paired["y"], paired["x"], confidence_level=confidence)
                    adf = test["residual_adf"]
                    row.update(formation_start=str(paired.index.min().date()),
                               formation_end=str(paired.index.max().date()), nobs=len(paired),
                               correlation=float(paired["y"].corr(paired["x"])),
                               alpha=test["intercept"], beta=test["hedge_ratio"],
                               quote_beta=test["hedge_ratio"], quote_date=str(paired.index.max().date()),
                               eg_statistic=test["test_statistic"], eg_pvalue=test["pvalue"],
                               eg_critical_5pct=test["critical_value_5pct"], eg_critical_10pct=test["critical_value_10pct"],
                               eg_passed_5pct=test["cointegrated_95"], eg_passed_10pct=test["cointegrated_90"],
                               eg_critical_value=test["critical_value"], eg_passed=test["cointegrated"],
                               residual_adf_statistic=adf["test_statistic"], residual_adf_critical_5pct=adf["critical_value_5pct"],
                               residual_adf_passed_5pct=adf["stationary_95"],
                               accepted_for_research=bool(adf["stationary_95"] and test["cointegrated"]))
                    if adf["stationary_95"] and test["cointegrated_95"]:
                        row["cointegration_status"] = "passed_5pct"
                    elif adf["stationary_95"] and test["cointegrated_90"]:
                        row["cointegration_status"] = "marginal_10pct"
                except (KeyError, ValueError, np.linalg.LinAlgError, FloatingPointError) as exc:
                    row["error"] = str(exc)
                rows.append(row)
        return pd.DataFrame(rows, columns=columns)

    def build(self, portfolio_a_tickers: list[str], *, candidate_overrides: dict[str, list[str]] | None = None,
              target_count: int | None = None, save: bool = True) -> dict:
        """Industry universe -> correlation > 0.90 -> ordered EG/OLS -> first two.

        Correlation, regression and tests all use 2022-2025 formation prices.
        Each A needs two independently passing substitutes. There is no B/C
        cointegration gate and no p-value reranking after correlation ordering.
        """
        config = self.config
        if target_count is not None and target_count < 1:
            raise ValueError("target_count must be positive")
        seeds = _normalize_tickers(portfolio_a_tickers)
        overrides = {_normalize_tickers([key])[0]: _normalize_tickers(value)
                     for key, value in (candidate_overrides or {}).items()}
        source = self.source or YahooFinanceSource(config)
        self.source = source
        output = Path(config.output_dir)
        train = config.cointegration
        formation_end = (pd.Timestamp(train.train_end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        as_of = pd.Timestamp(config.as_of_date) if config.as_of_date else pd.Timestamp.now(tz="America/New_York").tz_localize(None).normalize()
        end_exclusive = min(pd.Timestamp(config.data_end) + pd.Timedelta(days=1), as_of + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        matches, skipped, diagnostics, universe_rows, correlation_rows, seed_profiles = [], [], [], [], [], []
        selected_train_history, used_replacements = {}, set()
        def expand_share_classes(tickers):
            expanded = set(tickers)
            for group in config.equivalent_ticker_groups:
                if expanded.intersection(group):
                    expanded.update(group)
            return expanded

        for seed in seeds:
            if target_count is not None and len(matches) >= target_count:
                break
            if target_count is not None and config.disjoint_portfolios and seed in expand_share_classes(used_replacements | {r["A"] for r in matches}):
                skipped.append({"A": seed, "reason": "already_used_as_replacement", "detail": "Preserve disjoint A/B/C"})
                continue
            LOGGER.info("Screening %s", seed)
            reserved = {row["A"] for row in matches} if target_count is not None else set(seeds)
            excluded = reserved | {seed} if config.disjoint_portfolios else {seed}
            if config.disjoint_portfolios:
                excluded.update(used_replacements)
            excluded = expand_share_classes(excluded)
            try:
                if seed in overrides:
                    info = {"industry": "manual universe", "sector": None}
                    candidates = [{"ticker": ticker, "scope": "manual", "label": "candidate_overrides"}
                                  for ticker in overrides[seed] if ticker not in excluded]
                else:
                    info, candidates = source.get_candidates(seed, excluded)
                candidates = candidates[:config.max_candidates]
                universe_rows.extend(dict(A=seed, **row) for row in candidates)
                seed_profiles.append(dict(A=seed, **info))
            except Exception as exc:
                skipped.append({"A": seed, "reason": "candidate_discovery_error", "detail": str(exc)})
                continue
            if len(candidates) < 2:
                skipped.append({"A": seed, "reason": "fewer_than_two_candidates",
                                "detail": "Fewer than two available industry/related assets"})
                continue

            tickers = [row["ticker"] for row in candidates]
            histories = source.history([seed], train.train_start, formation_end)
            if seed not in histories:
                skipped.append({"A": seed, "reason": "seed_training_data_unavailable", "detail": "No Close training prices"})
                continue
            if len(self.training_prices(histories[seed]["Close"])) < train.min_observations:
                skipped.append({"A": seed, "reason": "insufficient_seed_training_observations", "detail": "Need at least {} formation observations".format(train.min_observations)})
                continue
            histories.update(source.history(tickers, train.train_start, formation_end))
            prices = self._prices(histories, [seed] + tickers)
            if seed not in prices:
                skipped.append({"A": seed, "reason": "seed_training_data_unavailable",
                                "detail": "No Close training prices"})
                continue
            seed_prices = self.training_prices(prices[seed])
            seed_correlations = []
            for candidate in candidates:
                name = candidate["ticker"]
                row = dict(A=seed, **candidate, correlation=np.nan, nobs=0,
                           correlation_rank=None, tested=False, selected=False,
                           status="training_data_unavailable")
                if name in prices:
                    paired = pd.concat([seed_prices.rename("A"),
                                        self.training_prices(prices[name]).rename("candidate")], axis=1).dropna()
                    row["nobs"] = len(paired)
                    if len(paired) >= train.min_observations:
                        values = paired if config.correlation_basis == "price" else paired.pct_change(fill_method=None).dropna()
                        row["correlation"] = float(values["A"].corr(values["candidate"]))
                        row["status"] = ("waiting_for_cointegration"
                                         if np.isfinite(row["correlation"]) and row["correlation"] > config.min_correlation
                                         else "correlation_not_above_threshold")
                    else:
                        row["status"] = "insufficient_training_observations"
                seed_correlations.append(row)
            # Strict > 0.90; rank before running any cointegration test.
            ranked = sorted(
                [row for row in seed_correlations if row["status"] == "waiting_for_cointegration"],
                key=lambda row: (-row["correlation"], row["ticker"]),
            )
            for rank, row in enumerate(ranked, 1):
                row["correlation_rank"] = rank
            accepted = []
            for row in ranked:
                if row["ticker"] in expand_share_classes({item[0]["ticker"] for item in accepted}):
                    row["status"] = "equivalent_share_class_already_accepted"
                    continue
                confidence_level = config.ab_confidence_level if not accepted else config.ac_confidence_level
                test = self.analyze_pair(prices[seed], prices[row["ticker"]], confidence_level=confidence_level)
                row["tested"] = True
                row["status"] = test["reason"]
                diagnostics.append(dict(test, seed=seed, family="A_vs_candidates",
                                        correlation=row["correlation"], correlation_rank=row["correlation_rank"]))
                if test["passed"]:
                    accepted.append((row, test))
                    if len(accepted) == 2:
                        break
            for row in ranked:
                if not row["tested"] and row["status"] == "waiting_for_cointegration":
                    row["status"] = "not_tested_after_two_found"
            correlation_rows.extend(seed_correlations)
            if len(accepted) < 2:
                skipped.append({"A": seed, "reason": "fewer_than_two_cointegrated_assets",
                                "detail": "{} passed; {} candidates above correlation > {}; {} in pool".format(
                                    len(accepted), len(ranked), config.min_correlation, len(candidates))})
                continue

            (b_row, ab), (c_row, ac) = accepted
            b, c = b_row["ticker"], c_row["ticker"]
            b_row["selected"] = c_row["selected"] = True
            row = {"A": seed, "B": b, "C": c, "industry": info.get("industry"), "sector": info.get("sector"),
                   "B_scope": b_row["scope"], "C_scope": c_row["scope"]}
            for prefix, test, candidate in (("AB", ab, b_row), ("AC", ac, c_row)):
                row[prefix + "_correlation"] = candidate["correlation"]
                row.update({prefix + "_" + key: test[key] for key in (
                    "pvalue", "test_statistic", "critical_value_5pct", "intercept", "hedge_ratio",
                    "ols_r_squared", "residual_adf_test_statistic", "residual_adf_critical_value_5pct",
                    "residual_adf_stationary_95", "nobs", "start", "end",
                    "used_lag", "residual_adf_used_lag",
                    "confidence_level", "critical_value", "cointegrated",
                )})
            matches.append(row)
            used_replacements.update([b, c])
            selected_train_history.update({ticker: histories[ticker] for ticker in (seed, b, c)})
            LOGGER.info("Selected %s / %s / %s; model ratios 1 / %.6f / %.6f",
                        seed, b, c, ab["hedge_ratio"], ac["hedge_ratio"])

        matches_df = pd.DataFrame(matches) if matches else pd.DataFrame(columns=MATCH_COLUMNS)
        skipped_df = pd.DataFrame(skipped, columns=["A", "reason", "detail"])
        diagnostic_df = pd.DataFrame(diagnostics) if diagnostics else pd.DataFrame(
            columns=["seed", "family", "y", "x", "correlation", "test_statistic", "critical_value_5pct", "passed", "reason"]
        )
        universe = pd.DataFrame(universe_rows, columns=["A", "ticker", "scope", "label"])
        correlation_df = pd.DataFrame(correlation_rows, columns=[
            "A", "ticker", "scope", "label", "correlation", "nobs", "correlation_rank", "tested", "selected", "status"
        ])
        ticker_lists = {name: matches_df[name].tolist() for name in ("A", "B", "C")}
        selected = list(dict.fromkeys(ticker for name in ("A", "B", "C") for ticker in ticker_lists[name]))
        # 2026 retrieval happens after matching and freezing all OLS hedge ratios.
        holdout_history = source.history(selected, formation_end, end_exclusive) if selected and formation_end < end_exclusive else {}
        train_prices = self._prices(selected_train_history, selected)
        holdout_prices = self._prices(holdout_history, selected).reindex(columns=selected)
        all_prices = pd.concat([train_prices, holdout_prices]).sort_index().reindex(columns=selected)
        all_prices.index.name = "Date"
        combined_history, availability = {}, []
        for ticker in selected:
            chunks = [mapping[ticker] for mapping in (selected_train_history, holdout_history) if ticker in mapping]
            combined_history[ticker] = pd.concat(chunks).sort_index()
            oos = holdout_history.get(ticker, pd.DataFrame())
            availability.append({"ticker": ticker, "holdout_observations": len(oos),
                                 "holdout_last_date": str(oos.index.max().date()) if not oos.empty else None})
        if combined_history:
            ohlcv = pd.concat(combined_history, names=["Ticker", "Date"]).reset_index()
        else:
            ohlcv = pd.DataFrame(columns=["Ticker", "Date", "Open", "High", "Low", "Close",
                                         "Volume", "Dividends", "Stock Splits"])

        portfolios, paths = {}, {}
        member_columns = ["ticker", "anchor_A", "ratio", "intercept", "correlation", "ols_r_squared"]
        for name in ("A", "B", "C"):
            members = []
            for match in matches:
                prefix = "A" + name
                members.append({
                    "ticker": match[name], "anchor_A": match["A"],
                    "ratio": 1.0 if name == "A" else match[prefix + "_hedge_ratio"],
                    "intercept": 0.0 if name == "A" else match[prefix + "_intercept"],
                    "correlation": 1.0 if name == "A" else match[prefix + "_correlation"],
                    "ols_r_squared": 1.0 if name == "A" else match[prefix + "_ols_r_squared"],
                })
            member_df = pd.DataFrame(members, columns=member_columns)
            data = member_df.merge(ohlcv.rename(columns={"Ticker": "ticker"}), on="ticker", how="inner")
            if not data.empty:
                data = data.sort_values(["anchor_A", "Date"])
            portfolio = Portfolio(name=name, members=member_df, data=data)
            portfolios[name] = portfolio

        # Match the ticker-first yfinance layout: (Ticker, Price) columns and
        # one row per date. Keep A, B, C ticker order without a third header level.
        if selected:
            open_close = pd.concat(
                [combined_history[ticker].reindex(columns=["Open", "Close"]) for ticker in selected],
                axis=1, keys=selected, names=["Ticker", "Price"],
            ).sort_index()
        else:
            open_close = pd.DataFrame(
                index=pd.DatetimeIndex([], name="Date"),
                columns=pd.MultiIndex.from_product([[], ["Open", "Close"]], names=["Ticker", "Price"]),
            )
        open_close.index.name = "Date"
        pair_betas = self.pair_beta_table(matches_df, train_prices)

        # Retrieve the benchmark after selection and regression. It is not an
        # A/B/C member and does not enter correlation or cointegration tests.
        if config.benchmark_ticker:
            benchmark = _normalize_tickers([config.benchmark_ticker])[0]
            benchmark_history = source.history([benchmark], train.train_start, end_exclusive)
            if benchmark not in benchmark_history:
                raise RuntimeError("Benchmark prices unavailable: " + benchmark)
            benchmark_prices = benchmark_history[benchmark].loc[:, ["Open", "Close"]]
            if benchmark_prices.dropna().empty:
                raise RuntimeError("Benchmark Open/Close unavailable: " + benchmark)
            if benchmark not in open_close.columns.get_level_values("Ticker"):
                open_close = pd.concat([
                    open_close,
                    pd.concat([benchmark_prices], axis=1, keys=[benchmark], names=["Ticker", "Price"]),
                ], axis=1).sort_index()
                open_close.index.name = "Date"

        metadata = {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "input_portfolio_A": seeds, "portfolios": ticker_lists,
            "target_count": target_count, "accepted_count": len(matches),
            "formation_start_inclusive": train.train_start, "formation_end_inclusive": train.train_end,
            "requested_data_end_inclusive": config.data_end, "download_end_exclusive": end_exclusive,
            "selection_uses_2026": False, "holdout_availability": availability,
            "ratio_convention": "A=1 Close-price unit; B=OLS beta from A~B; C=OLS beta from A~C. These are model ratios, not normalized capital weights or executed share quantities.",
            "yfinance_version": yf.__version__, "config": asdict(config),
            "package_versions": {name: version(name) for name in ("yfinance", "numpy", "pandas", "scipy", "statsmodels")},
            "universe_is_point_in_time": False,
            "universe_note": "Current Yahoo industry/sector snapshot; configured related industries/tickers are candidate expansions, not verified company-level supply-chain links.",
            "selection_note": "Training Pearson correlation strictly above the configured threshold, descending order; explicit OLS, EG and residual ADF; stop after two passing substitutes, otherwise skip A.",
            "adf_threshold": -2.86,
            "open_close_note": "Vendor Open and Close from auto_adjust=False; formation and dynamic OLS use Close only.",
            "benchmark_ticker": config.benchmark_ticker,
            "pair_beta_equations": {"AB": "A = alpha + beta*B + residual", "AC": "A = alpha + beta*C + residual", "BC": "B = alpha + beta*C + residual"},
            "eg_confidence_levels": {"AB": config.ab_confidence_level, "AC": config.ac_confidence_level, "BC": config.bc_confidence_level},
            "bc_policy": "Post-selection diagnostic at configured BC confidence; preserve separate 5% and 10% EG flags. Residual ADF remains at the fixed -2.86 threshold.",
            "lag_selection": "fixed; autolag=None for EG and ADF",
            "adf_note": "Fixed approximate large-sample 5% threshold for ADF with intercept and no time trend. EG retains its own cointegration critical value.",
            "tax_status": "not_evaluated", "source_events": getattr(source, "events", []),
        }
        if save:
            output.mkdir(parents=True, exist_ok=True)
            for name, table in (("portfolios_open_close", open_close), ("pair_betas", pair_betas)):
                paths[name] = output / (name + ".csv")
                table.to_csv(paths[name], index=(name == "portfolios_open_close"), encoding="utf-8-sig")
        self.result = {"portfolios": portfolios, "ticker_lists": ticker_lists, "triplets": matches_df,
                "open_close": open_close, "pair_betas": pair_betas,
                "skipped": skipped_df, "diagnostics": diagnostic_df,
                "seed_metadata": pd.DataFrame(seed_profiles),
                "candidate_universe": universe, "candidate_correlations": correlation_df,
                "train_prices": train_prices, "holdout_prices": holdout_prices,
                "prices": all_prices, "metadata": metadata, "paths": paths}
        return self.result
