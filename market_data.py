"""Shared Yahoo price downloads and formation-only pair analysis."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import json
import re
import time
import unicodedata

import numpy as np
import pandas as pd
import yfinance as yf
from cointegration import Cointegration

@dataclass(frozen=True)
class CointegrationConfig:
    train_start: str = "2022-01-01"
    train_end: str = "2024-12-31"  # inclusive
    min_observations: int = 750
    max_lag: int = 1
    check_i1: bool = False

    def __post_init__(self):
        if pd.Timestamp(self.train_start) >= pd.Timestamp(self.train_end):
            raise ValueError("train_start must be earlier than train_end")
        if self.min_observations < 20 or self.max_lag < 0:
            raise ValueError("min_observations >= 20 and max_lag >= 0 required")


@dataclass(frozen=True)
class PairResearchConfig:
    cointegration: CointegrationConfig = field(default_factory=CointegrationConfig)
    batch_size: int = 20
    confidence_level: float = 0.95
    related_industries: dict[str, list[str]] = field(default_factory=dict)
    equivalent_ticker_groups: tuple[tuple[str, ...], ...] = (
        ("GOOG", "GOOGL"), ("BRK-A", "BRK-B"), ("FOX", "FOXA"),
        ("NWS", "NWSA"), ("UA", "UAA"), ("HEI", "HEI-A"), ("BF-A", "BF-B"),
    )

    @classmethod
    def from_relationship_file(cls, path: Path, **settings):
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        related = {}
        for group in payload["groups"]:
            for industry in group["industries"]:
                peers = related.setdefault(industry, [])
                peers.extend(name for name in group["industries"] if name != industry and name not in peers)
        return cls(related_industries=related, **settings)

    def __post_init__(self):
        if self.batch_size < 1 or self.confidence_level not in (0.90, 0.95):
            raise ValueError("Use a positive batch size and confidence level 0.90 or 0.95.")


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
    """Download raw Open/Close/Volume in small batches; cache only in memory."""

    def __init__(self, config: PairResearchConfig | None = None):
        self.config = config or PairResearchConfig()
        self.events = []
        self._history_memo = {}

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
                    frame = frame.dropna(subset=[c for c in ("Open", "Close") if c in frame], how="all")
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


class PairAnalyzer:
    """Restrict prices to formation dates, then apply OLS, EG and ADF."""

    def __init__(self, config: PairResearchConfig):
        self.config = config
        self.cointegration = Cointegration(max_lag=config.cointegration.max_lag)

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
        confidence_level = self.config.confidence_level if confidence_level is None else confidence_level
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
