"""Build a liquid, Close-based pair candidate pool without A/B/C triplets."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
import logging
import re

import numpy as np
import pandas as pd
import yfinance as yf

from portfolio import CointegrationConfig, PortfolioBuilder, PortfolioConfig, YahooFinanceSource, _key

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class PairPoolConfig:
    target_pairs: int = 300
    max_assets: int = 1000
    min_market_cap: float = 10_000_000_000
    min_average_dollar_volume: float = 20_000_000
    liquidity_window: int = 60
    min_price: float = 5.0
    reference_long_amount: float = 100_000
    max_order_fraction_adv: float = 0.005
    min_correlation: float = 0.90
    confidence_level: float = 0.95
    batch_size: int = 20
    extra_equivalent_ticker_groups: tuple[tuple[str, ...], ...] = (
        ("PBR", "PBR-A"), ("LEN", "LEN-B"),
    )
    cointegration: CointegrationConfig = field(default_factory=CointegrationConfig)
    relationship_file: Path = Path("industry_relationships.json")
    output_file: Path = Path("data/pair_pool.csv")

    def __post_init__(self):
        for name in ("target_pairs", "max_assets", "liquidity_window", "batch_size"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(name + " must be a positive integer")
        if self.max_assets < 2:
            raise ValueError("max_assets must be at least 2")
        for name in ("min_market_cap", "min_average_dollar_volume", "min_price", "reference_long_amount"):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(name + " must be positive and finite")
        if not 0 < self.max_order_fraction_adv <= 1 or not 0 < self.min_correlation < 1:
            raise ValueError("Use an order fraction in (0, 1] and correlation in (0, 1)")
        if self.confidence_level not in (0.90, 0.95):
            raise ValueError("confidence_level must be 0.90 or 0.95")

    def source_config(self):
        return PortfolioConfig.from_relationship_file(
            self.relationship_file, cointegration=self.cointegration,
            min_correlation=self.min_correlation, batch_size=self.batch_size,
            ab_confidence_level=self.confidence_level,
        )


class PairPoolSource(YahooFinanceSource):
    """Discover current large-cap equities, reusing the existing price downloader."""

    def __init__(self, config: PairPoolConfig):
        super().__init__(config.source_config())
        self.pool_config = config

    def universe(self):
        sample = yf.EquityQuery("eq", ["region", "us"])
        labels = sample.valid_values["industry"]
        if isinstance(labels, dict):
            labels = [label for group in labels.values() for label in group]
        labels = sorted(set(labels))
        retrieved = datetime.now(timezone.utc).isoformat()
        rows = []
        for number, industry in enumerate(labels, 1):
            query = yf.EquityQuery("and", [
                yf.EquityQuery("eq", ["region", "us"]),
                yf.EquityQuery("is-in", ["exchange", "NYQ", "NMS", "NGM", "NCM", "ASE"]),
                yf.EquityQuery("eq", ["industry", industry]),
                yf.EquityQuery("gt", ["intradaymarketcap", self.pool_config.min_market_cap]),
            ])
            offset = 0
            seen = set()
            while True:
                try:
                    response = self._request(lambda: yf.screen(
                        query, offset=offset, size=250,
                        sortField="intradaymarketcap", sortAsc=False,
                    ), "pair universe " + industry)
                except RuntimeError:
                    break
                quotes = response.get("quotes", [])
                fresh = [q for q in quotes if q.get("symbol") not in seen]
                if not fresh:
                    break
                for quote in fresh:
                    ticker = str(quote.get("symbol", "")).upper()
                    seen.add(ticker)
                    market_cap = quote.get("marketCap")
                    if (not re.fullmatch(r"[A-Z0-9\-]{1,24}", ticker)
                            or re.search(r"-P[A-Z0-9]+$", ticker)
                            or ticker.endswith(("-WT", "-WS"))
                            or quote.get("quoteType") != "EQUITY"
                            or quote.get("currency") != "USD"
                            or market_cap is None or not np.isfinite(market_cap)
                            or market_cap < self.pool_config.min_market_cap):
                        continue
                    rows.append({"ticker": ticker, "name": quote.get("shortName", ticker),
                                 "industry": industry, "market_cap": float(market_cap),
                                 "market_cap_asof": retrieved})
                offset += len(quotes)
                if len(quotes) < 250 or offset >= response.get("total", offset):
                    break
            if number % 10 == 0 or number == len(labels):
                LOGGER.info("Industry screens %s/%s; %s equity records", number, len(labels), len(rows))
        if not rows:
            raise RuntimeError("No large-cap equity metadata returned; inspect source.events.")
        frame = pd.DataFrame(rows).sort_values("market_cap", ascending=False, kind="stable")
        frame = frame.drop_duplicates("ticker")
        return frame.head(self.pool_config.max_assets).reset_index(drop=True)


class PairPoolBuilder:
    """Select unique pairs on formation data; overlapping candidates are allowed."""

    def __init__(self, config: PairPoolConfig | None = None, *, source=None):
        self.config = config or PairPoolConfig()
        self.source = source if source is not None else PairPoolSource(self.config)
        self.legacy_config = self.config.source_config()
        self.tester = PortfolioBuilder(self.legacy_config)
        self.related = {
            _key(industry): {_key(name) for name in names}
            for industry, names in self.legacy_config.related_industries.items()
        }

    def build(self, *, save=True):
        config, train = self.config, self.config.cointegration
        universe = self.source.universe().copy()
        required = {"ticker", "industry", "market_cap", "market_cap_asof"}
        if not required.issubset(universe.columns) or universe.ticker.duplicated().any():
            raise ValueError("Universe needs unique tickers, industries and market-cap provenance.")
        universe = universe.loc[universe.market_cap.ge(config.min_market_cap)].copy()
        universe = universe.sort_values("market_cap", ascending=False, kind="stable").head(config.max_assets)
        end = (pd.Timestamp(train.train_end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        histories = self.source.history(universe.ticker.tolist(), train.train_start, end)
        assets, close_series, rejected = [], [], Counter()
        for row in universe.to_dict("records"):
            frame = histories.get(row["ticker"])
            if frame is None or not {"Close", "Volume"}.issubset(frame.columns):
                rejected["missing_close_or_volume"] += 1
                continue
            frame = frame.sort_index().loc[train.train_start:train.train_end, ["Close", "Volume"]]
            close = self.tester.training_prices(frame.Close)
            if len(close) < max(train.min_observations, config.liquidity_window):
                rejected["insufficient_training_history"] += 1
                continue
            tail = frame.reindex(close.index).tail(config.liquidity_window)
            if (pd.Timestamp(train.train_end) - close.index[-1]).days > 10:
                rejected["stale_formation_prices"] += 1
                continue
            if close.iloc[-1] < config.min_price:
                rejected["price_below_minimum"] += 1
                continue
            if close.iloc[-1] > config.reference_long_amount:
                rejected["whole_share_budget_too_small"] += 1
                continue
            volume = pd.to_numeric(tail.Volume, errors="coerce")
            if not (np.isfinite(volume) & volume.gt(0)).all():
                rejected["invalid_formation_volume"] += 1
                continue
            adv = float((tail.Close * volume).mean())
            if adv < config.min_average_dollar_volume:
                rejected["dollar_volume_below_minimum"] += 1
                continue
            row.update(average_dollar_volume=adv, liquidity_date=str(close.index[-1].date()),
                       formation_close=float(close.iloc[-1]))
            assets.append(row)
            close_series.append(close.rename(row["ticker"]))
        if len(assets) < 2:
            raise RuntimeError("Fewer than two assets passed formation liquidity/history screening.")
        assets = pd.DataFrame(assets).set_index("ticker", drop=False)
        # Retain the share class with the highest formation-period dollar volume.
        for group in self.legacy_config.equivalent_ticker_groups + config.extra_equivalent_ticker_groups:
            present = assets.index.intersection(group)
            if len(present) > 1:
                keep = assets.loc[present, "average_dollar_volume"].idxmax()
                assets = assets.drop(index=[ticker for ticker in present if ticker != keep])
        assets = assets.sort_values("market_cap", ascending=False, kind="stable")
        prices = pd.concat(close_series, axis=1).sort_index()
        prices = prices.loc[:, assets.index]
        correlation = prices.corr(min_periods=train.min_observations)
        tickers = assets.index.tolist()
        candidates = []
        for i, y in enumerate(tickers):
            industry_y = _key(assets.at[y, "industry"])
            for x in tickers[i + 1:]:
                industry_x = _key(assets.at[x, "industry"])
                same = industry_y == industry_x
                related = industry_x in self.related.get(industry_y, set()) or industry_y in self.related.get(industry_x, set())
                if not same and not related:
                    continue
                corr = correlation.at[y, x]
                if np.isfinite(corr) and corr > config.min_correlation:
                    candidates.append((float(corr), y, x, "same_industry" if same else "related_industry"))
        candidates.sort(key=lambda item: (-item[0], item[1], item[2]))
        accepted, pair_rejected, tested = [], Counter(), 0
        LOGGER.info("%s discovered assets; %s liquid assets; %s correlated industry/related pairs",
                    len(universe), len(assets), len(candidates))
        for corr, y, x, scope in candidates:
            tested += 1
            test = self.tester.analyze_pair(prices[y], prices[x], confidence_level=config.confidence_level)
            if not test["passed"] or test["hedge_ratio"] <= 0:
                pair_rejected[test["reason"] if not test["passed"] else "nonpositive_beta"] += 1
                continue
            paired = prices[[y, x]].dropna()
            alpha, beta = test["intercept"], test["hedge_ratio"]
            spread = paired[y] - alpha - beta * paired[x]
            std = float(spread.std(ddof=1))
            reference_short = config.reference_long_amount * beta * paired[x].iloc[-1] / paired[y].iloc[-1]
            reverse_short = config.reference_long_amount * paired[y].iloc[-1] / (beta * paired[x].iloc[-1])
            participation_y = max(config.reference_long_amount, reverse_short) / assets.at[y, "average_dollar_volume"]
            participation_x = max(config.reference_long_amount, reference_short) / assets.at[x, "average_dollar_volume"]
            if not np.isfinite(std) or std <= 1e-10:
                pair_rejected["degenerate_spread"] += 1
                continue
            if max(participation_y, participation_x) > config.max_order_fraction_adv:
                pair_rejected["reference_order_exceeds_adv_limit"] += 1
                continue
            row = dict(pair_id=y + "__" + x, y_ticker=y, x_ticker=x, scope=scope,
                       alpha=alpha, beta=beta, correlation=corr, price_basis="Close",
                       formation_start=test["start"], formation_end=test["end"], nobs=test["nobs"],
                       used_lag=test["used_lag"], confidence_level=config.confidence_level,
                       eg_statistic=test["test_statistic"], eg_pvalue=test["pvalue"],
                       eg_critical_value=test["critical_value"],
                       residual_adf_statistic=test["residual_adf_test_statistic"],
                       residual_adf_critical_value=-2.86, spread_mean=float(spread.mean()), spread_std=std,
                       reference_long_amount=config.reference_long_amount,
                       reference_short_amount=float(reference_short),
                       reference_reverse_short_amount=float(reverse_short),
                       reference_participation_y=float(participation_y),
                       reference_participation_x=float(participation_x),
                       max_order_fraction_adv=config.max_order_fraction_adv)
            for suffix, ticker in (("y", y), ("x", x)):
                for column in ("industry", "market_cap", "market_cap_asof", "average_dollar_volume", "liquidity_date", "formation_close"):
                    row[column + "_" + suffix] = assets.at[ticker, column]
            accepted.append(row)
            if len(accepted) % 25 == 0:
                LOGGER.info("Accepted %s/%s pairs after %s tests", len(accepted), config.target_pairs, tested)
            if len(accepted) >= config.target_pairs:
                break
        pairs = pd.DataFrame(accepted)
        if pairs.empty:
            raise RuntimeError("No pairs passed; inspect the correlation, liquidity and EG/ADF requirements.")
        if len(pairs) < config.target_pairs:
            LOGGER.warning("Found %s of target %s pairs; thresholds were not relaxed.", len(pairs), config.target_pairs)
        output = None
        if save:
            output = Path(config.output_file)
            output.parent.mkdir(parents=True, exist_ok=True)
            pairs.to_csv(output, index=False)
        return {"pairs": pairs, "universe": universe, "eligible_assets": assets.reset_index(drop=True),
                "candidate_count": len(candidates), "tested_count": tested,
                "asset_rejections": dict(rejected), "pair_rejections": dict(pair_rejected),
                "events": list(self.source.events), "output_file": output,
                "target_reached": len(pairs) >= config.target_pairs}
