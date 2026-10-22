"""Build a liquid, Close-based pair candidate pool without A/B/C triplets."""

from __future__ import annotations

from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
import logging
import json
import re

import numpy as np
import pandas as pd
import yfinance as yf

from market_data import CointegrationConfig, PairAnalyzer, PairResearchConfig, YahooFinanceSource, _key

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class PairPoolConfig:
    target_pairs: int = 300
    target_commodity_pairs: int = 50
    target_other_etf_pairs: int = 50
    max_etf_pairs_per_group: int = 5
    include_equities: bool = True
    include_etfs: bool = True
    require_etf_i1: bool = True
    max_assets: int = 1000
    min_market_cap: float = 10_000_000_000
    min_etf_assets: float = 100_000_000
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
    cointegration: CointegrationConfig = field(
        default_factory=lambda: CointegrationConfig(train_end="2024-12-31"))
    relationship_file: Path = Path("industry_relationships.json")
    etf_catalog_file: Path = Path("etf_universe.json")
    output_file: Path = Path("data/pair_pool.csv")

    def __post_init__(self):
        for name in ("target_pairs", "target_commodity_pairs", "target_other_etf_pairs",
                     "max_etf_pairs_per_group", "max_assets", "liquidity_window", "batch_size"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(name + " must be a positive integer")
        if self.max_assets < 2:
            raise ValueError("max_assets must be at least 2")
        for name in ("min_market_cap", "min_etf_assets", "min_average_dollar_volume", "min_price", "reference_long_amount"):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(name + " must be positive and finite")
        if not 0 < self.max_order_fraction_adv <= 1 or not 0 < self.min_correlation < 1:
            raise ValueError("Use an order fraction in (0, 1] and correlation in (0, 1)")
        if self.confidence_level not in (0.90, 0.95):
            raise ValueError("confidence_level must be 0.90 or 0.95")
        if not self.include_equities and not self.include_etfs:
            raise ValueError("Enable equities, ETFs or both.")

    def source_config(self):
        return PairResearchConfig.from_relationship_file(
            self.relationship_file, cointegration=self.cointegration,
            batch_size=self.batch_size,
            confidence_level=self.confidence_level,
        )


class PairPoolSource(YahooFinanceSource):
    """Discover current equities and curated ETFs, reusing the price downloader."""

    def __init__(self, config: PairPoolConfig):
        super().__init__(config.source_config())
        self.pool_config = config
        self.etf_catalog = pd.DataFrame()

    def universe(self):
        frames = []
        if self.pool_config.include_equities:
            frame = self.equity_universe()
            frame["instrument_type"] = "EQUITY"
            frame["pair_family"] = "equity"
            frame["asset_class"] = "equity"
            frames.append(frame)
        if self.pool_config.include_etfs:
            frames.append(self.etf_universe())
        frames = [frame for frame in frames if not frame.empty]
        if not frames:
            raise RuntimeError("No equity or ETF metadata returned.")
        return pd.concat(frames, ignore_index=True)

    def etf_universe(self):
        payload = json.loads(Path(self.pool_config.etf_catalog_file).read_text(encoding="utf-8"))
        entries = []
        for group in payload["groups"]:
            for ticker in group["tickers"]:
                entries.append({"ticker": ticker, "industry": "ETF: " + group["label"],
                                "instrument_type": "ETF", "pair_family": group["pair_family"],
                                "asset_class": group["asset_class"], "exposure_group": group["id"],
                                "implementation": group["implementation"],
                                "tracking_pair_group": group.get("tracking_pair_group", False),
                                "source_urls": " | ".join(group["source_urls"])})
        self.etf_catalog = pd.DataFrame(entries)
        if self.etf_catalog.ticker.duplicated().any():
            raise ValueError("ETF catalog tickers must be unique.")
        def fetch(entry):
            ticker = entry["ticker"]
            try:
                info = self._request(lambda: yf.Ticker(ticker).get_info(), "ETF metadata " + ticker)
            except RuntimeError:
                return None
            if info.get("quoteType") != "ETF" or info.get("currency") != "USD":
                self.events.append("ETF metadata excluded: " + ticker + " (type/currency)")
                return None
            row = dict(entry)
            row.update(name=info.get("shortName", ticker), market_cap=np.nan, market_cap_asof="",
                       fund_assets=info.get("totalAssets") or info.get("netAssets", np.nan),
                       fund_assets_asof=datetime.now(timezone.utc).isoformat(),
                       fund_family=info.get("fundFamily", ""))
            return row
        with ThreadPoolExecutor(max_workers=4) as executor:
            rows = [row for row in executor.map(fetch, entries) if row is not None]
        LOGGER.info("ETF metadata: %s/%s catalog entries returned USD ETF records", len(rows), len(entries))
        return pd.DataFrame(rows)

    def equity_universe(self):
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
        self.research_config = self.config.source_config()
        self.tester = PairAnalyzer(self.research_config)
        self.related = {
            _key(industry): {_key(name) for name in names}
            for industry, names in self.research_config.related_industries.items()
        }
        self.etf_related = {}
        if self.config.include_etfs:
            payload = json.loads(Path(self.config.etf_catalog_file).read_text(encoding="utf-8"))
            for group in payload.get("related_groups", []):
                for name in group:
                    self.etf_related.setdefault(name, set()).update(x for x in group if x != name)

    def build(self, *, save=True):
        config, train = self.config, self.config.cointegration
        universe = self.source.universe().copy()
        defaults = {"instrument_type": "EQUITY", "pair_family": "equity", "asset_class": "equity",
                    "exposure_group": "", "implementation": "equities", "tracking_pair_group": False,
                    "fund_assets": np.nan, "fund_assets_asof": "", "fund_family": "", "source_urls": "",
                    "market_cap": np.nan, "market_cap_asof": ""}
        for column, default in defaults.items():
            if column not in universe:
                universe[column] = default
            elif isinstance(default, bool):
                universe[column] = universe[column].eq(True)
            else:
                universe[column] = universe[column].fillna(default)
        required = {"ticker", "industry", "market_cap", "market_cap_asof"}
        if not required.issubset(universe.columns) or universe.ticker.duplicated().any():
            raise ValueError("Universe needs unique tickers, industries and market-cap provenance.")
        discovered = universe.copy()
        equities = universe.loc[universe.instrument_type.eq("EQUITY") & universe.market_cap.ge(config.min_market_cap)]
        equities = equities.sort_values("market_cap", ascending=False, kind="stable").head(config.max_assets)
        funds = universe.loc[universe.instrument_type.eq("ETF") & pd.to_numeric(universe.fund_assets, errors="coerce").ge(config.min_etf_assets)]
        funds = funds.sort_values("fund_assets", ascending=False, kind="stable")
        if not config.include_equities:
            equities = equities.iloc[:0]
        if not config.include_etfs:
            funds = funds.iloc[:0]
        universe = pd.concat([equities, funds], ignore_index=True)
        end = (pd.Timestamp(train.train_end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        histories = self.source.history(universe.ticker.tolist(), train.train_start, end)
        assets, close_series, rejected = [], [], Counter()
        rejected["ETF_missing_or_small_AUM"] = int(discovered.instrument_type.eq("ETF").sum() - len(funds))
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
            if row["instrument_type"] == "ETF" and config.require_etf_i1:
                try:
                    level = self.tester.cointegration.adf_test(close)
                    difference = self.tester.cointegration.adf_test(close.diff())
                except (ValueError, np.linalg.LinAlgError, FloatingPointError):
                    rejected["ETF_I1_test_error"] += 1
                    continue
                if level["stationary_95"] or not difference["stationary_95"]:
                    rejected["ETF_price_not_supported_as_I1"] += 1
                    continue
                row.update(level_adf_statistic=level["test_statistic"],
                           difference_adf_statistic=difference["test_statistic"])
            row.update(average_dollar_volume=adv, liquidity_date=str(close.index[-1].date()),
                       formation_close=float(close.iloc[-1]))
            assets.append(row)
            close_series.append(close.rename(row["ticker"]))
        if len(assets) < 2:
            raise RuntimeError("Fewer than two assets passed formation liquidity/history screening.")
        assets = pd.DataFrame(assets).set_index("ticker", drop=False)
        # Retain the share class with the highest formation-period dollar volume.
        for group in self.research_config.equivalent_ticker_groups + config.extra_equivalent_ticker_groups:
            present = assets.index.intersection(group)
            if len(present) > 1:
                keep = assets.loc[present, "average_dollar_volume"].idxmax()
                assets = assets.drop(index=[ticker for ticker in present if ticker != keep])
        assets["size_rank"] = assets.market_cap.where(assets.instrument_type.eq("EQUITY"), assets.fund_assets)
        assets = assets.sort_values("size_rank", ascending=False, kind="stable")
        prices = pd.concat(close_series, axis=1).sort_index()
        prices = prices.loc[:, assets.index]
        correlation = prices.corr(min_periods=train.min_observations)
        tickers = assets.index.tolist()
        candidates = []
        for i, y in enumerate(tickers):
            industry_y = _key(assets.at[y, "industry"])
            for x in tickers[i + 1:]:
                if assets.at[y, "pair_family"] != assets.at[x, "pair_family"]:
                    continue
                family = assets.at[y, "pair_family"]
                industry_x = _key(assets.at[x, "industry"])
                if family == "equity":
                    same = industry_y == industry_x
                    related = industry_x in self.related.get(industry_y, set()) or industry_y in self.related.get(industry_x, set())
                    scope = "same_industry" if same else "related_industry"
                else:
                    group_y, group_x = assets.at[y, "exposure_group"], assets.at[x, "exposure_group"]
                    same = group_y == group_x
                    related = group_x in self.etf_related.get(group_y, set())
                    scope = "same_etf_exposure" if same else "related_etf_exposure"
                if not same and not related:
                    continue
                corr = correlation.at[y, x]
                if np.isfinite(corr) and corr > config.min_correlation:
                    candidates.append((float(corr), y, x, scope, family))
        candidates.sort(key=lambda item: (-item[0], item[1], item[2]))
        accepted, pair_rejected, tested = [], Counter(), 0
        targets = {"equity": config.target_pairs, "commodity_etf": config.target_commodity_pairs,
                   "other_etf": config.target_other_etf_pairs}
        enabled = ({"equity"} if config.include_equities else set()) | ({"commodity_etf", "other_etf"} if config.include_etfs else set())
        counts, exposure_counts = Counter(), Counter()
        LOGGER.info("%s discovered assets; %s liquid assets; %s correlated industry/related pairs",
                    len(universe), len(assets), len(candidates))
        for corr, y, x, scope, family in candidates:
            if counts[family] >= targets[family]:
                continue
            exposure_key = tuple(sorted((assets.at[y, "exposure_group"], assets.at[x, "exposure_group"])))
            if family != "equity" and exposure_counts[exposure_key] >= config.max_etf_pairs_per_group:
                continue
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
            tracking_pair = (family != "equity" and scope == "same_etf_exposure"
                             and bool(assets.at[y, "tracking_pair_group"]))
            row = dict(pair_id=y + "__" + x, y_ticker=y, x_ticker=x, scope=scope, pair_family=family,
                       tracking_pair=tracking_pair,
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
                for column in ("industry", "market_cap", "market_cap_asof", "fund_assets", "fund_assets_asof",
                               "instrument_type", "asset_class", "exposure_group", "implementation", "source_urls",
                               "fund_family", "average_dollar_volume", "liquidity_date", "formation_close",
                               "level_adf_statistic", "difference_adf_statistic"):
                    row[column + "_" + suffix] = assets.loc[ticker].get(column, np.nan)
            accepted.append(row)
            counts[family] += 1
            if family != "equity":
                exposure_counts[exposure_key] += 1
            if len(accepted) % 25 == 0:
                LOGGER.info("Accepted %s total pairs; families=%s; after %s tests", len(accepted), dict(counts), tested)
            if all(counts[name] >= targets[name] for name in enabled):
                break
        pairs = pd.DataFrame(accepted)
        if pairs.empty:
            raise RuntimeError("No pairs passed; inspect the correlation, liquidity and EG/ADF requirements.")
        for name in sorted(enabled):
            if counts[name] < targets[name]:
                LOGGER.warning("%s: found %s of target %s; thresholds were not relaxed.", name, counts[name], targets[name])
        output = None
        if save:
            output = Path(config.output_file)
            output.parent.mkdir(parents=True, exist_ok=True)
            pairs.to_csv(output, index=False)
        catalog = getattr(self.source, "etf_catalog", pd.DataFrame())
        coverage = pd.DataFrame()
        if not catalog.empty:
            keys = ["pair_family", "exposure_group"]
            coverage = catalog.groupby(keys).size().rename("catalog_funds").to_frame()
            for column, frame in (("metadata_funds", discovered), ("size_eligible_funds", universe), ("formation_eligible_funds", assets)):
                count = frame.loc[frame.instrument_type.eq("ETF")].groupby(keys).size()
                coverage[column] = count.reindex(coverage.index, fill_value=0)
            coverage = coverage.reset_index()
            coverage["accepted_pairs_involving_group"] = [
                int(((pairs.exposure_group_y.eq(g) | pairs.exposure_group_x.eq(g)) & pairs.pair_family.eq(f)).sum())
                for f, g in coverage[keys].itertuples(index=False, name=None)]
        return {"pairs": pairs, "universe": universe, "eligible_assets": assets.reset_index(drop=True),
                "candidate_count": len(candidates), "tested_count": tested,
                "asset_rejections": dict(rejected), "pair_rejections": dict(pair_rejected),
                "family_counts": dict(counts), "targets": {name: targets[name] for name in enabled},
                "etf_coverage": coverage,
                "events": list(self.source.events), "output_file": output,
                "target_reached": all(counts[name] >= targets[name] for name in enabled)}
