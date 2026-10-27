"""Convert stock scores to signed target weights, independently of execution."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from collections.abc import Mapping

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ConstructionConfig:
    method: str = "demean_l1"
    group_by: str | None = None
    gross_exposure: float = 1.0  # NAV gross target (demean_l1), or cap (aqr_style).
    factor_columns: tuple[str, ...] = ("value", "quality", "momentum")
    risk_lookback: int = 252
    risk_min_observations: int = 120
    risk_lag_months: int = 1
    covariance_shrinkage: float = 0.1
    min_group_size: int = 2
    risk_aversion: float = 5.0
    alpha_scale: float = 1.0
    horizon_years: float = 1 / 12
    net_exposure: float = 0.0
    target_beta: float | None = 0.0
    max_weight: float = 0.05
    transaction_cost_rate: float = 0.0005
    borrow_fee_rate: float = 0.01

    def __post_init__(self):
        if self.method not in ("demean_l1", "aqr_style"):
            raise ValueError("Construction method must be demean_l1 or aqr_style")
        if self.group_by not in (None, "sector", "industry"):
            raise ValueError("Construction group_by must be sector, industry or None")
        if (isinstance(self.gross_exposure, bool) or not np.isfinite(self.gross_exposure)
                or self.gross_exposure <= 0):
            raise ValueError("gross_exposure must be positive and finite")
        for name in ("risk_aversion", "horizon_years", "max_weight"):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(name + " must be positive and finite")
        for name in ("alpha_scale", "transaction_cost_rate", "borrow_fee_rate"):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(name + " must be nonnegative and finite")
        for name, lower in (("risk_lookback", 3), ("risk_min_observations", 3), ("risk_lag_months", 0), ("min_group_size", 2)):
            if isinstance(getattr(self, name), bool) or not isinstance(getattr(self, name), int) or getattr(self, name) < lower:
                raise ValueError(name + " must be an integer of at least " + str(lower))
        if self.risk_min_observations > self.risk_lookback or not 0 <= self.covariance_shrinkage <= 1:
            raise ValueError("Invalid risk window or covariance shrinkage")
        if not np.isfinite(self.net_exposure) or abs(self.net_exposure) > self.gross_exposure:
            raise ValueError("Absolute net exposure must not exceed gross exposure")
        if self.target_beta is not None and not np.isfinite(self.target_beta):
            raise ValueError("target_beta must be finite or None")
        if not self.factor_columns or len(set(self.factor_columns)) != len(self.factor_columns):
            raise ValueError("Configure distinct factor columns")


@dataclass
class RiskModel:
    covariance: pd.DataFrame  # Annual covariance, in decimal-return units.
    beta: pd.Series
    as_of: pd.Timestamp
    source: str = "supplied"

    @staticmethod
    def cutoff(date, lag_months):
        date = pd.Timestamp(date).normalize().tz_localize(None)
        return min(date, (date.to_period("M") - lag_months).end_time.normalize())

    @classmethod
    def estimate(cls, prices, tickers, date, config, market="SPY"):
        """Estimate only from past split-consistent Close; never fill missing bars."""
        cutoff = cls.cutoff(date, config.risk_lag_months)
        prices = prices.copy()
        prices.index = pd.DatetimeIndex(prices.index).tz_localize(None).normalize()
        if prices.index.duplicated().any() or prices.columns.duplicated().any() or market not in prices:
            raise ValueError("Risk prices need unique dates/tickers and the market column")
        window = prices.sort_index().loc[:cutoff].tail(config.risk_lookback + 1)
        returns = window.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).where(lambda x: x > 0).pct_change(fill_method=None).iloc[1:]
        if len(returns) < config.risk_min_observations:
            raise ValueError("Insufficient price history before the lagged risk cutoff")
        tail = returns.tail(config.risk_min_observations)
        symbols = [ticker for ticker in tickers if ticker in returns and tail[ticker].notna().all()
                   and returns[ticker].std() > 1e-12]
        if not tail[market].notna().all() or returns[market].std() <= 1e-12:
            raise ValueError("Insufficient market returns for beta estimation")
        if not symbols:
            return cls(pd.DataFrame(), pd.Series(dtype=float), cutoff, "historical_close")
        common = returns[list(dict.fromkeys(symbols + [market]))].dropna()
        sample = common[symbols].cov().to_numpy() * 252
        covariance = (1 - config.covariance_shrinkage) * sample + config.covariance_shrinkage * np.diag(np.diag(sample))
        beta = common[symbols].apply(lambda values: values.cov(common[market]) / common[market].var())
        return cls(pd.DataFrame(covariance, index=symbols, columns=symbols), beta,
                   common.index.max(), "historical_close")

    def aligned(self, symbols, date, config):
        from scipy.linalg import eigvalsh
        stamp = pd.Timestamp(self.as_of).tz_localize(None)
        if pd.isna(stamp) or stamp.normalize() > self.cutoff(date, config.risk_lag_months):
            raise ValueError("Risk model is later than the permitted lagged cutoff")
        if self.covariance.index.duplicated().any() or self.covariance.columns.duplicated().any() or self.beta.index.duplicated().any():
            raise ValueError("Risk-model tickers must be unique")
        covariance = self.covariance.loc[symbols, symbols].to_numpy(dtype=float)
        beta = self.beta.reindex(symbols).to_numpy(dtype=float)
        if not np.isfinite(covariance).all() or not np.isfinite(beta).all() or not np.allclose(covariance, covariance.T, atol=1e-10):
            raise ValueError("Risk model needs finite symmetric covariance and finite betas")
        if eigvalsh(covariance, subset_by_index=[0, 0])[0] < -1e-10:
            raise ValueError("Covariance must be positive semidefinite")
        return (covariance + covariance.T) / 2, beta


class PortfolioConstructor:
    """One interface for demean/L1 weights and a simplified AQR-style optimizer.

    build() returns signed NAV targets. orders() converts one date's targets to
    whole-share trade deltas. Neither function runs a backtest or books fills.
    """

    def __init__(self, config=None):
        self.config = config or ConstructionConfig()

    def build(self, signals, *, score_column="score", prices=None, risk_model=None,
              current_weights=None, benchmark_weights=None, tax_buy_costs=None, tax_sell_costs=None):
        """Construct one date or a batch of dated targets.

        signals contains as_of/ticker, the configured group, and either score
        (demean_l1) or the raw factor columns (aqr_style). For the optimizer,
        supply historical Close columns including SPY, or a dated RiskModel.
        current_weights and benchmark_weights are ticker-indexed NAV weights.
        Per-date mappings are accepted; a batch requires actual holdings for
        each date. None means an initial empty portfolio, not prior targets.
        Tax inputs are nonnegative marginal costs per unit traded NAV weight;
        they are penalties supplied by the caller, not computed tax-lot results.
        """
        required = {"as_of", "ticker"}
        required.update([score_column] if self.config.method == "demean_l1" else self.config.factor_columns)
        if self.config.group_by is not None:
            required.add(self.config.group_by)
        if not required.issubset(signals):
            raise ValueError("Construction requires columns: " + ", ".join(sorted(required)))
        frame = signals.copy().reset_index(drop=True)
        frame["as_of"] = pd.to_datetime(frame["as_of"], errors="raise").dt.normalize()
        if frame["as_of"].isna().any() or frame["ticker"].isna().any():
            raise ValueError("Construction dates and tickers cannot be missing")
        frame["ticker"] = frame["ticker"].astype(str).str.strip().str.upper()
        if frame["ticker"].eq("").any() or frame.duplicated(["as_of", "ticker"]).any():
            raise ValueError("Use one nonempty ticker per construction date")
        result = frame[["as_of", "ticker"]].copy()
        result["score"] = pd.to_numeric(frame[score_column], errors="coerce") if score_column in frame else np.nan
        result["construction_group"] = "universe" if self.config.group_by is None else frame[self.config.group_by]
        valid = np.isfinite(result["score"])
        if "in_universe" in frame:
            if not frame["in_universe"].map(lambda value: isinstance(value, (bool, np.bool_))).all():
                raise ValueError("in_universe must contain booleans")
            valid &= frame["in_universe"]
        group_known = result["construction_group"].notna() & result["construction_group"].ne("")
        result["centered_score"] = 0.0
        result["target_weight"] = 0.0
        result["construction_status"] = "inactive_or_missing_score"
        result.loc[valid & ~group_known, "construction_status"] = "missing_group"
        valid &= group_known
        result.loc[valid, "construction_status"] = "zero_signal"
        reports = []
        if self.config.method == "aqr_style":
            if frame["as_of"].nunique() > 1 and not isinstance(current_weights, Mapping):
                raise ValueError("For multiple dates supply actual pre-trade current_weights per date; targets are not holdings")
            reports = self._aqr(frame, result, prices, risk_model, current_weights, benchmark_weights, tax_buy_costs, tax_sell_costs)
        else:
            self._demean(result, valid)
        result["target_side"] = np.select(
            [result["target_weight"].gt(0), result["target_weight"].lt(0)], ["long", "short"], default="flat",
        )
        result.loc[result["target_weight"].ne(0), "construction_status"] = "allocated"
        result = result.sort_values(["as_of", "target_weight", "ticker"], ascending=[True, False, True]).reset_index(drop=True)
        result.attrs["construction"] = asdict(self.config)
        result.attrs["weight_basis"] = "NAV"
        result.attrs["execution"] = "continuous_targets_only"
        result.attrs["risk_reports"] = reports
        return result

    def _demean(self, result, valid):
        for _, month in result.loc[valid].groupby("as_of", sort=True):
            magnitude = month["score"].abs().max()
            if magnitude == 0:
                continue
            # Scaling before centering keeps normalization stable and invariant
            # to signal units; it is algebraically the same as raw demeaning.
            scaled = month["score"] / magnitude
            centered = scaled - scaled.groupby(month["construction_group"], sort=False).transform("mean")
            # Recenter the small residuals to remove cancellation error. Drop
            # only whole near-constant groups; clipping individual names can
            # remove one side and destroy dollar neutrality.
            centered -= centered.groupby(month["construction_group"], sort=False).transform("mean")
            spread = centered.abs().groupby(month["construction_group"], sort=False).transform("max")
            centered = centered.where(spread > 32 * np.finfo(float).eps, 0.0)
            total = centered.abs().sum()
            if total == 0:
                continue
            result.loc[month.index, "centered_score"] = centered * magnitude
            result.loc[month.index, "target_weight"] = self.config.gross_exposure * centered / total

    @staticmethod
    def _at_date(value, date, name):
        if isinstance(value, Mapping):
            matches = [item for key, item in value.items() if pd.Timestamp(key).normalize() == date]
            if len(matches) != 1:
                raise ValueError("Supply exactly one " + name + " entry per decision date")
            return matches[0]
        return value

    @staticmethod
    def _weights(value, symbols, name):
        if value is None:
            return np.zeros(len(symbols))
        if not isinstance(value, pd.Series) or value.index.duplicated().any():
            raise ValueError(name + " must be a uniquely indexed ticker Series")
        value = pd.to_numeric(value, errors="raise")
        if not np.isfinite(value).all():
            raise ValueError(name + " must be finite")
        extras = value.loc[~value.index.isin(symbols)]
        if extras.abs().gt(1e-10).any():
            raise ValueError("Include all nonzero " + name + " tickers in the signal/risk universe")
        return value.reindex(symbols, fill_value=0).to_numpy(dtype=float)

    def _complete_factors(self, frame, groups):
        factors = frame[list(self.config.factor_columns)].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
        complete = factors.notna().all(axis=1)
        if "in_universe" in frame:
            complete &= frame["in_universe"]
        sizes = complete.groupby(groups).transform("sum")
        complete &= sizes.ge(self.config.min_group_size)
        return factors, complete

    def _model_alpha(self, frame, groups, covariance, beta):
        from .operators import transform
        factors, complete = self._complete_factors(frame, groups)
        eligible_beta = np.where(complete, beta, 0.0)
        model = np.zeros(len(frame))
        for column in self.config.factor_columns:
            factor = np.zeros(len(frame))
            zscores = transform(factors.loc[complete, column], ("group_rank", "group_zscore"), groups.loc[complete])
            factor[complete.to_numpy()] = zscores.to_numpy()
            if eligible_beta @ eligible_beta > 1e-14:
                factor -= eligible_beta * ((eligible_beta @ factor) / (eligible_beta @ eligible_beta))
            volatility = np.sqrt(max(float(factor @ covariance @ factor), 0))
            if volatility > 1e-12:
                model += factor / volatility / len(self.config.factor_columns)
        volatility = np.sqrt(max(float(model @ covariance @ model), 0))
        if volatility <= 1e-12:
            return np.zeros(len(frame))
        return self.config.alpha_scale * covariance @ (model / volatility)

    def _aqr(self, frame, result, prices, risk_model, current_weights, benchmark_weights, tax_buy_costs, tax_sell_costs):
        reports = []
        cfg = self.config
        result["model_alpha"] = 0.0
        result["centered_score"] = np.nan
        for date, raw in frame.groupby("as_of", sort=True):
            previous_input = self._at_date(current_weights, date, "current_weights")
            self._weights(previous_input, raw["ticker"].tolist(), "current_weights")
            model = self._at_date(risk_model, date, "risk_model")
            if model is None:
                if prices is None:
                    raise ValueError("aqr_style requires historical prices or a dated RiskModel")
                model = RiskModel.estimate(prices, raw["ticker"].tolist(), date, cfg)
            if not isinstance(model, RiskModel):
                raise ValueError("risk_model must be a RiskModel or a date-to-model mapping")
            known_group = result.loc[raw.index, "construction_group"].notna() & result.loc[raw.index, "construction_group"].ne("")
            covered = raw["ticker"].isin(model.covariance.index) & raw["ticker"].isin(model.beta.index) & known_group
            omitted = raw.loc[~covered, "ticker"]
            if previous_input is not None and previous_input.reindex(omitted, fill_value=0).abs().gt(1e-10).any():
                raise ValueError("Held stocks need risk history and group labels; supply a risk model before constructing")
            result.loc[raw.index[~covered], "construction_status"] = "missing_risk_or_group"
            selected = raw.loc[covered].copy()
            symbols = selected["ticker"].tolist()
            if not symbols:
                if previous_input is not None and previous_input.abs().gt(1e-10).any():
                    raise ValueError("No risk coverage for current holdings")
                if cfg.net_exposure != 0 or (cfg.target_beta is not None and cfg.target_beta != 0):
                    raise ValueError("No risk-covered assets can meet requested exposure")
                continue
            covariance, beta = model.aligned(symbols, date, cfg)
            selected.index = pd.RangeIndex(len(selected))
            groups = result.loc[raw.index[covered], "construction_group"].reset_index(drop=True)
            previous = self._weights(previous_input, symbols, "current_weights")
            benchmark = self._weights(self._at_date(benchmark_weights, date, "benchmark_weights"), symbols, "benchmark_weights")
            if not np.isclose(benchmark.sum(), cfg.net_exposure, atol=1e-8):
                raise ValueError("Provide benchmark weights summing to net_exposure; the default is a neutral component")
            buy_tax = self._weights(self._at_date(tax_buy_costs, date, "tax_buy_costs"), symbols, "tax_buy_costs")
            sell_tax = self._weights(self._at_date(tax_sell_costs, date, "tax_sell_costs"), symbols, "tax_sell_costs")
            if (buy_tax < 0).any() or (sell_tax < 0).any():
                raise ValueError("Marginal tax penalties must be nonnegative; tax-loss credits require a lot-aware model")
            alpha = self._model_alpha(selected, groups, covariance, beta)
            _, complete = self._complete_factors(selected, groups)
            tradable = complete.to_numpy()
            weights, details = self._optimize(alpha, covariance, beta, groups, previous, benchmark, tradable, buy_tax, sell_tax)
            result.loc[raw.index[covered], "target_weight"] = weights
            result.loc[raw.index[covered], "model_alpha"] = alpha
            result.loc[raw.index[covered], "construction_status"] = "optimized_zero"
            reports.append({"as_of": date, "risk_as_of": pd.Timestamp(model.as_of), "risk_source": model.source,
                            **details})
        return reports

    def _optimize(self, alpha, covariance, beta, groups, previous, benchmark, tradable, buy_tax, sell_tax):
        """Convex utility optimizer; risk penalty, gross cap and marginal taxes.

        This is an AQR-inspired public-data implementation, not the paper's
        Barra/HIFO optimizer or its hard tracking-error/leverage equalities.
        """
        import osqp
        from scipy import sparse
        cfg, n = self.config, len(alpha)
        eye = sparse.eye(n, format="csc")
        zero = sparse.csc_matrix((n, n))
        horizon = cfg.horizon_years
        risk = cfg.risk_aversion * horizon * covariance
        p = sparse.block_diag((sparse.csc_matrix(risk), zero, zero, zero), format="csc")
        q = np.r_[-horizon * alpha - risk @ benchmark - horizon * cfg.borrow_fee_rate / 2,
                  cfg.transaction_cost_rate + buy_tax, cfg.transaction_cost_rate + sell_tax,
                  np.full(n, horizon * cfg.borrow_fee_rate / 2)]
        # Variables: target weights, purchases, sales, absolute target weights.
        matrices = [sparse.hstack((eye, -eye, eye, zero)),
                    sparse.hstack((eye, zero, zero, -eye)),
                    sparse.hstack((-eye, zero, zero, -eye))]
        lower = [previous, np.full(n, -np.inf), np.full(n, -np.inf)]
        upper = [previous, np.zeros(n), np.zeros(n)]
        limits = np.where(tradable, cfg.max_weight, 0.0)
        matrices.append(sparse.eye(4 * n, format="csc"))
        lower.append(np.r_[-limits, np.zeros(3 * n)])
        upper.append(np.r_[limits, np.full(3 * n, np.inf)])
        def constraint(coefficients, lo, hi):
            matrices.append(sparse.csc_matrix(np.asarray(coefficients).reshape(1, -1)))
            lower.append(np.array([lo])); upper.append(np.array([hi]))
        constraint(np.r_[np.zeros(3 * n), np.ones(n)], -np.inf, cfg.gross_exposure)
        constraint(np.r_[np.ones(n), np.zeros(3 * n)], cfg.net_exposure, cfg.net_exposure)
        if cfg.target_beta is not None:
            constraint(np.r_[beta, np.zeros(3 * n)], cfg.target_beta, cfg.target_beta)
        if cfg.group_by is not None:
            for group in groups.unique():
                members = groups.eq(group).to_numpy(dtype=float)
                exposure = float(members @ benchmark)
                constraint(np.r_[members, np.zeros(3 * n)], exposure, exposure)
        problem = osqp.OSQP()
        problem.setup(P=sparse.triu(p, format="csc"), q=q, A=sparse.vstack(matrices, format="csc"),
                      l=np.concatenate(lower), u=np.concatenate(upper), verbose=False,
                      eps_abs=1e-8, eps_rel=1e-8, max_iter=100000, polishing=True)
        solved = problem.solve(raise_error=True)
        weights = solved.x[:n].copy()
        weights[np.abs(weights) < 1e-8] = 0
        tolerance = 1e-6
        if (abs(weights.sum() - cfg.net_exposure) > tolerance or abs(weights).sum() > cfg.gross_exposure + tolerance
                or np.any(abs(weights) > limits + tolerance)
                or (cfg.target_beta is not None and abs(beta @ weights - cfg.target_beta) > tolerance)):
            raise RuntimeError("Optimizer result violates exposure constraints")
        if cfg.group_by is not None:
            for group in groups.unique():
                members = groups.eq(group).to_numpy(dtype=float)
                if abs(members @ (weights - benchmark)) > tolerance:
                    raise RuntimeError("Optimizer result violates group exposure constraints")
        changes = weights - previous
        return weights, {"estimated_beta": float(beta @ weights),
                         "predicted_active_volatility": float(np.sqrt(max((weights - benchmark) @ covariance @ (weights - benchmark), 0))),
                         "turnover": float(abs(changes).sum()),
                         "estimated_transaction_cost": float(cfg.transaction_cost_rate * abs(changes).sum()),
                         "estimated_horizon_borrow_cost": float(cfg.borrow_fee_rate * horizon * -weights.clip(max=0).sum()),
                         "marginal_tax_penalty": float(buy_tax @ np.maximum(changes, 0) + sell_tax @ np.maximum(-changes, 0)),
                         "solver_status": solved.info.status}

    @staticmethod
    def cached_prices(database, symbols, start, end, market="SPY"):
        """Read cached quote units only; construction never fetches source data."""
        from .data import split_factors
        columns = {}
        for symbol in set(symbols) | {market}:
            frame = database.price_frame(symbol, start, end)
            if frame.empty and symbol == market and market == "SPY":
                frame = database.price_frame("__NYSE_CALENDAR__", start, end)
            if not frame.empty:
                columns[symbol] = frame["Close"] / split_factors(frame.index, database.split_series(symbol))
        return pd.DataFrame(columns).sort_index()

    def orders(self, targets, *, nav, prices, current_shares=None):
        """Turn one decision's signed NAV targets into whole-share trade deltas.

        Missing quotes defer orders. This plans trades; it does not book fills,
        holding fees, tax lots, or post-rounding neutrality.
        """
        if not np.isfinite(nav) or nav <= 0 or targets["as_of"].nunique() != 1:
            raise ValueError("Supply positive NAV and exactly one target date")
        if targets["ticker"].duplicated().any() or not np.isfinite(targets["target_weight"]).all():
            raise ValueError("Use unique ticker targets with finite weights")
        if not isinstance(prices, pd.Series) or prices.index.duplicated().any():
            raise ValueError("Execution prices must be a uniquely indexed ticker Series")
        if current_shares is not None and not isinstance(current_shares, pd.Series):
            raise ValueError("Current shares must be a ticker-indexed Series")
        shares = pd.Series(dtype=float) if current_shares is None else current_shares.astype(float)
        if (shares.index.duplicated().any() or not np.isfinite(shares).all()
                or not np.equal(shares, np.trunc(shares)).all() or shares.abs().ge(float(2**63)).any()):
            raise ValueError("Current shares must be finite signed integers")
        weights = targets.set_index("ticker")["target_weight"]
        symbols = weights.index.union(shares.index)
        quotes = pd.to_numeric(prices.reindex(symbols), errors="coerce")
        held = shares.reindex(symbols, fill_value=0)
        desired = weights.reindex(symbols, fill_value=0) * nav
        usable = np.isfinite(quotes) & quotes.gt(0)
        target_shares = held.copy()
        quantity = np.trunc(desired.loc[usable] / quotes.loc[usable])
        if quantity.abs().ge(float(2**63)).any():
            raise ValueError("Target share count exceeds int64 range")
        target_shares.loc[usable] = quantity
        delta = target_shares - held
        if delta.abs().ge(float(2**63)).any():
            raise ValueError("Trade share count exceeds int64 range")
        result = pd.DataFrame({"signal_date": targets["as_of"].iloc[0], "ticker": symbols,
                               "target_weight": weights.reindex(symbols, fill_value=0).to_numpy(), "target_amount": desired.to_numpy(),
                               "price": quotes.to_numpy(), "current_shares": held.to_numpy(dtype=np.int64),
                               "target_shares": target_shares.to_numpy(dtype=np.int64), "trade_shares": delta.to_numpy(dtype=np.int64)})
        result["order_amount"] = np.where(usable.to_numpy(), result["trade_shares"] * result["price"], 0.0)
        result["transaction_cost"] = result["order_amount"].abs() * self.config.transaction_cost_rate
        result["execution_status"] = np.where(~usable.to_numpy(), "missing_price", np.where(delta.to_numpy() == 0, "no_trade", "ready"))
        return result

    @staticmethod
    def summary(targets):
        """Audit target amounts; no market beta is inferred from net exposure."""
        rows = []
        for date, month in targets.groupby("as_of", sort=True):
            weights = month["target_weight"]
            rows.append({"as_of": date, "long_weight": weights.clip(lower=0).sum(),
                         "short_weight": weights.clip(upper=0).sum(), "net_weight": weights.sum(),
                         "gross_weight": weights.abs().sum(), "long_stocks": int(weights.gt(0).sum()),
                         "short_stocks": int(weights.lt(0).sum())})
        result = pd.DataFrame(rows, columns=["as_of", "long_weight", "short_weight", "net_weight",
                                             "gross_weight", "long_stocks", "short_stocks"])
        reports = targets.attrs.get("risk_reports", [])
        return result.merge(pd.DataFrame(reports), on="as_of", how="left") if reports else result
