"""Train pair signal parameters before 2025 and validate them during 2025."""

from __future__ import annotations

from dataclasses import dataclass, replace
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from pair_pool import PairPoolBuilder, PairPoolConfig, PairPoolSource

LOGGER = logging.getLogger(__name__)
METRICS = ("pnl", "return_on_budget", "sharpe", "max_drawdown", "trades", "win_rate",
           "forced_exits", "stop_exits", "observations")
OU_FIELDS = ("ar1_c", "ar1_b", "theta", "mu_e", "sigma_eq", "half_life_days")


@dataclass(frozen=True)
class SignalTrainingConfig:
    train_start: str = "2022-01-01"
    train_end: str = "2024-12-31"
    validation_start: str = "2025-01-01"
    validation_end: str = "2025-12-31"
    z_values: tuple[float, ...] = tuple(i / 4 for i in range(1, 15))
    exit_on_cross_mean: bool = False
    prevent_immediate_reentry: bool = True
    stop_loss: float = 0.12
    min_train_trades: int = 5
    min_validation_trades: int = 2
    min_validation_observations: int = 200
    long_amount: float = 100_000

    def __post_init__(self):
        dates = [pd.Timestamp(getattr(self, name)) for name in
                 ("train_start", "train_end", "validation_start", "validation_end")]
        if not dates[0] < dates[1] < dates[2] < dates[3] < pd.Timestamp("2026-01-01"):
            raise ValueError("Use nonoverlapping training/validation periods ending before 2026.")
        if not self.z_values or not np.isfinite(self.z_values).all() or min(self.z_values) <= 0:
            raise ValueError("z_values must contain positive, finite thresholds")
        if not np.isfinite(self.stop_loss) or not 0 < self.stop_loss <= 1:
            raise ValueError("stop_loss must be in (0, 1]")
        for name in ("min_train_trades", "min_validation_trades", "min_validation_observations"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(name + " must be a positive integer")
        if not np.isfinite(self.long_amount) or self.long_amount <= 0:
            raise ValueError("long_amount must be positive and finite")


class PairSignalTrainer:
    """Choose parameters in training, then evaluate the frozen choice once.

    Each simulation is an independent pair with a fixed 2*long_amount budget.
    At entry, both integer-share legs are capped at long_amount by one common
    scale. Shares then stay fixed until exit. Costs, borrow fees, distributions,
    financing and margin effects are excluded. This is not a multi-pair portfolio.
    """

    def __init__(self, config: SignalTrainingConfig | None = None, *,
                 pool_config: PairPoolConfig | None = None, source=None):
        self.config = config or SignalTrainingConfig()
        pool = pool_config or PairPoolConfig()
        formation = replace(pool.cointegration, train_start=self.config.train_start,
                            train_end=self.config.train_end)
        self.pool_config = replace(pool, cointegration=formation)
        self.source = source if source is not None else PairPoolSource(self.pool_config)
        self.pool_result = None

    def run(self, *, save=True):
        """Rebuild candidates without validation prices, train, and save one CSV."""
        self.pool_result = PairPoolBuilder(self.pool_config, source=self.source).build(save=False)
        return self.train(self.pool_result["pairs"], save=save)

    def train(self, pairs: pd.DataFrame, *, histories=None, save=True):
        """Add parameters and results; retain failed rows with explicit flags."""
        config = self.config
        required = {"pair_id", "y_ticker", "x_ticker", "formation_end"}
        if pairs.empty or not required.issubset(pairs.columns) or not pairs.pair_id.is_unique:
            raise ValueError("Provide nonempty, unique formation pairs with their date cutoff.")
        if not pd.to_datetime(pairs.formation_end).le(pd.Timestamp(config.train_end)).all():
            raise ValueError("Rebuild the pair pool through train_end; validation cannot select pairs.")
        if histories is None:
            tickers = sorted(set(pairs.y_ticker) | set(pairs.x_ticker))
            end = (pd.Timestamp(config.validation_end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
            histories = self.source.history(tickers, config.train_start, end)
        results = []
        for number, pair in enumerate(pairs.itertuples(index=False), 1):
            row = {"pair_id": pair.pair_id, "entry_z": np.nan, "exit_z": np.nan,
                   "training_status": "not_trained", "validation_status": "not_run",
                   "validation_passed": False, "train_alpha": np.nan, "train_beta": np.nan,
                   "train_nobs": 0, "refit_alpha": np.nan, "refit_beta": np.nan,
                   "refit_end": "", "refit_nobs": 0, "refit_status": "not_run"}
            for prefix in ("train_", "validation_"):
                row.update({prefix + name: (0 if name in ("trades", "forced_exits", "stop_exits", "observations") else np.nan)
                            for name in METRICS})
            for prefix in ("train_", "refit_"):
                row.update({prefix + name: np.nan for name in OU_FIELDS})
            try:
                prices = self._prices(histories, pair.y_ticker, pair.x_ticker)
                training = prices.loc[config.train_start:config.train_end]
                validation = prices.loc[config.validation_start:config.validation_end]
                if len(training) < self.pool_config.cointegration.min_observations:
                    raise ValueError("insufficient_training_prices")
                alpha, beta = self._fit(training)
                row.update(train_alpha=alpha, train_beta=beta, train_nobs=len(training))
                choices = []
                spread = prices.Close_y - alpha - beta * prices.Close_x
                ou = self.fit_legacy_scale(spread.loc[config.train_start:config.train_end])
                row.update({"train_" + name: value for name, value in ou.items()})
                z = self.z_score(spread, ou["mu_e"], ou["sigma_eq"])
                for entry in config.z_values:
                    exit_ = 0.0 if config.exit_on_cross_mean else entry / 2.0
                    stats = self._simulate(prices, z, beta, entry, exit_, config.train_start, config.train_end)
                    if stats["trades"] >= config.min_train_trades and np.isfinite(stats["sharpe"]):
                        choices.append((stats, entry, exit_))
                if not choices:
                    row["training_status"] = "too_few_trades_or_undefined_sharpe"
                else:
                    # Validation statistics never participate in parameter selection.
                    # Legacy auto_report ranks cumulative return first, Sharpe second.
                    stats, entry, exit_ = max(choices, key=lambda c: (
                        c[0]["return_on_budget"], c[0]["sharpe"], -c[1]))
                    row.update(entry_z=entry, exit_z=exit_, training_status="trained")
                    row.update({"train_" + name: value for name, value in stats.items()})
                    enough_validation = (len(validation) >= config.min_validation_observations
                        and (validation.index[0] - pd.Timestamp(config.validation_start)).days <= 10
                        and (pd.Timestamp(config.validation_end) - validation.index[-1]).days <= 10)
                    if not enough_validation:
                        row["validation_status"] = "insufficient_validation_prices"
                    else:
                        validation_stats = self._simulate(prices, z, beta, entry, exit_,
                                                          config.validation_start, config.validation_end)
                        row.update({"validation_" + name: value for name, value in validation_stats.items()})
                        if stats["pnl"] <= 0 or stats["sharpe"] <= 0:
                            row["validation_status"] = "training_not_profitable"
                        elif validation_stats["trades"] < config.min_validation_trades:
                            row["validation_status"] = "too_few_validation_trades"
                        elif validation_stats["pnl"] <= 0 or not validation_stats["sharpe"] > 0:
                            row["validation_status"] = "validation_not_profitable"
                        else:
                            row.update(validation_status="passed", validation_passed=True)
                # These later coefficients are for future deployment, never validation.
                try:
                    refit_alpha, refit_beta = self._fit(prices)
                    row.update(refit_alpha=refit_alpha, refit_beta=refit_beta, refit_status="ok",
                               refit_end=str(prices.index[-1].date()), refit_nobs=len(prices))
                    refit_spread = prices.Close_y - refit_alpha - refit_beta * prices.Close_x
                    row.update({"refit_" + name: value for name, value in self.fit_legacy_scale(refit_spread).items()})
                except ValueError as exc:
                    row["refit_status"] = str(exc)
            except ValueError as exc:
                row.update(training_status=str(exc), validation_status="data_or_model_error",
                           validation_passed=False)
            results.append(row)
            if number % 25 == 0 or number == len(pairs):
                LOGGER.info("Signal training %s/%s; validation passed=%s", number, len(pairs),
                            sum(record["validation_passed"] for record in results))
        scores = pd.DataFrame(results).set_index("pair_id")
        output = pairs.drop(columns=[c for c in scores if c in pairs] + ["window"], errors="ignore").join(scores, on="pair_id")
        for name in ("train_start", "train_end", "validation_start", "validation_end"):
            output[name] = getattr(config, name)
        output["signal_long_amount"] = config.long_amount
        output["signal_budget"] = 2 * config.long_amount
        output["transaction_cost_rate"] = 0.0
        output["borrow_fee_rate"] = 0.0
        output["signal_method"] = "legacy_mean_square"
        output["tau"] = 1 / 252
        output["stop_loss"] = config.stop_loss
        output["exit_on_cross_mean"] = config.exit_on_cross_mean
        output["prevent_immediate_reentry"] = config.prevent_immediate_reentry
        output["selection_metric"] = "training_return_then_sharpe"
        if save:
            path = Path(self.pool_config.output_file)
            path.parent.mkdir(parents=True, exist_ok=True)
            output.to_csv(path, index=False)
        return output

    def _prices(self, histories, y, x):
        frames = []
        for ticker, suffix in ((y, "y"), (x, "x")):
            frame = histories.get(ticker)
            if frame is None or not {"Open", "Close"}.issubset(frame.columns):
                raise ValueError("missing_open_or_close")
            frame = frame[["Open", "Close"]].copy()
            frame.index = pd.DatetimeIndex(pd.to_datetime(frame.index)).tz_localize(None).normalize()
            frame = frame.loc[~frame.index.duplicated(keep="last")].sort_index()
            # Discard 2026 and later before fitting or simulation.
            frame = frame.loc[self.config.train_start:self.config.validation_end]
            frames.append(frame.add_suffix("_" + suffix))
        prices = pd.concat(frames, axis=1).apply(pd.to_numeric, errors="coerce")
        return prices.replace([np.inf, -np.inf], np.nan).where(prices > 0).dropna()

    @staticmethod
    def _fit(prices):
        if len(prices) < 20 or prices.Close_x.std() <= 1e-10:
            raise ValueError("insufficient_or_constant_regression_prices")
        design = np.column_stack((np.ones(len(prices)), prices.Close_x.to_numpy()))
        alpha, beta = np.linalg.lstsq(design, prices.Close_y.to_numpy(), rcond=None)[0]
        if not np.isfinite([alpha, beta]).all() or beta <= 0:
            raise ValueError("nonpositive_or_invalid_beta")
        return float(alpha), float(beta)

    @staticmethod
    def fit_legacy_scale(spread: pd.Series):
        """Reproduce TS_implement/TS_lib mean_square scaling, without a window.

        Fit e_t = C + B*e_(t-1) + innovation, then use the legacy formula
        sigma_eq = sqrt(mean(e^2)*tau / (1-exp(-2*theta*tau))). The numerator
        deliberately matches the old spread second moment, not innovation variance.
        """
        e = pd.to_numeric(spread, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna().to_numpy()
        if len(e) < 20 or e.std() <= 1e-10:
            raise ValueError("insufficient_or_constant_ou_spread")
        design = np.column_stack((np.ones(len(e) - 1), e[:-1]))
        c, b = np.linalg.lstsq(design, e[1:], rcond=None)[0]
        if not np.isfinite([c, b]).all() or not 0 < b < 1:
            raise ValueError("ou_requires_0_lt_B_lt_1")
        tau = 1 / 252
        theta = -np.log(b) / tau
        mu = c / (1 - b)
        sigma = np.sqrt(np.mean(e ** 2) * tau / -np.expm1(-2 * theta * tau))
        if not np.isfinite([theta, mu, sigma]).all() or sigma <= 1e-10:
            raise ValueError("invalid_legacy_sigma")
        return {"ar1_c": float(c), "ar1_b": float(b), "theta": float(theta),
                "mu_e": float(mu), "sigma_eq": float(sigma),
                "half_life_days": float(np.log(2) / -np.log(b))}

    @staticmethod
    def z_score(spread: pd.Series, mu_e: float, sigma_eq: float):
        """Standardize with the frozen training mean and legacy scale."""
        return (spread - mu_e) / sigma_eq

    def _simulate(self, prices, z, beta, entry_z, exit_z, start, end, *, details=False):
        """Trade yesterday's signal at today's Open; flatten at period-end Close."""
        dates = prices.index
        selected = np.flatnonzero((dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end)))
        if not len(selected):
            raise ValueError("empty_simulation_period")
        opening = prices[["Open_y", "Open_x"]].to_numpy()
        closing = prices[["Close_y", "Close_x"]].to_numpy()
        z_values = z.reindex(dates).to_numpy()
        budget = 2 * self.config.long_amount
        cash, previous_nav = budget, budget
        quantity = np.zeros(2, dtype=np.int64)
        entry_price = np.zeros(2)
        direction, pending, entry_date = 0, None, None
        pending_reason, entry_gross = "signal", 0.0
        daily_pnl, trade_pnl, trades = [], [], []
        forced_exits, stop_exits = 0, 0

        def entry_signal(value):
            if not np.isfinite(value):
                return None
            return -1 if value > entry_z else (1 if value < -entry_z else None)

        # Validation starts flat but may use the final training Close as a signal.
        if selected[0] > 0:
            pending = entry_signal(z_values[selected[0] - 1])
        for i in selected:
            exited_today = False
            if pending is not None:
                if direction and pending == 0:
                    pnl = float(quantity @ (opening[i] - entry_price))
                    trade_pnl.append(pnl)
                    if details:
                        trades.append((entry_date, dates[i], direction, *quantity, pnl, pending_reason))
                    stop_exits += int(pending_reason == "stop")
                    cash += float(quantity @ opening[i])
                    quantity[:] = 0
                    direction = 0
                    exited_today = True
                elif not direction and pending:
                    # One common scale preserves the hedge ratio before whole-share rounding.
                    qy = int(np.floor(min(self.config.long_amount / opening[i, 0],
                                         self.config.long_amount / (beta * opening[i, 1]))))
                    qx = int(np.floor(beta * qy))
                    if qy > 0 and qx > 0:
                        direction = pending
                        quantity[:] = (direction * qy, -direction * qx)
                        entry_price = opening[i].copy()
                        entry_gross = float(np.abs(quantity) @ opening[i])
                        entry_date = dates[i]
                        cash -= float(quantity @ opening[i])
            pending = None
            pending_reason = "signal"
            if i == selected[-1] and direction:
                pnl = float(quantity @ (closing[i] - entry_price))
                trade_pnl.append(pnl)
                forced_exits += 1
                if details:
                    trades.append((entry_date, dates[i], direction, *quantity, pnl, "period_end"))
                cash += float(quantity @ closing[i])
                quantity[:] = 0
                direction = 0
            nav = cash + float(quantity @ closing[i])
            daily_pnl.append(nav - previous_nav)
            previous_nav = nav
            value = z_values[i]
            if direction:
                current_trade_pnl = float(quantity @ (closing[i] - entry_price))
                # Crossing zero beyond the exit band also closes the position.
                if current_trade_pnl <= -self.config.stop_loss * entry_gross:
                    pending, pending_reason = 0, "stop"
                elif np.isfinite(value) and ((direction == 1 and value >= -exit_z)
                                            or (direction == -1 and value <= exit_z)):
                    pending = 0
            elif not (exited_today and self.config.prevent_immediate_reentry):
                pending = entry_signal(value)
        pnl_array = np.asarray(daily_pnl)
        returns = pnl_array / budget
        volatility = float(returns.std(ddof=1)) if len(returns) > 1 else 0.0
        nav = np.r_[budget, budget + pnl_array.cumsum()]
        drawdown = nav / np.maximum.accumulate(nav) - 1
        result = {"pnl": float(pnl_array.sum()), "return_on_budget": float(returns.sum()),
                  "sharpe": float(np.sqrt(252) * returns.mean() / volatility) if volatility > 1e-12 else np.nan,
                  "max_drawdown": float(drawdown.min()), "trades": len(trade_pnl),
                  "win_rate": float(np.mean(np.asarray(trade_pnl) > 0)) if trade_pnl else np.nan,
                  "forced_exits": forced_exits, "stop_exits": stop_exits, "observations": len(selected)}
        if details:
            result["daily_pnl"] = pd.Series(pnl_array, index=dates[selected], name="daily_pnl")
            result["trade_log"] = pd.DataFrame(trades, columns=[
                "entry_date", "exit_date", "direction", "shares_y", "shares_x", "pnl", "exit_reason"])
        return result
