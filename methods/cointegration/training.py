"""Independent-pair calibration used by the frozen cointegration selector."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
from .execution import ExecutionModel
from .rules import (entry_scenario, payoff_statistics, spread_mode, spread_stop,
                    reversion_confirmed, within_entry_distance)
TRADING_DAYS_PER_YEAR = 252

@dataclass(frozen=True)
class HalfLifeFilter:
    """Screen an AR(1) spread half-life, measured in daily trading sessions.

    Reindex the spread to its trading calendar before estimating. Shift before
    dropping missing rows so a missing session is not silently compressed.
    Half-life is a decay estimate, not a deadline for crossing the exit level.
    """

    max_days: float | None = None
    min_days: float | None = None

    def __post_init__(self):
        for name in ("max_days", "min_days"):
            value = getattr(self, name)
            if value is not None and (isinstance(value, bool) or not np.isfinite(value) or value <= 0):
                raise ValueError(name + " must be positive and finite or None")
        if self.min_days is not None and self.max_days is not None and self.min_days > self.max_days:
            raise ValueError("The minimum half-life cannot exceed the maximum half-life.")

    @classmethod
    def from_limits(cls, *, min_years=None, max_years=None, max_days=None):
        """Use H=log(2)/theta in years with tau=1/252, as in TS_implement."""
        for name, value in (("min_half_life_years", min_years), ("max_half_life_years", max_years)):
            if value is not None and (isinstance(value, bool) or not np.isfinite(value) or value <= 0):
                raise ValueError(name + " must be positive and finite or None")
        if max_years is not None and max_days is not None:
            raise ValueError("Specify the maximum half-life in years or days, not both.")
        maximum = max_days if max_years is None else max_years * TRADING_DAYS_PER_YEAR
        return cls(max_days=maximum, min_days=None if min_years is None else min_years * TRADING_DAYS_PER_YEAR)

    @property
    def active(self):
        return self.min_days is not None or self.max_days is not None

    def accepts(self, days):
        return bool(np.isfinite(days) and days > 0
                    and (self.min_days is None or days >= self.min_days - 1e-10)
                    and (self.max_days is None or days <= self.max_days + 1e-10))

    @staticmethod
    def estimate(spread):
        values = pd.to_numeric(spread, errors="coerce").replace([np.inf, -np.inf], np.nan)
        pairs = pd.concat([values.rename("current"), values.shift(1).rename("lagged")], axis=1).dropna()
        if len(pairs) < 20 or pairs.lagged.std() <= 1e-10:
            raise ValueError("insufficient_half_life_observations")
        c, b = np.linalg.lstsq(np.column_stack([np.ones(len(pairs)), pairs.lagged]), pairs.current, rcond=None)[0]
        if not np.isfinite([c, b]).all() or not 0 < b < 1:
            raise ValueError("half_life_requires_0_lt_B_lt_1")
        half_life_days = float(np.log(2) / -np.log(b))
        return dict(half_life_days=half_life_days, half_life_years=half_life_days / TRADING_DAYS_PER_YEAR,
                    theta=float(-np.log(b) * TRADING_DAYS_PER_YEAR), ar1_c=float(c), ar1_b=float(b),
                    half_life_nobs=len(pairs), half_life_start=str(pd.Timestamp(pairs.index[0]).date()),
                    half_life_end=str(pd.Timestamp(pairs.index[-1]).date()))

    def apply(self, pool, field="selection_half_life_days"):
        if not self.active:
            return pool.copy(deep=True)
        if pool.empty:
            return pool.copy(deep=True)
        if field not in pool:
            raise ValueError("Half-life estimates are required before screening the pool.")
        values = pd.to_numeric(pool[field], errors="coerce")
        accepted = np.isfinite(values) & values.gt(0)
        if self.min_days is not None:
            accepted &= values.ge(self.min_days - 1e-10)
        if self.max_days is not None:
            accepted &= values.le(self.max_days + 1e-10)
        return pool.loc[accepted].copy()

@dataclass(frozen=True)
class SignalTrainingConfig:
    long_amount: float = 1_000_000 / 15
    borrow_fee_rate: float = 0.01
    transaction_cost_rate: float = 0.0005
    fx_financing_rate: float = 0.0
    execution_timing: str = "close"
    stop_loss: float = 0.12
    prevent_immediate_reentry: bool = True


class PairSignalTrainer:
    """Fit same-date OLS/OU and simulate one fixed-budget training pair."""

    def __init__(self, config=None):
        self.config = config or SignalTrainingConfig()
        self.execution = ExecutionModel(self.config.borrow_fee_rate,
            self.config.transaction_cost_rate, self.config.fx_financing_rate)

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

    @staticmethod
    def _simulation_inputs(prices, z, start, end, previous_z=None):
        """Read-only arrays reused across one pair's parameter grid in memory."""
        dates = prices.index
        selected = np.flatnonzero((dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end)))
        return (dates, selected, prices[["Open_y", "Open_x"]].to_numpy(),
                prices[["Close_y", "Close_x"]].to_numpy(), z.reindex(dates).to_numpy(),
                (previous_z if previous_z is not None else z.shift(1)).reindex(dates).to_numpy())

    def _simulate(self, prices, z, beta, entry_z, exit_z, start, end, *, details=False,
                  instrument_types=("EQUITY", "EQUITY"), unit_steps=(1, 1), risk=None, sigma_eq=None,
                  entry_policy=None, previous_z=None, _prepared=None):
        """Simulate fixed whole units with next-Open or same-Close execution.

        On a closed-market day the last known Close is the holding-cost mark.
        Close entries accrue from the next interval; existing Close holdings
        accrue before the exit decision. Settle at the final formation Close.
        The compact loop avoids building portfolio display tables for every Z.
        """
        close_execution = self.config.execution_timing == "close"
        risk = dict(risk or {}, entry_z=entry_z, exit_z=exit_z)
        use_spread_stop = spread_mode(risk)
        if use_spread_stop and (sigma_eq is None or not sigma_eq > 0):
            raise ValueError("Spread exits require a positive frozen sigma_eq.")
        if entry_policy is not None and not use_spread_stop:
            raise ValueError("Entry-policy calibration requires the spread risk model.")
        dates, selected, opening, closing, z_values, previous_values = (
            self._simulation_inputs(prices, z, start, end, previous_z) if _prepared is None else _prepared)
        if not len(selected):
            raise ValueError("empty_simulation_period")
        if close_execution and len(selected) < 2:
            raise ValueError("Close calibration requires at least two sessions")
        fills = closing if close_execution else opening
        budget = 2 * self.config.long_amount
        cash, previous_nav = budget, budget
        quantity = np.zeros(2, dtype=np.int64)
        entry_price = np.zeros(2)
        direction, pending, entry_date = 0, None, None
        pending_reason, entry_gross = "signal", 0.0
        daily_pnl, daily_costs, trade_pnl, trades, gross_reference = [], [], [], [], []
        natural_pnl = []
        entry_checks = entry_rejections = 0
        day_costs, trade_costs = np.zeros(3), np.zeros(3)  # Borrow, transaction, FX financing
        forced_exits, stop_exits = 0, 0
        previous_i = None
        pending_signal_i = None

        def fallback_signal(i, execution_number):
            if entry_policy is None or execution_number <= entry_policy["distance_fallback_after_days"]:
                return False
            value = z_values[i]
            return (np.isfinite(value) and exit_z < abs(value) < risk["stop_z"]
                    and within_entry_distance(value, entry_z, entry_policy["max_fallback_distance_z"]))

        def entry_signal(i):
            nonlocal pending_signal_i
            value = z_values[i]
            if not np.isfinite(value):
                return None
            if entry_policy is not None:
                if entry_policy["require_reversion"] and not reversion_confirmed(value, previous_values[i]):
                    return None
                execution_number = i - selected[0] + (1 if close_execution else 2)
                if abs(value) <= entry_z and not fallback_signal(i, execution_number):
                    return None
                pending_signal_i = i
                return -1 if value > 0 else 1
            return -1 if value > entry_z else (1 if value < -entry_z else None)

        def accrue(marks, days):
            nonlocal cash
            borrow, financing = self.execution.holding_costs(quantity, marks, instrument_types, days)
            costs = np.array([borrow.sum(), 0.0, financing.sum()])
            cash -= float(costs.sum())
            day_costs[:] += costs
            trade_costs[:] += costs

        def close_trade(marks, date, reason):
            nonlocal cash, direction, forced_exits, stop_exits
            gross = float(quantity @ (marks - entry_price))
            cost = float(self.execution.transaction_costs(quantity, marks).sum())
            day_costs[1] += cost
            trade_costs[1] += cost
            net = gross - float(trade_costs.sum())
            trade_pnl.append(net)
            if reason != "period_end":
                natural_pnl.append(net)
            if details:
                trades.append({"entry_date": entry_date, "exit_date": date, "direction": direction,
                    "shares_y": int(quantity[0]), "shares_x": int(quantity[1]), "entry_gross": entry_gross,
                    "gross_pnl": gross, "borrow_fees": trade_costs[0], "transaction_costs": trade_costs[1],
                    "fx_financing_fees": trade_costs[2], "total_costs": float(trade_costs.sum()),
                    "pnl": net, "exit_reason": reason})
            forced_exits += int(reason == "period_end")
            stop_exits += int(reason in ("stop", "z_stop"))
            cash += float(quantity @ marks) - cost
            quantity[:] = 0
            direction = 0

        if not close_execution and selected[0] > 0:
            pending = entry_signal(selected[0] - 1)
        for i in selected:
            day_costs[:] = 0
            exited_today = False
            reference_gross = float(np.abs(quantity) @ closing[i])
            if direction and previous_i is not None:
                closed_days = (dates[i] - dates[previous_i]).days - 1
                if closed_days > 0:
                    accrue(closing[previous_i], closed_days)
            if close_execution:
                if direction:
                    accrue(closing[i], 1)
                    net = float(quantity @ (closing[i] - entry_price)) - float(trade_costs.sum())
                    value = z_values[i]
                    if i == selected[-1]:
                        pending, pending_reason = 0, "period_end"
                    elif (spread_stop(risk, value, direction) if use_spread_stop
                          else net <= -self.config.stop_loss * entry_gross):
                        pending, pending_reason = 0, "z_stop" if use_spread_stop else "stop"
                    elif np.isfinite(value) and ((direction == 1 and value >= -exit_z)
                                                or (direction == -1 and value <= exit_z)):
                        pending, pending_reason = 0, "z_exit"
                elif i != selected[-1]:
                    pending = entry_signal(i)
            if pending is not None:
                if direction and pending == 0:
                    close_trade(fills[i], dates[i], pending_reason)
                    exited_today = True
                elif not direction and pending:
                    quantities = self.execution.size_pair(*fills[i], beta, self.config.long_amount,
                                                           pending, unit_steps)
                    allowed = True
                    if all(quantities) and use_spread_stop:
                        # Open fills must recheck the spread after any overnight gap.
                        fill_z = z_values[i] if close_execution else (
                            z_values[i - 1] + ((fills[i, 0] - closing[i - 1, 0])
                            - beta * (fills[i, 1] - closing[i - 1, 1])) / sigma_eq) if i > 0 else np.nan
                        check = entry_scenario(risk, fill_z, quantities, fills[i], sigma_eq, beta,
                                               self.execution, instrument_types)
                        if entry_policy is not None:
                            # Match a single pair's two passes: a below-threshold
                            # signal cannot become a primary signal at a later fill.
                            primary = abs(z_values[pending_signal_i]) > entry_z
                            if not primary:
                                check["allowed"] = False
                            if (not check["allowed"] and fallback_signal(pending_signal_i, i - selected[0] + 1)
                                    and within_entry_distance(fill_z, entry_z, entry_policy["max_fallback_distance_z"])):
                                check = entry_scenario(risk, fill_z, quantities, fills[i], sigma_eq, beta,
                                    self.execution, instrument_types, allow_nearby=True)
                        entry_checks += 1
                        allowed = check["allowed"]
                        entry_rejections += int(not allowed)
                    if all(quantities) and allowed:
                        direction = pending
                        quantity[:] = quantities
                        entry_price = fills[i].copy()
                        entry_gross = float(np.abs(quantity) @ fills[i])
                        entry_date = dates[i]
                        trade_costs[:] = 0
                        cost = float(self.execution.transaction_costs(quantity, fills[i]).sum())
                        trade_costs[1] = cost
                        day_costs[1] += cost
                        cash -= float(quantity @ fills[i]) + cost
            pending = None
            pending_reason = "signal"
            reference_gross = max(reference_gross, float(np.abs(quantity) @ closing[i]))
            if i == selected[-1] and direction:
                close_trade(closing[i], dates[i], "period_end")
            if direction and not close_execution:
                accrue(closing[i], 1)
            nav = cash + float(quantity @ closing[i])
            daily_pnl.append(nav - previous_nav)
            daily_costs.append(day_costs.copy())
            gross_reference.append(reference_gross)
            previous_nav, previous_i = nav, i
            if close_execution:
                continue
            value = z_values[i]
            if direction:
                current_trade_pnl = float(quantity @ (closing[i] - entry_price)) - float(trade_costs.sum())
                if (spread_stop(risk, value, direction) if use_spread_stop
                        else current_trade_pnl <= -self.config.stop_loss * entry_gross):
                    pending, pending_reason = 0, "z_stop" if use_spread_stop else "stop"
                elif np.isfinite(value) and ((direction == 1 and value >= -exit_z)
                                            or (direction == -1 and value <= exit_z)):
                    pending, pending_reason = 0, "z_exit"
            elif not (exited_today and self.config.prevent_immediate_reentry):
                pending = entry_signal(i)
        pnl_array, cost_array = np.asarray(daily_pnl), np.asarray(daily_costs)
        costs = cost_array.sum(axis=0)
        normalized = np.divide(pnl_array, gross_reference, out=np.zeros_like(pnl_array),
                               where=np.asarray(gross_reference) > 0)
        if not np.isclose(pnl_array.sum(), np.sum(trade_pnl), atol=1e-6):
            raise ArithmeticError("Training net daily P&L does not match completed net trades.")
        returns = pnl_array / budget
        volatility = float(returns.std(ddof=1)) if len(returns) > 1 else 0.0
        nav = np.r_[budget, budget + pnl_array.cumsum()]
        drawdown = nav / np.maximum.accumulate(nav) - 1
        result = {"pnl": float(pnl_array.sum()), "return_on_budget": float(returns.sum()),
                  "legacy_compound_return": float(np.prod(1 + normalized) - 1),
                  "sharpe": float(np.sqrt(252) * returns.mean() / volatility) if volatility > 1e-12 else np.nan,
                  "max_drawdown": float(drawdown.min()), "trades": len(trade_pnl),
                  "win_rate": float(np.mean(np.asarray(trade_pnl) > 0)) if trade_pnl else np.nan,
                  "forced_exits": forced_exits, "stop_exits": stop_exits, "observations": len(selected),
                  "gross_pnl": float(pnl_array.sum() + costs.sum()), "borrow_fees": float(costs[0]),
                  "transaction_costs": float(costs[1]), "fx_financing_fees": float(costs[2]),
                  "total_costs": float(costs.sum())}
        result.update(payoff_statistics(natural_pnl))
        result.update(entry_checks=entry_checks, entry_rejections=entry_rejections)
        if details:
            result["daily_pnl"] = pd.Series(pnl_array, index=dates[selected], name="daily_pnl")
            result["daily_costs"] = pd.DataFrame(cost_array, index=dates[selected],
                columns=["borrow_fee", "transaction_cost", "fx_financing_fee"])
            result["trade_log"] = pd.DataFrame(trades, columns=[
                "entry_date", "exit_date", "direction", "shares_y", "shares_x", "entry_gross",
                "gross_pnl", "borrow_fees", "transaction_costs", "fx_financing_fees", "total_costs", "pnl", "exit_reason"])
        return result
