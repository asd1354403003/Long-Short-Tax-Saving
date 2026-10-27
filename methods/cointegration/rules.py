"""Shared spread exits, conservative entry scenarios and realized net payoff."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PayoffRules:
    """Predeclared research settings; no threshold is selected using test data.

    Z is the project's legacy mean-square spread scale, not a normal quantile.
    Cost forecasts are scenarios, not guaranteed fills or bounded losses.
    """

    entries: tuple = (1.5, 2.0, 2.5, 3.0)
    exits: tuple = (0.0, 0.5)
    stop_gaps: tuple = (0.5, 1.0)
    min_expected_ratio: float = 1.5
    min_realized_payoff: float = 1.2
    min_winners: int = 2
    min_losers: int = 2
    overshoot_z: float = 0.25
    entry_band_z: float = 0.25
    rounding_move_fraction: float = 0.02
    min_win_rate: float = 0.0

    def __post_init__(self):
        if any(isinstance(value, bool) or not isinstance(value, int) or value < 1
               for value in (self.min_winners, self.min_losers)):
            raise ValueError("Winner and loser minimums must be positive integers.")
        if (isinstance(self.min_win_rate, bool) or not np.isfinite(self.min_win_rate)
                or not 0 <= self.min_win_rate <= 1):
            raise ValueError("Minimum natural net win rate must be in [0,1].")
        values = (*self.entries, *self.exits, *self.stop_gaps,
                  self.min_expected_ratio, self.min_realized_payoff,
                  self.overshoot_z, self.entry_band_z, self.rounding_move_fraction)
        if not np.isfinite(values).all() or min(values) < 0:
            raise ValueError("Payoff settings must be finite and nonnegative.")
        if (not self.entries or not self.exits or not self.stop_gaps
                or min(self.entries) <= max(self.exits) or min(self.stop_gaps) <= 0
                or self.min_expected_ratio < 1 or self.min_realized_payoff < 1
                or self.min_winners < 1 or self.min_losers < 1):
            raise ValueError("Use positive gaps, entry > exit and observed winners and losers.")

    def model_fields(self, entry, gap, half_life_days):
        # Three formation half-lives, converted from sessions to calendar days.
        # This only forecasts fees; actual fees accrue for every day held.
        horizon = float(np.clip(3 * half_life_days * 365.25 / 252, 7, 90))
        return dict(risk_mode="spread", stop_z=entry + gap,
                    min_entry_reward_risk=self.min_expected_ratio,
                    holding_cost_days=horizon, stop_overshoot_z=self.overshoot_z,
                    entry_band_z=min(self.entry_band_z, gap / 2),
                    rounding_move_fraction=self.rounding_move_fraction)


RISK_FIELDS = ("risk_mode", "stop_z", "min_entry_reward_risk", "holding_cost_days",
               "stop_overshoot_z", "entry_band_z", "rounding_move_fraction")


def spread_mode(model):
    return model.get("risk_mode") == "spread"


def spread_stop(model, z, direction, *, stop_z=None):
    threshold = model["stop_z"] if stop_z is None else stop_z
    return np.isfinite(z) and direction * z <= -float(threshold)


def reversion_confirmed(value, previous):
    """Require a same-side move toward the mean under one frozen model."""
    return bool(np.isfinite(value) and np.isfinite(previous)
                and np.sign(value) == np.sign(previous) and abs(value) < abs(previous))


def within_entry_distance(value, entry, cap):
    return cap is None or abs(abs(value) - entry) <= cap + 1e-12


def entry_scenario(model, z, quantities, prices, sigma, beta, execution,
                   instrument_types=("EQUITY", "EQUITY"), *, allow_nearby=False):
    """Hold X's price fixed and move Y to the frozen spread target/stop.

    Integer hedge mismatch gets a separate adverse 2% X-price allowance.
    Forecast round-trip transaction fees use scenario notionals. Holding fees
    use the largest scenario mark on each leg, for the forecast horizon.
    Stops receive an extra spread overshoot allowance. None is a loss limit.
    allow_nearby bypasses only the trained entry threshold/band; the mean-exit,
    spread-stop, direction, forecast cost and net reward/risk guards remain.
    """
    if not spread_mode(model):
        return dict(allowed=True, reason="legacy", expected_net_reward=np.nan,
                    expected_net_risk=np.nan, expected_reward_risk=np.nan)
    q = np.asarray(quantities, dtype=float)
    p = np.asarray(prices, dtype=float)
    value = abs(z)
    result = dict(allowed=False, reason="outside_exit_stop" if allow_nearby else "outside_entry_band", expected_net_reward=np.nan,
                  expected_net_risk=np.nan, expected_reward_risk=np.nan)
    if (not np.isfinite(z) or np.sign(z) != -np.sign(q[0])
            or not model["exit_z"] < value < model["stop_z"]):
        return result
    if not allow_nearby and (value <= model["entry_z"]
            or value > model["entry_z"] + model["entry_band_z"]):
        return result
    direction = np.sign(q[0])
    target, stop = p.copy(), p.copy()
    target[0] += direction * sigma * (value - model["exit_z"])
    stop[0] -= direction * sigma * (model["stop_z"] - value + model["stop_overshoot_z"])
    if min(*target, *stop) <= 0:
        result["reason"] = "nonpositive_scenario_price"
        return result
    entry_fee = execution.transaction_costs(q, p).sum()
    target_fee = execution.transaction_costs(q, target).sum()
    stop_fee = execution.transaction_costs(q, stop).sum()
    fee_marks = np.maximum.reduce([p, target, stop])
    borrow, financing = execution.holding_costs(q, fee_marks, instrument_types, model["holding_cost_days"])
    holding = borrow.sum() + financing.sum()
    rounding = abs(q[1] + beta * q[0]) * p[1] * model["rounding_move_fraction"]
    reward = float(q @ (target - p) - entry_fee - target_fee - holding - rounding)
    risk = float(-q @ (stop - p) + entry_fee + stop_fee + holding + rounding)
    ratio = reward / risk if risk > 0 else np.nan
    allowed = bool(reward > 0 and risk > 0 and ratio >= model["min_entry_reward_risk"])
    return dict(allowed=allowed, reason="passed" if allowed else "insufficient_net_reward_risk",
                expected_net_reward=reward, expected_net_risk=risk, expected_reward_risk=ratio)


def payoff_statistics(pnls):
    """Natural completed NET rounds only; no infinite score for zero losers."""
    values = np.asarray(pnls, dtype=float)
    wins, losses = values[values > 0], values[values < 0]
    mean_win = float(wins.mean()) if len(wins) else np.nan
    mean_loss = float(losses.mean()) if len(losses) else np.nan
    return dict(natural_trades=len(values), winners=len(wins), losers=len(losses),
                natural_win_rate=float(len(wins) / len(values)) if len(values) else np.nan,
                mean_winner_net_pnl=mean_win, mean_loser_net_pnl=mean_loss,
                payoff_ratio=mean_win / -mean_loss if len(wins) and len(losses) else np.nan,
                profit_factor=float(wins.sum() / -losses.sum()) if len(losses) else np.nan)


def passes_payoff(stats, rules):
    # Setting the configurable minimum to 1 still requires strictly >1.
    return (stats["winners"] >= rules.min_winners and stats["losers"] >= rules.min_losers
            and stats["payoff_ratio"] > 1.0
            and stats["payoff_ratio"] >= rules.min_realized_payoff
            and stats["profit_factor"] > 1
            and (rules.min_win_rate == 0 or stats["winners"] / max(1, stats["natural_trades"]) >= rules.min_win_rate))
