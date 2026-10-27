"""Execute the retained frozen cointegration pool with finite pair capacity."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from market_data import YahooFinanceSource
from .execution import ExecutionModel
from .rules import (RISK_FIELDS, entry_scenario, spread_mode, spread_stop,
                         reversion_confirmed, within_entry_distance)


@dataclass(frozen=True)
class PairBacktestConfig:
    max_pairs: int = 15
    long_amount: float = 1_000_000
    start_date: str = "2026-01-01"
    end_date: str | None = None
    allow_shared_tickers: bool = False
    borrow_fee_rate: float = 0.01
    transaction_cost_rate: float = 0.0005
    fx_financing_rate: float = 0.0
    execution_timing: str = "next_open"
    entry_selection: str = "sharpe"
    distance_fallback_after_days: int = 10
    require_reversion: bool = False
    max_fallback_distance_z: float | None = None
    use_pair_entry_filters: bool = False
    exit_stop_z: float | None = None
    exit_stop_loss_amount: float | None = None
    exit_take_profit_amount: float | None = None
    exit_on_mean_reversion: bool = True
    enable_stop_loss: bool = True

    def __post_init__(self):
        if isinstance(self.max_pairs, bool) or not isinstance(self.max_pairs, int) or self.max_pairs < 1:
            raise ValueError("max_pairs must be a positive integer")
        if not np.isfinite(self.long_amount) or self.long_amount <= 0:
            raise ValueError("long_amount must be positive and finite")
        pd.Timestamp(self.start_date)
        if self.end_date and pd.Timestamp(self.end_date) < pd.Timestamp(self.start_date):
            raise ValueError("end_date must not precede start_date")
        if self.execution_timing not in ("next_open", "close"):
            raise ValueError("execution_timing must be next_open or close")
        if self.entry_selection not in ("sharpe", "sharpe_then_distance"):
            raise ValueError("entry_selection must be sharpe or sharpe_then_distance")
        if (isinstance(self.distance_fallback_after_days, bool)
                or not isinstance(self.distance_fallback_after_days, int)
                or self.distance_fallback_after_days < 0):
            raise ValueError("distance_fallback_after_days must be a nonnegative integer")
        if not isinstance(self.require_reversion, bool):
            raise ValueError("require_reversion must be boolean")
        if not isinstance(self.use_pair_entry_filters, bool):
            raise ValueError("use_pair_entry_filters must be boolean")
        if self.exit_stop_z is not None and (isinstance(self.exit_stop_z, bool)
                or not np.isfinite(self.exit_stop_z) or self.exit_stop_z <= 0):
            raise ValueError("exit_stop_z must be positive and finite or None")
        if self.exit_stop_loss_amount is not None and (
                isinstance(self.exit_stop_loss_amount, bool)
                or not np.isfinite(self.exit_stop_loss_amount) or self.exit_stop_loss_amount <= 0):
            raise ValueError("exit_stop_loss_amount must be positive and finite or None")
        if self.exit_stop_z is not None and self.exit_stop_loss_amount is not None:
            raise ValueError("Choose either a dollar stop or a Z stop override.")
        if self.exit_take_profit_amount is not None and (
                isinstance(self.exit_take_profit_amount, bool)
                or not np.isfinite(self.exit_take_profit_amount) or self.exit_take_profit_amount <= 0):
            raise ValueError("exit_take_profit_amount must be positive and finite or None")
        if not isinstance(self.exit_on_mean_reversion, bool):
            raise ValueError("exit_on_mean_reversion must be boolean")
        if not isinstance(self.enable_stop_loss, bool):
            raise ValueError("enable_stop_loss must be boolean")
        if not self.enable_stop_loss and (self.exit_stop_z is not None or self.exit_stop_loss_amount is not None):
            raise ValueError("Disabled stops cannot have an active dollar/Z stop override.")
        if self.max_fallback_distance_z is not None and (
                isinstance(self.max_fallback_distance_z, bool)
                or not np.isfinite(self.max_fallback_distance_z)
                or self.max_fallback_distance_z < 0):
            raise ValueError("max_fallback_distance_z must be finite and nonnegative or None")
        ExecutionModel(self.borrow_fee_rate, self.transaction_cost_rate, self.fx_financing_rate)

    @property
    def leg_cap(self):
        return self.long_amount / self.max_pairs


class PairPoolBacktester:
    """Trade one frozen training-Sharpe pool with beta sizing and fixed units.

    Closed pairs can re-enter. Entry caps and whole-unit rounding precede each
    fill; quantities stay fixed until exit. Unfilled slots remain in cash.
    """

    def __init__(self, pairs: pd.DataFrame, config: PairBacktestConfig | None = None, *, source=None):
        self.config = config or PairBacktestConfig()
        self.execution = ExecutionModel(self.config.borrow_fee_rate, self.config.transaction_cost_rate,
                                        self.config.fx_financing_rate)
        self.pairs = self._prepare_pool(pairs, self.config.start_date)
        self.source = source if source is not None else YahooFinanceSource()

    @staticmethod
    def _prepare_pool(pairs, effective_date, *, allow_empty=False):
        required = {"pair_id", "y_ticker", "x_ticker", "selection_sharpe", "eligible",
                    "selection_end", "refit_status", "refit_end", "refit_alpha", "refit_beta",
                    "refit_mu_e", "refit_sigma_eq", "entry_z", "exit_z", "stop_loss"}
        if not required.issubset(pairs.columns) or pairs.pair_id.duplicated().any():
            raise ValueError("Provide the trained cointegration pool CSV with unique pair IDs and frozen model fields.")
        ranked = pairs.sort_values("selection_sharpe", ascending=False, na_position="last", kind="stable").copy()
        ranked["rank"] = np.arange(1, len(ranked) + 1)
        ranked = ranked.loc[ranked.eligible.eq(True) & ranked.refit_status.eq("ok")].copy()
        if ranked.empty and not allow_empty:
            raise ValueError("No pairs passed screening with a valid pre-start refit.")
        for field in ("selection_end", "refit_end"):
            if not pd.to_datetime(ranked[field]).lt(pd.Timestamp(effective_date)).all():
                raise ValueError("Model fitting and selection must finish before the backtest starts.")
        numeric = ["selection_sharpe", "refit_alpha", "refit_beta", "refit_mu_e", "refit_sigma_eq",
                   "entry_z", "exit_z", "stop_loss"]
        if not np.isfinite(ranked[numeric].to_numpy(dtype=float)).all():
            raise ValueError("Eligible pairs need finite model estimates and thresholds.")
        if (not ranked.refit_beta.gt(0).all() or not ranked.refit_sigma_eq.gt(0).all()
                or not ranked.entry_z.gt(ranked.exit_z).all() or not ranked.exit_z.ge(0).all()
                or not ranked.stop_loss.between(0, 1, inclusive="right").all()
                or ranked.y_ticker.eq(ranked.x_ticker).any()):
            raise ValueError("Invalid hedge ratios, scales, thresholds or pair membership.")
        ranked = ranked.reset_index(drop=True)
        for model in ranked.to_dict("records"):
            if spread_mode(model):
                fields = [name for name in RISK_FIELDS if name != "risk_mode"]
                if (not np.isfinite([model[name] for name in fields]).all()
                        or not model["stop_z"] > model["entry_z"]
                        or model["min_entry_reward_risk"] < 1
                        or min(model[name] for name in fields) < 0):
                    raise ValueError("Invalid spread stop or net reward/risk entry settings.")
        for suffix in ("y", "x"):
            if "instrument_type_" + suffix not in ranked:
                ranked["instrument_type_" + suffix] = "EQUITY"
            if "unit_step_" + suffix not in ranked:
                ranked["unit_step_" + suffix] = 1
            steps = pd.to_numeric(ranked["unit_step_" + suffix], errors="coerce")
            if not (np.isfinite(steps) & steps.ge(1) & steps.eq(np.floor(steps))).all():
                raise ValueError("Each instrument needs a positive integer unit_step.")
        return ranked


    @classmethod
    def from_csv(cls, path=Path("methods/cointegration/results/pool.csv"), config=None, **kwargs):
        return cls(pd.read_csv(path), config, **kwargs)

    @staticmethod
    def size_pair(open_y, open_x, beta, leg_cap, direction, unit_steps=(1, 1)):
        """One common continuous scale, then round both share magnitudes down."""
        return ExecutionModel.size_pair(open_y, open_x, beta, leg_cap, direction, unit_steps)

    @staticmethod
    def _clean(frame):
        if frame is None or not {"Open", "Close"}.issubset(frame.columns):
            return pd.DataFrame(columns=["Open", "Close"], index=pd.DatetimeIndex([]))
        frame = frame[["Open", "Close"]].copy()
        frame.index = pd.DatetimeIndex(pd.to_datetime(frame.index)).tz_localize(None).normalize()
        frame = frame.loc[~frame.index.duplicated(keep="last")].sort_index()
        frame = frame.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
        return frame.where(frame > 0)

    def _model_z(self, model, closing):
        return (closing[model["y_ticker"]] - model["refit_alpha"]
            - model["refit_beta"] * closing[model["x_ticker"]]
            - model["refit_mu_e"]) / model["refit_sigma_eq"]

    @staticmethod
    def ranked_entries(models, date, active, exited_today, *, distance_fallback=False,
                       require_reversion=False, max_fallback_distance_z=None):
        """Scan original signals by Sharpe, then nearby signals by Z distance.

        All ordering uses the completed signal bar. Admission still runs at
        the actual fill. A primary candidate can be retried in the second pass
        after an opening gap, but a held/accepted pair is never opened twice.
        """
        primary, nearby = [], []
        for model in models:
            pid = model["pair_id"]
            if pid in active or (pid in exited_today and model.get("prevent_immediate_reentry", True)):
                continue
            value = model["_z"].get(date, np.nan)
            if not np.isfinite(value):
                continue
            confirm = model.get("_require_reversion", require_reversion)
            cap = model.get("_max_fallback_distance_z", max_fallback_distance_z)
            if confirm:
                previous = model["_previous_z"].get(date, np.nan)
                if not reversion_confirmed(value, previous):
                    continue
            distance = abs(abs(value) - model["entry_z"])
            direction = -1 if value > 0 else 1
            if abs(value) > model["entry_z"]:
                primary.append((pid, direction, date, "sharpe", distance))
            if (distance_fallback and spread_mode(model)
                    and model["exit_z"] < abs(value) < model["stop_z"]
                    and within_entry_distance(value, model["entry_z"], cap)):
                nearby.append((distance, -model["selection_sharpe"], model["rank"],
                               (pid, direction, date, "distance", distance)))
        nearby.sort(key=lambda item: item[:3])
        return primary + [item[3] for item in nearby]

    @staticmethod
    def _event(state, date, signal_date, event, prices, reason, pnls=(0.0, 0.0), transaction_cost=0.0):
        positive = 0 if state["qy"] > 0 else 1
        negative = 1 - positive
        tickers, shares = (state["y"], state["x"]), (state["qy"], state["qx"])
        entry = (state["entry_y"], state["entry_x"])
        costs = state.get("costs", np.zeros(3))
        types = state.get("instrument_types", ("EQUITY", "EQUITY"))
        return {"execution_date": date, "signal_date": signal_date, "event": event,
                "trade_id": state["trade_id"], "pair_id": state["pair_id"],
                "rank": state["rank"], "entry_date": state["entry_date"],
                "model_effective_date": state.get("model_effective_date", pd.NaT),
                "model_training_end": state.get("model_training_end", pd.NaT),
                "entry_model_effective_date": state.get("entry_model_effective_date", pd.NaT),
                "entry_model_training_end": state.get("entry_model_training_end", pd.NaT),
                "signal_beta": state.get("model", {}).get("refit_beta", state["beta"]),
                "signal_alpha": state.get("model", {}).get("refit_alpha", np.nan),
                "signal_mu_e": state.get("model", {}).get("refit_mu_e", np.nan),
                "signal_sigma_eq": state.get("model", {}).get("refit_sigma_eq", np.nan),
                "entry_half_life_days": state.get("model", {}).get("selection_half_life_days", np.nan),
                "entry_z": state.get("model", {}).get("entry_z", np.nan),
                "exit_z": state.get("model", {}).get("exit_z", np.nan),
                "stop_loss": state.get("model", {}).get("stop_loss", np.nan),
                "risk_mode": state.get("model", {}).get("risk_mode", "notional"),
                "stop_z": state.get("model", {}).get("stop_z", np.nan),
                "exit_stop_z": state.get("exit_stop_z", state.get("model", {}).get("stop_z", np.nan)),
                "exit_stop_loss_amount": state.get("exit_stop_loss_amount", np.nan),
                "exit_take_profit_amount": state.get("exit_take_profit_amount", np.nan),
                "exit_on_mean_reversion": state.get("exit_on_mean_reversion", True),
                "enable_stop_loss": state.get("enable_stop_loss", True),
                "expected_net_reward": state.get("entry_scenario", {}).get("expected_net_reward", np.nan),
                "expected_net_risk": state.get("entry_scenario", {}).get("expected_net_risk", np.nan),
                "expected_reward_risk": state.get("entry_scenario", {}).get("expected_reward_risk", np.nan),
                "selection_stage": state.get("selection_stage", "sharpe"),
                "signal_distance_z": state.get("signal_distance_z", np.nan),
                "entry_distance_z": state.get("entry_distance_z", np.nan),
                "signal_z": state.get("signal_z", np.nan),
                "previous_signal_z": state.get("previous_signal_z", np.nan),
                "entry_require_reversion": state.get("model", {}).get("_require_reversion", False),
                "entry_max_fallback_distance_z": state.get("model", {}).get("_max_fallback_distance_z", np.nan),
                "long_ticker": tickers[positive], "short_ticker": tickers[negative],
                "long_instrument_type": types[positive], "short_instrument_type": types[negative],
                "long_shares": shares[positive], "short_shares": shares[negative],
                "long_price": prices[positive], "short_price": prices[negative],
                "long_amount": shares[positive] * prices[positive],
                "short_amount": shares[negative] * prices[negative],
                "long_entry_price": entry[positive], "short_entry_price": entry[negative],
                "long_realized_pnl": pnls[positive], "short_realized_pnl": pnls[negative],
                "realized_pnl": sum(pnls), "reason": reason,
                "transaction_cost": transaction_cost,
                "trade_borrow_fee": float(costs[0]), "trade_transaction_cost": float(costs[1]),
                "trade_fx_financing_fee": float(costs[2]), "trade_total_cost": float(costs.sum()),
                "net_realized_pnl": sum(pnls) - float(costs.sum()) if event == "close" else 0.0,
                "hedge_beta": state["beta"] if positive == 0 else 1 / state["beta"]}

    def run(self, *, histories=None):
        config = self.config
        close_execution = config.execution_timing == "close"
        start = pd.Timestamp(config.start_date)
        # Include today's bar only after 17:00 New York time, allowing a data-publication buffer.
        end = (pd.Timestamp(config.end_date) if config.end_date else
               (pd.Timestamp.now(tz="America/New_York") - pd.Timedelta(hours=17)).normalize().tz_localize(None))
        if end < start:
            raise ValueError("No completed backtest dates are available.")
        download_start = start - pd.Timedelta(days=10)
        tickers = list(dict.fromkeys(self.pairs[["y_ticker", "x_ticker"]].to_numpy().ravel()))
        if histories is None:
            histories = self.source.history(list(dict.fromkeys(tickers + ["SPY"])),
                download_start.strftime("%Y-%m-%d"), (end + pd.Timedelta(days=1)).strftime("%Y-%m-%d"))
        clean = {ticker: self._clean(histories.get(ticker)).loc[download_start:end]
                 for ticker in set(tickers + ["SPY"])}
        spy = clean["SPY"].dropna()
        dates = spy.index[spy.index >= start]
        if dates.empty:
            raise ValueError("SPY needs valid Open/Close prices over the backtest dates.")
        all_dates = spy.index
        opening = pd.DataFrame({t: clean[t].Open.reindex(all_dates) for t in tickers})
        closing = pd.DataFrame({t: clean[t].Close.reindex(all_dates) for t in tickers})
        fill_prices = closing if close_execution else opening
        def model_records(pool, effective_date):
            records = pool.to_dict("records")
            for model in records:
                if config.exit_stop_z is not None and (
                        not spread_mode(model) or config.exit_stop_z <= model["entry_z"]):
                    raise ValueError("The exit-only override requires a spread model and stop above entry Z.")
                model["_effective_date"] = effective_date
                model["_z"] = self._model_z(model, closing)
                # Both bars use the same frozen model.
                model["_previous_z"] = model["_z"].shift(1)
                confirm, cap = config.require_reversion, config.max_fallback_distance_z
                if config.use_pair_entry_filters:
                    confirm = model.get("pair_require_reversion")
                    if not isinstance(confirm, (bool, np.bool_)) or "pair_max_fallback_distance_z" not in model:
                        raise ValueError("Each model must provide its trained pair entry filters.")
                    raw_cap = model["pair_max_fallback_distance_z"]
                    cap = None if pd.isna(raw_cap) else float(raw_cap)
                    if cap is not None and (not np.isfinite(cap) or cap < 0):
                        raise ValueError("A trained fallback cap must be nonnegative or missing for unlimited.")
                model["_require_reversion"] = bool(confirm)
                model["_max_fallback_distance_z"] = cap
            return records

        models = model_records(self.pairs, start)
        by_id = {m["pair_id"]: m for m in models}
        z_scores = pd.DataFrame(np.nan, index=all_dates, columns=self.pairs.pair_id.unique())
        for model in models:
            z_scores[model["pair_id"]] = model["_z"]

        def scan(date, active, exited_today):
            execution_day = dates.searchsorted(date, side="right") + (0 if close_execution else 1)
            use_distance = (config.entry_selection == "sharpe_then_distance"
                and execution_day > config.distance_fallback_after_days)
            return self.ranked_entries(models, date, active, exited_today,
                distance_fallback=use_distance, require_reversion=config.require_reversion,
                max_fallback_distance_z=config.max_fallback_distance_z)

        long_positions = pd.DataFrame(0.0, index=dates, columns=tickers)
        short_positions = long_positions.copy()
        active, traded_pairs, pending_exits = {}, set(), {}
        entry_count = 0
        previous = all_dates[all_dates < dates[0]]
        pending_entries = scan(previous[-1], active, set()) if len(previous) else []
        cash = {"long": config.long_amount, "short": config.long_amount}
        previous_nav = cash.copy()
        realized_gain = realized_loss = 0.0
        cumulative_costs = np.zeros(3)  # Borrow, transaction, FX financing
        day_costs = np.zeros(3)
        cost_ledger = {}
        net_realized = 0.0
        previous_date = None
        events, snapshots, summaries, decision_log, entry_log = [], [], [], [], []

        def charge(state, kind, amounts):
            total = float(np.sum(amounts))
            state["costs"][kind] += total
            day_costs[kind] += total
            cost_ledger[state["pair_id"]][kind] += total
            for q, cost in zip((state["qy"], state["qx"]), amounts):
                cash["long" if q > 0 else "short"] -= cost
            return total

        def accrue(state, marks, days):
            borrow, financing = self.execution.holding_costs(
                (state["qy"], state["qx"]), marks, state["instrument_types"], days)
            charge(state, 0, borrow)
            charge(state, 2, financing)

        def exit_reason(state, date, net_pnl):
            model = state["model"]
            value = model["_z"].get(date, np.nan)
            direction = 1 if state["qy"] > 0 else -1
            liquidation_net = net_pnl
            if config.exit_stop_loss_amount is not None or config.exit_take_profit_amount is not None:
                marks = closing.loc[date, [state["y"], state["x"]]].to_numpy(dtype=float)
                closing_fee = self.execution.transaction_costs((state["qy"], state["qx"]), marks).sum()
                liquidation_net -= closing_fee
            if state["enable_stop_loss"]:
                if config.exit_stop_loss_amount is not None:
                    # Replace executed stops, retaining the entry forecast.
                    if liquidation_net <= -state["exit_stop_loss_amount"]:
                        return "dollar_stop"
                elif spread_mode(model):
                    if spread_stop(model, value, direction, stop_z=state["exit_stop_z"]):
                        return "z_stop"
                elif net_pnl <= -model["stop_loss"] * state["entry_gross"]:
                    return "stop"
            if (config.exit_take_profit_amount is not None
                    and liquidation_net >= state["exit_take_profit_amount"]):
                return "dollar_take_profit"
            if state["exit_on_mean_reversion"] and ((direction == 1 and value >= -model["exit_z"])
                    or (direction == -1 and value <= model["exit_z"])):
                return "z_exit"
            return None

        for date in dates:
            day_costs[:] = 0
            # Non-session calendar days use the last already-known Close mark.
            closed_days = (date - previous_date).days - 1 if previous_date is not None else 0
            if closed_days:
                for state in active.values():
                    accrue(state, (state["mark_y"], state["mark_x"]), closed_days)
            day_gain = day_loss = 0.0
            exited_today = set()
            for model in models:
                z_scores.at[date, model["pair_id"]] = model["_z"].get(date, np.nan)
            if close_execution:
                # Existing positions earn the Close-to-Close move and accrue
                # this session's borrow before the closing-auction decision.
                # New Close entries start accruing on the following interval.
                for pid, state in active.items():
                    quotes = closing.loc[date, [state["y"], state["x"]]].to_numpy(dtype=float)
                    for side, quote in zip(("y", "x"), quotes):
                        if np.isfinite(quote):
                            state["mark_" + side] = quote
                    marks = (state["mark_y"], state["mark_x"])
                    accrue(state, marks, 1)
                    net = (state["qy"] * (marks[0] - state["entry_y"])
                           + state["qx"] * (marks[1] - state["entry_x"]) - float(state["costs"].sum()))
                    z_scores.at[date, pid] = state["model"]["_z"].get(date, np.nan)
                    reason = exit_reason(state, date, net) if np.isfinite(quotes).all() else None
                    if pid not in pending_exits and reason:
                        pending_exits[pid] = (reason, date)
                        decision_log.append(dict(signal_date=date, trade_id=state["trade_id"], pair_id=pid,
                            reason=reason, net_pnl_before_exit=net, entry_gross=state["entry_gross"],
                            stop_loss=state["model"]["stop_loss"], z_score=state["model"]["_z"].get(date),
                            exit_z=state["model"]["exit_z"], direction=1 if state["qy"] > 0 else -1,
                            risk_mode=state["model"].get("risk_mode", "notional"),
                            stop_z=state["model"].get("stop_z", np.nan),
                            exit_stop_z=state["exit_stop_z"],
                            exit_stop_loss_amount=state["exit_stop_loss_amount"],
                            exit_take_profit_amount=state["exit_take_profit_amount"],
                            exit_on_mean_reversion=state["exit_on_mean_reversion"],
                            enable_stop_loss=state["enable_stop_loss"],
                            net_pnl_after_exit_cost=net - self.execution.transaction_costs(
                                (state["qy"], state["qx"]), marks).sum()))
                pending_entries = scan(date, active, set())
            for pid in list(pending_exits):
                state = active[pid]
                prices = fill_prices.loc[date, [state["y"], state["x"]]].to_numpy(dtype=float)
                if not np.isfinite(prices).all():
                    continue
                reason, signal_date = pending_exits.pop(pid)
                quantities = (state["qy"], state["qx"])
                pnls = tuple(q * (p - cost) for q, p, cost in zip(quantities, prices,
                                                               (state["entry_y"], state["entry_x"])))
                fee = charge(state, 1, self.execution.transaction_costs(quantities, prices))
                for q, price, pnl in zip(quantities, prices, pnls):
                    cash["long" if q > 0 else "short"] += q * price
                    day_gain += max(pnl, 0)
                    day_loss += min(pnl, 0)
                event = self._event(state, date, signal_date, "close", prices, reason, pnls, fee)
                events.append(event)
                net_realized += event["net_realized_pnl"]
                del active[pid]
                exited_today.add(pid)
            for pid, direction, signal_date, selection_stage, signal_distance in pending_entries:
                if len(active) >= config.max_pairs:
                    break
                model = by_id[pid]
                y, x = model["y_ticker"], model["x_ticker"]
                if pid in active:
                    continue
                occupied = {t for state in active.values() for t in (state["y"], state["x"])}
                if not config.allow_shared_tickers and ({y, x} & occupied):
                    continue
                prices = fill_prices.loc[date, [y, x]].to_numpy(dtype=float)
                if not np.isfinite(prices).all():
                    continue
                steps = (int(model["unit_step_y"]), int(model["unit_step_x"]))
                qy, qx = self.size_pair(*prices, model["refit_beta"], config.leg_cap, direction, steps)
                if not qy or not qx:
                    continue
                fill_z = ((prices[0] - model["refit_alpha"] - model["refit_beta"] * prices[1]
                           - model["refit_mu_e"]) / model["refit_sigma_eq"])
                check = entry_scenario(model, fill_z, (qy, qx), prices, model["refit_sigma_eq"],
                                       model["refit_beta"], self.execution,
                                       (model["instrument_type_y"], model["instrument_type_x"]),
                                       allow_nearby=selection_stage == "distance")
                fill_distance = abs(abs(fill_z) - model["entry_z"])
                if (selection_stage == "distance"
                        and not within_entry_distance(fill_z, model["entry_z"], model["_max_fallback_distance_z"])):
                    check.update(allowed=False, reason="fallback_distance_limit")
                signal_z = model["_z"].get(signal_date, np.nan)
                previous_z = model["_previous_z"].get(signal_date, np.nan)
                if spread_mode(model):
                    entry_log.append(dict(execution_date=date, signal_date=signal_date,
                        pair_id=pid, rank=model["rank"], z_score=fill_z,
                        selection_stage=selection_stage, signal_distance_z=signal_distance,
                        entry_distance_z=fill_distance, signal_z=signal_z,
                        previous_signal_z=previous_z,
                        entry_require_reversion=model["_require_reversion"],
                        entry_max_fallback_distance_z=model["_max_fallback_distance_z"], **check))
                if not check["allowed"]:
                    continue
                entry_count += 1
                state = {"trade_id": entry_count, "pair_id": pid, "rank": model["rank"],
                         "y": y, "x": x, "qy": qy, "qx": qx,
                         "entry_y": prices[0], "entry_x": prices[1], "mark_y": prices[0], "mark_x": prices[1],
                         "entry_date": date, "beta": model["refit_beta"], "model": model,
                          "exit_stop_z": (np.nan if not config.enable_stop_loss or config.exit_stop_loss_amount is not None else
                              config.exit_stop_z if config.exit_stop_z is not None else model.get("stop_z", np.nan)),
                          "exit_stop_loss_amount": (config.exit_stop_loss_amount
                              if config.exit_stop_loss_amount is not None else np.nan),
                          "exit_take_profit_amount": (config.exit_take_profit_amount
                              if config.exit_take_profit_amount is not None else np.nan),
                          "exit_on_mean_reversion": config.exit_on_mean_reversion,
                          "enable_stop_loss": config.enable_stop_loss,
                         "model_effective_date": model["_effective_date"],
                         "model_training_end": pd.Timestamp(model["refit_end"]),
                         "entry_model_effective_date": model["_effective_date"],
                         "entry_model_training_end": pd.Timestamp(model["refit_end"]),
                         "instrument_types": (model["instrument_type_y"], model["instrument_type_x"]),
                         "costs": np.zeros(3),
                         "entry_scenario": check,
                         "selection_stage": selection_stage, "signal_distance_z": signal_distance,
                          "signal_z": signal_z, "previous_signal_z": previous_z,
                         "entry_distance_z": abs(abs(fill_z) - model["entry_z"]),
                         "entry_gross": abs(qy) * prices[0] + abs(qx) * prices[1]}
                cost_ledger.setdefault(pid, np.zeros(3))
                fee = charge(state, 1, self.execution.transaction_costs((qy, qx), prices))
                for q, price in zip((qy, qx), prices):
                    cash["long" if q > 0 else "short"] -= q * price
                active[pid] = state
                traded_pairs.add(pid)
                events.append(self._event(state, date, signal_date, "open", prices, "z_entry", transaction_cost=fee))
            realized_gain += day_gain
            realized_loss += day_loss
            day_long = day_short = unrealized = 0.0
            net_unrealized = 0.0
            stale_legs = 0
            for pid, state in active.items():
                model = state["model"]
                quotes = closing.loc[date, [state["y"], state["x"]]].to_numpy(dtype=float)
                for side, quote in zip(("y", "x"), quotes):
                    if np.isfinite(quote):
                        state["mark_" + side] = quote
                    else:
                        stale_legs += 1
                marks = (state["mark_y"], state["mark_x"])
                if not close_execution:
                    accrue(state, marks, 1)
                pair_pnl = state["qy"] * (marks[0] - state["entry_y"]) + state["qx"] * (marks[1] - state["entry_x"])
                pair_net_pnl = pair_pnl - float(state["costs"].sum())
                unrealized += pair_pnl
                net_unrealized += pair_net_pnl
                snapshot = self._event(state, date, pd.NaT, "hold", marks, "")
                snapshot.update(Date=date, unrealized_pnl=pair_pnl, net_unrealized_pnl=pair_net_pnl,
                                z_score=model["_z"].get(date, np.nan),
                                holding_days=(date - state["entry_date"]).days)
                snapshots.append(snapshot)
                for ticker, q, mark in zip((state["y"], state["x"]), (state["qy"], state["qx"]), marks):
                    if q > 0:
                        long_positions.at[date, ticker] += q * mark
                        day_long += q * mark
                    else:
                        short_positions.at[date, ticker] += q * mark
                        day_short += q * mark
                value = model["_z"].get(date, np.nan)
                # Report the signal actually governing a held lot, including
                # lots no longer listed in the current entry pool.
                z_scores.at[date, pid] = value
                if not close_execution and pid not in pending_exits and np.isfinite(quotes).all():
                    reason = exit_reason(state, date, pair_net_pnl)
                    if reason:
                        pending_exits[pid] = (reason, date)
            nav_long, nav_short = cash["long"] + day_long, cash["short"] + day_short
            long_pnl, short_pnl = nav_long - previous_nav["long"], nav_short - previous_nav["short"]
            equity = nav_long + nav_short
            cumulative_pnl = equity - 2 * config.long_amount
            cumulative_costs += day_costs
            gross_cumulative_pnl = realized_gain + realized_loss + unrealized
            if (not np.isclose(cumulative_pnl, gross_cumulative_pnl - cumulative_costs.sum(), atol=1e-6)
                    or not np.isclose(cumulative_pnl, net_realized + net_unrealized, atol=1e-6)):
                raise ArithmeticError("Net cash/position P&L does not reconcile with price P&L minus costs.")
            summaries.append({"Date": date, "active_pairs": len(active), "entry_count": entry_count,
                "traded_pairs": len(traded_pairs),
                "available_pairs": len(models) - sum(pid in by_id for pid in active) - sum(
                    bool(by_id[pid].get("prevent_immediate_reentry", True))
                    for pid in exited_today if pid in by_id and pid not in active),
                "long_amount": day_long, "short_amount": day_short, "gross_amount": day_long - day_short,
                "net_amount": day_long + day_short, "cash": cash["long"] + cash["short"], "equity": equity,
                "long_pnl": long_pnl, "short_pnl": short_pnl, "daily_pnl": long_pnl + short_pnl,
                "gross_daily_pnl": long_pnl + short_pnl + float(day_costs.sum()),
                "borrow_fee": day_costs[0], "transaction_cost": day_costs[1],
                "fx_financing_fee": day_costs[2], "total_cost": float(day_costs.sum()),
                "cumulative_borrow_fee": cumulative_costs[0],
                "cumulative_transaction_cost": cumulative_costs[1],
                "cumulative_fx_financing_fee": cumulative_costs[2],
                "cumulative_total_cost": float(cumulative_costs.sum()),
                "gross_cumulative_pnl": gross_cumulative_pnl,
                "cumulative_net_realized_pnl": net_realized, "net_unrealized_pnl": net_unrealized,
                "daily_return": (long_pnl + short_pnl) / (2 * config.long_amount),
                "long_return": long_pnl / config.long_amount, "short_return": short_pnl / config.long_amount,
                "cumulative_pnl": cumulative_pnl, "cumulative_return": cumulative_pnl / (2 * config.long_amount),
                "realized_gain": day_gain, "realized_loss": day_loss,
                "cumulative_realized_gain": realized_gain, "cumulative_realized_loss": realized_loss,
                "cumulative_realized_pnl": realized_gain + realized_loss, "unrealized_pnl": unrealized,
                "stale_price_legs": stale_legs})
            previous_nav = {"long": nav_long, "short": nav_short}
            previous_date = date
            # Restart at the highest rank each Close; past trades do not blacklist a pair.
            if not close_execution:
                pending_entries = scan(date, active, exited_today)
        daily = pd.DataFrame(summaries).set_index("Date")
        daily["spy_close"] = spy.Close.reindex(dates)
        daily["spy_return"] = daily.spy_close.pct_change(fill_method=None)
        first_open = spy.at[dates[0], "Open"]
        daily.loc[dates[0], "spy_return"] = daily.spy_close.iloc[0] / first_open - 1
        daily["spy_cumulative_return"] = daily.spy_close / first_open - 1
        event_columns = list(self._event({"trade_id": 0, "qy": 1, "qx": -1, "y": "", "x": "", "entry_y": 0,
            "entry_x": 0, "pair_id": "", "rank": 0, "entry_date": pd.NaT, "beta": 1},
            pd.NaT, pd.NaT, "", (0, 0), ""))
        event_frame = pd.DataFrame(events, columns=event_columns)
        position_frame = pd.DataFrame(snapshots, columns=event_columns + [
            "Date", "unrealized_pnl", "net_unrealized_pnl", "z_score", "holding_days"])
        open_positions = position_frame.loc[position_frame.Date.eq(dates[-1])].sort_values("rank")
        closed_trades = event_frame.loc[event_frame.event.eq("close")].copy()
        realized = closed_trades.groupby("pair_id").agg(
            closed_trades=("trade_id", "size"), realized_pnl=("realized_pnl", "sum"),
            net_realized_pnl=("net_realized_pnl", "sum"))
        unrealized = open_positions.set_index("pair_id")[["unrealized_pnl", "net_unrealized_pnl", "holding_days"]]
        performance = realized.join(unrealized, how="outer").fillna(0)
        performance = performance.join(pd.DataFrame.from_dict(cost_ledger, orient="index",
            columns=["borrow_fee", "transaction_cost", "fx_financing_fee"])).fillna(0)
        performance["total_cost"] = performance[["borrow_fee", "transaction_cost", "fx_financing_fee"]].sum(axis=1)
        performance["gross_total_pnl"] = performance.realized_pnl + performance.unrealized_pnl
        performance["total_pnl"] = performance.gross_total_pnl - performance.total_cost
        performance = performance.join(self.pairs.set_index("pair_id")[["rank", "selection_sharpe"]])
        performance = performance.sort_values("total_pnl")
        if not np.isclose(performance.total_pnl.sum(), daily.cumulative_pnl.iloc[-1], atol=1e-6):
            raise ArithmeticError("Pair-level contributions do not reconcile with account P&L.")
        for frame in (long_positions, short_positions):
            frame.index.name, frame.columns.name = "Date", "ticker"
        return {"daily_summary": daily, "daily_positions_long": long_positions,
                "daily_positions_short": short_positions, "pair_positions": position_frame,
                "events": event_frame, "closed_trades": closed_trades,
                "pair_performance": performance,
                "open_positions": open_positions, "ranked_pairs": self.pairs,
                "z_scores": z_scores.reindex(dates), "pending_exits": pending_exits,
                "decision_log": pd.DataFrame(decision_log),
                "entry_checks": pd.DataFrame(entry_log, columns=["execution_date", "signal_date", "pair_id",
                    "rank", "z_score", "selection_stage", "signal_distance_z", "entry_distance_z",
                     "signal_z", "previous_signal_z", "entry_require_reversion", "entry_max_fallback_distance_z",
                    "allowed", "reason", "expected_net_reward", "expected_net_risk",
                    "expected_reward_risk"]),
                "missing_tickers": [t for t in tickers if clean[t].dropna(how="all").empty],
                "source_events": list(self.source.events)}
