"""Shared fee model and independent long/short execution."""
from dataclasses import dataclass
import numpy as np
import pandas as pd
from market_data import YahooFinanceSource
from selection import independent_pool


@dataclass(frozen=True)
class ExecutionModel:
    """Rates are decimals; transaction cost applies to each executed leg.

    Equity/ETF stock borrow accrues on short market value, ACT/360.
    Optional FX financing is a simplified gross-notional cost, not stock borrow
    or a reconstruction of broker swap points. Cash interest is excluded.
    """

    borrow_fee_rate: float = 0.01
    transaction_cost_rate: float = 0.0005
    fx_financing_rate: float = 0.0
    day_count: int = 360

    def __post_init__(self):
        for name in ("borrow_fee_rate", "transaction_cost_rate", "fx_financing_rate"):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(name + " must be nonnegative and finite")
        if self.transaction_cost_rate >= 1:
            raise ValueError("transaction_cost_rate must be less than 1")
        if self.day_count <= 0:
            raise ValueError("day_count must be positive")


    def transaction_costs(self, quantities, prices):
        return np.abs(np.asarray(quantities)) * np.asarray(prices) * self.transaction_cost_rate

    def holding_costs(self, quantities, prices, instrument_types=None, days=1):
        """Return per-leg stock borrow and FX financing costs for calendar days."""
        quantities, prices = np.asarray(quantities), np.asarray(prices)
        if instrument_types is None:
            instrument_types = np.full(quantities.shape, "EQUITY")
        fx = np.asarray(instrument_types) == "FX"
        borrow = np.where((quantities < 0) & ~fx, -quantities * prices, 0.0)
        financing = np.where(fx, np.abs(quantities) * prices, 0.0)
        return (borrow * self.borrow_fee_rate * days / self.day_count,
                financing * self.fx_financing_rate * days / self.day_count)


@dataclass(frozen=True)
class BacktestConfig:
    start_date: str = "2026-01-01"
    end_date: str | None = "2026-10-22"
    long_amount: float = 1_000_000
    short_amount: float = 1_000_000
    max_positions_per_side: int = 15
    execution_timing: str = "close"
    borrow_fee_rate: float = 0.01
    transaction_cost_rate: float = 0.0005
    fx_financing_rate: float = 0.0
    take_profit_amount: float | None = 2_000
    stop_loss_amount: float | None = None
    exit_on_signal: bool = True
    exit_on_pool_removal: bool = False
    allow_reentry: bool = True

    def __post_init__(self):
        if (isinstance(self.max_positions_per_side, bool)
                or not isinstance(self.max_positions_per_side, int) or self.max_positions_per_side < 1):
            raise ValueError("max_positions_per_side must be a positive integer.")
        for name in ("long_amount", "short_amount"):
            if isinstance(getattr(self, name), bool) or not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(name + " must be positive and finite.")
        for name in ("take_profit_amount", "stop_loss_amount"):
            value = getattr(self, name)
            if value is not None and (isinstance(value, bool) or not np.isfinite(value) or value <= 0):
                raise ValueError(name + " must be positive and finite, or None to disable.")
        for name in ("exit_on_signal", "exit_on_pool_removal", "allow_reentry"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(name + " must be boolean.")
        if self.execution_timing not in ("close", "next_open"):
            raise ValueError("execution_timing must be close or next_open.")
        if self.end_date and pd.Timestamp(self.end_date) < pd.Timestamp(self.start_date):
            raise ValueError("end_date must not precede start_date.")
        ExecutionModel(self.borrow_fee_rate, self.transaction_cost_rate, self.fx_financing_rate)

    @property
    def initial_capital(self):
        return self.long_amount + self.short_amount


class LongShortBacktester:
    """Rank long and short lists independently; no pair or hedge beta.

    Close fills use that date's snapshot; next-Open fills use only snapshots
    dated strictly before the fill session. Daily Close risk signals become
    pending exits for next Open. Fixed whole units remain until an exit.
    """

    def __init__(self, pool, config=None, *, source=None):
        if "target_weight" in pool:
            raise ValueError("This equal-slot backtest does not execute target weights. Use PortfolioConstructor.orders() in a weight-based execution loop.")
        self.pool = independent_pool(pool)
        self.config = config or BacktestConfig()
        self.source = source or YahooFinanceSource()
        self.execution = ExecutionModel(self.config.borrow_fee_rate,
            self.config.transaction_cost_rate, self.config.fx_financing_rate)

    def run(self, *, histories=None):
        cfg = self.config
        start = pd.Timestamp(cfg.start_date)
        end = (pd.Timestamp(cfg.end_date) if cfg.end_date else
            (pd.Timestamp.now(tz="America/New_York") - pd.Timedelta(hours=17)).normalize().tz_localize(None))
        tickers = list(dict.fromkeys(self.pool.ticker.tolist() + ["SPY"]))
        if histories is None:
            histories = self.source.history(tickers, str(start.date()), str((end + pd.Timedelta(days=1)).date()))
        clean = {}
        for ticker in tickers:
            frame = histories.get(ticker, pd.DataFrame(columns=["Open", "Close"]))
            frame = frame.reindex(columns=["Open", "Close"]).apply(pd.to_numeric, errors="coerce")
            frame.index = pd.DatetimeIndex(frame.index).tz_localize(None).normalize()
            clean[ticker] = frame.loc[~frame.index.duplicated(keep="last")].sort_index().replace([np.inf, -np.inf], np.nan).where(lambda value: value > 0)
        spy = clean["SPY"].dropna().loc[start:end]
        if spy.empty:
            raise ValueError("SPY needs valid Open and Close prices in the test period.")
        dates = spy.index
        opening = pd.DataFrame({t: clean[t].Open.reindex(dates) for t in tickers})
        closing = pd.DataFrame({t: clean[t].Close.reindex(dates) for t in tickers})
        close_fill = cfg.execution_timing == "close"
        cash, previous_equity, realized, net_realized = cfg.initial_capital, cfg.initial_capital, 0.0, 0.0
        realized_gain = realized_loss = 0.0
        cumulative_costs = np.zeros(3)
        active, events, positions, daily, pending, seen = {}, [], [], [], {}, set()
        trade_number, previous_date = 0, None
        snapshots = {date: group.set_index("ticker", drop=False) for date, group in self.pool.groupby("as_of", sort=True)}

        def candidates(date):
            available = [d for d in snapshots if d <= date] if close_fill else [d for d in snapshots if d < date]
            return snapshots[max(available)] if available else self.pool.iloc[:0].set_index("ticker", drop=False)

        def fee(state, price):
            return float(self.execution.transaction_costs([state["shares"]], [price])[0])

        def exit_reason(state, price, selected, date):
            net = state["shares"] * (price - state["entry_price"]) - state["total_cost"] - fee(state, price)
            if cfg.stop_loss_amount is not None and net <= -cfg.stop_loss_amount:
                return "dollar_stop"
            if cfg.take_profit_amount is not None and net >= cfg.take_profit_amount:
                return "dollar_take_profit"
            row = selected.loc[state["ticker"]] if state["ticker"] in selected.index else None
            if cfg.exit_on_signal and row is not None and (row.exit_signal or row.side != state["side"]):
                return "signal_exit"
            if cfg.exit_on_pool_removal and row is None:
                return "pool_removed"
            return None

        def event(state, date, price, action, reason, transaction_cost, signal_date):
            gross = state["shares"] * (price - state["entry_price"]) if action == "close" else 0.0
            return dict(trade_id=state["trade_id"], ticker=state["ticker"], side=state["side"],
                entry_date=state["entry_date"], execution_date=date, signal_date=signal_date,
                event=action, shares=state["shares"], price=price, amount=state["shares"] * price,
                entry_price=state["entry_price"], reason=reason, realized_pnl=gross,
                net_realized_pnl=gross-state["total_cost"] if action == "close" else 0.0,
                transaction_cost=transaction_cost, trade_total_cost=state["total_cost"])

        for date in dates:
            day_costs, exited = np.zeros(3), set()
            day_gain = day_loss = 0.0
            selected = candidates(date)
            fill_prices = closing.loc[date] if close_fill else opening.loc[date]
            # Weekends/holidays accrue at the last known Close; Close exits also
            # accrue the current day, whereas next-Open exits do not.
            for ticker, state in list(active.items()):
                closed_days = (date - previous_date).days - 1 if previous_date is not None else 0
                if closed_days > 0:
                    b, f = self.execution.holding_costs([state["shares"]], [state["last_close"]], [state["instrument_type"]], closed_days)
                    amount = np.array([b.sum(), 0.0, f.sum()])
                    cash -= amount.sum()
                    day_costs += amount
                    state["total_cost"] += amount.sum()
                mark = closing.at[date, ticker]
                if close_fill:
                    cost_mark = mark if pd.notna(mark) else state["last_close"]
                    b, f = self.execution.holding_costs([state["shares"]], [cost_mark], [state["instrument_type"]], 1)
                    amount = np.array([b.sum(), 0.0, f.sum()])
                    cash -= amount.sum()
                    day_costs += amount
                    state["total_cost"] += amount.sum()
                price = fill_prices[ticker]
                reason = exit_reason(state, price, selected, date) if close_fill and pd.notna(price) else pending.get(ticker, (None, None))[0]
                signal_date = date if close_fill else pending.get(ticker, (None, None))[1]
                if reason is not None and pd.notna(price):
                    cost = fee(state, price)
                    cash += state["shares"] * price - cost
                    day_costs[1] += cost
                    state["total_cost"] += cost
                    row = event(state, date, price, "close", reason, cost, signal_date)
                    events.append(row)
                    realized += row["realized_pnl"]
                    net_realized += row["net_realized_pnl"]
                    day_gain += max(row["realized_pnl"], 0.0)
                    day_loss += min(row["realized_pnl"], 0.0)
                    del active[ticker]
                    pending.pop(ticker, None)
                    exited.add(ticker)
            for row in selected.sort_values("score", ascending=False, kind="stable").itertuples(index=False):
                ticker = row.ticker
                if ticker in active or ticker in exited or not row.entry_signal or row.exit_signal:
                    continue
                if not cfg.allow_reentry and ticker in seen:
                    continue
                if sum(state["side"] == row.side for state in active.values()) >= cfg.max_positions_per_side:
                    continue
                price = fill_prices.get(ticker, np.nan)
                if not pd.notna(price):
                    continue
                cap = (cfg.long_amount if row.side == "long" else cfg.short_amount) / cfg.max_positions_per_side
                shares = int(np.floor(cap / price / row.unit_step)) * int(row.unit_step)
                if not shares:
                    continue
                shares *= 1 if row.side == "long" else -1
                trade_number += 1
                state = dict(trade_id=trade_number, ticker=ticker, side=row.side, shares=shares,
                    entry_date=date, entry_price=price, last_close=price, instrument_type=row.instrument_type,
                    total_cost=0.0)
                cost = fee(state, price)
                state["total_cost"] = cost
                cash -= shares * price + cost
                day_costs[1] += cost
                active[ticker] = state
                seen.add(ticker)
                events.append(event(state, date, price, "open", "ranked_entry", cost, row.as_of))
            marks, stale = {}, 0
            for ticker, state in active.items():
                mark = closing.at[date, ticker]
                if pd.isna(mark):
                    mark = state["last_close"]
                    stale += 1
                if not close_fill:
                    b, f = self.execution.holding_costs([state["shares"]], [mark], [state["instrument_type"]], 1)
                    amount = np.array([b.sum(), 0.0, f.sum()])
                    cash -= amount.sum()
                    day_costs += amount
                    state["total_cost"] += amount.sum()
                state["last_close"] = mark
                marks[ticker] = state["shares"] * mark
                unrealized = state["shares"] * (mark - state["entry_price"])
                positions.append(dict(Date=date, ticker=ticker, side=state["side"], shares=state["shares"],
                    amount=marks[ticker], trade_id=state["trade_id"], entry_date=state["entry_date"],
                    unrealized_pnl=unrealized, net_unrealized_pnl=unrealized-state["total_cost"],
                    holding_days=(date-state["entry_date"]).days))
                if not close_fill and ticker not in pending:
                    # This Close may schedule the next valid Open, never today's Open.
                    available = [d for d in snapshots if d <= date]
                    close_selection = snapshots[max(available)] if available else selected.iloc[:0]
                    reason = exit_reason(state, mark, close_selection, date)
                    if reason is not None:
                        pending[ticker] = (reason, date)
            equity = cash + sum(marks.values())
            pnl = equity - previous_equity
            cumulative_costs += day_costs
            realized_gain += day_gain
            realized_loss += day_loss
            unrealized = sum(state["shares"]*(state["last_close"]-state["entry_price"]) for state in active.values())
            net_unrealized = unrealized - sum(state["total_cost"] for state in active.values())
            long_value, short_value = sum(v for v in marks.values() if v > 0), sum(v for v in marks.values() if v < 0)
            daily.append(dict(Date=date, equity=equity, cash=cash, daily_pnl=pnl,
                daily_return=pnl/cfg.initial_capital, cumulative_pnl=equity-cfg.initial_capital,
                cumulative_return=(equity-cfg.initial_capital)/cfg.initial_capital,
                long_amount=long_value, short_amount=short_value, gross_amount=long_value-short_value,
                active_longs=sum(s["side"]=="long" for s in active.values()),
                active_shorts=sum(s["side"]=="short" for s in active.values()),
                cumulative_realized_pnl=realized, unrealized_pnl=unrealized,
                realized_gain=day_gain, realized_loss=day_loss,
                cumulative_realized_gain=realized_gain, cumulative_realized_loss=realized_loss,
                cumulative_net_realized_pnl=net_realized, net_unrealized_pnl=net_unrealized,
                gross_cumulative_pnl=realized+unrealized,
                borrow_fee=day_costs[0], transaction_cost=day_costs[1], fx_financing_fee=day_costs[2],
                total_cost=day_costs.sum(), cumulative_borrow_fee=cumulative_costs[0],
                cumulative_transaction_cost=cumulative_costs[1], cumulative_fx_financing_fee=cumulative_costs[2],
                cumulative_total_cost=cumulative_costs.sum(), stale_price_legs=stale))
            previous_equity, previous_date = equity, date
        summary = pd.DataFrame(daily).set_index("Date")
        summary["spy_close"] = spy.Close
        summary["spy_return"] = spy.Close.pct_change()
        summary.loc[dates[0], "spy_return"] = spy.Close.iloc[0] / spy.Open.iloc[0] - 1
        summary["spy_cumulative_return"] = spy.Close / spy.Open.iloc[0] - 1
        columns = ["trade_id", "ticker", "side", "entry_date", "execution_date", "signal_date", "event", "shares", "price", "amount", "entry_price", "reason", "realized_pnl", "net_realized_pnl", "transaction_cost", "trade_total_cost"]
        event_frame = pd.DataFrame(events, columns=columns)
        position_frame = pd.DataFrame(positions, columns=["Date", "ticker", "side", "shares", "amount", "trade_id", "entry_date", "unrealized_pnl", "net_unrealized_pnl", "holding_days"])
        def matrix(side):
            rows = position_frame.loc[position_frame.side.eq(side)]
            return rows.pivot(index="Date", columns="ticker", values="amount").reindex(index=dates, columns=self.pool.ticker.unique()).fillna(0)
        assert np.allclose(summary.cumulative_pnl, summary.gross_cumulative_pnl-summary.cumulative_total_cost, atol=1e-6)
        assert np.allclose(summary.cumulative_pnl, summary.cumulative_net_realized_pnl+summary.net_unrealized_pnl, atol=1e-6)
        return dict(daily_summary=summary, events=event_frame,
            closed_trades=event_frame.loc[event_frame.event.eq("close")].copy(),
            open_positions=position_frame.loc[position_frame.Date.eq(dates[-1])].copy(),
            daily_positions_long=matrix("long"), daily_positions_short=matrix("short"),
            positions=position_frame, source_events=list(self.source.events))
