"""Scheduled target-weight execution with integer shares and a cash/P&L ledger."""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtest import ExecutionModel
from methods.aqr_factors.construction import PortfolioConstructor


@dataclass(frozen=True)
class WeightedBacktestConfig:
    start_date: str = "2022-01-01"
    end_date: str = "2026-10-22"
    initial_nav: float = 1_000_000
    transaction_cost_rate: float = 0.0005
    borrow_fee_rate: float = 0.01
    rebalance_frequency: str = "monthly"

    def __post_init__(self):
        if not np.isfinite(self.initial_nav) or self.initial_nav <= 0:
            raise ValueError("initial_nav must be positive and finite")
        if pd.Timestamp(self.end_date) < pd.Timestamp(self.start_date):
            raise ValueError("end_date must not precede start_date")
        ExecutionModel(self.borrow_fee_rate, self.transaction_cost_rate)
        if self.rebalance_frequency not in {"monthly", "quarterly"}:
            raise ValueError("rebalance_frequency must be monthly or quarterly")


class WeightedBacktester:
    """Decide with completed month/quarter-end data; execute at the following Open.

    Rebalance existing holdings to target weights, including adds, trims and
    sign flips. Hold fixed integer shares between scheduled rebalances. Average cost is
    used for P&L attribution, not as a tax-lot or tax-liability model. A missing
    Open defers an order; a missing Close retains the last mark and is reported.
    Cash dividends, short dividend compensation, cash/margin interest and taxes
    are excluded. Splits adjust share counts and cost basis, not investment P&L.
    """

    def __init__(self, config=None, construction=None):
        self.config = config or WeightedBacktestConfig()
        self.constructor = PortfolioConstructor(construction)
        self.execution = ExecutionModel(self.config.borrow_fee_rate, self.config.transaction_cost_rate)
        if (self.constructor.config.transaction_cost_rate != self.config.transaction_cost_rate
                or self.constructor.config.borrow_fee_rate != self.config.borrow_fee_rate):
            raise ValueError("Construction and execution fees must use the same rates")

    @staticmethod
    def cached_histories(database, tickers, start, end):
        """Read canonical historical market share units, including split events."""
        histories = {t: database.price_frame(t, start, end) for t in tickers}
        if histories.get("SPY", pd.DataFrame()).empty:
            histories["SPY"] = database.price_frame("__NYSE_CALENDAR__", start, end)
        return histories

    def run(self, signals, *, histories=None):
        cfg = self.config
        frame = signals.signals.copy()
        if frame.empty:
            raise ValueError("Calculate monthly scores before the backtest")
        symbols = sorted(set(frame.ticker) | {"SPY"})
        if histories is None:
            histories = self.cached_histories(signals.database, symbols, cfg.start_date,
                str((pd.Timestamp(cfg.end_date)+pd.Timedelta(days=1)).date()))
        clean = {}
        for ticker in symbols:
            data = histories.get(ticker, pd.DataFrame()).copy()
            data.index = pd.DatetimeIndex(data.index).tz_localize(None).normalize()
            if data.index.duplicated().any():
                raise ValueError("Duplicate history dates: " + ticker)
            clean[ticker] = data.sort_index().reindex(columns=["Open", "Close", "Stock Splits"]).apply(pd.to_numeric,errors="coerce")
        spy = clean["SPY"].loc[cfg.start_date:cfg.end_date].dropna(subset=["Open","Close"])
        if spy.empty or spy[["Open","Close"]].le(0).any().any():
            raise ValueError("SPY needs valid Open/Close in the test period")
        dates, n = spy.index, len(symbols)
        symbol_index = {s:i for i,s in enumerate(symbols)}
        opening = pd.DataFrame({s:clean[s].Open.reindex(dates) for s in symbols}).replace([np.inf,-np.inf],np.nan).where(lambda x:x>0)
        closing = pd.DataFrame({s:clean[s].Close.reindex(dates) for s in symbols}).replace([np.inf,-np.inf],np.nan).where(lambda x:x>0)
        splits = pd.DataFrame({s:clean[s]["Stock Splits"].reindex(dates) for s in symbols}).fillna(0)
        snapshots = {d:g.copy() for d,g in frame.groupby("as_of",sort=True)}
        snapshot_dates = sorted(snapshots)
        if cfg.rebalance_frequency == "quarterly":
            snapshot_dates = [d for d in snapshot_dates if d.month in {3, 6, 9, 12}]
        states, events, daily, target_frames, order_frames, corporate = {}, [], [], [], [], []
        amount_matrix, share_matrix = np.zeros((len(dates),n)), np.zeros((len(dates),n))
        cash, previous_equity, realized, net_realized = cfg.initial_nav, cfg.initial_nav, 0., 0.
        cumulative_borrow = cumulative_tc = realized_gain = realized_loss = 0.
        previous_date = last_signal = None
        pending, trade_number, current_targets = {}, 0, None
        last_groups = {}
        risk_prices = None
        if self.constructor.config.method == "aqr_style":
            risk_prices = PortfolioConstructor.cached_prices(signals.database,symbols,
                str((pd.Timestamp(cfg.start_date)-pd.DateOffset(years=2)).date()),
                str((pd.Timestamp(cfg.end_date)+pd.Timedelta(days=1)).date()))

        def trade(ticker, delta, price, date, signal_date, reason="monthly_rebalance", cost_rate=None):
            nonlocal cash, realized, net_realized, realized_gain, realized_loss, trade_number
            rate = cfg.transaction_cost_rate if cost_rate is None else cost_rate
            fee = abs(delta)*price*rate
            cash -= delta*price+fee
            state = states.get(ticker)
            old = state["shares"] if state else 0.
            remaining = abs(delta)
            if old and old*delta < 0:
                quantity = min(abs(old),remaining)
                signed = np.sign(old)*quantity
                gross = signed*(price-state["entry_price"])
                allocated = state["open_cost"]*quantity/abs(old)
                exit_fee = quantity*price*rate
                realized += gross
                net_realized += gross-allocated-exit_fee
                realized_gain += max(gross,0.)
                realized_loss += min(gross,0.)
                events.append(dict(trade_id=state["trade_id"],ticker=ticker,side=state["side"],
                    entry_date=state["entry_date"],execution_date=date,signal_date=signal_date,event="close",
                    shares=signed,trade_shares=-signed,price=price,amount=signed*price,entry_price=state["entry_price"],
                    reason=reason,realized_pnl=gross,net_realized_pnl=gross-allocated-exit_fee,
                    transaction_cost=exit_fee,trade_total_cost=allocated+exit_fee))
                state["open_cost"] -= allocated
                state["shares"] -= signed
                remaining -= quantity
                if abs(state["shares"]) < 1e-10:
                    states.pop(ticker)
                    state = None
            if remaining > 1e-10:
                signed = np.sign(delta)*remaining
                if state is None:
                    trade_number += 1
                    state = dict(shares=0.,entry_price=price,entry_date=date,trade_id=trade_number,
                                 side="long" if signed>0 else "short",open_cost=0.,last_close=price)
                    states[ticker] = state
                quantity = abs(state["shares"])
                state["entry_price"] = (state["entry_price"]*quantity+price*remaining)/(quantity+remaining)
                state["shares"] += signed
                state["open_cost"] += remaining*price*rate
                events.append(dict(trade_id=state["trade_id"],ticker=ticker,side=state["side"],
                    entry_date=state["entry_date"],execution_date=date,signal_date=signal_date,event="open",
                    shares=signed,trade_shares=signed,price=price,amount=signed*price,entry_price=price,
                    reason=reason,realized_pnl=0.,net_realized_pnl=0.,transaction_cost=remaining*price*rate,
                    trade_total_cost=remaining*price*rate))
            return fee

        for day_i,date in enumerate(dates):
            borrow = tc = 0.
            # Calendar days without sessions use the previous session's mark.
            gap = (date-previous_date).days-1 if previous_date is not None else 0
            for ticker,state in list(states.items()):
                cost = max(-state["shares"],0)*state["last_close"]*cfg.borrow_fee_rate*gap/360
                state["open_cost"] += cost
                borrow += cost
                ratio = splits.at[date,ticker]
                if ratio and ratio != 1:
                    if not np.isfinite(ratio) or ratio<=0:
                        raise ValueError("Invalid split: "+ticker)
                    state["shares"] *= ratio
                    if abs(state["shares"]-round(state["shares"]))<1e-8:
                        state["shares"] = float(round(state["shares"]))
                    state["entry_price"] /= ratio
                    state["last_close"] /= ratio
                    fraction = state["shares"]-np.trunc(state["shares"])
                    if abs(fraction)>1e-8:
                        price = opening.at[date,ticker]
                        if not np.isfinite(price):
                            raise ValueError("Fractional split settlement needs an Open: "+ticker)
                        before = len(events)
                        trade(ticker,-fraction,price,date,date,"split_cash_in_lieu",cost_rate=0.)
                        corporate.extend(events[before:]); del events[before:]
            cash -= borrow
            available = [d for d in snapshot_dates if d < date]
            latest = max(available) if available else None
            if latest is not None and latest != last_signal:
                month = snapshots[latest].copy()
                group = self.constructor.config.group_by
                if group:
                    if signals.universe.proxy is not None and group=="sector":
                        labels = signals.universe.group_labels(latest,group)
                    else:
                        labels = signals.companies.set_index("ticker")[group]
                    last_groups.update(labels.dropna().to_dict())
                    month[group] = month.ticker.map(labels)
                    held_missing = month.ticker.isin(states) & month[group].isna()
                    month.loc[held_missing,group] = month.loc[held_missing,"ticker"].map(last_groups)
                nav_at_decision = cash+sum(s["shares"]*s["last_close"] for s in states.values())
                if nav_at_decision<=0:
                    raise RuntimeError("Nonpositive NAV before rebalance")
                held_weights = pd.Series({t:s["shares"]*s["last_close"]/nav_at_decision for t,s in states.items()},dtype=float)
                current_targets = self.constructor.build(month,current_weights=held_weights,prices=risk_prices)
                target_frames.append(current_targets)
                # Open sizing uses observed execution prices; the signal itself
                # remains frozen at the strictly earlier month-end cutoff.
                nav_open = cash+sum(s["shares"]*(opening.at[date,t] if pd.notna(opening.at[date,t]) else s["last_close"]) for t,s in states.items())
                orders = self.constructor.orders(current_targets,nav=nav_open,prices=opening.loc[date],
                    current_shares=pd.Series({t:s["shares"] for t,s in states.items()},dtype=float))
                orders["execution_date"] = date
                order_frames.append(orders)
                pending = {r.ticker:(r.target_amount,latest) for r in orders.itertuples()
                    if r.execution_status=="missing_price" and (r.target_weight!=0 or r.current_shares!=0)}
                for row in orders.loc[orders.execution_status.eq("ready")].itertuples():
                    tc += trade(row.ticker,int(row.trade_shares),row.price,date,latest,
                                reason=cfg.rebalance_frequency+"_rebalance")
                last_signal = latest
            else:
                for ticker,(amount,signal_date) in list(pending.items()):
                    price = opening.at[date,ticker]
                    if pd.notna(price):
                        quantity = np.trunc(amount/price)
                        delta = quantity-states.get(ticker,{}).get("shares",0)
                        if delta:
                            tc += trade(ticker,delta,price,date,signal_date,"deferred_rebalance")
                        pending.pop(ticker)
            stale = 0
            for ticker,state in states.items():
                price = closing.at[date,ticker]
                if pd.isna(price):
                    stale += 1
                else:
                    state["last_close"] = price
                cost = max(-state["shares"],0)*state["last_close"]*cfg.borrow_fee_rate/360
                state["open_cost"] += cost
                borrow += cost
                cash -= cost
                j = symbol_index[ticker]
                amount_matrix[day_i,j] = state["shares"]*state["last_close"]
                share_matrix[day_i,j] = state["shares"]
            amounts = amount_matrix[day_i]
            equity = cash+amounts.sum()
            if equity <= 0:
                raise RuntimeError("Nonpositive NAV during backtest")
            cumulative_borrow += borrow
            cumulative_tc += tc
            unrealized = sum(s["shares"]*(s["last_close"]-s["entry_price"]) for s in states.values())
            open_cost = sum(s["open_cost"] for s in states.values())
            pnl = equity-cfg.initial_nav
            daily.append(dict(Date=date,equity=equity,cash=cash,long_amount=amounts.clip(min=0).sum(),
                short_amount=amounts.clip(max=0).sum(),gross_amount=abs(amounts).sum(),
                net_amount=amounts.sum(),long_stocks=int((amounts>0).sum()),short_stocks=int((amounts<0).sum()),
                daily_pnl=equity-previous_equity,daily_return=equity/previous_equity-1,
                cumulative_pnl=pnl,cumulative_return=pnl/cfg.initial_nav,
                cumulative_realized_pnl=realized,cumulative_realized_gain=realized_gain,
                cumulative_realized_loss=realized_loss,cumulative_net_realized_pnl=net_realized,
                unrealized_pnl=unrealized,net_unrealized_pnl=unrealized-open_cost,
                gross_cumulative_pnl=realized+unrealized,borrow_fee=borrow,transaction_cost=tc,
                fx_financing_fee=0.,total_cost=borrow+tc,cumulative_borrow_fee=cumulative_borrow,
                cumulative_transaction_cost=cumulative_tc,cumulative_fx_financing_fee=0.,
                cumulative_total_cost=cumulative_borrow+cumulative_tc,stale_price_legs=stale,
                signal_date=last_signal,deferred_orders=len(pending)))
            previous_date,previous_equity = date,equity
        summary = pd.DataFrame(daily).set_index("Date")
        summary["spy_close"] = spy.Close
        summary["spy_return"] = spy.Close.pct_change()
        summary.loc[dates[0],"spy_return"] = spy.Close.iloc[0]/spy.Open.iloc[0]-1
        summary["spy_cumulative_return"] = spy.Close/spy.Open.iloc[0]-1
        columns = ["trade_id","ticker","side","entry_date","execution_date","signal_date","event","shares","trade_shares","price","amount","entry_price","reason","realized_pnl","net_realized_pnl","transaction_cost","trade_total_cost"]
        event_frame = pd.DataFrame(events,columns=columns)
        open_positions = pd.DataFrame([dict(ticker=t,side=s["side"],shares=s["shares"],entry_date=s["entry_date"],
            entry_price=s["entry_price"],close=s["last_close"],amount=s["shares"]*s["last_close"],
            unrealized_pnl=s["shares"]*(s["last_close"]-s["entry_price"]),
            net_unrealized_pnl=s["shares"]*(s["last_close"]-s["entry_price"])-s["open_cost"]) for t,s in states.items()])
        amounts = pd.DataFrame(amount_matrix,index=dates,columns=symbols)
        return dict(daily_summary=summary,events=event_frame,
            closed_trades=event_frame.loc[event_frame.event.eq("close")].copy(),open_positions=open_positions,
            daily_positions_long=amounts.clip(lower=0),daily_positions_short=amounts.clip(upper=0),
            daily_shares=pd.DataFrame(share_matrix,index=dates,columns=symbols),
            target_weights=pd.concat(target_frames,ignore_index=True) if target_frames else pd.DataFrame(),
            rebalance_orders=pd.concat(order_frames,ignore_index=True) if order_frames else pd.DataFrame(),
            corporate_events=pd.DataFrame(corporate,columns=columns),
            score_coverage=signals.coverage.copy(),source_events=list(signals.errors),
            execution=cfg.rebalance_frequency+"_next_open_target_weights",
            rebalance_frequency=cfg.rebalance_frequency)
