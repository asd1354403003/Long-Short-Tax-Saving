"""AQR net performance and reconciled annual long/short attribution."""
import numpy as np
import pandas as pd
from risk import benchmark_returns, excess_sharpe

def _summary(result, signals, name, cohort, initial_nav):
    daily, events = result["daily_summary"], result["events"]
    rf = benchmark_returns(daily.index, daily_risk_free=result.get("daily_risk_free"))
    first = events.execution_date.min() if not events.empty else pd.NaT
    active = daily.loc[first:] if pd.notna(first) else daily.iloc[:0]
    stats = {"sharpe": np.nan, "max_drawdown": np.nan, "beta": np.nan, "volatility": np.nan}
    if len(active):
        returns = active.daily_return
        std = returns.std(ddof=1)
        nav = active.equity / initial_nav
        market = active.spy_return
        stats = dict(sharpe=excess_sharpe(returns, rf.reindex(returns.index)),
            volatility=std*np.sqrt(252), max_drawdown=(nav/nav.cummax().clip(lower=1)-1).min(),
            beta=returns.cov(market)/market.var(ddof=1) if market.var(ddof=1)>0 else np.nan)
    final = daily.iloc[-1]
    full_returns = daily.daily_return
    full_std = full_returns.std(ddof=1)
    full_nav = daily.equity/initial_nav
    return dict(variant=name, cohort=cohort, factors=" + ".join(signals.config.factor_operators),
        configured_start=str(daily.index[0].date()), sessions=len(daily),
        first_trade=str(first.date()) if pd.notna(first) else None, active_sessions=len(active),
        end=str(daily.index[-1].date()), net_return=final.equity/initial_nav-1,
        net_pnl=final.cumulative_pnl, gross_pnl=final.gross_cumulative_pnl,
        costs=final.cumulative_total_cost, transaction_costs=final.cumulative_transaction_cost,
        borrow_fees=final.cumulative_borrow_fee, latest_ranked=int(signals.coverage.ranked_stocks.iloc[-1]),
        ranked_stock_months=int(signals.coverage.ranked_stocks.sum()),
        stale_price_legs=int(daily.stale_price_legs.sum()),
        full_window_sharpe=excess_sharpe(full_returns, rf),
        zero_rate_return_volatility_ratio=full_returns.mean()/full_std*np.sqrt(252) if full_std>0 else np.nan,
        full_window_max_drawdown=(full_nav/full_nav.cummax().clip(lower=1)-1).min(), **stats)


def side_attribution(result, backtest_config):
    daily=result['daily_summary']
    events=pd.concat([result['events'],result['corporate_events']],ignore_index=True)
    events['cash_spent']=events.trade_shares*events.price
    def matrix(frame,field,template):
        if frame.empty:
            return template*0
        data=frame.pivot_table(index='execution_date',columns='ticker',values=field,aggfunc='sum',fill_value=0)
        return data.reindex(index=template.index,columns=template.columns,fill_value=0)
    longs,shorts=result['daily_positions_long'],result['daily_positions_short']
    long_pnl=longs-longs.shift(1,fill_value=0)-matrix(events.loc[events.side.eq('long')],'cash_spent',longs)
    short_pnl=shorts-shorts.shift(1,fill_value=0)-matrix(events.loc[events.side.eq('short')],'cash_spent',shorts)
    transaction=matrix(events,'transaction_cost',longs)
    gaps=daily.index.to_series().diff().dt.days.sub(1).fillna(0)
    borrow=-shorts*backtest_config.borrow_fee_rate/360
    borrow+=(-shorts.shift(1,fill_value=0)).mul(gaps,axis=0)*backtest_config.borrow_fee_rate/360
    assert np.allclose((long_pnl+short_pnl-transaction-borrow).sum(axis=1),daily.daily_pnl,atol=1e-6)
    assert np.allclose(borrow.sum(axis=1),daily.borrow_fee,atol=1e-6)
    parts=pd.DataFrame(dict(long_pnl=long_pnl.sum(axis=1),short_pnl=short_pnl.sum(axis=1),
        transaction_cost=transaction.sum(axis=1),borrow_fee=borrow.sum(axis=1),net_pnl=daily.daily_pnl))
    previous_nav=daily.equity.shift(1,fill_value=backtest_config.initial_nav)
    rows=[]
    for year,frame in parts.groupby(parts.index.year):
        nav=previous_nav.loc[frame.index[0]]
        row=dict(year=year,opening_nav=nav,**frame.sum().to_dict())
        row.update({name+'_contribution':row[name]/nav for name in frame})
        rows.append(row)
    return pd.DataFrame(rows)
