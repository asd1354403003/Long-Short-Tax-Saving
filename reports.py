"""Shared performance comparison and accounting checks for either engine."""
import numpy as np
import pandas as pd
from risk import benchmark_returns, excess_sharpe


class PerformanceReport:
    @staticmethod
    def by_year(result, *, daily_risk_free=None):
        """Calendar-year comparisons, including each year's opening NAV base."""
        daily = result["daily_summary"]
        initial = daily.equity.iloc[0]-daily.cumulative_pnl.iloc[0]
        nav = pd.DataFrame({"Strategy":daily.equity/initial,"SPY":1+daily.spy_cumulative_return})
        returns = nav/nav.shift(1).fillna(1)-1
        rf = benchmark_returns(returns.index, daily_risk_free=(
            daily_risk_free if daily_risk_free is not None else result.get("daily_risk_free")))
        rows=[]
        for year,frame in returns.groupby(returns.index.year):
            cumulative=(1+frame).cumprod()
            volatility=frame.std(ddof=1)
            for name in frame:
                rows.append(dict(year=year,portfolio=name,**{
                    "return":cumulative[name].iloc[-1]-1,
                    "sharpe":excess_sharpe(frame[name], rf.reindex(frame.index)),
                    "volatility":volatility[name]*np.sqrt(252),
                    "max_drawdown":(cumulative[name]/cumulative[name].cummax().clip(lower=1)-1).min()}))
        return pd.DataFrame(rows).set_index(["year","portfolio"])

    @staticmethod
    def compare(result, annual_risk_free_rate=None, *, daily_risk_free=None):
        daily = result["daily_summary"]
        initial = daily.equity.iloc[0] - daily.cumulative_pnl.iloc[0]
        nav = pd.DataFrame({"Strategy": daily.equity/initial,
                            "SPY": 1+daily.spy_cumulative_return})
        returns = nav/nav.shift(1).fillna(1)-1
        supplied = daily_risk_free
        if supplied is None and annual_risk_free_rate is None:
            supplied = result.get("daily_risk_free")
        rf = benchmark_returns(returns.index, daily_risk_free=supplied,
                               annual_risk_free_rate=annual_risk_free_rate)
        excess = returns.sub(rf, axis=0)
        volatility = returns.std(ddof=1)
        return pd.DataFrame({"return": nav.iloc[-1]-1,
            "sharpe": excess.mean()/excess.std(ddof=1).replace(0, np.nan)*np.sqrt(252),
            "volatility": volatility*np.sqrt(252),
            "max_drawdown": (nav/nav.cummax().clip(lower=1)-1).min()})

    @staticmethod
    def trades(result):
        closed = result["closed_trades"]
        pnl = closed.net_realized_pnl.astype(float)
        winners, losers = pnl[pnl>0], pnl[pnl<0]
        return pd.Series(dict(entries=int(result["events"].event.eq("open").sum()),
            closed_trades=len(closed), open_positions=len(result["open_positions"]),
            closed_win_rate=float(pnl.gt(0).mean()) if len(pnl) else np.nan,
            mean_winner=float(winners.mean()), mean_loser=float(losers.mean()),
            payoff_ratio=float(winners.mean()/-losers.mean()) if len(winners) and len(losers) else np.nan,
            profit_factor=float(winners.sum()/-losers.sum()) if len(losers) else np.nan))

    @staticmethod
    def audit(result):
        daily, events = result["daily_summary"], result["events"]
        assert np.allclose(daily.cumulative_pnl, daily.gross_cumulative_pnl-daily.cumulative_total_cost, atol=1e-6)
        assert np.allclose(daily.cumulative_pnl, daily.cumulative_net_realized_pnl+daily.net_unrealized_pnl, atol=1e-6)
        assert np.isclose(daily.daily_pnl.sum(), daily.cumulative_pnl.iloc[-1], atol=1e-6)
        assert np.isclose(events.transaction_cost.sum(), daily.transaction_cost.sum(), atol=1e-6)
        if result.get("execution") in {"monthly_next_open_target_weights", "quarterly_next_open_target_weights"}:
            corporate = result["corporate_events"]
            closed = events.loc[events.event.eq("close")]
            assert np.isclose(closed.realized_pnl.sum()+corporate.realized_pnl.sum(),daily.cumulative_realized_pnl.iloc[-1],atol=1e-6)
            assert np.isclose(closed.net_realized_pnl.sum()+corporate.net_realized_pnl.sum(),daily.cumulative_net_realized_pnl.iloc[-1],atol=1e-6)
            assert events.signal_date.lt(events.execution_date).all()
            quantities = result["daily_shares"].to_numpy()
            assert np.allclose(quantities,np.trunc(quantities),atol=1e-8)
        if "shares" in events:
            shares = pd.to_numeric(events.shares).to_numpy(dtype=float)
            assert np.allclose(shares, np.trunc(shares))
            assert events.loc[events.side.eq("short"), "amount"].le(0).all()
            assert events.loc[events.side.eq("long"), "amount"].ge(0).all()
        else:
            for side in ("long", "short"):
                shares = pd.to_numeric(events[side+"_shares"]).to_numpy(dtype=float)
                assert np.allclose(shares, np.trunc(shares))
            assert events.short_amount.le(0).all() and events.long_amount.ge(0).all()
        return pd.Series(dict(accounting="passed", sessions=len(daily),
            stale_price_legs=int(daily.stale_price_legs.sum())))
