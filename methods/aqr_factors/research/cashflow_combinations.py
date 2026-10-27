"""Recorded small-combination search anchored on operating cashflow yield.

All trials use fixed factor weights throughout 2022-2026. Batch calculations
retain daily integer positions, corporate splits, next-Open executions, actual
turnover costs and calendar-day short borrow. Winners are independently
replayed with the existing full transaction-ledger engine.
"""
from __future__ import annotations

from copy import copy, deepcopy
from dataclasses import asdict, replace
from itertools import combinations
from pathlib import Path
import hashlib
import json
import time

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from ..factors import balance_ratios
from ..signals import FactorSignals, SignalConfig
from ..construction import ConstructionConfig
from .multifactor_experiment import side_attribution
from .experiments import _summary
from methods.aqr_factors.backtest import WeightedBacktestConfig, WeightedBacktester
from reports import PerformanceReport
from risk import historical_risk_free


CORE='cashflow_yield'
OTHERS=('value','quality','momentum','earnings_yield','roa','low_volatility',
    'gross_profitability','average_asset_roa','low_accruals','profit_stability',
    'fcf_yield','momentum_6_1','residual_momentum')
FACTOR_NAMES=(CORE,)+OTHERS


from ..portfolio import target_arrays

def specifications():
    result=[]
    for size in range(4):
        for other in combinations(OTHERS,size):
            core_weights=[1.] if not size else sorted({.25,.5,.75,1/(size+1)})
            for core_weight in core_weights:
                factors=(CORE,)+other
                weights=(core_weight,)+((1-core_weight)/size,)*size if size else (1.,)
                result.append(dict(variant=f'cf_{len(result):04d}',factors=factors,weights=weights,
                    cashflow_weight=core_weight,factor_count=len(factors)))
    assert len(result)==1210
    return result


def batch_backtest(weights, decision_dates, market, config):
    """Daily cash/NAV engine, vectorized over strategies, without P&L tax lots.

    market arrays are exchange-session aligned and contain raw historical
    share-unit Open/Close/splits. Missing Close carries an existing holding's
    last mark and is flagged; missing Open defers the frozen dollar target.
    This matches the existing execution engine; it never fills signal prices.
    """
    dates=market['dates']
    opening,closing,splits=market['open'],market['close'],market['splits']
    c,_,n=weights.shape
    quantities=np.zeros((c,n)); marks=np.zeros((c,n))
    cash=np.full(c,config.initial_nav,dtype=float)
    pending=np.zeros((c,n)); pending_active=np.zeros((c,n),dtype=bool)
    equity=np.zeros((len(dates),c)); costs=np.zeros_like(equity); borrows=np.zeros_like(equity)
    stales=np.zeros(c,dtype=np.int64)
    execution={int(dates.searchsorted(date,side='right')):i for i,date in enumerate(decision_dates)}
    previous_date=None
    for day,date in enumerate(dates):
        old_short=np.minimum(quantities,0)
        gap=(date-previous_date).days-1 if previous_date is not None else 0
        borrow=-(old_short*marks).sum(axis=1)*config.borrow_fee_rate*gap/360
        cash-=borrow
        ratio=splits[day]
        split_names=np.flatnonzero(ratio!=1)
        for j in split_names:
            quantities[:,j]*=ratio[j]
            marks[:,j]/=ratio[j]
            rounded=np.rint(quantities[:,j])
            quantities[:,j]=np.where(abs(quantities[:,j]-rounded)<1e-8,rounded,quantities[:,j])
            fraction=quantities[:,j]-np.trunc(quantities[:,j])
            if np.any(abs(fraction)>1e-8):
                if not np.isfinite(opening[day,j]):
                    raise ValueError('Fractional split settlement lacks an Open')
                cash+=fraction*opening[day,j]
                quantities[:,j]-=fraction
        valid_open=np.isfinite(opening[day]) & (opening[day]>0)
        transaction=np.zeros(c)
        if day in execution:
            mark_open=np.where(valid_open[None,:],opening[day][None,:],marks)
            nav_open=cash+(quantities*mark_open).sum(axis=1)
            if np.any(nav_open<=0):
                raise ValueError('Nonpositive NAV')
            desired=weights[:,execution[day],:]*nav_open[:,None]
            target=quantities.copy()
            target[:,valid_open]=np.trunc(desired[:,valid_open]/opening[day,valid_open])
            pending=desired.copy()
            pending_active=(~valid_open[None,:]) & ((weights[:,execution[day],:]!=0) | (quantities!=0))
            ready=np.broadcast_to(valid_open,(c,n))
        else:
            ready=pending_active & valid_open[None,:]
            target=quantities.copy()
            good_rows,good_cols=np.where(ready)
            target[good_rows,good_cols]=np.trunc(pending[good_rows,good_cols]/opening[day,good_cols])
            pending_active[ready]=False
        delta=np.where(ready,target-quantities,0)
        trade_cash=delta*np.where(valid_open,opening[day],0)[None,:]
        transaction=abs(trade_cash).sum(axis=1)*config.transaction_cost_rate
        cash-=trade_cash.sum(axis=1)+transaction
        new_state=(target!=0) & ((quantities==0) | (target*quantities<0)) & ready
        marks=np.where(new_state,opening[day][None,:],marks)
        quantities+=delta
        valid_close=np.isfinite(closing[day]) & (closing[day]>0)
        marks=np.where(valid_close[None,:] & (quantities!=0),closing[day][None,:],marks)
        stales+=((quantities!=0) & ~valid_close[None,:]).sum(axis=1)
        today_borrow=-(np.minimum(quantities,0)*marks).sum(axis=1)*config.borrow_fee_rate/360
        borrow+=today_borrow
        cash-=today_borrow
        equity[day]=cash+(quantities*marks).sum(axis=1)
        costs[day]=transaction
        borrows[day]=borrow
        previous_date=date
    return dict(equity=equity,transaction_cost=costs,borrow_fee=borrows,stales=stales)




def market_arrays(histories, tickers, config):
    spy=histories['SPY'].loc[config.start_date:config.end_date].dropna(subset=['Open','Close'])
    dates=spy.index
    opening=np.full((len(dates),len(tickers)),np.nan)
    closing=opening.copy(); splits=np.ones_like(opening)
    for j,ticker in enumerate(tickers):
        frame=histories.get(ticker,pd.DataFrame())
        if frame.empty:
            continue
        frame=frame.reindex(dates)
        opening[:,j]=frame.Open.where(frame.Open.gt(0))
        closing[:,j]=frame.Close.where(frame.Close.gt(0))
        splits[:,j]=frame['Stock Splits'].where(frame['Stock Splits'].gt(0),1)
    return dict(dates=dates,open=opening,close=closing,splits=splits,spy=spy)


def saved_target_array(folder,tickers,decisions):
    weights=pd.read_csv(folder/'target_weights.csv',parse_dates=['as_of'])
    return weights.pivot(index='as_of',columns='ticker',values='target_weight').reindex(index=decisions,columns=tickers).fillna(0).to_numpy()


def summarize_batch(specs,result,counts,market,config):
    equity=result['equity']
    previous=np.vstack([np.full((1,equity.shape[1]),config.initial_nav),equity[:-1]])
    returns=equity/previous-1
    std=returns.std(axis=0,ddof=1)
    rf=historical_risk_free(market['dates']).to_numpy()
    excess=returns-rf[:,None]
    excess_std=excess.std(axis=0,ddof=1)
    sharpe=np.divide(excess.mean(axis=0)*np.sqrt(252),excess_std,
                     out=np.full_like(excess_std,np.nan),where=excess_std>0)
    high=np.maximum.accumulate(np.vstack([np.full((1,equity.shape[1]),config.initial_nav),equity]),axis=0)[1:]
    drawdown=(equity/high-1).min(axis=0)
    spy=market['spy']
    spy_return=spy.Close.pct_change()
    spy_return.iloc[0]=spy.Close.iloc[0]/spy.Open.iloc[0]-1
    market_returns=spy_return.to_numpy()
    beta=((returns-returns.mean(axis=0))*(market_returns-market_returns.mean())[:,None]).sum(axis=0)/(len(returns)-1)/market_returns.var(ddof=1)
    year_returns={}
    for year in np.unique(market['dates'].year):
        positions=np.flatnonzero(market['dates'].year==year)
        year_returns[int(year)]=equity[positions[-1]]/previous[positions[0]]-1
    rows=[]
    for i,spec in enumerate(specs):
        annual={f'return_{year}':float(values[i]) for year,values in year_returns.items()}
        full_years=[annual[f'return_{year}'] for year in [2022,2023,2024,2025]]
        rows.append(dict(variant=spec['variant'],factors=' + '.join(spec['factors']),
            weights=json.dumps(dict(zip(spec['factors'],spec['weights']))),cashflow_weight=spec['cashflow_weight'],
            factor_count=spec['factor_count'],net_return=equity[-1,i]/config.initial_nav-1,
            sharpe=sharpe[i],beta=beta[i],max_drawdown=drawdown[i],
            mean_ranked=counts[i].mean(),min_ranked=counts[i].min(),stale_price_legs=int(result['stales'][i]),
            transaction_costs=result['transaction_cost'][:,i].sum(),borrow_fees=result['borrow_fee'][:,i].sum(),
            min_full_year_return=min(full_years),positive_full_years=sum(value>0 for value in full_years),
            min_including_partial_2026=min(annual.values()),**annual))
    return pd.DataFrame(rows),returns


def run(output_dir='methods/aqr_factors/research/results/cashflow_combinations',progress=print):
    source=Path('methods/aqr_factors/research/results/factor_comparison')
    trial=Path('methods/aqr_factors/research/results/ten_factor_trial')
    out=Path(output_dir); out.mkdir(parents=True,exist_ok=True)
    meta=json.loads((source/'comparison_settings.json').read_text(encoding='utf-8'))
    base=FactorSignals(SignalConfig(**meta['signal_settings']))
    base.metadata=deepcopy(meta['source_metadata'])
    raw=pd.read_csv(source/'monthly_raw_factors.csv.gz',parse_dates=['as_of'],low_memory=False)
    raw=raw.loc[raw.as_of.dt.month.isin([3,6,9,12])].reset_index(drop=True)
    for name,values in balance_ratios(raw).items():
        raw[name]=values
    extra_names=list(FACTOR_NAMES[7:])
    extra=pd.read_csv(trial/'quarterly_scores.csv.gz',parse_dates=['as_of'],usecols=['as_of','ticker']+extra_names)
    raw=raw.merge(extra,on=['as_of','ticker'],how='left')
    # Explicitly retain the denominator rules of the original factor reader.
    raw.loc[~raw.total_assets.gt(0),['quality','roa']]=np.nan
    base.signals=raw.copy()
    profiles={}
    for r in base.database.read("SELECT ticker,payload FROM snapshots WHERE dataset='company' ORDER BY observed_at").itertuples():
        profiles[r.ticker]=json.loads(r.payload)
    base.companies=pd.DataFrame([dict(ticker=t,industry=i.get('industry'),sector=i.get('sector')) for t,i in profiles.items()])
    config=WeightedBacktestConfig(**meta['backtest_settings'])
    construction=ConstructionConfig(**meta['construction'])
    tickers=sorted(set(raw.ticker)|{'SPY'})
    decisions=pd.DatetimeIndex(sorted(raw.as_of.unique()))
    specs=specifications()
    (out/'specifications.json').write_text(json.dumps(specs,indent=2),encoding='utf-8')
    manifest=dict(trials=len(specs),mandatory_factor=CORE,optional_factors=OTHERS,max_total_factors=4,
        cashflow_weights='Equal weight and cashflow 25%, 50%, 75%; other included factors split the remaining weight equally; duplicates removed.',
        backtest=asdict(config),construction=asdict(construction),source_requests=0,
        eligibility='Each decision independently drops only that combination selected missing factors; minimum ten stocks per sector; no full-history intersection.',
        ranking='Daily net Sharpe descending; also report maximin complete-calendar-year return. Full 2022-2026 sample is used for selection, so results are in-sample research, not independent validation.',
        limitations='Latest vendor financial versions and period-end +90-day missing-publication-date assumption; no dividends, cash interest or taxes; 2026 through October 22.',
        source_sha256=hashlib.sha256((source/'monthly_raw_factors.csv.gz').read_bytes()).hexdigest())
    (out/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    progress('Read cached execution prices once')
    histories=WeightedBacktester.cached_histories(base.database,tickers,config.start_date,
        str((pd.Timestamp(config.end_date)+pd.Timedelta(days=1)).date()))
    market=market_arrays(histories,tickers,config)
    # Independent execution parity before ranking a large set of strategies.
    checks={}
    for label,folder in [('original_three',source/'own/baseline'),('cashflow_reference',source/'own/plus_cashflow'),
            ('low_vol_reference',source/'own/plus_low_vol'),('risk_reference',trial/'ten_factor_risk')]:
        target=saved_target_array(folder,tickers,decisions)[None,:,:]
        batch=batch_backtest(target,decisions,market,config)
        saved=pd.read_csv(folder/'daily_summary.csv',index_col='Date',parse_dates=True)
        differences={field:float(np.max(abs(batch[field][:,0]-saved[column].to_numpy()))) for field,column in
            [('equity','equity'),('transaction_cost','transaction_cost'),('borrow_fee','borrow_fee')]}
        assert differences['equity']<1e-4 and differences['transaction_cost']<1e-6 and differences['borrow_fee']<1e-6,(label,differences)
        checks[label]=dict(**differences,all_1206_sessions_reproduced=True)
    progress('Four saved full-ledger portfolios reproduced by batch execution')
    (out/'execution_parity.json').write_text(json.dumps(checks,indent=2),encoding='utf-8')
    weights,counts=target_arrays(raw,specs,tickers,decisions,progress=progress)
    # Verify the original equally weighted four-factor cashflow construction.
    ref=next(i for i,s in enumerate(specs) if s['factors']==(CORE,'value','quality','momentum') and s['cashflow_weight']==.25)
    original_weights=saved_target_array(source/'own/plus_cashflow',tickers,decisions)
    assert np.allclose(weights[ref],original_weights,rtol=0,atol=1e-11)
    rows=[]; navs=[]
    for start in range(0,len(specs),64):
        finish=min(start+64,len(specs))
        result=batch_backtest(weights[start:finish],decisions,market,config)
        summary,_=summarize_batch(specs[start:finish],result,counts[start:finish],market,config)
        rows.append(summary); navs.append(result['equity'])
        pd.concat(rows,ignore_index=True).to_csv(out/'search_results.csv',index=False)
        progress(f'Executed {finish}/{len(specs)} combinations; current best Sharpe {pd.concat(rows).sharpe.max():.3f}')
    table=pd.concat(rows,ignore_index=True)
    nav=np.concatenate(navs,axis=1)
    pd.DataFrame(nav,index=market['dates'],columns=[s['variant'] for s in specs]).to_csv(out/'all_nav_curves.csv.gz',index_label='Date')
    valid=table.loc[table.stale_price_legs.eq(0)&table.sharpe.notna()].copy()
    ranked=valid.sort_values(['sharpe','min_full_year_return','net_return'],ascending=False)
    ranked.to_csv(out/'ranked_results.csv',index=False)
    maximin=valid.sort_values(['min_full_year_return','sharpe'],ascending=False)
    maximin.to_csv(out/'ranked_by_worst_year.csv',index=False)
    # Replay leading combinations, the best worst-year result, cashflow alone
    # and the previously discussed cashflow reference using the original ledger.
    selected=list(dict.fromkeys(list(ranked.variant.head(3))+list(maximin.variant.head(1))+[specs[0]['variant'],specs[ref]['variant']]))
    full_rows=[]; annual_rows=[]; side_rows=[]; verification={}
    for name in selected:
        i=next(i for i,s in enumerate(specs) if s['variant']==name)
        spec=specs[i]
        progress('Full ledger replay '+name+' '+str(dict(zip(spec['factors'],spec['weights']))))
        signals=copy(base); signals.metadata=deepcopy(base.metadata); signals.signals=raw.copy()
        signals.signals['score']=np.nan; signals.signals['rank']=np.nan
        # Use the same raw-factor ranking routine. The score used by the
        # constructor can be the already sector-centered signed weight because
        # a further demeaning/L1 pass is algebraically invariant.
        weight_frame=pd.DataFrame(weights[i],index=decisions,columns=tickers)
        lookup=weight_frame.stack().rename('score')
        keys=pd.MultiIndex.from_frame(signals.signals[['as_of','ticker']])
        score=lookup.reindex(keys).to_numpy()
        signals.signals['score']=np.where(abs(score)>0,score,np.nan)
        signals.signals['rank']=signals.signals.groupby('as_of').score.rank(ascending=False,method='min')
        signals.coverage=pd.DataFrame(dict(as_of=decisions,ranked_stocks=counts[i]))
        result=WeightedBacktester(config,construction).run(signals,histories=histories)
        PerformanceReport.audit(result)
        saved_nav=nav[:,i]
        error=float(np.max(abs(result['daily_summary'].equity.to_numpy()-saved_nav)))
        assert error<1e-4,(name,error)
        folder=out/'verified'/name; folder.mkdir(parents=True,exist_ok=True)
        for key in ('daily_summary','target_weights','events','rebalance_orders'):
            result[key].to_csv(folder/(key+'.csv'),index=key=='daily_summary')
        row=_summary(result,signals,name,'own',config.initial_nav)
        row['factors']=' + '.join(spec['factors']); row['weights']=json.dumps(dict(zip(spec['factors'],spec['weights'])))
        full_rows.append(row)
        year=PerformanceReport.by_year(result).reset_index(); year.insert(0,'variant',name); annual_rows.append(year)
        side=side_attribution(result,config); side.insert(0,'variant',name); side_rows.append(side)
        verification[name]=dict(accounting='passed',batch_daily_nav_max_error=error,sessions=len(result['daily_summary']),
            source_requests=0,daily_long_short_reconciled=True,first_trade=str(result['events'].execution_date.min().date()))
        pd.DataFrame(full_rows).to_csv(out/'verified_comparison.csv',index=False)
        pd.concat(annual_rows,ignore_index=True).to_csv(out/'verified_annual_returns.csv',index=False)
        pd.concat(side_rows,ignore_index=True).to_csv(out/'verified_side_attribution.csv',index=False)
        (out/'verification.json').write_text(json.dumps(verification,indent=2),encoding='utf-8')
    progress('BEST BY SHARPE\n'+ranked.head(10).to_string(index=False))
    progress('BEST BY WORST FULL YEAR\n'+maximin.head(3).to_string(index=False))
    return ranked


if __name__=='__main__':
    run()
