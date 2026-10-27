"""Fixed, prospective-rule ten-factor trial using only the existing local cache.

This module reads imported quarterly facts; it never parses/fetches company
filings, changes source facts, or selects weights using realized test returns.
"""
from __future__ import annotations

from copy import copy, deepcopy
from dataclasses import asdict, replace
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd

from ..signals import FactorSignals, SignalConfig
from ..operators import transform
from ..construction import PortfolioConstructor, ConstructionConfig
from .experiments import _summary
from methods.aqr_factors.backtest import WeightedBacktester, WeightedBacktestConfig
from reports import PerformanceReport


FAMILIES = {
    'momentum': ('momentum_12_1', 'momentum_6_1', 'residual_momentum'),
    'quality': ('gross_profitability', 'average_asset_roa', 'low_accruals', 'profit_stability'),
    'value': ('ocf_yield', 'fcf_yield', 'ep_yield'),
}
FAMILY_WEIGHTS = {'momentum': .40, 'quality': .35, 'value': .25}
FACTORS = tuple(factor for names in FAMILIES.values() for factor in names)
DEFINITIONS = {
    'momentum_12_1': 'Split-consistent Close[t-21]/Close[t-252]-1; exchange sessions.',
    'momentum_6_1': 'Split-consistent Close[t-21]/Close[t-126]-1; exchange sessions.',
    'residual_momentum': 'Standardized sum of last 105 residuals from a trailing 252-session market/sector regression ending t-21; minimum 200 observations.',
    'gross_profitability': 'Four consecutive available quarters gross profit / average total assets at fiscal end and one fiscal year earlier.',
    'average_asset_roa': 'Four consecutive available quarters net income / average fiscal-year assets.',
    'low_accruals': '-(TTM net income - TTM operating cashflow)/average fiscal-year assets.',
    'profit_stability': '-Sample std of eight consecutive quarterly net income/average quarter-start and quarter-end total assets.',
    'ocf_yield': 'Four consecutive available quarters operating cashflow / historical market cap.',
    'fcf_yield': 'Four consecutive available quarters free cashflow / historical market cap; missing reported FCF derived from OCF - absolute capex only when both are present.',
    'ep_yield': 'Four consecutive available quarters common-stockholder net income / historical market cap; negative earnings retain their sign.',
}
METRICS = ('TotalAssets', 'GrossProfit', 'TotalRevenue', 'CostOfRevenue',
           'NetIncome', 'NetIncomeCommonStockholders', 'OperatingCashFlow',
           'CapitalExpenditure', 'FreeCashFlow')


from ..performance import side_attribution

def consecutive(periods):
    return all(60 <= (b-a).days <= 120 for a, b in zip(periods, periods[1:]))


def financial_features(known):
    """Use the latest jointly available fiscal quarter and consecutive history.

    known contains only records already public (or assumed public under the
    existing date policy). Missing quarters/values remain missing, never zero.
    """
    common = set(known['income']) & set(known['balance']) & set(known['cashflow'])
    if not common:
        return {}
    end = max(common)
    income_periods = sorted(d for d in known['income'] if d <= end)
    used = []
    def fact(statement, period, metric):
        record = known[statement].get(period)
        if record is None:
            return np.nan
        used.append(record)
        values = record['values']
        value = values.get(metric, np.nan)
        if not np.isfinite(value) and metric == 'GrossProfit':
            value = values.get('TotalRevenue', np.nan)-values.get('CostOfRevenue', np.nan)
        if not np.isfinite(value) and metric == 'FreeCashFlow':
            value = values.get('OperatingCashFlow', np.nan)-abs(values.get('CapitalExpenditure', np.nan))
        return value
    four = income_periods[-4:]
    eight = income_periods[-8:]
    def total(statement, metric):
        if len(four) != 4 or not consecutive(four):
            return np.nan
        values = [fact(statement, d, metric) for d in four]
        return sum(values) if np.isfinite(values).all() else np.nan
    assets = fact('balance', end, 'TotalAssets')
    previous = [d for d in known['balance'] if 330 <= (end-d).days <= 400]
    year_ago = min(previous, key=lambda d: abs((end-d).days-365)) if previous else None
    previous_assets = fact('balance', year_ago, 'TotalAssets') if year_ago is not None else np.nan
    average_assets = (assets+previous_assets)/2 if assets>0 and previous_assets>0 else np.nan
    gp = total('income', 'GrossProfit')
    ni = total('income', 'NetIncome')
    common_ni = total('income', 'NetIncomeCommonStockholders')
    ocf = total('cashflow', 'OperatingCashFlow')
    fcf = total('cashflow', 'FreeCashFlow')
    roa_quarters = []
    if len(eight)==8 and consecutive(eight):
        for period in eight:
            earlier = [d for d in known['balance'] if 60 <= (period-d).days <= 120]
            before = max(earlier) if earlier else None
            a1 = fact('balance', period, 'TotalAssets')
            a0 = fact('balance', before, 'TotalAssets') if before is not None else np.nan
            quarterly_income = fact('income', period, 'NetIncome')
            roa_quarters.append(quarterly_income/((a1+a0)/2) if a1>0 and a0>0 else np.nan)
    stability = -np.std(roa_quarters, ddof=1) if len(roa_quarters)==8 and np.isfinite(roa_quarters).all() else np.nan
    return dict(financial_period_end=end, financial_available_from=max(r['effective_available'] for r in used),
        financial_date_assumed=any(r['date_assumed'] for r in used),
        gross_profit_ttm=gp, net_income_ttm_new=ni, common_net_income_ttm_new=common_ni,
        operating_cashflow_ttm_new=ocf, free_cashflow_ttm=fcf, average_assets=average_assets,
        gross_profitability=gp/average_assets, average_asset_roa=ni/average_assets,
        low_accruals=-(ni-ocf)/average_assets, profit_stability=stability)


def quarterly_inputs(database, rows, lag_days=90, progress=print):
    reports = database.read("""SELECT r.report_id,r.ticker,r.statement,r.period_end,
        r.available_from,r.observed_at,r.financial_currency,p.reference_available_from
        FROM fundamental_reports r JOIN yahoo_csv_report_provenance p USING(report_id)
        WHERE r.frequency='quarterly' AND r.statement IN ('balance','income','cashflow')
        AND r.period_end>='2017-01-01' AND r.financial_currency='USD'
        AND EXISTS (SELECT 1 FROM yahoo_csv_periods c WHERE c.report_id=r.report_id
                    AND c.alignment_status NOT LIKE 'hold_%')""")
    reports['period_end']=pd.to_datetime(reports.period_end)
    reports['effective_available']=pd.to_datetime(reports.available_from.fillna(reports.reference_available_from),errors='coerce')
    reports['date_assumed']=reports.effective_available.isna()
    reports.loc[reports.date_assumed,'effective_available']=reports.loc[reports.date_assumed,'period_end']+pd.Timedelta(days=lag_days)
    reports=reports.loc[reports.effective_available.gt(reports.period_end)
        & reports.effective_available.le(rows.as_of.max()) & reports.ticker.isin(rows.ticker)]
    identifiers=reports.report_id.tolist()
    values={}
    with database.connection() as db:
        for offset in range(0,len(identifiers),500):
            part=identifiers[offset:offset+500]
            query='SELECT report_id,metric,value FROM fundamental_facts WHERE report_id IN ({}) AND metric IN ({})'.format(
                ','.join('?' for _ in part),','.join('?' for _ in METRICS))
            for identifier,metric,value in db.execute(query,(*part,*METRICS)):
                if value is not None:
                    values.setdefault(identifier,{})[metric]=value
    progress(f'Read {len(reports):,} local quarterly report versions')
    dates=sorted(rows.as_of.unique())
    records=[]
    for ticker, group in reports.groupby('ticker',sort=True):
        ordered=group.sort_values(['effective_available','period_end','observed_at','report_id']).to_dict('records')
        known={name:{} for name in ('balance','income','cashflow')}
        index=0
        for date in dates:
            while index<len(ordered) and ordered[index]['effective_available']<=date:
                record=ordered[index]
                record['values']=values.get(record['report_id'],{})
                known[record['statement']][record['period_end']]=record
                index+=1
            features=financial_features(known)
            if features:
                assert features['financial_available_from']<=date
                records.append(dict(as_of=date,ticker=ticker,**features))
    return pd.DataFrame(records)


def residual_trend(stock, market, sector, *, minimum=200):
    """Fit trailing market/peer exposures; score only the last 105 residuals.

    Inputs end 21 sessions before the decision. Using a proper subset of the
    regression window avoids the zero-sum residual/intercept degeneracy.
    """
    frame=pd.concat([stock,market,sector],axis=1,keys=['stock','market','sector']).tail(252)
    if len(frame)<minimum or not frame.iloc[-105:].notna().all().all():
        return np.nan
    fit=frame.dropna()
    if len(fit)<minimum:
        return np.nan
    x=np.c_[np.ones(len(fit)),fit[['market','sector']].to_numpy()]
    coefficients=np.linalg.lstsq(x,fit.stock.to_numpy(),rcond=None)[0]
    tail=frame.iloc[-105:]
    residuals=tail.stock.to_numpy()-np.c_[np.ones(105),tail[['market','sector']].to_numpy()]@coefficients
    deviation=residuals.std(ddof=1)
    return float(residuals.sum()/(deviation*np.sqrt(105))) if deviation>1e-12 else 0.


def estimate_risk(window):
    """Pairwise beta uses exactly each stock's observed market dates."""
    window=window.loc[window.SPY.notna()]
    market=window.SPY
    present=window.notna()
    count=present.sum()
    market_mean=present.mul(market,axis=0).sum()/count
    cross=window.mul(market,axis=0).sum()-window.sum()*market_mean
    market_square=present.mul(market.pow(2),axis=0).sum()-count*market_mean.pow(2)
    beta=(cross/market_square).where(count.ge(200) & market_square.gt(0))
    sigma=window.std(ddof=1)*np.sqrt(252)
    return beta,sigma,count


def price_inputs(rows, closes, progress=print):
    returns=closes.pct_change(fill_method=None)
    market=returns.SPY
    records=[]
    for number,(date,month) in enumerate(rows.groupby('as_of',sort=True),1):
        position=closes.index.get_loc(date)
        risk=returns.iloc[max(0,position-251):position+1]
        beta,sigma,valid_counts=estimate_risk(risk)
        active=month.loc[month.in_universe & ~month.input_status.eq('download_failed')]
        peer_series={}
        for sector, members in active.groupby('score_group'):
            names=[name for name in members.ticker if name in returns]
            peers=returns[names]
            peer_series[sector]=(peers.sum(axis=1,min_count=1),peers.notna().sum(axis=1),set(names))
        for row in month.itertuples():
            if row.ticker not in closes:
                continue
            price=closes[row.ticker]
            item=dict(as_of=date,ticker=row.ticker,estimated_beta=beta.get(row.ticker,np.nan),
                estimated_volatility=sigma.get(row.ticker,np.nan),risk_observations=int(valid_counts.get(row.ticker,0)))
            if position>=252:
                item['momentum_12_1']=price.iloc[position-21]/price.iloc[position-252]-1
            if position>=126:
                item['momentum_6_1']=price.iloc[position-21]/price.iloc[position-126]-1
            peers=peer_series.get(row.score_group)
            if peers is not None and len(peers[2])>2 and row.ticker in peers[2]:
                # Leave the stock out of its own equal-weight peer benchmark.
                own_returns=returns[row.ticker]
                peer_count=peers[1]-own_returns.notna().astype(int)
                sector_return=(peers[0]-own_returns.fillna(0))/peer_count.where(peer_count>=2)
                end=position-21
                item['residual_momentum']=residual_trend(returns[row.ticker].iloc[:end+1],market.iloc[:end+1],sector_return.iloc[:end+1])
            records.append(item)
        progress(f'Price factors {number}/{rows.as_of.nunique()}: {date.date()}')
    return pd.DataFrame(records)


def score_inputs(raw, mask=None):
    frame=raw.copy()
    complete=np.isfinite(frame[list(FACTORS)]).all(axis=1)
    active=frame.in_universe & ~frame.input_status.isin(['download_failed','missing_month_end_price'])
    valid=active & complete & frame.close.gt(0) & frame.market_cap.gt(0)
    valid &= frame.risk_observations.ge(200) & frame.estimated_volatility.gt(0) & frame.estimated_beta.notna()
    if mask is not None:
        valid &= mask
    frame['score']=np.nan
    frame['rank']=np.nan
    coverage=[]
    for date,month in frame.groupby('as_of',sort=True):
        eligible=month.index[valid.loc[month.index] & month.score_group.notna()]
        sizes=frame.loc[eligible].groupby('score_group').ticker.transform('size')
        usable=eligible[sizes.ge(10)]
        groups=frame.loc[usable,'score_group']
        for factor in FACTORS:
            frame.loc[usable,factor+'_score']=transform(frame.loc[usable,factor],('group_rank','group_zscore'),groups)
        for family,factors in FAMILIES.items():
            frame.loc[usable,family+'_family_score']=frame.loc[usable,[f+'_score' for f in factors]].mean(axis=1)
        frame.loc[usable,'score']=sum(FAMILY_WEIGHTS[family]*frame.loc[usable,family+'_family_score'] for family in FAMILIES)
        frame.loc[usable,'rank']=frame.loc[usable,'score'].rank(ascending=False,method='min')
        coverage.append(dict(as_of=date,sample_size=int(month.in_universe.sum()),ranked_stocks=len(usable),
            complete_inputs=int((active.loc[month.index]&complete.loc[month.index]).sum())))
    return frame,pd.DataFrame(coverage)


class RiskConstrainedConstructor(PortfolioConstructor):
    """Project inverse-volatility scores onto ex-ante constraints, no fitting.

    Preserve score direction, gross 1, net/sector neutrality, beta band +/-0.05,
    max absolute weight 1%. A fixed 5bp turnover penalty is included.
    """
    def __init__(self, config, max_weight=.01, beta_band=.05):
        super().__init__(config)
        self.weight_cap=max_weight
        self.beta_band=beta_band
        self.checks=[]

    def build(self, signals, **kwargs):
        import osqp
        from scipy import sparse
        targets=super().build(signals,**kwargs)
        data=signals.set_index('ticker')
        for date,month in targets.groupby('as_of',sort=True):
            eligible=month.index[month.target_weight.ne(0)]
            names=targets.loc[eligible,'ticker']
            groups=targets.loc[eligible,'construction_group'].reset_index(drop=True)
            volatility=data.loc[names,'estimated_volatility'].to_numpy()
            beta=data.loc[names,'estimated_beta'].to_numpy()
            base=targets.loc[eligible,'target_weight'].to_numpy()
            signs=np.sign(base)
            ideal=base/volatility
            # Preserve the original long/short signal directions in the QP.
            ideal/=abs(ideal).sum()
            previous=kwargs.get('current_weights',pd.Series(dtype=float)).reindex(names,fill_value=0).to_numpy()
            n=len(names)
            eye=sparse.eye(n,format='csc')
            zero=sparse.csc_matrix((n,n))
            matrices=[sparse.hstack((eye,-eye)),sparse.hstack((-eye,-eye)),sparse.eye(2*n,format='csc')]
            lower=[np.full(n,-np.inf),np.full(n,-np.inf),np.r_[np.where(signs>0,0,-self.weight_cap),np.zeros(n)]]
            upper=[previous,-previous,np.r_[np.where(signs>0,self.weight_cap,0),np.full(n,np.inf)]]
            def constraint(coefficients,lo,hi):
                matrices.append(sparse.csc_matrix(np.r_[coefficients,np.zeros(n)].reshape(1,-1)))
                lower.append(np.array([lo])); upper.append(np.array([hi]))
            constraint(np.ones(n),0,0)
            constraint(signs,1,1)
            constraint(beta,-self.beta_band,self.beta_band)
            for group in groups.unique():
                constraint(groups.eq(group).to_numpy(dtype=float),0,0)
            problem=osqp.OSQP()
            problem.setup(P=sparse.block_diag((eye,zero),format='csc'),q=np.r_[-ideal,np.full(n,self.config.transaction_cost_rate)],
                A=sparse.vstack(matrices,format='csc'),l=np.concatenate(lower),u=np.concatenate(upper),
                verbose=False,eps_abs=1e-8,eps_rel=1e-8,max_iter=100000,polishing=True)
            solution=problem.solve(raise_error=True)
            weights=solution.x[:n]
            weights[abs(weights)<1e-8]=0
            assert abs(weights.sum())<1e-6 and abs(abs(weights).sum()-1)<1e-6
            assert abs(beta@weights)<=self.beta_band+1e-6 and abs(weights).max()<=self.weight_cap+1e-6
            assert np.all(weights*signs>=-1e-7)
            assert all(abs(weights[groups.eq(g).to_numpy()].sum())<1e-6 for g in groups.unique())
            targets.loc[eligible,'target_weight']=weights
            self.checks.append(dict(as_of=date,estimated_beta=float(beta@weights),gross=float(abs(weights).sum()),
                net=float(weights.sum()),max_weight=float(abs(weights).max()),solver_status=solution.info.status))
        targets['target_side']=np.select([targets.target_weight.gt(0),targets.target_weight.lt(0)],['long','short'],default='flat')
        return targets




def run(output_dir='methods/aqr_factors/research/results/ten_factor_trial', progress=print):
    source=Path('methods/aqr_factors/research/results/factor_comparison')
    out=Path(output_dir); out.mkdir(parents=True,exist_ok=True)
    meta=json.loads((source/'comparison_settings.json').read_text(encoding='utf-8'))
    base=FactorSignals(SignalConfig(**meta['signal_settings']))
    base.metadata=deepcopy(meta['source_metadata'])
    original=pd.read_csv(source/'own/baseline/monthly_scores.csv.gz',parse_dates=['as_of'],low_memory=False)
    original=original.loc[original.as_of.dt.month.isin([3,6,9,12])].reset_index(drop=True)
    base.signals=original.copy()
    profiles={}
    for r in base.database.read("SELECT ticker,payload FROM snapshots WHERE dataset='company' ORDER BY observed_at").itertuples():
        profiles[r.ticker]=json.loads(r.payload)
    base.companies=pd.DataFrame([dict(ticker=t,industry=i.get('industry'),sector=i.get('sector')) for t,i in profiles.items()])
    cfg=WeightedBacktestConfig(**meta['backtest_settings'])
    construction=ConstructionConfig(**meta['construction'])
    manifest=dict(families=FAMILIES,definitions=DEFINITIONS,weights=FAMILY_WEIGHTS,backtest=asdict(cfg),source_requests=0,
        missing_policy='Drop each decision row missing any of the ten selected inputs; no full-history intersection.',
        financial_alignment='Latest fiscal quarter jointly available in income, balance and cashflow; four/eight consecutive fiscal quarters; TTM assets averaged over the prior fiscal year.',
        date_policy='Existing source/referenced dates retained; missing dates use fiscal period end +90 days; latest vendor versions may be restated.',
        residual_momentum='252 daily observations ending 21 sessions earlier, minimum 200; fit intercept + SPY + leave-one-out sector peer returns known at that cutoff; standardized sum of last 105 residuals.',
        risk='Own candidates unchanged: inverse volatility score projection, fixed alpha sign, gross=1/net=0/sector net=0; beta band +/-0.05, per-name cap 1%, 5bp turnover penalty.',
        validation='Fixed 40/35/25 weights from the proposal; no retrospective parameter search, no claim of independent out-of-sample validation.',
        source_scores_sha256=hashlib.sha256((source/'own/baseline/monthly_scores.csv.gz').read_bytes()).hexdigest())
    manifest_path=out/'manifest.json'
    if manifest_path.exists():
        old_manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
        if old_manifest.get('source_scores_sha256')!=manifest['source_scores_sha256']:
            raise ValueError('Existing checkpoints use a different source snapshot; choose a fresh output directory')
    manifest_path.write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    financial_path=out/'financial_inputs.csv.gz'
    if financial_path.exists():
        financial=pd.read_csv(financial_path,parse_dates=['as_of','financial_period_end','financial_available_from'])
    else:
        financial=quarterly_inputs(base.database,original,progress=progress)
        financial.to_csv(financial_path,index=False)
    progress('Reading cached prices with warmup; no source downloads')
    histories=WeightedBacktester.cached_histories(base.database,sorted(set(original.ticker)|{'SPY'}),'2020-12-01','2026-10-23')
    calendar=histories['SPY'].index
    close_columns={}
    for ticker,frame in histories.items():
        if frame.empty:
            continue
        ratios=frame['Stock Splits'].where(frame['Stock Splits'].gt(0),1.)
        future=ratios.iloc[::-1].cumprod().iloc[::-1]/ratios
        close_columns[ticker]=(frame.Close/future).reindex(calendar)
    closes=pd.DataFrame(close_columns,index=calendar)
    price_path=out/'price_inputs.csv.gz'
    if price_path.exists():
        prices=pd.read_csv(price_path,parse_dates=['as_of'])
        # Recalculate small risk statistics from current cached windows. Never
        # mix a stock's observed-date covariance with all-date market variance.
        returns=closes.pct_change(fill_method=None)
        for date,part in prices.groupby('as_of'):
            beta,sigma,count=estimate_risk(returns.loc[:date].tail(252))
            prices.loc[part.index,'estimated_beta']=part.ticker.map(beta)
            prices.loc[part.index,'estimated_volatility']=part.ticker.map(sigma)
            prices.loc[part.index,'risk_observations']=part.ticker.map(count)
        prices.to_csv(price_path,index=False)
    else:
        prices=price_inputs(original,closes,progress=progress)
        prices.to_csv(price_path,index=False)
    raw=original.merge(financial,on=['as_of','ticker'],how='left').merge(prices,on=['as_of','ticker'],how='left')
    raw['ocf_yield']=raw.operating_cashflow_ttm_new/raw.market_cap.where(raw.market_cap.gt(0))
    raw['fcf_yield']=raw.free_cashflow_ttm/raw.market_cap.where(raw.market_cap.gt(0))
    raw['ep_yield']=raw.common_net_income_ttm_new/raw.market_cap.where(raw.market_cap.gt(0))
    scored,coverage=score_inputs(raw)
    scored.to_csv(out/'quarterly_scores.csv.gz',index=False)
    coverage.to_csv(out/'candidate_coverage.csv',index=False)
    coverage_fields=[]
    for date,month in raw.loc[raw.in_universe].groupby('as_of'):
        coverage_fields.append(dict(as_of=date,**{f:int(np.isfinite(month[f]).sum()) for f in FACTORS}))
    pd.DataFrame(coverage_fields).to_csv(out/'factor_coverage.csv',index=False)
    progress('CANDIDATE COVERAGE\n'+coverage.to_string(index=False))
    usable=scored.score.notna()
    assert scored.loc[usable,'financial_available_from'].le(scored.loc[usable,'as_of']).all()
    # Historical price units must reproduce the saved 12-1 momentum.
    compare=raw.loc[usable,['momentum','momentum_12_1']].dropna()
    assert np.allclose(compare.momentum,compare.momentum_12_1,rtol=1e-8,atol=1e-8)
    # Legacy signals count ticker bars; this experiment counts exchange sessions.
    # Audit inactive/missing-bar differences rather than requiring unavailable
    # observations outside the tradable candidate set to match.
    finite=raw.momentum.notna() & raw.momentum_12_1.notna()
    different=finite & ~np.isclose(raw.momentum,raw.momentum_12_1,rtol=1e-8,atol=1e-8)
    raw.loc[different,['as_of','ticker','in_universe','momentum','momentum_12_1']].to_csv(out/'price_alignment_exceptions.csv',index=False)
    shared=usable & original.score.notna()
    size=raw.loc[shared].groupby(['as_of','score_group']).ticker.transform('size')
    shared.loc[size.index[size.lt(10)]]=False
    cases=('ten_factor_plain','ten_factor_risk','ten_factor_common','original_on_ten_factor_pool')
    summaries=[]; annual=[]; sides=[]; checks={}
    for name in cases:
        progress('Backtest '+name)
        signals=copy(base); signals.metadata=deepcopy(base.metadata)
        if name=='original_on_ten_factor_pool':
            signals.signals=original.copy()
            signals.rescore(base.config,eligibility_mask=shared)
        elif name=='ten_factor_common':
            signals.signals,signals.coverage=score_inputs(raw,mask=shared)
        else:
            signals.signals=scored.copy(); signals.coverage=coverage.copy()
        if name in {'ten_factor_common','original_on_ten_factor_pool'}:
            keys=signals.signals.loc[signals.signals.score.notna(),['as_of','ticker']].sort_values(['as_of','ticker']).reset_index(drop=True)
            expected=raw.loc[shared,['as_of','ticker']].sort_values(['as_of','ticker']).reset_index(drop=True)
            assert keys.equals(expected)
        engine=WeightedBacktester(cfg,construction)
        if name=='ten_factor_risk':
            engine.constructor=RiskConstrainedConstructor(construction)
        result=engine.run(signals,histories=histories)
        PerformanceReport.audit(result)
        if name=='ten_factor_risk':
            pd.DataFrame(engine.constructor.checks).to_csv(out/'risk_constraint_audit.csv',index=False)
        summary=_summary(result,signals,name,'common' if name in {'ten_factor_common','original_on_ten_factor_pool'} else 'own',cfg.initial_nav)
        summary['factors']=' + '.join(base.config.factor_operators if name=='original_on_ten_factor_pool' else FACTORS)
        summary['mean_ranked']=signals.coverage.ranked_stocks.mean()
        summaries.append(summary)
        year=PerformanceReport.by_year(result).reset_index(); year.insert(0,'variant',name); annual.append(year)
        side=side_attribution(result,cfg); side.insert(0,'variant',name); sides.append(side)
        folder=out/name; folder.mkdir(exist_ok=True)
        for key in ('daily_summary','target_weights','events','rebalance_orders'):
            result[key].to_csv(folder/(key+'.csv'),index=key=='daily_summary')
        checks[name]=dict(accounting='passed',daily_long_short_reconciled=True,sessions=len(result['daily_summary']),
            first_trade=str(result['events'].execution_date.min().date()),stale_price_legs=int(result['daily_summary'].stale_price_legs.sum()),
            source_requests=0,financial_inputs_available_by_cutoff=True)
        pd.DataFrame(summaries).to_csv(out/'comparison.csv',index=False)
        pd.concat(annual,ignore_index=True).to_csv(out/'performance_by_year.csv',index=False)
        pd.concat(sides,ignore_index=True).to_csv(out/'annual_side_attribution.csv',index=False)
        (out/'verification.json').write_text(json.dumps(checks,indent=2),encoding='utf-8')
        progress(f"  Net {summary['net_return']:.3%}; Sharpe {summary['sharpe']:.3f}; beta {summary['beta']:.3f}")
    progress(pd.DataFrame(summaries)[['variant','net_return','sharpe','max_drawdown','beta','mean_ranked']].to_string(index=False))
    progress(pd.concat(annual).query("portfolio == 'Strategy'").pivot(index='variant',columns='year',values='return').to_string())
    return pd.DataFrame(summaries)


if __name__=='__main__':
    run()
