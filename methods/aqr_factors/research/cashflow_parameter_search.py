"""Recorded parameter tuning around cashflow strategies, on the existing sample.

No yearly switching, return clipping, cost removal, price-gap substitution or
source downloads. All settings are global constants for each candidate. This
is explicitly in-sample optimization, not out-of-sample performance evidence.
"""
from __future__ import annotations

from copy import copy, deepcopy
from dataclasses import asdict, replace
from itertools import product
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd

from .cashflow_combinations import (CORE, FACTOR_NAMES, target_arrays, market_arrays,
    batch_backtest, summarize_batch)
from ..factors import balance_ratios
from ..signals import FactorSignals, SignalConfig
from ..construction import ConstructionConfig, PortfolioConstructor
from .experiments import _summary
from .multifactor_experiment import side_attribution
from methods.aqr_factors.backtest import WeightedBacktestConfig, WeightedBacktester
from reports import PerformanceReport








from ..portfolio import FrozenTargetsConstructor, decision_rows, transform_targets

class ParameterSearch:
    def __init__(self, output_dir='methods/aqr_factors/research/results/cashflow_parameter_search', progress=print,
                 *, rebalance_frequency='quarterly', extra_path=None):
        self.out=Path(output_dir); self.out.mkdir(parents=True,exist_ok=True)
        self.progress=progress
        source=Path('aqr_research/data/backtest_inputs')
        meta=json.loads((source/'metadata.json').read_text(encoding='utf-8'))
        self.base=FactorSignals(SignalConfig(**meta['signal_settings']))
        self.base.metadata=deepcopy(meta['source_metadata'])
        raw=pd.read_csv(source/'monthly_raw_factors.csv.gz',parse_dates=['as_of'],low_memory=False)
        raw=decision_rows(raw,rebalance_frequency)
        for name,values in balance_ratios(raw).items(): raw[name]=values
        extra_columns=list(FACTOR_NAMES[7:])+['estimated_volatility','estimated_beta']
        if extra_path is None:
            extra_path=source/'monthly_extra_inputs.csv.gz'
        extra=pd.read_csv(extra_path,
            parse_dates=['as_of'],usecols=['as_of','ticker']+extra_columns)
        if extra.duplicated(['as_of','ticker']).any():
            raise ValueError('Duplicate dated factor inputs')
        if not set(raw.as_of).issubset(set(extra.as_of)):
            raise ValueError('Factor/risk inputs do not cover every requested decision month')
        self.raw=raw.merge(extra,on=['as_of','ticker'],how='left')
        self.raw.loc[~self.raw.total_assets.gt(0),['quality','roa']]=np.nan
        self.base.signals=self.raw.copy()
        profiles={}
        for r in self.base.database.read("SELECT ticker,payload FROM snapshots WHERE dataset='company' ORDER BY observed_at").itertuples():
            profiles[r.ticker]=json.loads(r.payload)
        self.base.companies=pd.DataFrame([dict(ticker=t,industry=i.get('industry'),sector=i.get('sector')) for t,i in profiles.items()])
        self.config=replace(WeightedBacktestConfig(**meta['backtest_settings']),rebalance_frequency=rebalance_frequency)
        self.construction=ConstructionConfig(**meta['construction'])
        self.tickers=sorted(set(self.raw.ticker)|{'SPY'})
        self.decisions=pd.DatetimeIndex(sorted(self.raw.as_of.unique()))
        self.histories=WeightedBacktester.cached_histories(self.base.database,self.tickers,self.config.start_date,
            str((pd.Timestamp(self.config.end_date)+pd.Timedelta(days=1)).date()))
        self.market=market_arrays(self.histories,self.tickers,self.config)
        sectors=sorted(self.raw.score_group.dropna().unique())
        self.sectors=sectors
        self.groups=[]; self.volatility=[]; self.beta=[]
        for date in self.decisions:
            month=self.raw.loc[self.raw.as_of.eq(date)].set_index('ticker').reindex(self.tickers)
            self.groups.append(pd.Categorical(month.score_group,categories=sectors).codes)
            self.volatility.append(month.estimated_volatility.to_numpy(dtype=float))
            self.beta.append(month.estimated_beta.to_numpy(dtype=float))
        self.groups=np.asarray(self.groups); self.volatility=np.asarray(self.volatility); self.beta=np.asarray(self.beta)
        self.results=[]; self.parameters={}; self.bases={}; self.base_specs={}; self.navs={}; self.target_cache={}
        self.trial_count=0
        manifest=dict(goal_sharpe=1.58,require_positive_annual_returns=True,backtest=asdict(self.config),source_requests=0,
            permitted_parameters='Factor weights, per-side score tails, score power, inverse volatility sizing, sector budgets, estimated-beta projection, ex-ante short volatility filters.',
            rebalance_frequency=rebalance_frequency,decision_count=len(self.decisions),
            safeguards=f'Fixed candidate parameters across years; {rebalance_frequency} decisions, original universe, NAV, fees and raw dates; gross 1/sector and dollar neutrality; stale price portfolios excluded; no guessed prices.',
            interpretation='Optimization on already inspected 2022-2026 returns. Any target achievement is in-sample only, not independently validated expected performance.',
            source_sha256=hashlib.sha256((source/'monthly_raw_factors.csv.gz').read_bytes()).hexdigest(),
            extra_inputs_path=str(extra_path),extra_inputs_sha256=hashlib.sha256(Path(extra_path).read_bytes()).hexdigest())
        (self.out/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')

    def add_bases(self, specs):
        new=[{**s,'factors':tuple(s['factors']),'weights':tuple(s['weights'])} for s in specs if s['variant'] not in self.bases]
        if not new: return
        weights,counts=target_arrays(self.raw,new,self.tickers,self.decisions,progress=lambda *_:None)
        for i,spec in enumerate(new):
            self.bases[spec['variant']]=weights[i]
            self.base_specs[spec['variant']]=spec
        self.progress(f'Prepared {len(new)} new factor-weight bases')

    def evaluate(self, cases, stage):
        if not cases: return pd.DataFrame()
        stage_rows=[]
        for start in range(0,len(cases),48):
            batch=cases[start:start+48]
            targets=[]; specs=[]; params=[]
            for base_name,settings in batch:
                name=f'param_{self.trial_count:06d}'; self.trial_count+=1
                settings=dict(settings)
                sector_mask=settings.pop('sector_mask',None)
                w=transform_targets(self.bases[base_name],self.groups,self.volatility,self.beta,
                    sector_mask=sector_mask,**settings)
                if sector_mask is not None: settings['sector_mask']=list(sector_mask)
                selected=(abs(w)>1e-10).sum(axis=1)
                # Reject degenerate or excessively concentrated target designs.
                min_longs=int((w>1e-10).sum(axis=1).min()); min_shorts=int((w<-1e-10).sum(axis=1).min())
                valid=(np.allclose(abs(w).sum(axis=1),1,atol=1e-8) and abs(w).max()<=.05+1e-8
                       and min_longs>=10 and min_shorts>=10)
                if not valid:
                    continue
                base=self.base_specs[base_name]
                spec=dict(variant=name,**{k:base[k] for k in ['factors','weights','cashflow_weight','factor_count']})
                targets.append(w); specs.append(spec)
                params.append(dict(variant=name,base_name=base_name,stage=stage,settings=settings,min_longs=min_longs,
                    min_shorts=min_shorts,max_weight=float(abs(w).max())))
            if not targets: continue
            targets=np.asarray(targets)
            result=batch_backtest(targets,self.decisions,self.market,self.config)
            counts=(abs(targets)>1e-10).sum(axis=2)
            table,_=summarize_batch(specs,result,counts,self.market,self.config)
            table['stage']=stage
            for i,param in enumerate(params):
                self.parameters[param['variant']]=param
                table.loc[table.variant.eq(param['variant']),'parameters']=json.dumps(param['settings'])
                table.loc[table.variant.eq(param['variant']),'base_name']=param['base_name']
                self.navs[param['variant']]=result['equity'][:,i].copy()
            stage_rows.append(table); self.results.append(table)
            total=pd.concat(self.results,ignore_index=True)
            total.to_csv(self.out/'all_results.csv',index=False)
            reliable=total.loc[total.stale_price_legs.eq(0)].sort_values('sharpe',ascending=False)
            reliable.to_csv(self.out/'ranked_results.csv',index=False)
            (self.out/'parameters.json').write_text(json.dumps(self.parameters,indent=2),encoding='utf-8')
            best=reliable.iloc[0] if len(reliable) else None
            self.progress(f'{stage} {min(start+48,len(cases))}/{len(cases)}: best reliable Sharpe '+(f'{best.sharpe:.4f}' if best is not None else 'unavailable'))
        return pd.concat(stage_rows,ignore_index=True) if stage_rows else pd.DataFrame()

    def targets_for(self,name):
        param=self.parameters[name]
        return transform_targets(self.bases[param['base_name']],self.groups,self.volatility,self.beta,**param['settings'])

    def verify(self,names):
        rows=[]; annual=[]; sides=[]; checks={}
        for name in names:
            self.progress('Full ledger verification '+name)
            w=self.targets_for(name)
            signals=copy(self.base); signals.metadata=deepcopy(self.base.metadata); signals.signals=self.raw.copy()
            base_weights=self.bases[self.parameters[name]['base_name']]
            base_scores=pd.DataFrame(base_weights,index=self.decisions,columns=self.tickers).stack()
            keys=pd.MultiIndex.from_frame(signals.signals[['as_of','ticker']])
            signals.signals['score']=base_scores.reindex(keys).to_numpy()
            signals.coverage=pd.DataFrame(dict(as_of=self.decisions,ranked_stocks=(abs(w)>1e-10).sum(axis=1)))
            engine=WeightedBacktester(self.config,self.construction)
            engine.constructor=FrozenTargetsConstructor(self.construction,w,self.tickers,self.decisions)
            result=engine.run(signals,histories=self.histories)
            PerformanceReport.audit(result)
            error=float(np.max(abs(result['daily_summary'].equity.to_numpy()-self.navs[name])))
            assert error<1e-4,(name,error)
            assert result['daily_summary'].stale_price_legs.sum()==0
            folder=self.out/'verified'/name; folder.mkdir(parents=True,exist_ok=True)
            for key in ('daily_summary','target_weights','events','rebalance_orders'):
                result[key].to_csv(folder/(key+'.csv'),index=key=='daily_summary')
            base=self.base_specs[self.parameters[name]['base_name']]
            row=_summary(result,signals,name,'tuned_own',self.config.initial_nav)
            row['factors']=' + '.join(base['factors']); row['weights']=json.dumps(dict(zip(base['factors'],base['weights'])))
            row['parameters']=json.dumps(self.parameters[name]['settings']); rows.append(row)
            year=PerformanceReport.by_year(result).reset_index(); year.insert(0,'variant',name); annual.append(year)
            side=side_attribution(result,self.config); side.insert(0,'variant',name); sides.append(side)
            constraints=[]
            for t,date in enumerate(self.decisions):
                net_by_sector=[float(w[t,self.groups[t]==group].sum()) for group in np.unique(self.groups[t])]
                constraints.append(dict(as_of=date,long_weight=float(w[t].clip(min=0).sum()),
                    short_weight=float(w[t].clip(max=0).sum()),gross=float(abs(w[t]).sum()),net=float(w[t].sum()),
                    estimated_beta=float(w[t]@np.nan_to_num(self.beta[t])),max_weight=float(abs(w[t]).max()),
                    max_abs_sector_net=max(abs(value) for value in net_by_sector),
                    target_longs=int((w[t]>1e-10).sum()),target_shorts=int((w[t]<-1e-10).sum())))
            audit=pd.DataFrame(constraints)
            assert np.allclose(audit.gross,1,atol=1e-8) and audit.net.abs().max()<1e-8
            assert audit.max_abs_sector_net.max()<1e-8
            assert audit.max_weight.max()<=.05+1e-8
            if self.parameters[name]['settings'].get('beta_neutral')==1:
                assert audit.estimated_beta.abs().max()<1e-8
            audit.to_csv(folder/'target_constraint_audit.csv',index=False)
            checks[name]=dict(accounting='passed',source_requests=0,batch_nav_max_error=error,source_prices_complete=True,
                sessions=len(result['daily_summary']),positive_all_years=bool((year.loc[year.portfolio.eq('Strategy'),'return']>0).all()),
                sharpe=row['sharpe'],goal_met=bool(row['sharpe']>=1.58 and (year.loc[year.portfolio.eq('Strategy'),'return']>0).all()))
            pd.DataFrame(rows).to_csv(self.out/'verified_comparison.csv',index=False)
            pd.concat(annual).to_csv(self.out/'verified_annual_returns.csv',index=False)
            pd.concat(sides).to_csv(self.out/'verified_side_attribution.csv',index=False)
            (self.out/'verification.json').write_text(json.dumps(checks,indent=2),encoding='utf-8')
        return pd.DataFrame(rows)


def replay_candidate(preset=None):
    """Recompute the selected strategy without repeating the parameter search."""
    if preset is None:
        from ..settings import AQRSettings
        from ..run import replay_file
        return replay_file(AQRSettings.selected_parameter_file)
    output=Path('methods/aqr_factors/research/results/cashflow_parameter_search/candidate_replay')
    search=ParameterSearch(output_dir=output)
    factors=tuple(preset['factor_weights'])
    weights=tuple(preset['factor_weights'].values())
    assert factors[0]==CORE and abs(sum(weights)-1)<1e-12
    base_name='candidate_factors'
    search.add_bases([dict(variant=base_name,factors=factors,weights=weights,
        cashflow_weight=weights[0],factor_count=len(factors))])
    name=preset['variant']
    search.parameters[name]=dict(variant=name,base_name=base_name,stage='replay',settings=preset['construction'])
    targets=search.targets_for(name)
    search.navs[name]=batch_backtest(targets[None,:,:],search.decisions,search.market,search.config)['equity'][:,0]
    result=search.verify([name])
    (output/'candidate_configuration.json').write_text(json.dumps(preset,indent=2),encoding='utf-8')
    print(result[['variant','net_return','sharpe','max_drawdown','beta']].to_string(index=False),flush=True)
    return result


def run():
    search=ParameterSearch()
    ranked=pd.read_csv('methods/aqr_factors/research/results/cashflow_combinations/ranked_results.csv')
    original=json.loads(Path('methods/aqr_factors/research/results/cashflow_combinations/specifications.json').read_text())
    seed_names=list(dict.fromkeys(ranked.variant.head(10).tolist()+['cf_0352','cf_1126']))
    specs=[s for s in original if s['variant'] in seed_names]
    search.add_bases(specs)
    cases=[]
    for spec in specs:
        for tail,power,inverse_vol,budget,neutral in product([1.,.3,.2,.1,.05],[0.,1.,2.],[0.,1.],['signal','equal'],[0.,1.]):
            cases.append((spec['variant'],dict(tail=tail,power=power,inverse_vol=inverse_vol,
                sector_budget=budget,beta_neutral=neutral)))
    search.evaluate(cases,'construction_coarse')
    table=pd.concat(search.results,ignore_index=True)
    valid=table.loc[table.stale_price_legs.eq(0)].sort_values('sharpe',ascending=False)
    print('COARSE TOP\n'+valid.head(10).to_string(index=False),flush=True)
    # Persist reconstruction inputs for subsequent stages in this same run.
    np.savez_compressed(search.out/'base_targets.npz',**search.bases)
    (search.out/'base_specifications.json').write_text(json.dumps(search.base_specs,indent=2),encoding='utf-8')
    selected=valid.variant.head(3).tolist()
    search.verify(selected)
    return search


if __name__=='__main__':
    import sys
    replay_candidate() if '--candidate' in sys.argv else run()
