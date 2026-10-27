"""Fixed monthly portfolio combinations selected for complete-year consistency.

Everything is retrospective research on the previously inspected sample. A
blend has constant model allocations across all years. Each month its targets
are combined, netted and normalized to gross 1, then actually traded with the
unchanged integer-share/cost ledger. Proxy return blends only propose trials;
they are never reported as executable performance.
"""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from itertools import combinations
import hashlib
import json

import numpy as np
import pandas as pd
from scipy.optimize import linprog, minimize, LinearConstraint, Bounds

from .cashflow_parameter_search import ParameterSearch, transform_targets
from .cashflow_combinations import batch_backtest
from .monthly_cashflow_search import prepare_inputs
from methods.aqr_factors.backtest import WeightedBacktester


OUTPUT=Path('methods/aqr_factors/research/results/monthly_stability_search')
SMOOTHED_OUTPUT=Path('methods/aqr_factors/research/results/monthly_smoothed_stability')
COMBINED_OUTPUT=Path('methods/aqr_factors/research/results/monthly_combined_stability')
RISK_OUTPUT=Path('methods/aqr_factors/research/results/monthly_risk_stability')
PRIOR=Path('methods/aqr_factors/research/results/monthly_cashflow_parameter_search')
YEARS=tuple(f'return_{year}' for year in range(2022,2026))


from ..portfolio import combine_targets, smooth_targets

def nominal_cashflow_share(model):
    if 'factor_weights' in model:return model['factor_weights'].get('cashflow_yield',0)
    return sum(weight*nominal_cashflow_share(model['component_models'][key])
               for key,weight in model['component_allocations'].items())




def add_stability(table,benchmark):
    frame=table.copy()
    frame['annual_mean']=frame[list(YEARS)].mean(axis=1)
    frame['annual_std']=frame[list(YEARS)].std(axis=1,ddof=1)
    frame['annual_range']=frame[list(YEARS)].max(axis=1)-frame[list(YEARS)].min(axis=1)
    frame['stability_goal_met']=(frame.sharpe.ge(1.58)&frame.min_full_year_return.ge(benchmark['minimum'])
        &frame.annual_std.le(benchmark['std'])&frame.return_2026.gt(0)&frame.stale_price_legs.eq(0))
    return frame




def component_candidates():
    table=pd.read_csv(PRIOR/'all_results.csv')
    parameters=json.loads((PRIOR/'parameters.json').read_text())
    neutral=table.variant.map(lambda name:parameters[name]['settings'].get('beta_neutral',0)==1)
    table=table.loc[table.stale_price_legs.eq(0)&neutral&table.min_full_year_return.gt(-.10)].copy()
    table['annual_std']=table[list(YEARS)].std(axis=1,ddof=1)
    table['annual_mean']=table[list(YEARS)].mean(axis=1)
    picks=[]
    def take(frame,count):
        picks.extend(frame.drop_duplicates('factors').head(count).variant.tolist())
    take(table.sort_values('sharpe',ascending=False),10)
    take(table.sort_values('min_full_year_return',ascending=False),12)
    take(table.loc[table.min_full_year_return.gt(.025)&table.annual_mean.gt(.04)].sort_values('annual_std'),8)
    for year in YEARS:
        take(table.sort_values(year,ascending=False),7)
    names=list(dict.fromkeys(picks))
    # Keep the previous selected and a low-dispersion cashflow/earnings model.
    names=list(dict.fromkeys(names+['param_010825','param_005911','param_002968']))
    return table.set_index('variant').loc[names].reset_index(),parameters


class StabilitySearch(ParameterSearch):
    def __init__(self,output_dir=OUTPUT):
        super().__init__(output_dir,rebalance_frequency='monthly',extra_path=prepare_inputs())
        self.reference_beta=self.beta.copy()
        annual=pd.read_csv('result/aqr_quarterly_sharpe_1_5939/verified_side_attribution.csv')
        values=annual.loc[annual.year.lt(2026),'net_pnl_contribution']
        self.benchmark=dict(minimum=float(values.min()),std=float(values.std(ddof=1)),mean=float(values.mean()))
        self.components={};self.component_models={};self.ensemble_specs={};self.ensemble_count=0
        manifest=json.loads((self.out/'manifest.json').read_text())
        manifest.update(objective='At least quarterly minimum full-year return, no greater complete-year standard deviation, net Sharpe >=1.58, positive partial 2026.',
                        annual_comparison_years=[2022,2023,2024,2025],partial_2026_compared_as_full_year=False,
                        quarter_benchmark=self.benchmark,ensemble_rule='Fixed component allocations; monthly target netting then gross normalization to 1; actual executable ledger, no annual switching.')
        (self.out/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')

    def prepare_components(self):
        chosen,parameters=component_candidates()
        original=json.loads((PRIOR/'base_specifications.json').read_text())
        names=chosen.variant.tolist()
        bases=list(dict.fromkeys(parameters[name]['base_name'] for name in names))
        self.add_bases([original[name] for name in bases])
        for name in names:
            param=parameters[name]
            self.components[name]=transform_targets(self.bases[param['base_name']],self.groups,self.volatility,self.beta,**param['settings'])
            self.component_models[name]=dict(factor_weights=dict(zip(original[param['base_name']]['factors'],original[param['base_name']]['weights'])),construction=param['settings'])
        chosen.to_csv(self.out/'component_source_metrics.csv',index=False)
        (self.out/'component_models.json').write_text(json.dumps(self.component_models,indent=2),encoding='utf-8')
        np.savez_compressed(self.out/'component_targets.npz',**self.components)
        self.bases.clear()
        self.evaluate_ensembles([{name:1.} for name in names],'component_controls')
        controls=pd.concat(self.results,ignore_index=True).copy()
        for row in controls.itertuples():
            source=next(iter(self.ensemble_specs[self.parameters[row.variant]['base_name']]))
            expected=chosen.set_index('variant').loc[source]
            assert abs(expected.sharpe-row.sharpe)<1e-8,(source,expected.sharpe,row.sharpe)
            assert abs(expected.net_return-row.net_return)<1e-8
        self.component_controls=controls
        return names

    def blend(self,weights):
        names=list(weights)
        return combine_targets(np.array([self.components[name] for name in names]),list(weights.values()))

    def evaluate_ensembles(self,weights_list,stage):
        for offset in range(0,len(weights_list),48):
            cases=[];temporary=[]
            for weights in weights_list[offset:offset+48]:
                name=f'fixed_blend_{self.ensemble_count:06d}';self.ensemble_count+=1
                weights={k:float(v) for k,v in weights.items() if v>1e-8}
                total=sum(weights.values());weights={k:v/total for k,v in weights.items()}
                self.ensemble_specs[name]=weights
                try:
                    self.bases[name]=self.blend(weights)
                except ValueError:
                    continue
                self.base_specs[name]=dict(variant=name,factors=tuple(weights),weights=tuple(weights.values()),
                    cashflow_weight=sum(v*nominal_cashflow_share(self.component_models[k]) for k,v in weights.items()),
                    factor_count=len(weights))
                cases.append((name,dict(beta_neutral=1.)))
                temporary.append(name)
            super().evaluate(cases,stage)
            for name in temporary:self.bases.pop(name,None)
            (self.out/'ensemble_specifications.json').write_text(json.dumps(self.ensemble_specs,indent=2),encoding='utf-8')
            table=add_stability(pd.concat(self.results,ignore_index=True),self.benchmark)
            table.to_csv(self.out/'stability_results.csv',index=False)
            stable=table.loc[table.stability_goal_met].sort_values(['annual_std','sharpe'],ascending=[True,False])
            stable.to_csv(self.out/'stability_goal_candidates.csv',index=False)
            self.progress(f'{stage}: {min(offset+48,len(weights_list))}/{len(weights_list)}, quarterly-stability+Sharpe qualifiers {len(stable)}')

    def targets_for(self,name):
        param=self.parameters[name]
        base=self.blend(self.ensemble_specs[param['base_name']])
        return transform_targets(base,self.groups,self.volatility,self.beta,**param['settings'])

    def verify_selected(self,names):
        for name in names:
            base_name=self.parameters[name]['base_name']
            self.bases[base_name]=self.blend(self.ensemble_specs[base_name])
        return super().verify(names)

    def save_selected(self,name):
        param=self.parameters[name]
        weights=self.ensemble_specs[param['base_name']]
        metrics=pd.read_csv(self.out/'stability_results.csv').set_index('variant').loc[name].to_dict()
        value=dict(variant=name,rebalance_frequency='monthly',construction='fixed_monthly_target_blend',
                   component_allocations=weights,component_models={key:self.component_models[key] for key in weights},
                   netting='Combine component targets each month then normalize account gross to 1; projection only removes numerical beta residual.',
                   backtest=asdict(self.config),quarter_benchmark=self.benchmark,metrics=metrics,
                   interpretation='In-sample selection on previously inspected 2022-2026 history, not independent future validation.',
                   assumptions='Existing date references or period-end +90 days; latest vendor versions; no dividends, short dividend compensation, cash interest or taxes.')
        value['final_risk_lookback']=getattr(self,'risk_lookback',252)
        value['annual_dispersion_at_most_quarter']=bool(metrics['annual_std']<=self.benchmark['std'])
        value['worst_complete_year_at_least_quarter']=bool(metrics['min_full_year_return']>=self.benchmark['minimum'])
        value['sharpe_goal_met']=bool(metrics['sharpe']>=1.58)
        value['selection_policy']='Prefer lower complete-year dispersion among clean positive-year portfolios with net Sharpe >=1.58; the stricter quarter minimum-year floor remains a separate diagnostic.'
        (self.out/'selected_candidate.json').write_text(json.dumps(value,indent=2),encoding='utf-8')
        return value


def propose_optimized_blends(search,names):
    """In-sample proxies propose fixed weights; executable trials decide results."""
    control=search.component_controls
    by_component={next(iter(search.ensemble_specs[search.parameters[row.variant]['base_name']])):row.variant for row in control.itertuples()}
    nav=np.array([search.navs[by_component[name]] for name in names]).T
    returns=nav/np.vstack([np.full((1,len(names)),search.config.initial_nav),nav[:-1]])-1
    annual=np.array([control.set_index('variant').loc[by_component[name],list(YEARS)].to_numpy(dtype=float) for name in names])
    mean=returns.mean(axis=0);cov=np.cov(returns,rowvar=False)
    proposals=[];records=[]
    for cap in (.25,.4,.6):
        for floor in (.025,.04,.05,.06,.07,.08):
            for spread in (.005,.015,.03,.06):
                matrix=np.vstack([np.ones(len(names)),annual.T])
                low=np.r_[1,np.full(4,floor)];high=np.r_[1,np.full(4,floor+spread)]
                feasible=linprog(np.zeros(len(names)),A_ub=np.r_[annual.T,-annual.T],
                    b_ub=np.r_[np.full(4,floor+spread),np.full(4,-floor)],A_eq=np.ones((1,len(names))),
                    b_eq=[1],bounds=[(0,cap)]*len(names),method='highs')
                if not feasible.success:
                    records.append(dict(cap=cap,floor=floor,spread=spread,status='proxy_constraints_infeasible'))
                    continue
                def objective(w):
                    sigma=np.sqrt(max(w@cov@w,1e-16))
                    return -(mean@w)/sigma*np.sqrt(252)
                def gradient(w):
                    sigma=np.sqrt(max(w@cov@w,1e-16))
                    return -np.sqrt(252)*(mean/sigma-(mean@w)*(cov@w)/sigma**3)
                result=minimize(objective,feasible.x,jac=gradient,method='SLSQP',bounds=Bounds(0,cap),
                    constraints=[LinearConstraint(np.ones((1,len(names))),1,1),
                                 LinearConstraint(annual.T,low[1:],high[1:])],options=dict(maxiter=200,ftol=1e-10))
                w=result.x if result.success else feasible.x
                w=np.maximum(w,0);w/=w.sum()
                weights={name:float(v) for name,v in zip(names,w) if v>1e-7}
                proposals.append(weights)
                records.append(dict(cap=cap,floor=floor,spread=spread,status=result.message,proxy_sharpe=-objective(w),
                                    weights=json.dumps(weights),proxy_min_year=float((annual.T@w).min()),proxy_year_std=float((annual.T@w).std(ddof=1))))
    pd.DataFrame(records).to_csv(search.out/'proxy_proposals.csv',index=False)
    return proposals


def select_for_verification(search):
    table=add_stability(pd.concat(search.results,ignore_index=True),search.benchmark)
    reliable=table.loc[table.stale_price_legs.eq(0)&table.return_2026.gt(0)]
    goals=reliable.loc[reliable.stability_goal_met].sort_values(['annual_std','sharpe'],ascending=[True,False])
    if len(goals):
        return goals.variant.head(3).tolist()
    positive=reliable.loc[reliable.min_full_year_return.gt(0)]
    consistent=positive.loc[positive.sharpe.ge(1.58)&positive.min_full_year_return.ge(.05)].sort_values(['annual_std','min_full_year_return'],ascending=[True,False])
    if len(consistent):return consistent.variant.head(3).tolist()
    strong=positive.loc[positive.sharpe.ge(1.58)].sort_values(['annual_std','min_full_year_return'],ascending=[True,False])
    fallback=positive.sort_values(['min_full_year_return','sharpe'],ascending=False)
    return list(dict.fromkeys(strong.variant.head(2).tolist()+fallback.variant.head(1).tolist()))


def run():
    search=StabilitySearch()
    names=search.prepare_components()
    proposals=propose_optimized_blends(search,names)
    search.evaluate_ensembles(proposals,'optimized_fixed_allocations')
    # Pair controls expose how much improvement requires a complex mixture.
    pairs=[]
    for first,second in combinations(names,2):
        for weight in (.25,.5,.75):pairs.append({first:weight,second:1-weight})
    search.evaluate_ensembles(pairs,'fixed_pair_blends')
    selected=select_for_verification(search)
    search.verify_selected(selected)
    value=search.save_selected(selected[0])
    print('SELECTED\n'+json.dumps(value,indent=2),flush=True)
    return search


def reconstruct_component(search,model,cache):
    signature=hashlib.sha256(json.dumps(model,sort_keys=True).encode()).hexdigest()
    if signature in cache:return cache[signature]
    if 'component_allocations' in model:
        names=list(model['component_allocations'])
        targets=combine_targets(np.array([reconstruct_component(search,model['component_models'][name],cache) for name in names]),
                                list(model['component_allocations'].values()))
        targets=transform_targets(targets,search.groups,search.volatility,search.reference_beta,beta_neutral=1.)
    else:
        name='replay_leaf_'+signature[:16]
        factors=tuple(model['factor_weights']);weights=tuple(model['factor_weights'].values())
        search.add_bases([dict(variant=name,factors=factors,weights=weights,cashflow_weight=weights[0],factor_count=len(factors))])
        targets=transform_targets(search.bases[name],search.groups,search.volatility,search.reference_beta,**model['construction'])
        if 'smoothing' in model:
            targets=smooth_targets(targets,abs(search.bases[name])>1e-12,search.groups,search.reference_beta,**model['smoothing'])
    if 'risk_overlay' in model:
        risk=dated_risk_betas(search)[model['risk_overlay']['lookback']]
        targets=transform_targets(targets,search.groups,search.volatility,risk,beta_neutral=1.)
    cache[signature]=targets
    return targets


def replay(output_dir=OUTPUT,preset_file=None):
    if preset_file is not None and Path(preset_file).suffix == '.py':
        from ..run import replay_file
        return replay_file(str(preset_file))
    saved=json.loads(Path(preset_file or output_dir/'selected_candidate.json').read_text())
    search=StabilitySearch(output_dir/'candidate_replay')
    if asdict(search.config)!=saved['backtest']:
        raise ValueError('The local backtest date/NAV/cost settings differ from the saved candidate')
    search.component_models=saved['component_models']
    cache={}
    for component,model in search.component_models.items():
        search.components[component]=reconstruct_component(search,model,cache)
    if saved.get('final_risk_lookback',252)!=252:
        search.risk_lookback=saved['final_risk_lookback']
        search.beta=dated_risk_betas(search)[search.risk_lookback]
    search.evaluate_ensembles([saved['component_allocations']],'replay')
    name=next(iter(search.parameters))
    search.verify_selected([name])
    value=search.save_selected(name)
    print(json.dumps(value['metrics'],indent=2),flush=True)
    return search


def resume_search(output_dir=OUTPUT):
    search=StabilitySearch(output_dir)
    search.results=[pd.read_csv(output_dir/'all_results.csv')]
    search.parameters=json.loads((output_dir/'parameters.json').read_text())
    search.ensemble_specs=json.loads((output_dir/'ensemble_specifications.json').read_text())
    search.component_models=json.loads((output_dir/'component_models.json').read_text())
    with np.load(output_dir/'component_targets.npz') as bank:
        search.components={key:bank[key] for key in bank.files}
    search.trial_count=1+max(int(name.split('_')[-1]) for name in search.parameters)
    search.ensemble_count=1+max(int(name.split('_')[-1]) for name in search.ensemble_specs)
    for name,weights in search.ensemble_specs.items():
        search.base_specs[name]=dict(variant=name,factors=tuple(weights),weights=tuple(weights.values()),
            cashflow_weight=sum(value*nominal_cashflow_share(search.component_models[key]) for key,value in weights.items()),
            factor_count=len(weights))
    return search


def reselect(output_dir):
    search=resume_search(output_dir)
    selected=select_for_verification(search)
    for name in selected:
        targets=search.targets_for(name)
        search.navs[name]=batch_backtest(targets[None],search.decisions,search.market,search.config)['equity'][:,0]
    search.verify_selected(selected)
    value=search.save_selected(selected[0])
    print(json.dumps(value['metrics'],indent=2),flush=True)
    return search


def refine_allocations(search=None):
    """Fit actual netted-target backtests around balanced fixed allocations.

    A seeded Dirichlet neighborhood changes only one allocation vector used
    throughout the sample. Selection never changes allocations by calendar year.
    """
    if search is None:search=resume_search()
    rng=np.random.default_rng(20261026)
    for generation in range(3):
        table=add_stability(pd.concat(search.results,ignore_index=True),search.benchmark)
        valid=table.loc[table.stale_price_legs.eq(0)&table.return_2026.gt(0)&table.min_full_year_return.gt(.04)].copy()
        valid['shortfall']=(1.58-valid.sharpe).clip(lower=0)*.25
        valid['shortfall']+=(search.benchmark['minimum']-valid.min_full_year_return).clip(lower=0)*10
        valid['shortfall']+=(valid.annual_std-search.benchmark['std']).clip(lower=0)*10
        leaders=valid.sort_values(['shortfall','annual_std','sharpe'],ascending=[True,True,False]).head(6)
        proposals=[]
        for row in leaders.itertuples():
            weights=search.ensemble_specs[search.parameters[row.variant]['base_name']]
            names=list(weights);center=np.array(list(weights.values()))
            if len(names)==1:continue
            for concentration in (60.,200.,800.):
                for sample in rng.dirichlet(np.maximum(center*concentration,.1),size=32):
                    if sample.max()>.65:continue
                    proposals.append(dict(zip(names,sample)))
            for denominator in (20,40,80):
                rounded=np.rint(center*denominator)
                if rounded.sum()>0:
                    rounded=rounded/rounded.sum()
                    proposals.append(dict(zip(names,rounded)))
                    if denominator==40:
                        for first,second in combinations(range(len(names)),2):
                            for delta in (-.075,-.05,-.025,.025,.05,.075):
                                transferred=rounded.copy()
                                transferred[first]+=delta;transferred[second]-=delta
                                if transferred.min()>=0 and transferred.max()<=.65:
                                    proposals.append(dict(zip(names,transferred)))
        search.evaluate_ensembles(proposals,f'actual_allocation_refinement_{generation}')
        table=add_stability(pd.concat(search.results,ignore_index=True),search.benchmark)
        if table.stability_goal_met.any():break
    selected=select_for_verification(search)
    for name in selected:
        if name not in search.navs:
            weights=search.targets_for(name)
            search.navs[name]=batch_backtest(weights[None],search.decisions,search.market,search.config)['equity'][:,0]
    search.verify_selected(selected)
    value=search.save_selected(selected[0])
    print('SELECTED\n'+json.dumps(value,indent=2),flush=True)
    return search


def run_smoothed():
    """Monthly screens/trades with explicitly lagged target-signal smoothing."""
    search=StabilitySearch(SMOOTHED_OUTPUT)
    originals=json.loads((OUTPUT/'component_models.json').read_text())
    specs=[]
    for name,model in originals.items():
        factors=tuple(model['factor_weights']);weights=tuple(model['factor_weights'].values())
        specs.append(dict(variant=name,factors=factors,weights=weights,cashflow_weight=weights[0],factor_count=len(factors)))
    search.add_bases(specs)
    with np.load(OUTPUT/'component_targets.npz') as targets:
        for name,model in originals.items():
            eligible=abs(search.bases[name])>1e-12
            for smoothing in (dict(window=2),dict(window=3),dict(window=6),dict(decay=.5),dict(decay=.75)):
                suffix='_'.join(f'{key}_{value}' for key,value in smoothing.items())
                key=name+'_'+suffix
                search.components[key]=smooth_targets(targets[name],eligible,search.groups,search.beta,**smoothing)
                search.component_models[key]={**model,'smoothing':smoothing}
    search.bases.clear()
    (search.out/'component_models.json').write_text(json.dumps(search.component_models,indent=2),encoding='utf-8')
    np.savez_compressed(search.out/'component_targets.npz',**search.components)
    search.evaluate_ensembles([{name:1.} for name in search.components],'smoothed_component_controls')
    controls=pd.concat(search.results,ignore_index=True)
    search.component_controls=controls
    reliable=add_stability(controls.loc[controls.stale_price_legs.eq(0)],search.benchmark)
    candidates=[]
    def take(frame,count):
        # Keep diverse original model/smoothing choices rather than copies.
        candidates.extend(frame.head(count).variant.tolist())
    take(reliable.sort_values('sharpe',ascending=False),15)
    take(reliable.sort_values('min_full_year_return',ascending=False),15)
    take(reliable.loc[reliable.min_full_year_return.gt(.025)].sort_values('annual_std'),10)
    for year in YEARS:take(reliable.sort_values(year,ascending=False),8)
    names=[next(iter(search.ensemble_specs[search.parameters[name]['base_name']])) for name in dict.fromkeys(candidates)]
    proposals=propose_optimized_blends(search,names)
    search.evaluate_ensembles(proposals,'smoothed_fixed_allocations')
    return refine_allocations(search)


def run_combined():
    """Mix complementary unsmoothed/smoothed fixed models, preserving both levels."""
    search=StabilitySearch(COMBINED_OUTPUT)
    source_rows=[];families={}
    for family,folder in [('raw',OUTPUT),('smoothed',SMOOTHED_OUTPUT)]:
        table=pd.read_csv(folder/'stability_results.csv')
        table=table.loc[table.stale_price_legs.eq(0)&table.return_2026.gt(0)]
        stable=table.loc[table.min_full_year_return.ge(search.benchmark['minimum'])
                         &table.annual_std.le(search.benchmark['std'])].sort_values('sharpe',ascending=False).head(8)
        strong=table.loc[table.sharpe.ge(1.58)&table.min_full_year_return.gt(.04)].sort_values('annual_std').head(8)
        chosen=pd.concat([stable,strong]).drop_duplicates('variant')
        models=json.loads((folder/'component_models.json').read_text())
        ensembles=json.loads((folder/'ensemble_specifications.json').read_text())
        parameters=json.loads((folder/'parameters.json').read_text())
        with np.load(folder/'component_targets.npz') as bank:
            for row in chosen.itertuples():
                weights=ensembles[parameters[row.variant]['base_name']]
                key=family+'_'+row.variant
                targets=combine_targets(np.array([bank[name] for name in weights]),list(weights.values()))
                search.components[key]=transform_targets(targets,search.groups,search.volatility,search.beta,beta_neutral=1.)
                search.component_models[key]=dict(component_allocations=weights,component_models={name:models[name] for name in weights})
                families[key]=family
                source_rows.append(dict(component=key,source_folder=str(folder),source_variant=row.variant,
                                        source_sharpe=row.sharpe,source_return=row.net_return))
    (search.out/'component_models.json').write_text(json.dumps(search.component_models,indent=2),encoding='utf-8')
    pd.DataFrame(source_rows).to_csv(search.out/'component_sources.csv',index=False)
    np.savez_compressed(search.out/'component_targets.npz',**search.components)
    search.evaluate_ensembles([{name:1.} for name in search.components],'combined_parent_controls')
    controls=pd.concat(search.results,ignore_index=True)
    search.component_controls=controls
    expected=pd.DataFrame(source_rows).set_index('component')
    for row in controls.itertuples():
        key=next(iter(search.ensemble_specs[search.parameters[row.variant]['base_name']]))
        assert abs(row.sharpe-expected.loc[key,'source_sharpe'])<1e-8
        assert abs(row.net_return-expected.loc[key,'source_return'])<1e-8
    proposals=propose_optimized_blends(search,list(search.components))
    search.evaluate_ensembles(proposals,'combined_optimized_allocations')
    pairs=[]
    for first,second in combinations(search.components,2):
        if families[first]==families[second]:continue
        for weight in np.arange(.1,1.,.1):pairs.append({first:float(weight),second:float(1-weight)})
    search.evaluate_ensembles(pairs,'cross_family_fixed_pairs')
    table=add_stability(pd.concat(search.results,ignore_index=True),search.benchmark)
    if not table.stability_goal_met.any():return refine_allocations(search)
    selected=select_for_verification(search)
    search.verify_selected(selected)
    value=search.save_selected(selected[0])
    print('SELECTED\n'+json.dumps(value['metrics'],indent=2),flush=True)
    return search


def dated_risk_betas(search):
    """Observed-date betas from past cached split-consistent daily closes only."""
    if hasattr(search,'risk_beta_bank'):return search.risk_beta_bank
    path=RISK_OUTPUT/'dated_betas.csv.gz'
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        data=pd.read_csv(path,parse_dates=['as_of'])
    else:
        histories=WeightedBacktester.cached_histories(search.base.database,search.tickers,'2020-12-01','2026-10-23')
        calendar=histories['SPY'].index
        prices={}
        for name,frame in histories.items():
            if frame.empty:continue
            ratios=frame['Stock Splits'].where(frame['Stock Splits'].gt(0),1.)
            future=ratios.iloc[::-1].cumprod().iloc[::-1]/ratios
            prices[name]=(frame.Close/future).reindex(calendar)
        returns=pd.DataFrame(prices,index=calendar).pct_change(fill_method=None)
        rows=[]
        for date in search.decisions:
            part=pd.DataFrame({'ticker':search.tickers});part['as_of']=date
            for lookback,minimum in ((63,45),(126,100),(252,200)):
                window=returns.loc[:date].tail(lookback)
                window=window.loc[window.SPY.notna()]
                market=window.SPY;present=window.notna();count=present.sum()
                market_mean=present.mul(market,axis=0).sum()/count
                numerator=window.mul(market,axis=0).sum()-window.sum()*market_mean
                denominator=present.mul(market.pow(2),axis=0).sum()-count*market_mean.pow(2)
                beta=(numerator/denominator).where(count.ge(minimum)&denominator.gt(0))
                part[f'beta_{lookback}']=part.ticker.map(beta)
            rows.append(part)
        data=pd.concat(rows,ignore_index=True);data.to_csv(path,index=False)
    bank={}
    for lookback in (63,126,252):
        bank[lookback]=data.pivot(index='as_of',columns='ticker',values=f'beta_{lookback}').reindex(index=search.decisions,columns=search.tickers).to_numpy()
    stocks=np.array(search.tickers)!='SPY'  # The benchmark has no stock-score input row.
    assert np.allclose(bank[252][:,stocks],search.reference_beta[:,stocks],equal_nan=True,rtol=1e-9,atol=1e-10)
    search.risk_beta_bank=bank
    return bank


def run_risk(lookback=126):
    """Check a fixed shorter risk window without changing financial inputs/fees."""
    search=StabilitySearch(RISK_OUTPUT/str(lookback))
    risk=dated_risk_betas(search)[lookback]
    for family,folder in [('raw',OUTPUT),('smoothed',SMOOTHED_OUTPUT)]:
        table=pd.read_csv(folder/'stability_results.csv')
        table=table.loc[table.stale_price_legs.eq(0)&table.return_2026.gt(0)]
        stable=table.loc[table.min_full_year_return.ge(search.benchmark['minimum'])&table.annual_std.le(search.benchmark['std'])].sort_values('sharpe',ascending=False).head(6)
        strong=table.loc[table.sharpe.ge(1.58)&table.min_full_year_return.gt(.04)].sort_values('annual_std').head(6)
        chosen=pd.concat([stable,strong]).drop_duplicates('variant')
        models=json.loads((folder/'component_models.json').read_text())
        ensembles=json.loads((folder/'ensemble_specifications.json').read_text())
        parameters=json.loads((folder/'parameters.json').read_text())
        with np.load(folder/'component_targets.npz') as bank:
            for row in chosen.itertuples():
                weights=ensembles[parameters[row.variant]['base_name']]
                key=f'{family}_{row.variant}_risk_{lookback}'
                targets=combine_targets(np.array([bank[name] for name in weights]),list(weights.values()))
                targets=transform_targets(targets,search.groups,search.volatility,search.reference_beta,beta_neutral=1.)
                search.components[key]=transform_targets(targets,search.groups,search.volatility,risk,beta_neutral=1.)
                search.component_models[key]=dict(component_allocations=weights,component_models={name:models[name] for name in weights},risk_overlay=dict(lookback=lookback))
    search.risk_lookback=lookback;search.beta=risk
    (search.out/'component_models.json').write_text(json.dumps(search.component_models,indent=2),encoding='utf-8')
    np.savez_compressed(search.out/'component_targets.npz',**search.components)
    search.evaluate_ensembles([{name:1.} for name in search.components],'shorter_risk_window_controls')
    search.component_controls=pd.concat(search.results,ignore_index=True)
    proposals=propose_optimized_blends(search,list(search.components))
    search.evaluate_ensembles(proposals,'shorter_risk_fixed_allocations')
    table=add_stability(pd.concat(search.results,ignore_index=True),search.benchmark)
    if not table.stability_goal_met.any():return refine_allocations(search)
    selected=select_for_verification(search)
    search.verify_selected(selected)
    value=search.save_selected(selected[0])
    print('SELECTED\n'+json.dumps(value['metrics'],indent=2),flush=True)
    return search


if __name__=='__main__':
    import sys
    if '--candidate' in sys.argv:replay()
    elif '--refine' in sys.argv:refine_allocations()
    elif '--smooth' in sys.argv:run_smoothed()
    elif '--smoothed-candidate' in sys.argv:replay(SMOOTHED_OUTPUT)
    elif '--combine' in sys.argv:run_combined()
    elif '--combined-candidate' in sys.argv:replay(COMBINED_OUTPUT)
    elif '--select-combined' in sys.argv:reselect(COMBINED_OUTPUT)
    elif '--risk126' in sys.argv:run_risk(126)
    elif '--risk63' in sys.argv:run_risk(63)
    elif '--risk126-candidate' in sys.argv:replay(RISK_OUTPUT/'126')
    elif '--risk63-candidate' in sys.argv:replay(RISK_OUTPUT/'63')
    elif '--stable-candidate' in sys.argv:
        from ..settings import AQRSettings
        preset=Path(AQRSettings.stable_monthly_candidate_file)
        replay(preset.parent,preset_file=preset)
    else:run()
