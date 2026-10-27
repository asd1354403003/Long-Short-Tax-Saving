"""Monthly cashflow strategy research, entirely from the existing local cache.

Recompute fundamentals, momentum and risk at each month end. Do not forward-fill
quarter-end scores, change source facts, or fetch source data. Parameter tuning
uses the already inspected sample and is explicitly not independent validation.
"""
from __future__ import annotations

from dataclasses import asdict
from itertools import product
from pathlib import Path
from shutil import copy2
import hashlib
import json

import numpy as np
import pandas as pd

from .cashflow_parameter_search import ParameterSearch
from .cashflow_combinations import CORE, FACTOR_NAMES, batch_backtest, specifications
from .multifactor_experiment import quarterly_inputs, price_inputs
from ..signals import FactorSignals, SignalConfig
from methods.aqr_factors.backtest import WeightedBacktester


OUTPUT=Path('methods/aqr_factors/research/results/monthly_cashflow_parameter_search')
INPUTS=Path('methods/aqr_factors/research/results/monthly_cashflow_inputs')
SOURCE=Path('methods/aqr_factors/research/results/factor_comparison')


def prepare_inputs(progress=print):
    destination=Path('aqr_research/data/backtest_inputs/monthly_extra_inputs.csv.gz')
    if not destination.exists():
        raise FileNotFoundError('Missing fixed monthly input snapshot')
    progress('Using fixed monthly factor inputs; no downloads')
    return destination


def save_selected(search,name):
    base=search.base_specs[search.parameters[name]['base_name']]
    summary=pd.read_csv(search.out/'verified_comparison.csv').set_index('variant').loc[name]
    candidate=dict(variant=name,rebalance_frequency='monthly',
                   factor_weights=dict(zip(base['factors'],base['weights'])),
                   construction=search.parameters[name]['settings'],
                   interpretation='Selected on 2022-2026 history; in-sample, not independent future validation.')
    selected=dict(candidate=candidate,backtest=asdict(search.config),goal_sharpe=1.58,
                  net_sharpe=float(summary.sharpe),net_return=float(summary.net_return),
                  max_drawdown=float(summary.max_drawdown),realized_beta=float(summary.beta),
                  parameter_trials=len(pd.concat(search.results)) if search.results else 1,
                  monthly_decisions=len(search.decisions),source_requests=0,
                  full_ledger_verified=True,missing_held_price_marks=int(summary.stale_price_legs),
                  data_limitations='Latest vendor versions and missing-publication-date +90-day assumptions; no dividends, short dividend compensation, cash interest or taxes.')
    checks=json.loads((search.out/'verification.json').read_text())[name]
    selected.update(goal_met_in_sample=checks['goal_met'],all_years_positive=checks['positive_all_years'])
    (search.out/'selected_candidate.json').write_text(json.dumps(selected,indent=2),encoding='utf-8')
    return selected


def replay_candidate():
    from ..run import replay_file
    from ..settings import AQRSettings
    return replay_file(AQRSettings.selected_parameter_file)


def run():
    from .historical_rf_search import Search, WORK
    search=Search()
    if (WORK/'targets.npz').exists():search.resume()
    else:search.seed_controls()
    search.factor_combinations()
    search.finish_checkpoint()
    return search


def extend_monthly_factors():
    """Test small factor combinations for monthly trading, retaining all trials.

    Quarter-selected factor seeds can miss the best monthly combination. Use
    the same 1,210 pre-existing cashflow-anchored combinations, with the three
    monthly-leading construction rules. No annual parameter switching.
    """
    search=ParameterSearch(OUTPUT,rebalance_frequency='monthly',extra_path=prepare_inputs())
    prior=pd.read_csv(OUTPUT/'all_results.csv')
    search.results=[prior]
    search.parameters=json.loads((OUTPUT/'parameters.json').read_text())
    search.trial_count=1+max(int(name.split('_')[-1]) for name in search.parameters)
    old_specs=json.loads((OUTPUT/'base_specifications.json').read_text())
    specs=specifications()
    search.add_bases(list(old_specs.values())+[s for s in specs if s['variant'] not in old_specs])
    already={(p['base_name'],json.dumps(p['settings'],sort_keys=True)) for p in search.parameters.values()}
    cases=[]
    for spec in specs:
        for power in (0.,1.,2.):
            settings=dict(tail=.05,power=power,inverse_vol=0.,sector_budget='signal',beta_neutral=1.)
            if (spec['variant'],json.dumps(settings,sort_keys=True)) not in already:
                cases.append((spec['variant'],settings))
    search.evaluate(cases,'monthly_factor_combinations')
    total=pd.concat(search.results,ignore_index=True)
    reliable=total.loc[total.stale_price_legs.eq(0)].sort_values('sharpe',ascending=False)
    eligible=reliable.loc[reliable.min_including_partial_2026.gt(0)]
    names=(eligible if len(eligible) else reliable).variant.head(3).tolist()
    for name in names:
        if name not in search.navs:
            targets=search.targets_for(name)
            search.navs[name]=batch_backtest(targets[None],search.decisions,search.market,search.config)['equity'][:,0]
    search.verify(names)
    save_selected(search,names[0])
    (search.out/'base_specifications.json').write_text(json.dumps(search.base_specs,indent=2),encoding='utf-8')
    return search


def refine_monthly_weights():
    """Refine leading monthly sets and the quarter winner with fixed weights."""
    search=ParameterSearch(OUTPUT,rebalance_frequency='monthly',extra_path=prepare_inputs())
    prior=pd.read_csv(OUTPUT/'all_results.csv')
    search.results=[prior]
    search.parameters=json.loads((OUTPUT/'parameters.json').read_text())
    search.trial_count=1+max(int(name.split('_')[-1]) for name in search.parameters)
    old_specs=json.loads((OUTPUT/'base_specifications.json').read_text())
    search.base_specs.update(old_specs)
    reliable=prior.loc[prior.stale_price_legs.eq(0) & prior.min_including_partial_2026.gt(0)].sort_values('sharpe',ascending=False)
    seeds=pd.concat([reliable.drop_duplicates('factors').head(3),
                     reliable.loc[reliable.base_name.eq('cf_0692')].head(1)]).drop_duplicates('factors')
    specs=[]
    for row in seeds.itertuples():
        factors=tuple(old_specs[row.base_name]['factors'])
        for others in product(range(1,10),repeat=len(factors)-1):
            core=10-sum(others)
            if core<2:
                continue
            weights=tuple(number/10 for number in (core,*others))
            specs.append(dict(variant=f'monthly_weights_{len(specs):04d}',factors=factors,weights=weights,
                              cashflow_weight=weights[0],factor_count=len(factors)))
    search.add_bases(specs)
    cases=[]
    for spec in specs:
        for tail,power in product((.02,.035,.05,.075,.1),(0.,1.,2.)):
            cases.append((spec['variant'],dict(tail=tail,power=power,inverse_vol=0.,sector_budget='signal',beta_neutral=1.)))
    search.evaluate(cases,'monthly_weight_tail_refinement')
    total=pd.concat(search.results,ignore_index=True)
    reliable=total.loc[total.stale_price_legs.eq(0)].sort_values('sharpe',ascending=False)
    eligible=reliable.loc[reliable.min_including_partial_2026.gt(0)]
    names=(eligible if len(eligible) else reliable).variant.head(3).tolist()
    for name in names:
        param=search.parameters[name]
        if param['base_name'] not in search.bases:
            search.add_bases([old_specs[param['base_name']]])
        if name not in search.navs:
            targets=search.targets_for(name)
            search.navs[name]=batch_backtest(targets[None],search.decisions,search.market,search.config)['equity'][:,0]
    search.verify(names)
    save_selected(search,names[0])
    (search.out/'base_specifications.json').write_text(json.dumps(search.base_specs,indent=2),encoding='utf-8')
    return search


def polish_monthly_weights(*,denominator=40,radius=4,tails=(.04,.045,.05,.055,.06),
                           stage='monthly_local_weight_grid',prefix='monthly_local'):
    """Record a local constant-weight grid around the best monthly factor set."""
    search=ParameterSearch(OUTPUT,rebalance_frequency='monthly',extra_path=prepare_inputs())
    prior=pd.read_csv(OUTPUT/'all_results.csv')
    search.results=[prior]
    search.parameters=json.loads((OUTPUT/'parameters.json').read_text())
    search.trial_count=1+max(int(name.split('_')[-1]) for name in search.parameters)
    old_specs=json.loads((OUTPUT/'base_specifications.json').read_text())
    search.base_specs.update(old_specs)
    best=prior.loc[prior.stale_price_legs.eq(0) & prior.min_including_partial_2026.gt(0)].sort_values('sharpe',ascending=False).iloc[0]
    source=old_specs[best.base_name]
    factors=tuple(source['factors'])
    center=np.rint(np.array(source['weights'])*denominator).astype(int)
    assert center.sum()==denominator
    specs=[]
    ranges=[range(max(1,int(value)-radius),min(denominator,int(value)+radius)+1) for value in center[1:]]
    for others in product(*ranges):
        core=denominator-sum(others)
        if core<1 or abs(core-center[0])>radius:
            continue
        weights=tuple(value/denominator for value in (core,*others))
        specs.append(dict(variant=f'{prefix}_{search.trial_count}_{len(specs):04d}',factors=factors,weights=weights,
                          cashflow_weight=weights[0],factor_count=len(factors)))
    search.add_bases(specs)
    cases=[(spec['variant'],dict(tail=tail,power=0.,inverse_vol=0.,sector_budget='signal',beta_neutral=1.))
           for spec in specs for tail in tails]
    search.evaluate(cases,stage)
    total=pd.concat(search.results,ignore_index=True)
    reliable=total.loc[total.stale_price_legs.eq(0)].sort_values('sharpe',ascending=False)
    eligible=reliable.loc[reliable.min_including_partial_2026.gt(0)]
    names=(eligible if len(eligible) else reliable).variant.head(3).tolist()
    for name in names:
        param=search.parameters[name]
        if param['base_name'] not in search.bases:
            search.add_bases([old_specs[param['base_name']]])
        if name not in search.navs:
            targets=search.targets_for(name)
            search.navs[name]=batch_backtest(targets[None],search.decisions,search.market,search.config)['equity'][:,0]
    search.verify(names)
    save_selected(search,names[0])
    (search.out/'base_specifications.json').write_text(json.dumps(search.base_specs,indent=2),encoding='utf-8')
    return search


if __name__=='__main__':
    import sys
    if '--candidate' in sys.argv:
        replay_candidate()
    elif '--extend' in sys.argv:
        extend_monthly_factors()
    elif '--refine' in sys.argv:
        refine_monthly_weights()
    elif '--polish' in sys.argv:
        polish_monthly_weights()
    elif '--fine' in sys.argv:
        polish_monthly_weights(denominator=80,radius=2,tails=(.045,.05,.055),
                               stage='monthly_half_step_weights',prefix='monthly_fine')
    else:
        run(baseline_only='--baseline' in sys.argv)
