"""Local, fixed-parameter monthly search using the corrected historical RF.

All trials retain the same dated universe, NAV, gross=1, integer executions and
fees. This is in-sample research. Only a full-ledger replay can certify a result.
"""
from pathlib import Path
from dataclasses import asdict
import hashlib
import json
import time
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from risk import historical_risk_free
from .cashflow_parameter_search import ParameterSearch, transform_targets
from .cashflow_combinations import batch_backtest
from ..portfolio import combine_targets, smooth_targets, cap_targets

ROOT = Path(__file__).resolve().parents[3]
WORK = ROOT / "methods/aqr_factors/research/rf_search"
OLD = ROOT / "methods/aqr_factors/research/results"


def signature(model):
    return hashlib.sha256(json.dumps(model, sort_keys=True).encode()).hexdigest()


class Search:
    def __init__(self):
        WORK.mkdir(exist_ok=True)
        self.inputs = ParameterSearch(WORK, rebalance_frequency="monthly")
        self.rf = historical_risk_free(self.inputs.market["dates"]).to_numpy()
        self.models, self.targets, self.rows, self.returns = {}, {}, [], {}
        self.started = time.monotonic()

    def evaluate(self, proposals, stage, *, component_only=False):
        fresh = [(model, target) for model, target in proposals if signature(model) not in self.models]
        for offset in range(0, len(fresh), 40):
            batch = fresh[offset:offset + 40]
            models, targets, keys = [], [], []
            for model, target in batch:
                key = signature(model)
                if key in self.models:
                    continue
                if not np.allclose(abs(target).sum(axis=1), 1, atol=1e-8):
                    continue
                valid=(abs(target).max() <= .05 + 1e-8 and
                       (target>1e-10).sum(axis=1).min()>=10 and (target<-1e-10).sum(axis=1).min()>=10)
                if not component_only and not valid:
                    continue
                self.models[key] = model
                keys.append(key); models.append(model); targets.append(target)
            if not targets:
                continue
            result = batch_backtest(np.array(targets), self.inputs.decisions, self.inputs.market, self.inputs.config)
            nav = result["equity"]
            previous = np.vstack([np.full((1, len(keys)), self.inputs.config.initial_nav), nav[:-1]])
            returns = nav / previous - 1
            excess = returns - self.rf[:, None]
            score = excess.mean(axis=0) / excess.std(axis=0, ddof=1) * np.sqrt(252)
            for i, key in enumerate(keys):
                valid=(abs(targets[i]).max()<=.05+1e-8 and (targets[i]>1e-10).sum(axis=1).min()>=10
                       and (targets[i]<-1e-10).sum(axis=1).min()>=10)
                row = dict(key=key, stage=stage, eligible_candidate=valid, sharpe=float(score[i]), net_return=float(nav[-1, i]/self.inputs.config.initial_nav-1),
                    volatility=float(returns[:, i].std(ddof=1)*np.sqrt(252)),
                    stale=int(result["stales"][i]), max_weight=float(abs(targets[i]).max()),
                    transaction_costs=float(result["transaction_cost"][:, i].sum()),
                    borrow_fees=float(result["borrow_fee"][:, i].sum()))
                for year in range(2022, 2027):
                    pos = np.flatnonzero(self.inputs.market["dates"].year == year)
                    row[f"return_{year}"] = float(nav[pos[-1], i] / previous[pos[0], i] - 1)
                row["min_full_year"] = min(row[f"return_{year}"] for year in range(2022, 2026))
                self.rows.append(row)
                self.returns[key] = returns[:, i]
                self.targets[key] = targets[i]
            self.save(stage)
            table=pd.DataFrame(self.rows)
            eligible=table.eligible_candidate.ne(False) if 'eligible_candidate' in table else True
            leaders = table.loc[table.stale.eq(0)&eligible].sort_values("sharpe", ascending=False)
            if len(leaders):
                best = leaders.iloc[0]
                print(f"{stage}: {min(offset+40,len(fresh))}/{len(fresh)}; total {len(self.rows)}; best historical-RF Sharpe {best.sharpe:.4f}; minimum year {best.min_full_year:.2%}", flush=True)

    def save(self, stage):
        table = pd.DataFrame(self.rows)
        table.to_csv(WORK / "trials.csv", index=False)
        (WORK / "models.json").write_text(json.dumps(self.models, indent=2), encoding="utf-8")
        eligible=table.eligible_candidate.ne(False) if 'eligible_candidate' in table else True
        clean = table.loc[table.stale.eq(0)&eligible]
        positive = clean.loc[clean.min_full_year.gt(0) & clean.return_2026.gt(0)]
        state = dict(stage=stage, trials=len(table), elapsed_seconds=time.monotonic()-self.started,
            best_sharpe=float(clean.sharpe.max()) if len(clean) else None,
            best_positive_year_sharpe=float(positive.sharpe.max()) if len(positive) else None,
            goal_met_in_batch=bool((positive.sharpe >= 1.58).any()),
            full_ledger_verified=False, source_requests=0, risk_free="historical DGS1MO daily accrual proxy",
            backtest=asdict(self.inputs.config))
        (WORK / "state.json").write_text(json.dumps(state, indent=2), encoding="utf-8")

    def seed_controls(self):
        specs=[dict(variant='seed_cf_quality_momentum',factors=('cashflow_yield','quality','momentum_6_1'),
                    weights=(.5,.25,.25),cashflow_weight=.5,factor_count=3)]
        self.inputs.add_bases(specs)
        proposals=[]
        for tail in (.025,.05,.1,.25,1.):
            settings=dict(tail=tail,power=0.,inverse_vol=0.,sector_budget='signal',beta_neutral=1.)
            model=dict(factor_weights=dict(zip(specs[0]['factors'],specs[0]['weights'])),construction=settings)
            target=transform_targets(self.inputs.bases[specs[0]['variant']],self.inputs.groups,self.inputs.volatility,self.inputs.beta,**settings)
            proposals.append((model,target))
        self.evaluate(proposals,'historical_rf_seed_controls')

    def resume(self):
        self.models=json.loads((WORK/"models.json").read_text())
        self.rows=pd.read_csv(WORK/"trials.csv").to_dict("records")
        with np.load(WORK/"returns.npz") as bank:
            self.returns={key:bank[key].copy() for key in bank.files}
        with np.load(WORK/"targets.npz") as bank:
            self.targets={key:bank[key].copy() for key in bank.files}

    def prune_targets(self):
        table=pd.DataFrame(self.rows).query("stale == 0")
        keys=[]
        for column,number in [("sharpe",120),("net_return",80),("min_full_year",80)]:
            keys.extend(table.sort_values(column,ascending=False).head(number).key)
        for year in range(2022,2027):
            keys.extend(table.sort_values(f"return_{year}",ascending=False).head(16).key)
        keep=set(keys)
        self.targets={key:target for key,target in self.targets.items() if key in keep}

    def factor_combinations(self):
        from ..portfolio import FACTOR_NAMES
        rng=np.random.default_rng(20261026)
        others=[f for f in FACTOR_NAMES if f!="cashflow_yield"]
        specs=[]
        for count in (1,2,3):
            for chosen in combinations(others,count):
                for cashflow in (.25,.5,.75):
                    factors=("cashflow_yield",)+chosen
                    weights=(cashflow,)+((1-cashflow)/count,)*count
                    name="fresh_"+hashlib.sha256(repr((factors,weights)).encode()).hexdigest()[:16]
                    specs.append(dict(variant=name,factors=factors,weights=weights,
                        cashflow_weight=cashflow,factor_count=len(factors)))
        specs.append(dict(variant="cashflow_only",factors=("cashflow_yield",),weights=(1.,),cashflow_weight=1.,factor_count=1))
        for offset in range(0,len(specs),48):
            part=specs[offset:offset+48]
            self.inputs.add_bases(part)
            proposals=[]
            for spec in part:
                factors=dict(zip(spec["factors"],spec["weights"]))
                for number in range(6):
                    settings=dict(tail=float(rng.choice([.015,.025,.035,.05,.075,.1,.2,.35,1.])),
                        power=float(rng.choice([0.,.5,1.,2.])),inverse_vol=float(rng.choice([0.,.5,1.])),
                        sector_budget=str(rng.choice(["signal","equal"])),beta_neutral=1.)
                    model=dict(factor_weights=factors,construction=settings)
                    target=transform_targets(self.inputs.bases[spec["variant"]],self.inputs.groups,
                        self.inputs.volatility,self.inputs.beta,**settings)
                    proposals.append((model,target))
            self.evaluate(proposals,f"fresh_factor_combinations_{offset//48+1}")
            for spec in part:
                self.inputs.bases.pop(spec["variant"],None)
            self.prune_targets()
        self.optimize_blends("fresh_factor_optimized_blends",count=80)

    def scoring_variants(self):
        rng=np.random.default_rng(20261027)
        table=pd.DataFrame(self.rows).query('stale == 0')
        leaves=[];seen=set()
        for column,count in [('sharpe',100),('net_return',60),('min_full_year',40)]:
            for key in table.sort_values(column,ascending=False).head(count).key:
                model=self.models[key]
                if 'factor_weights' not in model:continue
                identity=json.dumps(model['factor_weights'],sort_keys=True)
                if identity not in seen:leaves.append(model);seen.add(identity)
        for offset in range(0,len(leaves),18):
            specs=[];lookup={}
            for i,model in enumerate(leaves[offset:offset+18]):
                for operator in ['group_zscore','group_winsor_zscore','global_rank_zscore']:
                    key=f'operator_{offset+i}_{operator}'
                    scoring=dict(score_operator=operator,winsor_quantile=.025)
                    specs.append(dict(variant=key,factors=tuple(model['factor_weights']),weights=tuple(model['factor_weights'].values()),
                        cashflow_weight=model['factor_weights'].get('cashflow_yield',0),factor_count=len(model['factor_weights']),**scoring))
                    lookup[key]=(model,scoring)
            self.inputs.add_bases(specs)
            proposals=[]
            for spec in specs:
                source,scoring=lookup[spec['variant']]
                for number in range(6):
                    settings=dict(tail=float(rng.choice([.025,.035,.05,.075,.1,.2,.35,1.])),power=float(rng.choice([0.,1.,2.])),
                        inverse_vol=float(rng.choice([0.,.5,1.])),sector_budget=str(rng.choice(['signal','equal'])),beta_neutral=1.)
                    model=dict(factor_weights=source['factor_weights'],construction=settings,scoring=scoring)
                    target=transform_targets(self.inputs.bases[spec['variant']],self.inputs.groups,self.inputs.volatility,self.inputs.beta,**settings)
                    proposals.append((model,target))
            self.evaluate(proposals,f'raw_zscore_and_global_rank_{offset//18+1}')
            for spec in specs:self.inputs.bases.pop(spec['variant'],None)
            self.prune_targets()
        self.optimize_blends('alternate_scoring_optimized_blends',count=90)

    def sector_components(self):
        table=pd.DataFrame(self.rows).query('stale == 0')
        seeds=[]
        for col in ['sharpe','net_return','min_full_year']:
            seeds.extend(table.sort_values(col,ascending=False).head(5).key)
        names=list(dict.fromkeys(seeds))
        # Source portfolios are held fixed over the complete sample; each block
        # stays dollar/sector/beta neutral. Only final mixtures must pass caps.
        proposals=[]
        for name in names:
            if name not in self.targets:continue
            for code,label in enumerate(self.inputs.sectors):
                model=dict(component_allocations={'source':1.},component_models={'source':self.models[name]},sector_filter=[label])
                target=transform_targets(self.targets[name],self.inputs.groups,self.inputs.volatility,self.inputs.beta,
                                         sector_mask=[code],beta_neutral=1.)
                proposals.append((model,target))
        # Wider within-sector tails make final diversified allocations feasible
        # under the 5% single-name account cap.
        leaf_names=[]
        for column in ['sharpe','net_return','min_full_year']:
            leaves=table.loc[table.key.map(lambda key:'factor_weights' in self.models[key])]
            leaf_names.extend(leaves.sort_values(column,ascending=False).head(3).key)
        specs=[];lookup={}
        for key in dict.fromkeys(leaf_names):
            model=self.models[key]
            spec=dict(variant='sector_base_'+key,factors=tuple(model['factor_weights']),weights=tuple(model['factor_weights'].values()),
                      cashflow_weight=model['factor_weights'].get('cashflow_yield',0),factor_count=len(model['factor_weights']),**model.get('scoring',{}))
            specs.append(spec);lookup[spec['variant']]=model
        self.inputs.add_bases(specs)
        for spec in specs:
            source=lookup[spec['variant']]
            for tail in [.15,.35,1.]:
                for code,label in enumerate(self.inputs.sectors):
                    settings=dict(tail=tail,power=0.,inverse_vol=.5,sector_budget='signal',beta_neutral=1.)
                    base=transform_targets(self.inputs.bases[spec['variant']],self.inputs.groups,self.inputs.volatility,
                        self.inputs.beta,**settings)
                    model=dict(factor_weights=source['factor_weights'],construction=settings,sector_filter=[label])
                    if 'scoring' in source:model['scoring']=source['scoring']
                    target=transform_targets(base,self.inputs.groups,self.inputs.volatility,self.inputs.beta,sector_mask=[code],beta_neutral=1.)
                    proposals.append((model,target))
        self.evaluate(proposals,'constant_sector_component_controls',component_only=True)
        self.optimize_blends('constant_sector_optimized_blends',count=130)


    def optimize_blends(self, stage, count=60):
        table = pd.DataFrame(self.rows).query("stale == 0")
        chosen = []
        def take(frame, n):
            chosen.extend(frame.head(n).key.tolist())
        take(table.sort_values("sharpe", ascending=False), count//2)
        take(table.sort_values("net_return", ascending=False), count//4)
        take(table.sort_values("min_full_year", ascending=False), count//4)
        for year in range(2022, 2027):
            take(table.sort_values(f"return_{year}", ascending=False), 4)
        names = list(dict.fromkeys(chosen))
        matrix = np.array([self.returns[name] for name in names]).T
        mean = matrix.mean(axis=0) - self.rf.mean()
        covariance = np.cov(matrix-self.rf[:, None], rowvar=False)
        covariance += np.eye(len(names))*1e-12
        def objective(weights):
            variance = weights@covariance@weights
            numerator = mean@weights
            return -numerator/np.sqrt(variance)
        def gradient(weights):
            cw = covariance@weights
            variance = weights@cw
            return -mean/np.sqrt(variance) + (mean@weights)*cw/variance**1.5
        proposals = []
        for cap in [.15, .25, .4, .65, 1.]:
            fit = minimize(objective, np.full(len(names), 1/len(names)), jac=gradient,
                bounds=[(0, cap)]*len(names), constraints=[dict(type="eq", fun=lambda w: w.sum()-1,
                    jac=lambda w: np.ones(len(w)))], method="SLSQP", options=dict(maxiter=500, ftol=1e-12))
            allocations = {names[i]: float(v) for i, v in enumerate(fit.x) if v > 1e-7}
            total = sum(allocations.values()); allocations = {k:v/total for k,v in allocations.items()}
            print(f"{stage}: convex-return proxy cap {cap}, Sharpe {-objective(fit.x)*np.sqrt(252):.4f}, {len(allocations)} components", flush=True)
            for exponent in [1., .5, 2.]:
                weights = {k:v**exponent for k,v in allocations.items()}
                total = sum(weights.values()); weights={k:v/total for k,v in weights.items()}
                model=dict(component_allocations=weights, component_models={k:self.models[k] for k in weights})
                target=combine_targets(np.array([self.targets[k] for k in weights]), list(weights.values()))
                target=transform_targets(target,self.inputs.groups,self.inputs.volatility,self.inputs.beta,beta_neutral=1.)
                proposals.append((model,target))
                if abs(target).max()>.05+1e-8:
                    try:
                        capped=cap_targets(target,self.inputs.groups,self.inputs.beta,max_weight=.05)
                    except ValueError:
                        continue
                    proposals.append(({**model,'weight_cap':.05},capped))
        self.evaluate(proposals,stage)

    def finish_checkpoint(self):
        np.savez_compressed(WORK/"targets.npz", **self.targets)
        np.savez_compressed(WORK/"returns.npz", **self.returns)


if __name__ == "__main__":
    import sys
    search=Search()
    if "--scoring" in sys.argv:
        search.resume()
        search.scoring_variants()
    elif "--sectors" in sys.argv:
        search.resume()
        search.sector_components()
    elif "--factors" in sys.argv:
        search.resume()
        search.factor_combinations()
    else:
        if (WORK/'targets.npz').exists():search.resume()
        else:search.seed_controls()
        search.optimize_blends("historical_rf_initial_blends")
    search.finish_checkpoint()
