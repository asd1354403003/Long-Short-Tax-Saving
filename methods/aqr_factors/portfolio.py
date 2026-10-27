"""Dated factor ranking, portfolio construction and fixed target blending."""
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from scipy.optimize import linprog, minimize
from scipy.linalg import qr
from .construction import PortfolioConstructor

FACTOR_NAMES = ('cashflow_yield', 'value', 'quality', 'momentum',
    'earnings_yield', 'roa', 'low_volatility', 'gross_profitability',
    'average_asset_roa', 'low_accruals', 'profit_stability', 'fcf_yield',
    'momentum_6_1', 'residual_momentum')


def cap_targets(weights, groups, beta, max_weight=.05):
    """Closest capped dated weights, preserving side, sector and beta neutrality.

    Signs and the selected stock set are fixed before this projection. Linear
    constraints enforce long +.5, short -.5, zero sector and estimated beta.
    Infeasible selections are rejected rather than relaxing gross or caps.
    """
    if not 0 < max_weight <= .5:
        raise ValueError('Invalid single-name weight cap')
    output=weights.copy()
    for t,row in enumerate(weights):
        if abs(row).max() <= max_weight+1e-10:
            continue
        chosen=np.flatnonzero(abs(row)>1e-12)
        original=row[chosen]
        long=original>0
        sector=groups[t,chosen]
        risk=beta[t,chosen]
        if not np.isfinite(risk).all() or (sector<0).any():
            raise ValueError('Capping requires known historical sector and beta')
        matrix=[long.astype(float),(~long).astype(float)]
        right=[.5,-.5]
        for group in np.unique(sector):
            matrix.append((sector==group).astype(float));right.append(0.)
        matrix.append(risk);right.append(0.)
        a=np.asarray(matrix);b=np.asarray(right)
        _,tri,pivot=qr(a.T,mode='economic',pivoting=True)
        rank=int((abs(np.diag(tri))>1e-10).sum())
        keep=sorted(pivot[:rank]);a=a[keep];b=b[keep]
        bounds=[(0,max_weight) if value else (-max_weight,0) for value in long]
        feasible=linprog(np.zeros(len(chosen)),A_eq=a,b_eq=b,bounds=bounds,method='highs')
        if not feasible.success:
            raise ValueError('Selected names cannot satisfy gross/sector/beta/cap constraints')
        fit=minimize(lambda w:.5*np.sum((w-original)**2),feasible.x,
            jac=lambda w:w-original,bounds=bounds,
            constraints=[dict(type='eq',fun=lambda w:a@w-b,jac=lambda w:a)],
            method='SLSQP',options=dict(maxiter=100,ftol=1e-12))
        if not fit.success or abs(a@fit.x-b).max()>1e-8 or abs(fit.x).max()>max_weight+1e-8:
            raise ValueError('Capped target projection did not converge')
        output[t]=0.;output[t,chosen]=fit.x
    return output

def transform_targets(base, groups, volatility, beta, *, tail=1., power=1.,
                      inverse_vol=0., sector_budget='signal', beta_neutral=0.,
                      short_vol_quantile=1., sector_mask=None):
    """Sector-balanced tails and inverse-volatility sizing, using past risk only.

    Beta projection is restricted to selected names and is sector-demeaned,
    preserving zero sector/net exposure. It may change selected names' signs.
    """
    output=np.zeros_like(base)
    for t in range(len(base)):
        original=base[t]
        valid=(original!=0) & (groups[t]>=0)
        if inverse_vol:
            valid &= np.isfinite(volatility[t]) & (volatility[t]>0)
        if beta_neutral:
            valid &= np.isfinite(beta[t])
        if sector_mask is not None:
            valid &= np.isin(groups[t],sector_mask)
        members=[]; budgets=[]; magnitudes=[]
        for group in np.unique(groups[t,valid]):
            chosen=np.flatnonzero(valid & (groups[t]==group))
            longs=chosen[original[chosen]>0]
            shorts=chosen[original[chosen]<0]
            if short_vol_quantile<1:
                finite_vol=volatility[t,chosen]
                finite_vol=finite_vol[np.isfinite(finite_vol)]
                if len(finite_vol):
                    threshold=np.quantile(finite_vol,short_vol_quantile)
                    shorts=shorts[np.isfinite(volatility[t,shorts]) & (volatility[t,shorts]<=threshold)]
            if not len(longs) or not len(shorts):
                continue
            if tail<1:
                k=max(2,int(np.ceil(len(chosen)*tail)))
                longs=longs[np.argsort(-original[longs],kind='stable')[:k]]
                shorts=shorts[np.argsort(original[shorts],kind='stable')[:k]]
            long_mag=abs(original[longs])**power
            short_mag=abs(original[shorts])**power
            if inverse_vol:
                long_mag/=volatility[t,longs]**inverse_vol
                short_mag/=volatility[t,shorts]**inverse_vol
            budget=abs(original[chosen]).sum() if sector_budget=='signal' else (1. if sector_budget=='equal' else np.sqrt(len(chosen)))
            members.append((longs,shorts)); magnitudes.append((long_mag,short_mag)); budgets.append(budget)
        if not budgets:
            continue
        budgets=np.asarray(budgets)/sum(budgets)
        for budget,(longs,shorts),(long_mag,short_mag) in zip(budgets,members,magnitudes):
            output[t,longs]=.5*budget*long_mag/long_mag.sum()
            output[t,shorts]=-.5*budget*short_mag/short_mag.sum()
        if beta_neutral:
            risk_direction=np.zeros(base.shape[1])
            for longs,shorts in members:
                chosen=np.r_[longs,shorts]
                risk_direction[chosen]=beta[t,chosen]-beta[t,chosen].mean()
            denominator=risk_direction@risk_direction
            exposure=output[t]@np.nan_to_num(beta[t])
            if denominator>1e-12:
                output[t]-=beta_neutral*exposure/denominator*risk_direction
        gross=abs(output[t]).sum()
        if gross>1e-12:
            output[t]/=gross
    return output


class FrozenTargetsConstructor(PortfolioConstructor):
    """Execute audited dated targets through the unchanged transaction ledger."""
    def __init__(self, config, weights, tickers, decisions):
        super().__init__(config)
        self.weights=pd.DataFrame(weights,index=decisions,columns=tickers)

    def build(self, signals, **kwargs):
        targets=super().build(signals,**kwargs)
        for date,month in targets.groupby('as_of'):
            targets.loc[month.index,'target_weight']=month.ticker.map(self.weights.loc[date]).fillna(0)
        targets['target_side']=np.select([targets.target_weight.gt(0),targets.target_weight.lt(0)],['long','short'],default='flat')
        targets['construction_status']=np.where(targets.target_weight.ne(0),'allocated_tuned','tuned_zero')
        return targets


def decision_rows(raw, frequency):
    """Select dated decisions, preserving monthly changing candidate pools."""
    if frequency not in {'monthly', 'quarterly'}:
        raise ValueError('rebalance_frequency must be monthly or quarterly')
    mask=raw.as_of.dt.month.isin([3,6,9,12]) if frequency=='quarterly' else np.ones(len(raw),dtype=bool)
    return raw.loc[mask].reset_index(drop=True)


def target_arrays(raw, specs, tickers, decisions, progress=print):
    """Recompute ranking inside each combination's own dated candidate pool."""
    targets=np.zeros((len(specs),len(decisions),len(tickers)))
    counts=np.zeros((len(specs),len(decisions)),dtype=int)
    masks={}
    for i,spec in enumerate(specs):
        operator=spec.get('score_operator','group_rank_zscore')
        quantile=float(spec.get('winsor_quantile',.01))
        if operator not in {'group_rank_zscore','group_zscore','group_winsor_zscore','global_rank_zscore'} or not 0<=quantile<.5:
            raise ValueError('Unsupported scoring operator or winsor quantile')
        masks.setdefault((spec['factors'],operator,quantile),[]).append(i)
    factor_index={name:i for i,name in enumerate(FACTOR_NAMES)}
    for date_i,date in enumerate(decisions):
        month=raw.loc[raw.as_of.eq(date)].set_index('ticker').reindex(tickers)
        groups,labels=pd.factorize(month.score_group,sort=True)
        values=month[list(FACTOR_NAMES)].to_numpy(dtype=float)
        active=month.in_universe.eq(True).to_numpy(dtype=bool)
        active &= ~month.input_status.isin(['download_failed','missing_month_end_price']).to_numpy()
        active &= month.close.gt(0).to_numpy() & month.market_cap.gt(0).to_numpy() & (groups>=0)
        cache={}
        for (factors,operator,quantile),indices in masks.items():
            columns=[factor_index[f] for f in factors]
            eligible=active & np.isfinite(values[:,columns]).all(axis=1)
            sizes=np.bincount(groups[eligible],minlength=len(labels))
            eligible &= np.where(groups>=0,sizes[np.maximum(groups,0)]>=10,False)
            key=eligible.tobytes()
            chosen=np.flatnonzero(eligible)
            transformed=[]
            for column in columns:
                item=(key,column,operator,quantile)
                if item not in cache:
                    z=np.zeros(len(tickers))
                    member_sets=[chosen] if operator=='global_rank_zscore' else [chosen[groups[chosen]==group] for group in np.unique(groups[chosen])]
                    for members in member_sets:
                        rank=values[members,column].copy()
                        if operator in {'group_rank_zscore','global_rank_zscore'}:
                            rank=rankdata(rank,method='average')
                        elif operator=='group_winsor_zscore':
                            rank=np.clip(rank,*np.quantile(rank,[quantile,1-quantile]))
                        deviation=rank.std(ddof=0)
                        z[members]=(rank-rank.mean())/deviation if deviation>0 else 0.
                    cache[item]=z
                transformed.append(cache[item])
            matrix=np.asarray(transformed)
            for spec_i in indices:
                score=np.round(np.asarray(specs[spec_i]['weights'])@matrix,12)
                centered=np.zeros(len(tickers))
                for group in np.unique(groups[chosen]):
                    members=chosen[groups[chosen]==group]
                    centered[members]=score[members]-score[members].mean()
                    centered[members]-=centered[members].mean()
                denominator=abs(centered).sum()
                targets[spec_i,date_i]=centered/denominator if denominator>0 else centered
                counts[spec_i,date_i]=len(chosen)
        progress(f'Ranked combinations {date_i+1}/{len(decisions)}: {date.date()}')
    return targets,counts


def combine_targets(targets, allocations):
    """Net constant, nonnegative component allocations, gross-normalized by date."""
    allocations=np.asarray(allocations,dtype=float)
    if not np.isfinite(allocations).all() or (allocations<0).any() or not np.isclose(allocations.sum(),1):
        raise ValueError('Component allocations must be nonnegative and sum to 1')
    blended=np.einsum('c,ctn->tn',allocations,targets)
    gross=abs(blended).sum(axis=1)
    if (gross<1e-10).any():
        raise ValueError('Component cancellation leaves no portfolio')
    return blended/gross[:,None]


def smooth_targets(original,eligible,groups,beta,*,window=None,decay=None):
    """Use past targets only, with current eligibility, sector and beta controls."""
    if (window is None)==(decay is None):raise ValueError('Specify window or decay')
    output=np.zeros_like(original)
    state=np.zeros(original.shape[1])
    for t in range(len(original)):
        if window is not None:
            row=original[max(0,t-window+1):t+1].mean(axis=0).copy()
        else:
            state=original[t].copy() if t==0 else decay*state+(1-decay)*original[t]
            row=state.copy()
        valid=eligible[t]&np.isfinite(beta[t])&(groups[t]>=0)
        row[~valid]=0
        state[~valid]=0
        risk=np.zeros(original.shape[1])
        for group in np.unique(groups[t,valid]):
            chosen=np.flatnonzero(valid&(groups[t]==group)&(abs(row)>1e-12))
            longs=chosen[row[chosen]>0];shorts=chosen[row[chosen]<0]
            if not len(longs) or not len(shorts):
                row[chosen]=0
                continue
            budget=abs(row[chosen]).sum()
            row[longs]*=.5*budget/row[longs].sum()
            row[shorts]*=.5*budget/abs(row[shorts]).sum()
            risk[chosen]=beta[t,chosen]-beta[t,chosen].mean()
        if risk@risk>1e-12:
            row-=(row@np.nan_to_num(beta[t]))/(risk@risk)*risk
        gross=abs(row).sum()
        if gross>1e-12:output[t]=row/gross
    return output
