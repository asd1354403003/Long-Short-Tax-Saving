import unittest
import numpy as np
import pandas as pd

from methods.aqr_factors.portfolio import cap_targets, target_arrays, FACTOR_NAMES


class PortfolioProjectionTests(unittest.TestCase):
    def inputs(self):
        half=np.r_[.12,np.full(29,.38/29)]
        row=np.r_[half,-half]
        groups=np.tile(np.arange(60)%3,(2,1))
        risk=np.linspace(.3,1.7,30)
        return np.vstack([row,row]),groups,np.tile(np.r_[risk,risk],(2,1))

    def test_cap_keeps_side_gross_sector_and_beta_constraints(self):
        weights,groups,beta=self.inputs()
        result=cap_targets(weights,groups,beta,.05)
        self.assertLessEqual(abs(result).max(),.05+1e-8)
        np.testing.assert_allclose(result.clip(min=0).sum(axis=1),.5,atol=1e-8)
        np.testing.assert_allclose(result.clip(max=0).sum(axis=1),-.5,atol=1e-8)
        np.testing.assert_allclose((result*beta).sum(axis=1),0,atol=1e-8)
        for group in range(3):
            np.testing.assert_allclose(result[:,groups[0]==group].sum(axis=1),0,atol=1e-8)

    def test_later_targets_cannot_change_earlier_capped_weights(self):
        weights,groups,beta=self.inputs()
        first=cap_targets(weights,groups,beta,.05)
        weights[1]=np.roll(weights[1],3)
        later=cap_targets(weights,groups,beta,.05)
        np.testing.assert_allclose(first[0],later[0],atol=1e-10)

    def test_infeasible_selected_stock_set_is_rejected(self):
        with self.assertRaises(ValueError):
            cap_targets(np.array([[.5,-.5]]),np.array([[0,0]]),np.array([[1.,1.]]),.05)

    def test_score_operators_keep_same_month_eligibility_and_sector_neutrality(self):
        symbols=[f'T{i:02}' for i in range(40)]
        date=pd.Timestamp('2022-01-31')
        rng=np.random.default_rng(7)
        frame=pd.DataFrame({name:rng.normal(size=40) for name in FACTOR_NAMES})
        frame=frame.assign(as_of=date,ticker=symbols,score_group=np.repeat(['A','B'],20),
                           in_universe=True,input_status='ok',close=10.,market_cap=100.)
        frame.loc[0,'cashflow_yield']=np.nan
        specs=[dict(variant=operator,factors=('cashflow_yield','quality'),weights=(.5,.5),score_operator=operator)
               for operator in ['group_rank_zscore','group_zscore','group_winsor_zscore','global_rank_zscore']]
        weights,counts=target_arrays(frame,specs,symbols,pd.DatetimeIndex([date]),progress=lambda *_:None)
        self.assertTrue((weights[:,:,0]==0).all())
        self.assertTrue((counts==39).all())
        np.testing.assert_allclose(abs(weights).sum(axis=2),1)
        for group in [slice(0,20),slice(20,40)]:
            np.testing.assert_allclose(weights[:,:,group].sum(axis=2),0,atol=1e-12)


if __name__=='__main__':
    unittest.main()
