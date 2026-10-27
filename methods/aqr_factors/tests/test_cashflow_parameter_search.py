"""Risk/candidate transformations must preserve constraints and date isolation."""
import unittest
import numpy as np
import pandas as pd
from methods.aqr_factors.research.cashflow_parameter_search import transform_targets, decision_rows


class ParameterSearchTests(unittest.TestCase):
    def test_monthly_decisions_keep_new_members_and_non_quarter_months(self):
        rows=pd.DataFrame({'as_of':pd.to_datetime(['2021-12-31','2022-01-31','2022-02-28','2022-03-31']),
                           'ticker':['OLD','NEW_JAN','NEW_FEB','NEW_MAR']})
        self.assertEqual(decision_rows(rows,'monthly').ticker.tolist(),rows.ticker.tolist())
        self.assertEqual(decision_rows(rows,'quarterly').ticker.tolist(),['OLD','NEW_MAR'])
        with self.assertRaises(ValueError):
            decision_rows(rows,'weekly')

    def fixture(self):
        base=np.array([[-.10,-.15,.05,.20,-.20,-.05,.10,.15]]*2)
        groups=np.array([[0,0,0,0,1,1,1,1]]*2)
        volatility=np.array([[.1,.4,.2,.3,.3,.2,.5,.15]]*2)
        beta=np.array([[.4,1.6,.8,1.5,.7,1.8,.5,1.2]]*2)
        return base,groups,volatility,beta

    def test_default_reproduces_unmodified_targets(self):
        base,groups,vol,beta=self.fixture()
        result=transform_targets(base,groups,vol,beta)
        np.testing.assert_allclose(result,base,atol=1e-14)

    def test_beta_projection_and_equal_sector_budget_preserve_neutrality(self):
        base,groups,vol,beta=self.fixture()
        result=transform_targets(base,groups,vol,beta,tail=.2,power=0,inverse_vol=1,
            sector_budget='equal',beta_neutral=1)
        np.testing.assert_allclose(abs(result).sum(axis=1),1,atol=1e-12)
        np.testing.assert_allclose((result*beta).sum(axis=1),0,atol=1e-12)
        np.testing.assert_allclose(result[:,:4].sum(axis=1),0,atol=1e-12)
        np.testing.assert_allclose(result[:,4:].sum(axis=1),0,atol=1e-12)

    def test_later_risk_and_scores_cannot_change_earlier_targets(self):
        base,groups,vol,beta=self.fixture()
        original=transform_targets(base,groups,vol,beta,tail=.2,inverse_vol=1,beta_neutral=1)
        base[1]*=-20; vol[1]*=7; beta[1]*=3
        changed=transform_targets(base,groups,vol,beta,tail=.2,inverse_vol=1,beta_neutral=1)
        np.testing.assert_array_equal(original[0],changed[0])


if __name__=='__main__':
    unittest.main()
