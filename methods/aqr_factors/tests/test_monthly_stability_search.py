"""Fixed blends must retain dated neutrality and not use later target rows."""
import unittest
import numpy as np
import pandas as pd
from methods.aqr_factors.research.monthly_stability_search import combine_targets, smooth_targets, add_stability, select_for_verification


class FixedBlendTests(unittest.TestCase):
    def test_net_before_gross_normalization_preserves_sector_and_beta(self):
        base=np.array([[[.25,-.25,.25,-.25],[.3,-.3,.2,-.2]],
                       [[-.1,.1,.4,-.4],[.4,-.4,.1,-.1]]])
        w=combine_targets(base,[.4,.6])
        np.testing.assert_allclose(abs(w).sum(axis=1),1)
        np.testing.assert_allclose(w[:,:2].sum(axis=1),0,atol=1e-14)
        np.testing.assert_allclose(w[:,2:].sum(axis=1),0,atol=1e-14)
        beta=np.array([.8,.8,1.2,1.2])
        np.testing.assert_allclose(w@beta,0,atol=1e-14)
        np.testing.assert_allclose(w.clip(min=0).sum(axis=1),.5)
        changed=base.copy();changed[:,1]*=-7
        np.testing.assert_array_equal(w[0],combine_targets(changed,[.4,.6])[0])

    def test_invalid_allocations_and_total_cancellation_rejected(self):
        base=np.array([[[.5,-.5]],[[-.5,.5]]])
        for allocations in ([1,1],[-.1,1.1],[.5,.5]):
            with self.assertRaises(ValueError):combine_targets(base,allocations)

    def test_smoothing_uses_current_eligibility_and_no_later_targets(self):
        targets=np.array([[.25,-.25,.25,-.25],[.1,-.1,.4,-.4],[.2,-.2,.3,-.3]])
        groups=np.array([[0,0,1,1]]*3)
        beta=np.array([[.8,.8,1.2,1.2]]*3)
        eligible=np.ones_like(targets,dtype=bool);eligible[1,0]=False
        for parameters in (dict(window=3),dict(decay=.75)):
            result=smooth_targets(targets,eligible,groups,beta,**parameters)
            self.assertEqual(result[1,0],0)
            np.testing.assert_allclose(abs(result).sum(axis=1),1)
            np.testing.assert_allclose(result.sum(axis=1),0,atol=1e-14)
            changed=targets.copy();changed[2]*=-100
            other=smooth_targets(changed,eligible,groups,beta,**parameters)
            np.testing.assert_array_equal(result[:2],other[:2])

    def test_partial_year_is_not_compared_as_another_complete_year(self):
        row=dict(sharpe=1.6,min_full_year_return=.07,return_2022=.07,return_2023=.08,
                 return_2024=.07,return_2025=.08,return_2026=.20,stale_price_legs=0)
        result=add_stability(pd.DataFrame([row]),dict(minimum=.067,std=.011))
        self.assertTrue(result.stability_goal_met.iloc[0])
        self.assertLess(result.annual_std.iloc[0],.006)
        row['return_2026']=-.01
        self.assertFalse(add_stability(pd.DataFrame([row]),dict(minimum=.067,std=.011)).stability_goal_met.iloc[0])

    def test_evenness_selection_keeps_quarter_floor_as_separate_diagnostic(self):
        rows=[]
        for name,values in [('even',[.06,.07,.08,.07]),('uneven',[.07,.18,.07,.18])]:
            rows.append(dict(variant=name,sharpe=1.6,min_full_year_return=min(values),
                             return_2026=.10,stale_price_legs=0,**dict(zip(['return_2022','return_2023','return_2024','return_2025'],values))))
        class Study:pass
        study=Study();study.results=[pd.DataFrame(rows)];study.benchmark=dict(minimum=.067,std=.0103)
        self.assertEqual(select_for_verification(study)[0],'even')
        # Lower annual dispersion does not imply a higher worst-year return.
        diagnostics=add_stability(study.results[0],study.benchmark)
        self.assertFalse(diagnostics.stability_goal_met.any())


if __name__=='__main__':unittest.main()
