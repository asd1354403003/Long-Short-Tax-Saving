"""Execution parity under deferred orders, splits, missing marks and weekends."""
import unittest
from types import SimpleNamespace
import numpy as np
import pandas as pd

from methods.aqr_factors.research.cashflow_combinations import specifications, batch_backtest, market_arrays
from methods.aqr_factors.construction import ConstructionConfig
from methods.aqr_factors.backtest import WeightedBacktester, WeightedBacktestConfig
from reports import PerformanceReport


class CashflowCombinationTests(unittest.TestCase):
    def test_search_is_cashflow_anchored_and_has_unique_fixed_weights(self):
        specs=specifications()
        self.assertEqual(len(specs),1210)
        seen=set()
        for spec in specs:
            self.assertEqual(spec['factors'][0],'cashflow_yield')
            self.assertLessEqual(len(spec['factors']),4)
            self.assertAlmostEqual(sum(spec['weights']),1.)
            key=(spec['factors'],spec['weights'])
            self.assertNotIn(key,seen)
            seen.add(key)

    def test_batch_matches_full_ledger_with_deferred_split_and_missing_close(self):
        dates=pd.to_datetime(['2023-03-30','2023-03-31','2023-04-03','2023-04-04','2023-04-05'])
        histories={
            'A':pd.DataFrame(dict(Open=[np.nan,100.,99.,98.,102.],Close=[101.,102.,98.,100.,103.],**{'Stock Splits':[0.]*5}),index=dates),
            'B':pd.DataFrame(dict(Open=[200.,203.,204.,205.,206.],Close=[202.,204.,206.,np.nan,207.],**{'Stock Splits':[0.]*5}),index=dates),
            'C':pd.DataFrame(dict(Open=[333.,331.,220.,226.,228.],Close=[332.,330.,225.,227.,229.],**{'Stock Splits':[0.,0.,1.5,0.,0.]}),index=dates),
            'SPY':pd.DataFrame(dict(Open=[100.,101.,102.,103.,104.],Close=[101.,102.,103.,104.,105.],**{'Stock Splits':[0.]*5}),index=dates),
        }
        decisions=pd.to_datetime(['2022-12-31','2023-03-31'])
        frame=pd.DataFrame([dict(as_of=date,ticker=ticker,score=score,in_universe=True)
            for date,values in [(decisions[0],[-.4,.3,.1]),(decisions[1],[.3,-.4,.1])]
            for ticker,score in zip(['A','B','C'],values)])
        signals=SimpleNamespace(signals=frame,errors=[],coverage=pd.DataFrame(),
            universe=SimpleNamespace(proxy=None),companies=pd.DataFrame(dict(ticker=['A','B','C'],sector='Tech')))
        config=WeightedBacktestConfig(start_date='2023-03-30',end_date='2023-04-05',initial_nav=10_000,
            rebalance_frequency='quarterly')
        result=WeightedBacktester(config,ConstructionConfig(group_by='sector')).run(signals,histories=histories)
        PerformanceReport.audit(result)
        tickers=['A','B','C','SPY']
        weights=result['target_weights'].pivot(index='as_of',columns='ticker',values='target_weight').reindex(index=decisions,columns=tickers).fillna(0).to_numpy()[None,:,:]
        batch=batch_backtest(weights,decisions,market_arrays(histories,tickers,config),config)
        for field in ['equity','transaction_cost','borrow_fee']:
            np.testing.assert_allclose(batch[field][:,0],result['daily_summary'][field],atol=1e-7,rtol=0)
        self.assertEqual(int(batch['stales'][0]),int(result['daily_summary'].stale_price_legs.sum()))
        self.assertGreater(batch['stales'][0],0)
        self.assertTrue(result['events'].reason.eq('deferred_rebalance').any())
        self.assertFalse(result['corporate_events'].empty)


if __name__=='__main__':
    unittest.main()
