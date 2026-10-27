"""Financial alignment, price cutoff and portfolio-risk checks for the trial."""
import unittest
import sqlite3
from contextlib import contextmanager
import numpy as np
import pandas as pd

from methods.aqr_factors.research.multifactor_experiment import financial_features, quarterly_inputs, price_inputs, estimate_risk, RiskConstrainedConstructor
from methods.aqr_factors.construction import ConstructionConfig


class MultifactorTests(unittest.TestCase):
    def fixture(self):
        known={name:{} for name in ('income','balance','cashflow')}
        dates=pd.date_range('2018-12-31',periods=12,freq='QE')
        for date in dates:
            for statement,values in [('income',dict(GrossProfit=10.,NetIncome=2.,NetIncomeCommonStockholders=1.5)),
                    ('balance',dict(TotalAssets=100.)),('cashflow',dict(OperatingCashFlow=3.,CapitalExpenditure=-1.))]:
                known[statement][date]=dict(values=values,effective_available=date+pd.Timedelta(days=90),date_assumed=True)
        return known,dates

    def test_consecutive_ttm_signed_cashflow_and_average_assets(self):
        known,dates=self.fixture()
        result=financial_features(known)
        self.assertEqual(result['financial_period_end'],dates[-1])
        self.assertEqual(result['gross_profit_ttm'],40.)
        self.assertEqual(result['free_cashflow_ttm'],8.)
        self.assertEqual(result['average_assets'],100.)
        self.assertAlmostEqual(result['gross_profitability'],.4)
        self.assertAlmostEqual(result['average_asset_roa'],.08)
        self.assertAlmostEqual(result['low_accruals'],.04)
        self.assertAlmostEqual(result['profit_stability'],0.)
        del known['cashflow'][dates[-2]]
        missing=financial_features(known)
        self.assertTrue(np.isnan(missing['free_cashflow_ttm']))
        self.assertTrue(np.isnan(missing['operating_cashflow_ttm_new']))
        self.assertTrue(np.isnan(missing['low_accruals']))

    def test_future_prices_cannot_change_decision_factors_or_risk(self):
        rng=np.random.default_rng(71)
        dates=pd.bdate_range('2020-01-01',periods=440)
        returns=rng.normal(0,.01,(440,5))
        prices=pd.DataFrame(100*np.exp(returns.cumsum(axis=0)),index=dates,columns=['SPY','A','B','C','D'])
        rows=pd.DataFrame(dict(as_of=[dates[330]]*4,ticker=['A','B','C','D'],
            in_universe=True,input_status='eligible',score_group='Tech'))
        first=price_inputs(rows,prices.iloc[:350],progress=lambda *_:None)
        prices.loc[dates[350]:]*=1000
        later=price_inputs(rows,prices,progress=lambda *_:None)
        pd.testing.assert_frame_equal(first,later)
        self.assertTrue(first.residual_momentum.notna().all())

    def test_source_availability_blocks_future_quarter(self):
        known,dates=self.fixture()
        future=dates[-1]+pd.offsets.QuarterEnd()
        for statement in known:
            known[statement][future]=dict(values=known[statement][dates[-1]]['values'].copy(),
                effective_available=future+pd.Timedelta(days=90),date_assumed=True)
        known['income'][future]['values']['GrossProfit']=1000.
        db=sqlite3.connect(':memory:')
        db.execute('CREATE TABLE fundamental_facts(report_id TEXT, metric TEXT, value REAL)')
        reports=[]
        for statement,periods in known.items():
            for period,record in periods.items():
                identifier=statement+str(period.date())
                reports.append(dict(report_id=identifier,ticker='A',statement=statement,period_end=str(period.date()),
                    available_from=None,reference_available_from=None,observed_at='2026-10-26',financial_currency='USD'))
                db.executemany('INSERT INTO fundamental_facts VALUES(?,?,?)',[(identifier,k,v) for k,v in record['values'].items()])
        class Database:
            def read(self, query):
                return pd.DataFrame(reports)
            @contextmanager
            def connection(self):
                yield db
        rows=pd.DataFrame(dict(as_of=pd.to_datetime(['2021-12-31','2022-03-31']),ticker='A'))
        features=quarterly_inputs(Database(),rows,progress=lambda *_:None)
        self.assertEqual(features.iloc[0].financial_period_end,dates[-1])
        self.assertEqual(features.iloc[0].gross_profit_ttm,40.)
        self.assertEqual(features.iloc[1].financial_period_end,future)
        self.assertEqual(features.iloc[1].gross_profit_ttm,1030.)
        self.assertTrue(features.financial_available_from.le(features.as_of).all())
        db.close()

    def test_beta_uses_matching_observation_dates(self):
        rng=np.random.default_rng(75)
        market=pd.Series(rng.normal(0,.01,300))
        stock=2*market+.005
        stock.iloc[220:]=np.nan
        market.iloc[220:]*=20
        beta,_,counts=estimate_risk(pd.DataFrame(dict(SPY=market,A=stock)))
        self.assertEqual(counts.A,220)
        self.assertAlmostEqual(beta.A,2.,places=10)

    def test_risk_projection_keeps_gross_sector_beta_and_position_limits(self):
        rows=pd.DataFrame(dict(as_of=[pd.Timestamp('2023-03-31')]*8,ticker=list('ABCDEFGH'),
            score=[-2.,-1.,1.,2.]*2,sector=['Tech']*4+['Health']*4,
            estimated_beta=[.5,2.,.6,1.5]*2,estimated_volatility=[.4,.2,.3,.5]*2))
        model=RiskConstrainedConstructor(ConstructionConfig(group_by='sector'),max_weight=.20)
        targets=model.build(rows,current_weights=pd.Series(dtype=float))
        weights=targets.set_index('ticker').target_weight.reindex(rows.ticker).to_numpy()
        self.assertAlmostEqual(abs(weights).sum(),1.,places=6)
        self.assertAlmostEqual(weights.sum(),0.,places=6)
        self.assertLessEqual(abs(weights).max(),.20+1e-6)
        self.assertLessEqual(abs(rows.estimated_beta.to_numpy()@weights),.05+1e-6)
        self.assertTrue(np.all(weights*np.sign(rows.score)>=-1e-8))
        self.assertAlmostEqual(weights[:4].sum(),0.,places=6)
        self.assertAlmostEqual(weights[4:].sum(),0.,places=6)


if __name__=='__main__':
    unittest.main()
