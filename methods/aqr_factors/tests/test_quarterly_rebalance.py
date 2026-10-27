"""Quarter-end eligibility is local to that decision, with later re-entry."""
from types import SimpleNamespace
import unittest

import numpy as np
import pandas as pd

from methods.aqr_factors.construction import ConstructionConfig
from methods.aqr_factors.backtest import WeightedBacktestConfig, WeightedBacktester
from reports import PerformanceReport


class QuarterlyExecutionTests(unittest.TestCase):
    def test_only_quarter_snapshots_trade_and_missing_stock_can_return(self):
        tickers = ("A", "B", "C", "D")
        rows = []
        for date in ("2021-12-31", "2022-01-31", "2022-02-28", "2022-03-31", "2022-06-30", "2022-09-30"):
            scores = [1., 2., 3., 4.]
            if date in ("2022-01-31", "2022-02-28"):
                scores = scores[::-1]  # Must not trigger a mid-quarter trade.
            if date == "2022-03-31":
                scores[1] = np.nan
            rows.extend(dict(as_of=pd.Timestamp(date), ticker=ticker, score=score, in_universe=True)
                        for ticker, score in zip(tickers, scores))
        signals = SimpleNamespace(signals=pd.DataFrame(rows), universe=SimpleNamespace(proxy=None),
            companies=pd.DataFrame(), database=None, coverage=pd.DataFrame(), errors=[])
        dates = pd.bdate_range("2022-01-03", "2022-10-04")
        histories = {ticker: pd.DataFrame({"Open":100., "Close":100., "Stock Splits":0.},index=dates)
                     for ticker in (*tickers, "SPY")}
        config = WeightedBacktestConfig(start_date="2022-01-01", end_date="2022-10-04",
            transaction_cost_rate=0., borrow_fee_rate=0., rebalance_frequency="quarterly")
        construction = ConstructionConfig(group_by=None, min_group_size=2,
            transaction_cost_rate=0., borrow_fee_rate=0.)
        result = WeightedBacktester(config, construction).run(signals, histories=histories)
        self.assertTrue(set(result["events"].execution_date.dt.month).issubset({1, 4, 7, 10}))
        self.assertEqual(set(pd.to_datetime(result["target_weights"].as_of).dt.month), {3, 6, 9, 12})
        self.assertTrue(result["events"].signal_date.dt.month.isin([3, 6, 9, 12]).all())
        self.assertLess(result["daily_shares"].at[pd.Timestamp("2022-01-03"), "B"], 0)
        self.assertEqual(result["daily_shares"].at[pd.Timestamp("2022-04-01"), "B"], 0)
        self.assertLess(result["daily_shares"].at[pd.Timestamp("2022-07-01"), "B"], 0)
        self.assertEqual(result["execution"], "quarterly_next_open_target_weights")
        PerformanceReport.audit(result)


if __name__ == "__main__":
    unittest.main()
