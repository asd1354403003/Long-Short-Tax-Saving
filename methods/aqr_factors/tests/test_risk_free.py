"""Benchmark units, chronology and Sharpe convention tests."""
import unittest
import numpy as np
import pandas as pd

from risk import treasury_risk_free, benchmark_returns, excess_sharpe
from reports import PerformanceReport


class HistoricalRiskFreeTests(unittest.TestCase):
    def test_annual_percent_is_accrued_over_calendar_gaps_without_future_quotes(self):
        dates = pd.to_datetime(["2024-01-05", "2024-01-08"])
        quotes = pd.Series([5.0, 50.0, 80.0],
            index=pd.to_datetime(["2024-01-03", "2024-01-04", "2024-01-08"]))
        actual = treasury_risk_free(dates, quotes)
        # First day: Jan 4 interval start uses Jan 3 quote, strictly earlier.
        # Weekend: Jan 5 interval start uses Jan 4 quote; Jan 8 cannot leak back.
        np.testing.assert_allclose(actual, [5 / 100 / 365, (1 + 50 / 100 / 365) ** 3 - 1])

    def test_missing_and_stale_history_raise_instead_of_zero_fill(self):
        dates = pd.to_datetime(["2024-01-10", "2024-01-11"])
        for quote_date in ["2024-01-10", "2023-12-01"]:
            with self.assertRaises(ValueError):
                treasury_risk_free(dates, pd.Series([5.0], index=pd.to_datetime([quote_date])))

    def test_nonconstant_benchmark_uses_excess_standard_deviation(self):
        dates = pd.date_range("2024-01-01", periods=4)
        returns = pd.Series([.001, -.002, .003, .0015], index=dates)
        rf = pd.Series([.0001, .0002, .0004, .0003], index=dates)
        excess = returns - rf
        expected = np.sqrt(252) * excess.mean() / excess.std(ddof=1)
        self.assertAlmostEqual(excess_sharpe(returns, rf), expected)
        self.assertNotAlmostEqual(expected, np.sqrt(252) * excess.mean() / returns.std(ddof=1))

    def test_missing_daily_benchmark_does_not_silently_align_or_zero_fill(self):
        dates = pd.date_range("2024-01-01", periods=3)
        with self.assertRaises(ValueError):
            benchmark_returns(dates, daily_risk_free=pd.Series([.001], index=dates[:1]))

    def test_shared_full_and_annual_reports_use_the_same_benchmark(self):
        dates = pd.to_datetime(["2023-12-29", "2024-01-02", "2024-01-03", "2024-01-04"])
        returns = pd.Series([.001, -.002, .003, .0015], index=dates)
        rf = pd.Series([.0001, .0004, .0001, .0001], index=dates)
        equity = 100 * (1 + returns).cumprod()
        daily = pd.DataFrame(dict(equity=equity, cumulative_pnl=equity - 100,
                                  spy_cumulative_return=equity / 100 - 1))
        result = dict(daily_summary=daily, daily_risk_free=rf)
        comparison = PerformanceReport.compare(result)
        self.assertAlmostEqual(comparison.loc["Strategy", "sharpe"], excess_sharpe(returns, rf))
        annual = PerformanceReport.by_year(result)
        self.assertAlmostEqual(annual.loc[(2024, "Strategy"), "sharpe"],
                               excess_sharpe(returns.iloc[1:], rf.iloc[1:]))
        # An explicitly requested constant rate overrides the cached benchmark.
        zero = PerformanceReport.compare(result, annual_risk_free_rate=0.)
        self.assertAlmostEqual(zero.loc["Strategy", "sharpe"], returns.mean()/returns.std(ddof=1)*np.sqrt(252))


if __name__ == "__main__":
    unittest.main()
