"""Cross-sectional missing values must not borrow from future/outside members."""
from contextlib import contextmanager
from dataclasses import replace
import sqlite3
import unittest

import numpy as np
import pandas as pd

from methods.aqr_factors.signals import FactorSignals, SignalConfig
from methods.aqr_factors.fundamentals import FundamentalStore
from methods.aqr_factors.factors import trailing_four


class MissingFactorTests(unittest.TestCase):
    def example(self, method):
        config = SignalConfig(universe="custom", tickers=("A", "B", "C", "E", "X", "OUT"),
                              start="2023-01-01", end_exclusive="2023-04-01",
                              min_stocks=2, missing_value_method=method)
        class Universe:
            proxy = None
            tickers = config.tickers
            def members(self, date):
                return {"A", "B", "C", "E", "X"}
        signals = FactorSignals.__new__(FactorSignals)
        signals.config = config
        signals._universe = lambda cfg: Universe()
        signals.metadata = {"start":config.start, "requested_end_exclusive":config.end_exclusive}
        signals.companies = pd.DataFrame({"ticker":config.tickers,
            "sector":["Tech"]*4+["Other", "Tech"], "industry":["Example"]*6})
        records = []
        for date, scale in [("2023-01-31", 1), ("2023-02-28", 1000)]:
            for ticker, value in zip(config.tickers, [1,np.nan,3,100,1000,99999]):
                records.append(dict(as_of=pd.Timestamp(date), ticker=ticker, close=10.,
                    status="eligible", input_status="eligible", value=value*scale,
                    quality=2., momentum=.1))
        signals.signals = pd.DataFrame(records)
        return signals

    def test_mean_uses_only_same_month_active_group(self):
        signals = self.example("mean").rescore()
        b = signals.signals.query("ticker == 'B'").iloc[0]
        self.assertAlmostEqual(b.value, (1+3+100)/3)
        self.assertTrue(b.value_imputed)
        self.assertTrue(pd.isna(b.value_raw))

    def test_median_and_rescore_back_to_drop(self):
        signals = self.example("median").rescore()
        self.assertEqual(signals.signals.query("ticker == 'B'").iloc[0].value, 3)
        signals.rescore(replace(signals.config, missing_value_method="drop"))
        b = signals.signals.query("ticker == 'B'")
        self.assertTrue(b.value.isna().all())
        self.assertTrue(b.score.isna().all())
        self.assertEqual(int(signals.coverage.imputed_stocks.sum()), 0)

    def test_no_donors_stays_missing(self):
        signals = self.example("mean")
        signals.signals.loc[signals.signals.ticker.eq("X"), "value"] = np.nan
        signals.rescore()
        self.assertTrue(signals.signals.query("ticker == 'X'").value.isna().all())

    def test_universe_scope_remains_same_month_and_excludes_nonmembers(self):
        signals = self.example("mean")
        signals.rescore(replace(signals.config, missing_value_scope="universe"))
        self.assertEqual(signals.signals.query("ticker == 'B'").iloc[0].value, (1+3+100+1000)/4)

    def test_only_selected_factors_control_drop_and_ranking(self):
        signals = self.example("drop")
        signals.signals["earnings_yield"] = [-.02, .03, .04, .05, .06, .07]*2
        signals.rescore(replace(signals.config, factor_operators={"earnings_yield": ("group_rank",)}))
        b = signals.signals.query("ticker == 'B'").iloc[0]
        a = signals.signals.query("ticker == 'A'").iloc[0]
        self.assertTrue(pd.isna(b.value))  # Inactive value does not cause a drop.
        self.assertTrue(pd.notna(b.score))
        self.assertLess(a.score, b.score)  # Losses do not rank as cheap positive PE.
        with self.assertRaises(ValueError):
            replace(signals.config, factor_operators={"made_up": ("rank",)})

    def test_comparison_cohort_is_applied_before_ranking_and_group_minimum(self):
        signals = self.example("drop")
        mask = ~signals.signals.ticker.isin(["C", "X"])
        signals.rescore(eligibility_mask=mask)
        # Only A/E are complete, active Tech rows; OUT must not become a donor.
        month = signals.signals.loc[signals.signals.as_of.eq(pd.Timestamp("2023-01-31"))].set_index("ticker")
        self.assertEqual(month.at["A", "value_score"], -1.)
        self.assertEqual(month.at["E", "value_score"], 1.)
        self.assertTrue(pd.isna(month.at["C", "score"]))
        self.assertEqual(signals.coverage.ranked_stocks.iloc[0], 2)

    def test_missing_in_one_month_does_not_remove_later_complete_month(self):
        signals = self.example("drop")
        complete_later = signals.signals.ticker.eq("B") & signals.signals.as_of.eq(pd.Timestamp("2023-02-28"))
        signals.signals.loc[complete_later, "value"] = 2000.
        signals.rescore()
        b = signals.signals.loc[signals.signals.ticker.eq("B")].set_index("as_of")
        self.assertTrue(pd.isna(b.loc[pd.Timestamp("2023-01-31"), "score"]))
        self.assertTrue(pd.notna(b.loc[pd.Timestamp("2023-02-28"), "score"]))

    def balance_example(self, method="drop"):
        signals = self.example(method)
        signals.signals["book_equity"] = signals.signals.ticker.map({"A":-20., "B":10., "C":30., "E":20., "X":100., "OUT":500.})
        signals.signals["total_assets"] = 100.
        signals.signals["gross_profit"] = 10.
        signals.signals["market_cap"] = 100.
        # Simulate old exports where negative equity suppressed both factors.
        signals.signals.loc[signals.signals.ticker.eq("A"), ["value", "quality"]] = np.nan
        signals.signals.loc[signals.signals.ticker.eq("A"), "input_status"] = "nonpositive_equity_or_assets"
        return signals

    def test_negative_equity_is_finite_and_policy_toggle_is_reversible(self):
        signals = self.balance_example().rescore()
        a = signals.signals.loc[signals.signals.ticker.eq("A")].iloc[0]
        self.assertAlmostEqual(a.value, -.2)
        self.assertAlmostEqual(a.quality, .1)
        self.assertTrue(pd.notna(a.score))
        signals.rescore(replace(signals.config, require_positive_book_equity=True))
        a = signals.signals.loc[signals.signals.ticker.eq("A")].iloc[0]
        self.assertTrue(pd.isna(a.score))
        self.assertEqual(a.value_raw, -.2)
        signals.rescore(replace(signals.config, require_positive_book_equity=False))
        self.assertTrue(signals.signals.loc[signals.signals.ticker.eq("A"), "score"].notna().all())

    def test_book_policy_does_not_exclude_quality_only_or_get_mean_imputed(self):
        signals = self.balance_example("mean")
        signals.rescore(replace(signals.config, require_positive_book_equity=True))
        self.assertTrue(signals.signals.loc[signals.signals.ticker.eq("A"), "score"].isna().all())
        signals.rescore(replace(signals.config, factor_operators={"quality": ("group_rank",)}, require_positive_book_equity=True))
        self.assertTrue(signals.signals.loc[signals.signals.ticker.eq("A"), "score"].notna().all())

    def test_invalid_assets_only_block_ratios_that_use_assets(self):
        signals = self.balance_example("mean")
        signals.signals.loc[signals.signals.ticker.eq("B"), "total_assets"] = 0.
        signals.rescore()
        b = signals.signals.loc[signals.signals.ticker.eq("B")].iloc[0]
        self.assertTrue(pd.isna(b.score))
        self.assertEqual(b.status, "invalid_total_assets")
        signals.rescore(replace(signals.config, factor_operators={"value": ("group_rank",)}))
        self.assertTrue(signals.signals.loc[signals.signals.ticker.eq("B"), "score"].notna().all())


class TrailingFourTests(unittest.TestCase):
    def test_missing_quarter_value_or_gap_cannot_be_skipped(self):
        dates = pd.to_datetime(["2024-03-31", "2024-06-30", "2024-09-30", "2024-12-31"])
        known = {d: {"report_id": str(i)} for i, d in enumerate(dates)}
        values = {str(i): {"NetIncome": float(i+1)} for i in range(4)}
        self.assertEqual(trailing_four(known, values, "NetIncome")[0], 10.)
        self.assertIsNone(trailing_four(dict(list(known.items())[:3]), values, "NetIncome")[0])
        values["2"]["NetIncome"] = None
        self.assertIsNone(trailing_four(known, values, "NetIncome")[0])
        values["2"]["NetIncome"] = 3.
        del known[dates[1]]
        known[pd.Timestamp("2023-12-31")] = {"report_id": "0"}
        self.assertIsNone(trailing_four(known, values, "NetIncome")[0])

    def test_fourth_quarter_is_unavailable_until_its_reference_date(self):
        connection = sqlite3.connect(":memory:")
        connection.executescript("""
        CREATE TABLE fundamental_reports(report_id,ticker,statement,frequency,period_end,
            available_from,observed_at,financial_currency);
        CREATE TABLE yahoo_csv_report_provenance(report_id,reference_available_from,reference_availability_method);
        CREATE TABLE yahoo_csv_periods(report_id,alignment_status);
        CREATE TABLE fundamental_facts(report_id,metric,value);
        """)
        periods = ["2024-03-31", "2024-06-30", "2024-09-30", "2024-12-31"]
        available = ["2024-05-01", "2024-08-01", "2024-11-01", "2025-02-01"]
        for index, (period, date) in enumerate(zip(periods, available)):
            connection.execute("INSERT INTO fundamental_reports VALUES(?, 'A','income','quarterly',?,NULL,'2026-10-26','USD')", (str(index),period))
            connection.execute("INSERT INTO yahoo_csv_report_provenance VALUES(?,?,'existing_proxy')", (str(index),date))
            connection.execute("INSERT INTO yahoo_csv_periods VALUES(?,'verified_source')", (str(index),))
            connection.execute("INSERT INTO fundamental_facts VALUES(?,'NetIncomeCommonStockholders',?)", (str(index), -float(index+1)))
        class Database:
            def read(self, query):
                return pd.read_sql_query(query, connection)
            @contextmanager
            def connection(self):
                yield connection
        events, _, _ = FundamentalStore(Database()).website_quarterly_events()
        self.assertTrue(all(e["common_net_income_ttm"] is None for e in events["A"][:-1]))
        self.assertEqual(events["A"][-1]["common_net_income_ttm"], -10.)
        self.assertEqual(events["A"][-1]["available_from"], pd.Timestamp("2025-02-01"))
        connection.close()


class WebsiteDateTests(unittest.TestCase):
    def test_unknown_date_is_excluded_unless_lag_is_explicit(self):
        connection = sqlite3.connect(":memory:")
        connection.executescript("""
        CREATE TABLE fundamental_reports(report_id,ticker,statement,frequency,period_end,
            available_from,observed_at,financial_currency);
        CREATE TABLE yahoo_csv_report_provenance(report_id,reference_available_from,reference_availability_method);
        CREATE TABLE yahoo_csv_periods(report_id,alignment_status);
        CREATE TABLE fundamental_facts(report_id,metric,value);
        INSERT INTO fundamental_reports VALUES('known','A','balance','quarterly','2021-09-30',NULL,'2026-10-26','USD');
        INSERT INTO fundamental_reports VALUES('unknown','A','income','quarterly','2021-09-30',NULL,'2026-10-26','USD');
        INSERT INTO yahoo_csv_report_provenance VALUES('known','2021-11-01','existing_proxy');
        INSERT INTO yahoo_csv_report_provenance VALUES('unknown',NULL,NULL);
        INSERT INTO yahoo_csv_periods VALUES('known','verified_source');
        INSERT INTO yahoo_csv_periods VALUES('unknown','verified_source');
        INSERT INTO fundamental_facts VALUES('known','CommonStockEquity',50);
        INSERT INTO fundamental_facts VALUES('known','TotalAssets',100);
        INSERT INTO fundamental_facts VALUES('unknown','GrossProfit',20);
        """)
        class Database:
            def read(self, query):
                return pd.read_sql_query(query, connection)
            @contextmanager
            def connection(self):
                yield connection
        store = FundamentalStore(Database())
        events, _, meta = store.website_quarterly_events()
        self.assertIsNone(events['A'][0]['gross_profit'])
        self.assertEqual(meta['excluded_unknown_date_versions'], 1)
        events, _, meta = store.website_quarterly_events(availability="period_lag", lag_days=90)
        self.assertEqual(events['A'][-1]['gross_profit'], 20)
        self.assertEqual(events['A'][-1]['available_from'], pd.Timestamp('2021-12-29'))
        self.assertIsNone(connection.execute("SELECT available_from FROM fundamental_reports WHERE report_id='unknown'").fetchone()[0])
        connection.close()


if __name__ == "__main__":
    unittest.main()
