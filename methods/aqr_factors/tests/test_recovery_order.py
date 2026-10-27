"""Offline checks of ticker direction, historical cutoffs and checkpoint retention."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
import uuid

import pandas as pd

from methods.aqr_factors.database import AQRDatabase
from methods.aqr_factors.fundamentals import FundamentalStore
from methods.aqr_factors.recovery import FundamentalRecovery
from methods.aqr_factors.recovery_order import apply_forward_order,export_frontiers,forward_priority


class RecoveryOrderTests(unittest.TestCase):
    def test_later_comparative_cannot_erase_2021_publication(self):
        database=AQRDatabase("aqr_research/data/vintage-test-"+uuid.uuid4().hex+".sqlite3")
        try:
            with database.connection() as sql:
                for identifier,available,observed in [
                    ("original_2021","2021-02-16","2026-10-25T01:00:00+00:00"),
                    ("comparative_2022","2022-02-16","2026-10-25T02:00:00+00:00")]:
                    sql.execute("INSERT INTO fundamental_reports VALUES(?,?,?,?,?,?,?,?,?,?)",
                        (identifier,"TEST","income","yearly","2020-12-31",available,
                         "next_session_after_verified_official_filing","USD",observed,"original-issuer-test"))
            FundamentalStore(database).build_monthly("2021-12-01","2022-02-28",["TEST"])
            linked=database.read("SELECT month_end,report_id FROM monthly_fundamentals WHERE ticker='TEST' AND statement='income' AND frequency='yearly' ORDER BY month_end")
            self.assertEqual(linked.report_id.tolist(),["original_2021","original_2021","comparative_2022"])
        finally:
            for suffix in ("","-wal","-shm"):
                Path(str(database.path)+suffix).unlink(missing_ok=True)

    def test_ticker_order_takes_precedence_over_dates(self):
        jobs=[dict(ticker="ZZZ",dataset="statement_income_yearly",attempts=7,status="retry_pending"),
              dict(ticker="AAA",dataset="statement_income_quarterly",attempts=2,status="complete"),
              dict(ticker="AAA",dataset="statement_income_yearly",attempts=3,status="retry_pending")]
        members=pd.DataFrame([dict(ticker="AAA",first_cutoff="2025-12-31"),
                              dict(ticker="ZZZ",first_cutoff="2021-12-31")])
        apply_forward_order(jobs,members)
        self.assertEqual([(j["ticker"],j["dataset"]) for j in jobs],
            [("AAA","statement_income_yearly"),("AAA","statement_income_quarterly"),("ZZZ","statement_income_yearly")])
        self.assertEqual([j["attempts"] for j in jobs],[3,2,7])
        self.assertEqual(jobs[1]["status"],"complete")
        self.assertEqual(min(jobs,key=forward_priority)["ticker"],"AAA")

    def test_backwards_scope_extension_retains_checkpoint(self):
        worker=FundamentalRecovery.__new__(FundamentalRecovery)
        worker.start_date="2022-01-01"
        old=dict(ticker="ZZZ",dataset="statement_income_yearly",attempts=9,status="complete")
        worker.state=dict(start_date="2023-01-01",jobs=[deepcopy(old)],
            requests_attempted=1331,sources_recovered=2,global_next_request_at=9999999999,official_fallback_at=1234)
        worker._missing_source_jobs=lambda:[dict(ticker="ZZZ",dataset="statement_income_yearly",attempts=0,status="pending"),
            dict(ticker="AAA",dataset="statement_balance_yearly",attempts=0,status="pending")]
        worker._export_queue=lambda:None
        worker._extend_scope()
        self.assertEqual(worker.state["jobs"][0],old)
        self.assertEqual(len(worker.state["jobs"]),2)
        self.assertEqual(worker.state["requests_attempted"],1331)
        self.assertEqual(worker.state["global_next_request_at"],9999999999)
        self.assertEqual(worker.state["official_fallback_at"],1234)
        self.assertEqual(worker.state["scope_extensions"][0]["previous_month_cutoff"],"2021-12-31")
        worker._extend_scope()
        self.assertEqual(len(worker.state["jobs"]),2)

    def test_frontiers_use_2021_cutoff_and_reverse_tickers(self):
        database=AQRDatabase("aqr_research/data/order-test-"+uuid.uuid4().hex+".sqlite3")
        output=database.path.parent/("order-test-"+uuid.uuid4().hex)
        output.mkdir()
        try:
            with database.connection() as sql:
                for ticker in ("AAA","ZZZ"):
                    sql.execute("INSERT INTO universe_members VALUES(?,?,?,?,?,?,?,?,?)",
                        ("iwb_holdings_proxy","2021-12-31",ticker,ticker,ticker,"Test",1,"US","USD"))
                    sql.execute("INSERT INTO monthly_fundamentals VALUES(?,?,?,?,?,?,?,?)",
                        (ticker,"2021-12-31","income","yearly",None,"no_usable_report","test","2026-10-25"))
            jobs=[dict(ticker=t,dataset="statement_income_yearly",status="retry_pending",attempts=4,next_attempt_at=1) for t in ("ZZZ","AAA")]
            plan=export_frontiers(database,output,jobs,"2022-01-01","2026-10-22")
            forward=pd.read_csv(output/"fundamental_yahoo_forward_queue.csv")
            reverse=pd.read_csv(output/"fundamental_official_reverse_queue.csv")
            self.assertEqual(forward.ticker.tolist(),["AAA","ZZZ"])
            self.assertEqual(reverse.ticker.tolist(),["ZZZ","AAA"])
            self.assertTrue(reverse.first_cutoff.eq("2021-12-31").all())
            self.assertTrue(reverse.latest_missing_cutoff.eq("2021-12-31").all())
            self.assertEqual(plan["start_date"],"2022-01-01")
            self.assertFalse(plan["all_complete"])
        finally:
            for path in output.iterdir():
                path.unlink()
            output.rmdir()
            for suffix in ("","-wal","-shm"):
                Path(str(database.path)+suffix).unlink(missing_ok=True)


if __name__=="__main__":
    unittest.main()
