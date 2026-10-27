"""Resumable, paced Yahoo source recovery; official-source handoff after a deadline.

Run from the project folder: python -m methods.aqr_factors.recovery --worker
This prepares source records only. It never runs or changes the strategy.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import logging
import math
import os
from pathlib import Path
import time

import pandas as pd
import yfinance as yf
from yfinance.config import YfConfig
from yfinance.data import YfData

from .database import AQRDatabase, utc_now
from .fundamentals import FundamentalStore, SERIES, STATEMENTS, validate_yahoo_info
from .universe import Universe
from .recovery_order import forward_priority, export_frontiers


class FundamentalRecovery:
    """One process owns the queue; each failed source has its own retry time."""

    def __init__(self, settings, *, result_dir="methods/aqr_factors/research/results"):
        self.settings = settings
        self.start_date = getattr(settings,"recovery_start_date",settings.start_date)
        self.database = AQRDatabase(settings.database_path)
        self.store = FundamentalStore(self.database)
        self.root = Path(__file__).resolve().parents[2]
        relative = Path(result_dir)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Use a project-relative recovery results directory")
        self.output = self.root / relative
        self.output.mkdir(parents=True, exist_ok=True)
        self.path = self.output / "fundamental_recovery_state.json"
        self.universe = Universe(settings.universe, settings.tickers, settings.membership_file,
            database=self.database, start_date=self.start_date, end_date=settings.end_date)
        self.minimum_interval = settings.recovery_request_interval_seconds
        self.rate_limit_cooldown = settings.recovery_rate_limit_cooldown_seconds
        self.fallback_hours = settings.recovery_official_fallback_after_hours
        self.state = self._read() if self.path.exists() else self._initialize()
        if self.state["end_date"] != settings.end_date or self.start_date > self.state["start_date"]:
            raise ValueError("Recovery cannot shrink or change its end date without preserving its queue")

    def _read(self):
        return json.loads(self.path.read_text(encoding="utf-8"))

    def _save(self):
        self.state["updated_at"] = utc_now()
        temp = self.path.with_suffix(".tmp")
        temp.write_text(json.dumps(self.state, ensure_ascii=False, indent=2), encoding="utf-8")
        temp.replace(self.path)

    def _missing_source_jobs(self):
        now = time.time()
        snapshots = self.database.read("""SELECT dataset,ticker,verified_through,observed_at,
            CASE WHEN dataset='yahoo_info' THEN payload END payload
            FROM snapshots WHERE dataset='yahoo_info' OR dataset='earnings_dates'
            OR dataset LIKE 'statement_%' ORDER BY observed_at""")
        latest = {(row.ticker, row.dataset): row for row in
                  snapshots.drop_duplicates(["ticker", "dataset"], keep="last").itertuples()}
        records = self.database.read("SELECT DISTINCT ticker,statement,frequency FROM fundamental_reports")
        ingested = set(records.itertuples(index=False, name=None))
        newest = self.database.read("""SELECT ticker,weight,name FROM universe_members
            WHERE provider='iwb_holdings_proxy' AND as_of=(SELECT MAX(as_of) FROM universe_members
            WHERE provider='iwb_holdings_proxy' AND as_of<=?)""", (self.settings.end_date,))
        weights = dict(zip(newest.ticker, pd.to_numeric(newest.weight, errors="coerce").fillna(0)))
        names = self.database.read("SELECT ticker,name FROM universe_members ORDER BY as_of").drop_duplicates("ticker",keep="last")
        names = dict(zip(names.ticker,names.name))
        datasets = ["statement_income_yearly", "statement_balance_yearly", "earnings_dates", "yahoo_info"]
        datasets += ["statement_{}_{}".format(*key) for key in SERIES if "statement_{}_{}".format(*key) not in datasets]
        jobs = []
        for symbol in self.universe.tickers:
            for priority, dataset in enumerate(datasets):
                saved = latest.get((symbol, dataset))
                missing = saved is None or saved.verified_through < self.settings.end_date
                if dataset == "yahoo_info" and saved is not None:
                    try:
                        validate_yahoo_info(json.loads(saved.payload), ("country","currency","financialCurrency"))
                    except ValueError:
                        missing = True
                if dataset.startswith("statement_"):
                    _, statement, frequency = dataset.split("_",2)
                    missing |= (symbol,statement,frequency) not in ingested
                if missing:
                    jobs.append(dict(ticker=symbol,name=names.get(symbol,symbol),dataset=dataset,
                        priority=[0 if symbol in weights else 1,-float(weights.get(symbol,0)),priority,symbol],
                        attempts=0,status="pending",next_attempt_at=now,last_error=None))
        jobs.sort(key=lambda job:job["priority"])
        return jobs

    def _initialize(self):
        now = time.time()
        jobs = self._missing_source_jobs()
        state = dict(version=1,requested_at=utc_now(),start_date=self.start_date,
            end_date=self.settings.end_date,phase="yahoo_retry",jobs=jobs,
            initial_target_tickers=len({job["ticker"] for job in jobs}),
            official_fallback_at=now+self.fallback_hours*3600,
            global_next_request_at=now+self.settings.recovery_initial_cooldown_seconds,
            rate_limit_streak=0,requests_attempted=0,sources_recovered=0,
            automation_id="aqr",worker_pid=None,all_complete=False)
        self.state = state
        self._export_queue()
        self._save()
        return state

    def _calendar(self, start, end):
        # The historical NYSE calendar is already cached. No extra Yahoo
        # request is hidden inside availability-date attribution.
        return self.database.price_frame("__NYSE_CALENDAR__",start,end).index

    def _extend_scope(self):
        """Extend backwards without resetting attempts, cooldowns or successes."""
        if self.start_date >= self.state["start_date"]:
            return
        previous_start = self.state["start_date"]
        existing = {(job["ticker"],job["dataset"]) for job in self.state["jobs"]}
        new_jobs = [job for job in self._missing_source_jobs()
            if (job["ticker"],job["dataset"]) not in existing]
        self.state["jobs"].extend(new_jobs)
        self.state["start_date"] = self.start_date
        self.state.setdefault("scope_extensions",[]).append(dict(
            previous_start=previous_start,new_start=self.start_date,
            previous_month_cutoff=str((pd.Timestamp(self.start_date)-pd.Timedelta(days=1)).date()),
            new_source_jobs=len(new_jobs),extended_at=utc_now()))
        self._export_queue()

    def _synchronize(self, symbol):
        info = self.database.latest_snapshot("yahoo_info",symbol)
        currency = (info or {}).get("payload",{}).get("financialCurrency")
        earnings = self.database.latest_snapshot("earnings_dates",symbol)
        releases = (earnings or {}).get("payload",[])
        frames = {}
        for statement, frequency in SERIES:
            saved = self.database.latest_snapshot("statement_{}_{}".format(statement,frequency),symbol)
            if saved is None:
                continue
            self.store.ingest(symbol,statement,frequency,saved["payload"],releases,self._calendar,
                currency,max(saved["observed_at"],(earnings or {}).get("observed_at",saved["observed_at"])))
            if frequency == "yearly":
                frames[statement] = {row["period_end"]:row["values"] for row in saved["payload"]}
        annual = []
        for period in sorted(set(frames.get("income",{})) & set(frames.get("balance",{}))):
            income,balance = frames["income"][period],frames["balance"][period]
            equity,assets = balance.get("CommonStockEquity"),balance.get("TotalAssets")
            profit = income.get("GrossProfit")
            if profit is None and income.get("TotalRevenue") is not None and income.get("CostOfRevenue") is not None:
                profit = income["TotalRevenue"]-income["CostOfRevenue"]
            if any(value is None or not math.isfinite(value) for value in (equity,assets,profit)):
                continue
            dates = sorted(pd.Timestamp(row["date"]) for row in releases
                if pd.Timestamp(period) < pd.Timestamp(row["date"]).tz_localize(None).normalize()
                <= pd.Timestamp(period)+pd.Timedelta(days=120))
            release = dates[0] if dates else None
            available = None
            if release is not None:
                day = release.tz_localize(None).normalize()
                calendar = self._calendar(str(day.date()),str(pd.Timestamp.now(tz="America/New_York").date()))
                later = calendar[calendar>day]
                available = str(later[0].date()) if len(later) else None
            annual.append(dict(ticker=symbol,period_end=period,
                earnings_proxy_at=release.isoformat() if release is not None else None,
                available_from=available,availability_method="next_session_after_annual_earnings_proxy",
                book_equity=equity,total_assets=assets,gross_profit=profit))
        prior = self.database.latest_snapshot("financials",symbol)
        if annual and (prior is None or prior["payload"] != annual):
            self.database.put_snapshot("financials",symbol,annual,self.settings.end_date)
        monthly_start = str((pd.Timestamp(self.start_date).to_period("M")-1).start_time.date())
        self.store.build_monthly(monthly_start,self.settings.end_date,[symbol])

    def _fetch(self, job):
        symbol,dataset = job["ticker"],job["dataset"]
        ticker = yf.Ticker(symbol)
        self.store.stats.clear()
        if dataset == "yahoo_info":
            saved = self.database.latest_snapshot(dataset,symbol)
            complete = False
            if saved:
                try:
                    validate_yahoo_info(saved["payload"],("country","currency","financialCurrency"))
                    complete = True
                except ValueError:
                    pass
            source = self.store._source(dataset,symbol,self.settings.end_date,ticker.get_info,
                lambda info:validate_yahoo_info(info,("country","currency","financialCurrency")),force=not complete)
            profile = {name:source["payload"].get(name) for name in
                ("shortName","industry","sector","country","currency","financialCurrency")}
            self.database.put_snapshot("company",symbol,profile,source["verified_through"])
        elif dataset == "earnings_dates":
            self.store._source(dataset,symbol,self.settings.end_date,
                lambda:ticker.get_earnings_dates(limit=100),self.store._encode_earnings)
        else:
            _,statement,frequency = dataset.split("_",2)
            self.store._source(dataset,symbol,self.settings.end_date,
                lambda:getattr(ticker,STATEMENTS[statement])(pretty=False,freq=frequency),
                self.store._encode_statement)
        self._synchronize(symbol)

    def _export_queue(self):
        pending = [job for job in self.state["jobs"] if job["status"] != "complete"]
        rows=[]
        for symbol in sorted({job["ticker"] for job in pending}):
            jobs = [job for job in pending if job["ticker"]==symbol]
            rows.append(dict(ticker=symbol,company_name=jobs[0]["name"],
                missing_sources="; ".join(job["dataset"] for job in jobs),
                attempts=sum(job["attempts"] for job in jobs),
                latest_error=next((job["last_error"] for job in reversed(jobs) if job["last_error"]),None),
                next_attempt_at=pd.Timestamp(min(job["next_attempt_at"] for job in jobs),unit="s",tz="UTC").isoformat(),
                official_fallback_due=time.time()>=self.state["official_fallback_at"]))
        pd.DataFrame(rows,columns=["ticker","company_name","missing_sources","attempts","latest_error","next_attempt_at","official_fallback_due"]).to_csv(
            self.output/"fundamental_research_queue.csv",index=False)
        self.state["remaining_source_tickers"] = len(rows)
        self.state["remaining_sources"] = len(pending)
        historical = self._historical_gaps()
        historical.to_csv(self.output/"fundamental_historical_gaps.csv",index=False)
        self.state["historical_review_tickers"] = len(historical)
        self.state["remaining_fundamental_tickers"] = len(
            {job["ticker"] for job in pending} | set(historical.ticker))

    def _historical_gaps(self):
        lower = str((pd.Timestamp(self.start_date).to_period("M")-1).start_time.date())
        needed = self.database.read("""SELECT ticker,MIN(as_of) cutoff,MAX(name) name
            FROM universe_members WHERE provider='iwb_holdings_proxy' AND as_of>=? AND as_of<=?
            GROUP BY ticker""",(lower,self.settings.end_date)).set_index("ticker")
        saved = self.database.read("SELECT ticker,payload FROM snapshots WHERE dataset='financials' ORDER BY observed_at")
        dates = {}
        for row in saved.itertuples():
            valid = [item["available_from"] for item in json.loads(row.payload) if item.get("available_from")]
            if valid:
                dates[row.ticker] = min(dates.get(row.ticker,min(valid)),min(valid))
        rows=[]
        for symbol in self.universe.tickers:
            cutoff = str(needed.loc[symbol,"cutoff"]) if symbol in needed.index else str((pd.Timestamp(lower)+pd.offsets.MonthEnd(0)).date())
            first = dates.get(symbol)
            if first is None or first > cutoff:
                rows.append(dict(ticker=symbol,company_name=needed.loc[symbol,"name"] if symbol in needed.index else symbol,
                    first_universe_cutoff=cutoff,first_available_annual_inputs=first,
                    reason="no_annual_inputs_at_first_historical_cutoff; verify filing coverage or field applicability"))
        return pd.DataFrame(rows,columns=["ticker","company_name","first_universe_cutoff","first_available_annual_inputs","reason"])

    def step(self, *, now=None):
        now = time.time() if now is None else now
        if now >= self.state["official_fallback_at"]:
            self.state["phase"] = "official_fallback_due"
        jobs = [job for job in self.state["jobs"] if job["status"] != "complete"]
        if not jobs:
            self._export_queue()
            complete = self.state["historical_review_tickers"] == 0
            self.state.update(phase="source_queue_complete" if complete else
                ("official_fallback_due" if now>=self.state["official_fallback_at"] else "awaiting_official_fallback"),
                all_complete=complete)
            self._save()
            return 60
        ready = [job for job in jobs if job["next_attempt_at"] <= now]
        if now < self.state["global_next_request_at"] or not ready:
            due = max(self.state["global_next_request_at"],min(job["next_attempt_at"] for job in jobs))
            self._export_queue(); self._save()
            return max(1,min(60,due-now))
        job = min(ready,key=forward_priority)
        job["attempts"] += 1
        self.state["requests_attempted"] += 1
        # A fresh object plus a cleared HTTP response cache makes this a real
        # retry; successful source snapshots in SQLite remain reusable.
        YfData().cache_get.cache_clear()
        old_debug = YfConfig.debug.hide_exceptions
        try:
            YfConfig.debug.hide_exceptions = False
            self._fetch(job)
            job.update(status="complete",last_error=None,completed_at=utc_now())
            self.state["sources_recovered"] += 1
            self.state["rate_limit_streak"] = 0
            self.state["global_next_request_at"] = now+self.minimum_interval
        except Exception as exc:
            message = f"{type(exc).__name__}: {exc}"
            job.update(status="retry_pending",last_error=message,last_attempt_at=utc_now())
            limited = "rate limit" in message.lower() or "429" in message
            if limited:
                self.state["rate_limit_streak"] += 1
                cooldown = min(3600,self.rate_limit_cooldown*2**min(self.state["rate_limit_streak"]-1,5))
                self.state["global_next_request_at"] = now+cooldown
                job["next_attempt_at"] = now+cooldown
            else:
                self.state["global_next_request_at"] = now+self.minimum_interval
                job["next_attempt_at"] = now+min(21600,900*2**min(job["attempts"]-1,5))
            logging.warning("%s %s: %s",job["ticker"],job["dataset"],message)
        finally:
            YfConfig.debug.hide_exceptions = old_debug
        self._export_queue(); self._save()
        return max(1,min(60,self.state["global_next_request_at"]-time.time()))

    @contextmanager
    def _lease(self):
        import msvcrt
        path = self.output/"fundamental_recovery.lock"
        with path.open("a+b") as handle:
            if handle.tell()==0:
                handle.write(b"0"); handle.flush()
            handle.seek(0)
            try:
                msvcrt.locking(handle.fileno(),msvcrt.LK_NBLCK,1)
            except OSError as exc:
                raise RuntimeError("A fundamental recovery worker is already running") from exc
            try:
                yield
            finally:
                handle.seek(0)
                msvcrt.locking(handle.fileno(),msvcrt.LK_UNLCK,1)

    def run(self):
        with self._lease():
            # Reload after taking the lease, so another completed owner cannot
            # have its progress replaced by an earlier in-memory state.
            self.state = self._read()
            self._extend_scope()
            export_frontiers(self.database,self.output,self.state["jobs"],
                self.start_date,self.settings.end_date)
            self.state["recovery_order"] = "ticker_A_to_Z"
            self.state.update(worker_pid=os.getpid(),worker_started_at=utc_now(),worker_status="running")
            self._save()
            try:
                while not self.state["all_complete"]:
                    time.sleep(self.step())
            finally:
                self.state.update(worker_status="stopped",worker_pid=None)
                self._save()

    def status(self):
        self.state = self._read()
        return {key:value for key,value in self.state.items() if key != "jobs"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker",action="store_true")
    parser.add_argument("--plan",action="store_true",help="Refresh cached A-to-Z and Z-to-A work queues only")
    args = parser.parse_args()
    from .settings import AQRSettings
    worker = FundamentalRecovery(AQRSettings)
    logging.basicConfig(filename=worker.output/"fundamental_recovery_worker.log",level=logging.WARNING,encoding="utf-8")
    if args.plan:
        print(json.dumps(export_frontiers(worker.database,worker.output,worker.state["jobs"],
            worker.start_date,worker.settings.end_date),ensure_ascii=False,indent=2))
    elif args.worker:
        worker.run()
    else:
        print(json.dumps(worker.status(),ensure_ascii=False,indent=2))


if __name__ == "__main__":
    main()
