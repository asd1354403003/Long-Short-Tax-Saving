"""Opposite recovery frontiers, built entirely from cached source evidence.

Yahoo endpoints return their supported history, not arbitrary start/end pages.
The forward order selects unresolved tickers A-to-Z; the official-source queue
selects tickers Z-to-A. Dates identify coverage gaps, not the ticker ordering.
Neither queue treats a partial source response as complete historical coverage.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .database import utc_now


SOURCE_ORDER = {
    "statement_income_yearly": 0, "statement_balance_yearly": 1,
    "statement_cashflow_yearly": 2, "statement_income_quarterly": 3,
    "statement_balance_quarterly": 4, "statement_cashflow_quarterly": 5,
    "earnings_dates": 6, "yahoo_info": 7,
    "statement_income_trailing": 8, "statement_cashflow_trailing": 9,
}


def forward_priority(job):
    return (job["ticker"], SOURCE_ORDER.get(job["dataset"], 99))


def apply_forward_order(jobs, members):
    cutoffs = dict(zip(members.ticker, members.first_cutoff))
    for job in jobs:
        job["first_universe_cutoff"] = cutoffs.get(job["ticker"], "9999-12-31")
    jobs.sort(key=forward_priority)


def export_frontiers(database, output, jobs, start_date, end_date):
    """Export explicit work orders; no downloads, strategy or state-file writes."""
    output = Path(output)
    lower = str((pd.Timestamp(start_date).to_period("M")-1).start_time.date())
    members = database.read("""SELECT ticker,MAX(name) company_name,
        MIN(as_of) first_cutoff,MAX(as_of) last_cutoff
        FROM universe_members WHERE provider='iwb_holdings_proxy'
        AND as_of>=? AND as_of<=? GROUP BY ticker""", (lower,end_date))
    apply_forward_order(jobs, members)
    pending = [job for job in jobs if job["status"]!="complete"]
    forward = pd.DataFrame([dict(ticker=j["ticker"],dataset=j["dataset"],
        first_universe_cutoff=j["first_universe_cutoff"],status=j["status"],
        attempts=j["attempts"],next_attempt_at=j["next_attempt_at"]) for j in pending],
        columns=["ticker","dataset","first_universe_cutoff","status","attempts","next_attempt_at"])
    forward.to_csv(output/"fundamental_yahoo_forward_queue.csv",index=False)
    # Match only months in which the security belongs to the historical universe.
    missing_rows = database.read("""SELECT ticker,month_end FROM monthly_fundamentals
        WHERE status='no_usable_report' AND frequency IN ('yearly','quarterly')
        AND month_end>=? AND month_end<=?""", (lower,end_date))
    membership_months = database.read("""SELECT DISTINCT ticker,substr(as_of,1,7) month
        FROM universe_members WHERE provider='iwb_holdings_proxy'
        AND as_of>=? AND as_of<=?""", (lower,end_date))
    missing_rows["month"] = missing_rows.month_end.str[:7]
    matched = missing_rows.merge(membership_months,on=["ticker","month"],how="inner")
    missing = matched.groupby("ticker").agg(first_missing_cutoff=("month_end","min"),
        latest_missing_cutoff=("month_end","max"),missing_statement_months=("month_end","size")).reset_index()
    counts = database.read("SELECT ticker,COUNT(*) raw_statement_reports FROM fundamental_reports GROUP BY ticker")
    reverse = members.merge(missing,how="left",on="ticker").merge(counts,how="left",on="ticker")
    reverse["raw_statement_reports"] = pd.to_numeric(reverse.raw_statement_reports,errors="coerce").fillna(0).astype(int)
    reverse["missing_statement_months"] = pd.to_numeric(reverse.missing_statement_months,errors="coerce").fillna(0).astype(int)
    source_jobs = {}
    for job in pending:
        source_jobs.setdefault(job["ticker"],[]).append(job["dataset"])
    reverse["missing_yahoo_sources"] = reverse.ticker.map(lambda t:"; ".join(source_jobs.get(t,[])))
    # Retain the annual-input/field-applicability review alongside raw statements.
    historical_file = output/"fundamental_historical_gaps.csv"
    historical = pd.read_csv(historical_file) if historical_file.exists() else pd.DataFrame(columns=["ticker"])
    historical_tickers = set(historical.ticker)
    reverse["annual_input_review"] = reverse.ticker.isin(historical_tickers)
    reverse = reverse.loc[(reverse.missing_statement_months>0) | reverse.annual_input_review |
                          reverse.missing_yahoo_sources.ne("")].copy()
    reverse["no_raw_statements"] = reverse.raw_statement_reports.eq(0)
    empty = reverse.no_raw_statements & reverse.latest_missing_cutoff.isna()
    reverse.loc[empty,"first_missing_cutoff"] = reverse.loc[empty,"first_cutoff"]
    reverse.loc[empty,"latest_missing_cutoff"] = reverse.loc[empty,"last_cutoff"]
    reverse["latest_missing_cutoff"] = reverse.latest_missing_cutoff.fillna(reverse.first_cutoff)
    reverse = reverse.sort_values("ticker",ascending=False)
    reverse.to_csv(output/"fundamental_official_reverse_queue.csv",index=False)
    plan = dict(updated_at=utc_now(),start_date=start_date,end_date=end_date,
        yahoo=dict(direction="forward",order="ticker A-to-Z, annual before quarterly statements",
            queue="fundamental_yahoo_forward_queue.csv",pending_sources=len(forward),
            history_limit="Use only history actually returned by Yahoo. A requested start date cannot extend statement history; old unavailable periods remain official-source work."),
        official=dict(direction="reverse",order="ticker Z-to-A; verify all required historical periods for each issuer",
            queue="fundamental_official_reverse_queue.csv",review_tickers=len(reverse),
            requirements="Original issuer/CIK, original annual or quarterly filing, explicit units/currency/period/publication, original document and accession; do not backfill older months from newer publications."),
        shared_database=str(database.path.relative_to(Path(__file__).resolve().parents[2])),
        successful_sources_reused=True,whole_range_verification_required=True,all_complete=False)
    (output/"fundamental_recovery_plan.json").write_text(json.dumps(plan,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    return plan
