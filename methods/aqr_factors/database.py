"""Method-local SQLite storage for AQR source data, membership and calculations."""
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import json
import sqlite3

import pandas as pd


def utc_now():
    return pd.Timestamp.now(tz="UTC").isoformat()


class AQRDatabase:
    def __init__(self, path="aqr_research/data/aqr.sqlite3"):
        relative = Path(path)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Use a project-relative AQR database path")
        self.path = Path(__file__).resolve().parents[2] / relative
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connection() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.executescript("""
                CREATE TABLE IF NOT EXISTS prices (
                    ticker TEXT, date TEXT, open REAL, high REAL, low REAL,
                    close REAL, volume REAL, dividend REAL, split REAL,
                    observed_at TEXT, PRIMARY KEY(ticker,date));
                CREATE TABLE IF NOT EXISTS shares (
                    ticker TEXT, date TEXT, shares REAL, observed_at TEXT,
                    PRIMARY KEY(ticker,date));
                CREATE TABLE IF NOT EXISTS splits (
                    ticker TEXT, date TEXT, ratio REAL, observed_at TEXT,
                    PRIMARY KEY(ticker,date));
                CREATE TABLE IF NOT EXISTS coverage (
                    dataset TEXT, ticker TEXT, start_date TEXT, end_exclusive TEXT,
                    observed_at TEXT, PRIMARY KEY(dataset,ticker,start_date,end_exclusive));
                CREATE TABLE IF NOT EXISTS snapshots (
                    dataset TEXT, ticker TEXT, observed_at TEXT, verified_through TEXT,
                    payload TEXT, PRIMARY KEY(dataset,ticker,observed_at));
                CREATE TABLE IF NOT EXISTS monthly_calculations (
                    config_key TEXT, month TEXT, payload TEXT, observed_at TEXT,
                    PRIMARY KEY(config_key,month));
                CREATE TABLE IF NOT EXISTS universe_snapshots (
                    provider TEXT, as_of TEXT, requested_as_of TEXT, source_url TEXT,
                    source_sha256 TEXT, observed_at TEXT, equity_count INTEGER,
                    total_rows INTEGER, publication_time_verified INTEGER DEFAULT 0,
                    PRIMARY KEY(provider,as_of));
                CREATE TABLE IF NOT EXISTS universe_members (
                    provider TEXT, as_of TEXT, ticker TEXT, provider_ticker TEXT,
                    name TEXT, sector TEXT, weight REAL, location TEXT, currency TEXT,
                    PRIMARY KEY(provider,as_of,ticker));
                CREATE TABLE IF NOT EXISTS monthly_universe (
                    provider TEXT, trade_month TEXT, decision_cutoff TEXT,
                    snapshot_as_of TEXT, publication_assumption TEXT,
                    PRIMARY KEY(provider,trade_month));
                CREATE TABLE IF NOT EXISTS source_documents (
                    sha256 TEXT PRIMARY KEY, source_url TEXT, observed_at TEXT,
                    body_zlib BLOB);
                CREATE TABLE IF NOT EXISTS fundamental_reports (
                    report_id TEXT PRIMARY KEY, ticker TEXT, statement TEXT,
                    frequency TEXT, period_end TEXT, available_from TEXT,
                    availability_method TEXT, financial_currency TEXT,
                    observed_at TEXT, source TEXT);
                CREATE INDEX IF NOT EXISTS fundamental_report_lookup
                    ON fundamental_reports(ticker,statement,frequency,period_end,observed_at);
                CREATE TABLE IF NOT EXISTS fundamental_facts (
                    report_id TEXT, metric TEXT, value REAL,
                    PRIMARY KEY(report_id,metric),
                    FOREIGN KEY(report_id) REFERENCES fundamental_reports(report_id));
                CREATE TABLE IF NOT EXISTS monthly_fundamentals (
                    ticker TEXT, month_end TEXT, statement TEXT, frequency TEXT,
                    report_id TEXT, status TEXT, version_policy TEXT, calculated_at TEXT,
                    PRIMARY KEY(ticker,month_end,statement,frequency),
                    FOREIGN KEY(report_id) REFERENCES fundamental_reports(report_id));
                CREATE VIEW IF NOT EXISTS monthly_fundamental_values AS
                    SELECT m.ticker,m.month_end,m.statement,m.frequency,m.status,
                        m.version_policy,r.period_end,r.available_from,
                        r.availability_method,r.financial_currency,r.observed_at,
                        r.source,f.metric,f.value
                    FROM monthly_fundamentals m
                    LEFT JOIN fundamental_reports r ON r.report_id=m.report_id
                    LEFT JOIN fundamental_facts f ON f.report_id=m.report_id;
            """)

    @contextmanager
    def connection(self):
        db = sqlite3.connect(self.path, timeout=30)
        db.execute("PRAGMA busy_timeout=30000")
        db.execute("PRAGMA foreign_keys=ON")
        try:
            with db:
                yield db
        finally:
            db.close()

    def read(self, sql, parameters=()):
        with self.connection() as db:
            return pd.read_sql_query(sql, db, params=parameters)

    def missing_ranges(self, dataset, ticker, start, end_exclusive):
        start, end = pd.Timestamp(start), pd.Timestamp(end_exclusive)
        ranges = self.read(
            "SELECT start_date,end_exclusive FROM coverage WHERE dataset=? AND ticker=? ORDER BY start_date",
            (dataset, ticker),
        )
        cursor, missing = start, []
        for lower, upper in ranges.itertuples(index=False, name=None):
            lower, upper = pd.Timestamp(lower), pd.Timestamp(upper)
            if upper <= cursor or lower >= end:
                continue
            if lower > cursor:
                missing.append((str(cursor.date()), str(min(lower, end).date())))
            cursor = max(cursor, upper)
            if cursor >= end:
                break
        if cursor < end:
            missing.append((str(cursor.date()), str(end.date())))
        return missing

    def store_range(self, dataset, ticker, start, end_exclusive, frame):
        """Commit rows and successful request coverage together; failures call neither."""
        observed = utc_now()
        with self.connection() as db:
            if dataset == "prices":
                rows = []
                for date, row in frame.iterrows():
                    values = [row.get(name) for name in ("Open", "High", "Low", "Close", "Volume", "Dividends", "Stock Splits")]
                    values = [None if pd.isna(value) else float(value) for value in values]
                    rows.append((ticker, str(date.date()), *values, observed))
                db.executemany("INSERT OR REPLACE INTO prices VALUES(?,?,?,?,?,?,?,?,?,?)", rows)
            elif dataset == "shares":
                db.executemany("INSERT OR REPLACE INTO shares VALUES(?,?,?,?)", [
                    (ticker, str(date.date()), float(value), observed) for date, value in frame.items()
                ])
            else:
                raise ValueError("Unsupported range dataset: " + dataset)
            db.execute("INSERT OR REPLACE INTO coverage VALUES(?,?,?,?,?)", (
                dataset, ticker, start, end_exclusive, observed,
            ))

    def price_frame(self, ticker, start, end_exclusive):
        frame = self.read("SELECT * FROM prices WHERE ticker=? AND date>=? AND date<? ORDER BY date", (ticker, start, end_exclusive))
        if frame.empty:
            return pd.DataFrame()
        frame["date"] = pd.to_datetime(frame["date"])
        return frame.set_index("date").rename(columns={
            "open": "Open", "high": "High", "low": "Low", "close": "Close",
            "volume": "Volume", "dividend": "Dividends", "split": "Stock Splits",
        }).drop(columns=["ticker", "observed_at"])

    def share_series(self, ticker, start, end_exclusive):
        # Keep one earlier record for an as-of join at the left boundary.
        frame = self.read("""SELECT date,shares FROM shares WHERE ticker=? AND date<?
            AND (date>=? OR date=(SELECT MAX(date) FROM shares WHERE ticker=? AND date<?)) ORDER BY date""",
            (ticker, end_exclusive, start, ticker, start))
        return pd.Series(frame["shares"].to_numpy(), index=pd.to_datetime(frame["date"]), dtype=float)

    def put_snapshot(self, dataset, ticker, payload, verified_through, *, observed_at=None):
        with self.connection() as db:
            db.execute("INSERT OR REPLACE INTO snapshots VALUES(?,?,?,?,?)", (
                dataset, ticker, observed_at or utc_now(), verified_through,
                json.dumps(payload, ensure_ascii=False, allow_nan=False),
            ))

    def latest_snapshot(self, dataset, ticker):
        rows = self.read("SELECT * FROM snapshots WHERE dataset=? AND ticker=? ORDER BY observed_at DESC LIMIT 1", (dataset, ticker))
        if rows.empty:
            return None
        row = rows.iloc[0].to_dict()
        row["payload"] = json.loads(row["payload"])
        return row

    def all_financials(self, ticker):
        snapshots = self.read("SELECT payload FROM snapshots WHERE dataset='financials' AND ticker=? ORDER BY observed_at", (ticker,))
        by_period = {}
        for payload in snapshots["payload"]:
            for record in json.loads(payload):
                by_period[str(pd.Timestamp(record["period_end"]).date())] = record
        result = list(by_period.values())
        for record in result:
            record["period_end"] = pd.Timestamp(record["period_end"])
            record["available_from"] = pd.Timestamp(record["available_from"]) if record["available_from"] else pd.NaT
        return result

    def replace_splits(self, ticker, splits):
        with self.connection() as db:
            db.execute("DELETE FROM splits WHERE ticker=?", (ticker,))
            db.executemany("INSERT INTO splits VALUES(?,?,?,?)", [
                (ticker, str(date.date()), float(ratio), utc_now()) for date, ratio in splits.items()
            ])

    def split_series(self, ticker):
        frame = self.read("SELECT date,ratio FROM splits WHERE ticker=? ORDER BY date", (ticker,))
        return pd.Series(frame["ratio"].to_numpy(), index=pd.to_datetime(frame["date"]), dtype=float)

    def get_month(self, key, month):
        rows = self.read("SELECT payload FROM monthly_calculations WHERE config_key=? AND month=?", (key, month))
        return None if rows.empty else json.loads(rows.iloc[0, 0])

    def put_month(self, key, month, rows):
        payload = rows.to_json(orient="records", date_format="iso", double_precision=15)
        with self.connection() as db:
            db.execute("INSERT OR REPLACE INTO monthly_calculations VALUES(?,?,?,?)", (key, month, payload, utc_now()))

    def invalidate_calculations(self):
        with self.connection() as db:
            db.execute("DELETE FROM monthly_calculations")

    def summary(self):
        statements = {
            "prices": "SELECT COUNT(*) rows,COUNT(DISTINCT ticker) tickers,MIN(date) first_date,MAX(date) last_date FROM prices",
            "shares": "SELECT COUNT(*) rows,COUNT(DISTINCT ticker) tickers,MIN(date) first_date,MAX(date) last_date FROM shares",
            "universe_snapshots": "SELECT COUNT(*) rows,MIN(as_of) first_date,MAX(as_of) last_date FROM universe_snapshots",
            "universe_members": "SELECT COUNT(*) rows,COUNT(DISTINCT ticker) tickers FROM universe_members",
            "monthly_calculations": "SELECT COUNT(*) rows,MIN(month) first_month,MAX(month) last_month FROM monthly_calculations",
            "fundamental_reports": "SELECT COUNT(*) rows,COUNT(DISTINCT ticker) tickers,MIN(period_end) first_period,MAX(period_end) last_period FROM fundamental_reports",
            "fundamental_facts": "SELECT COUNT(*) rows,COUNT(DISTINCT metric) metrics FROM fundamental_facts",
            "monthly_fundamentals": "SELECT COUNT(*) rows,COUNT(DISTINCT ticker) tickers,MIN(month_end) first_month,MAX(month_end) last_month FROM monthly_fundamentals",
        }
        return {name: self.read(sql).iloc[0].to_dict() for name, sql in statements.items()}


def restore_snapshot(folder=None):
    """Restore the Git LFS snapshot without overwriting an existing database.

    Stream the archive, verify its full uncompressed size and SHA-256, then
    publish the verified file atomically. No source downloads or recalculations.
    """
    import gzip
    import hashlib
    import os
    import uuid

    folder = Path(folder) if folder is not None else Path(__file__).resolve().parents[2] / "aqr_research/data"
    target = folder / "aqr.sqlite3"
    archive = folder / "aqr.sqlite3.gz"
    if target.exists():
        raise FileExistsError(f"Existing database is preserved: {target}")
    with archive.open("rb") as source:
        if source.read(2) != b"\x1f\x8b":
            raise ValueError("Database archive is not downloaded; run git lfs pull first")
    expected = json.loads((folder / "aqr.sqlite3.manifest.json").read_text(encoding="utf-8"))
    temporary = folder / (".aqr-restore-" + uuid.uuid4().hex + ".sqlite3")
    digest = hashlib.sha256()
    size = 0
    try:
        with gzip.open(archive, "rb") as source, temporary.open("xb") as destination:
            while chunk := source.read(8 * 1024 * 1024):
                digest.update(chunk)
                size += len(chunk)
                destination.write(chunk)
        if size != expected["database_bytes"] or digest.hexdigest() != expected["database_sha256"]:
            raise ValueError("Database snapshot checksum mismatch; nothing was restored")
        # A hard link publishes the same verified bytes and fails if a database
        # appeared during extraction. It never replaces an existing file.
        os.link(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return target


if __name__ == "__main__":
    print(f"Restored verified database: {restore_snapshot()}")
