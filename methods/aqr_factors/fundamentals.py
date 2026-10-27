"""Complete reported financial fields, versioned sources and monthly as-of links.

Yahoo statements are latest downloaded versions, potentially restated. Earnings
dates provide availability proxies, not verified original filing timestamps.
Current info is stored separately and never backfilled into historical months.
"""
from __future__ import annotations

from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import math

import numpy as np
import pandas as pd
import yfinance as yf

from .database import AQRDatabase, utc_now
from .factors import trailing_four


STATEMENTS = {"income": "get_income_stmt", "balance": "get_balance_sheet", "cashflow": "get_cashflow"}
FREQUENCIES = ("yearly", "quarterly", "trailing")
SERIES = tuple((statement, frequency) for statement in STATEMENTS for frequency in FREQUENCIES
               if not (statement == "balance" and frequency == "trailing"))
SOURCE = "https://finance.yahoo.com/"


def validate_yahoo_info(info, required=("financialCurrency",)):
    """Reject partial/error responses before treating a profile as cached data."""
    missing = [name for name in required if not isinstance(info, dict) or not info.get(name)]
    if missing:
        raise ValueError("Incomplete Yahoo company info: missing " + ", ".join(missing))
    return json_value(info)


def json_value(value):
    """Preserve every returned info field without nonstandard NaN JSON."""
    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [json_value(item) for item in value]
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if isinstance(value, np.generic):
        return json_value(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


class FundamentalStore:
    def __init__(self, database: AQRDatabase):
        self.database = database
        self.stats = Counter()
        self.errors = []

    def website_quarterly_events(self, *, availability="existing_dates_only", lag_days=90):
        """Read native website versions, retaining unknown publication dates as unknown.

        existing_dates_only uses the separately retained prior-report date
        references (which may be earnings proxies). period_lag is an explicit
        in-memory research assumption, not a correction to the source database.
        Each event contains available balance, income and cash-flow quarters,
        plus trailing-four sums using only quarters available by that event.
        """
        if availability not in {"existing_dates_only", "period_lag"}:
            raise ValueError("Invalid financial availability policy")
        reports = self.database.read("""SELECT r.report_id,r.ticker,r.statement,r.period_end,
            r.available_from,r.observed_at,r.financial_currency,
            p.reference_available_from,p.reference_availability_method
            FROM fundamental_reports r JOIN yahoo_csv_report_provenance p USING(report_id)
            WHERE r.frequency='quarterly' AND r.statement IN ('balance','income','cashflow')
            AND EXISTS (SELECT 1 FROM yahoo_csv_periods c WHERE c.report_id=r.report_id
                AND c.alignment_status NOT LIKE 'hold_%')""")
        if reports.empty:
            raise ValueError("No imported website quarterly statements")
        currencies = reports.groupby("ticker")["financial_currency"].agg(lambda x: set(x.dropna()))
        reports["period_end"] = pd.to_datetime(reports["period_end"])
        reports["effective_available"] = pd.to_datetime(reports["available_from"].fillna(
            reports["reference_available_from"]), errors="coerce")
        unknown = reports["effective_available"].isna()
        reports["date_assumed"] = unknown & (availability == "period_lag")
        if availability == "period_lag":
            reports.loc[unknown, "effective_available"] = reports.loc[unknown, "period_end"] + pd.Timedelta(days=lag_days)
        metadata = dict(source="Yahoo Finance member-site native quarterly CSV",
            availability_policy=availability, reporting_lag_days=lag_days if availability=="period_lag" else None,
            report_versions=int(len(reports)), missing_date_references=int(unknown.sum()),
            excluded_unknown_date_versions=int(reports.effective_available.isna().sum()),
            assumed_date_versions=int(reports.date_assumed.sum()),
            assumed_date_rule="For missing dates only, fiscal period end plus lag days; not a verified publication date" if availability=="period_lag" else None,
            quality_definition="latest available quarterly gross profit / latest available total assets",
            value_definition="latest available common stock equity / historical market capitalization",
            vintage="latest vendor versions; date references are not proof of original historical values")
        reports = reports.loc[reports.effective_available.notna()
            & reports.effective_available.gt(reports.period_end)
            & reports.financial_currency.eq("USD")].copy()
        metrics = ("CommonStockEquity", "TotalAssets", "GrossProfit", "TotalRevenue", "CostOfRevenue",
                   "NetIncomeCommonStockholders", "NetIncome", "OperatingCashFlow")
        values = {}
        identifiers = reports.report_id.tolist()
        with self.database.connection() as db:
            for offset in range(0, len(identifiers), 500):
                part = identifiers[offset:offset+500]
                query = "SELECT report_id,metric,value FROM fundamental_facts WHERE report_id IN ({}) AND metric IN ({})".format(
                    ",".join("?" for _ in part), ",".join("?" for _ in metrics))
                for report, metric, value in db.execute(query, (*part, *metrics)):
                    values.setdefault(report, {})[metric] = value
        events = {}
        for ticker, group in reports.groupby("ticker", sort=False):
            latest, records = {}, []
            known = {name: {} for name in ("balance", "income", "cashflow")}
            for available, announced in group.sort_values(["effective_available","period_end","observed_at","report_id"]).groupby("effective_available", sort=True):
                for record in announced.to_dict("records"):
                    known[record["statement"]][record["period_end"]] = record
                    old = latest.get(record["statement"])
                    # An old comparative/version arriving later cannot replace
                    # the most recent fiscal period with an older one.
                    if old is None or record["period_end"] >= old["period_end"]:
                        latest[record["statement"]] = record
                balance, income = latest.get("balance"), latest.get("income")
                b = values.get(balance["report_id"], {}) if balance else {}
                i = values.get(income["report_id"], {}) if income else {}
                profit = i.get("GrossProfit")
                if profit is None and i.get("TotalRevenue") is not None and i.get("CostOfRevenue") is not None:
                    profit = i["TotalRevenue"] - i["CostOfRevenue"]
                common_income, ttm_start, ttm_end = trailing_four(known["income"], values, "NetIncomeCommonStockholders")
                net_income, _, _ = trailing_four(known["income"], values, "NetIncome")
                cashflow, cf_start, cf_end = trailing_four(known["cashflow"], values, "OperatingCashFlow")
                records.append(dict(ticker=ticker,
                    period_end=max(r["period_end"] for r in latest.values()),
                    available_from=available, earnings_proxy_at=None,
                    availability_method=availability,
                    balance_period_end=balance["period_end"] if balance else pd.NaT,
                    income_period_end=income["period_end"] if income else pd.NaT,
                    cashflow_period_end=latest["cashflow"]["period_end"] if "cashflow" in latest else pd.NaT,
                    book_equity=b.get("CommonStockEquity"), total_assets=b.get("TotalAssets"),
                    common_net_income_ttm=common_income, net_income_ttm=net_income,
                    operating_cashflow_ttm=cashflow, earnings_ttm_start=ttm_start, earnings_ttm_end=ttm_end,
                    cashflow_ttm_start=cf_start, cashflow_ttm_end=cf_end,
                    balance_date_assumed=bool(balance["date_assumed"]) if balance else False,
                    income_date_assumed=bool(income["date_assumed"]) if income else False,
                    cashflow_date_assumed=bool(latest["cashflow"]["date_assumed"]) if "cashflow" in latest else False,
                    gross_profit=profit, financial_frequency="quarterly"))
            events[ticker] = records
        metadata["usable_tickers_with_dated_reports"] = len(events)
        return events, currencies.to_dict(), metadata

    def _source(self, dataset, symbol, through, fetch, encode, *, force=False):
        saved = self.database.latest_snapshot(dataset, symbol)
        if not force and saved is not None and saved["verified_through"] >= through:
            return saved
        self.stats[dataset + "_requests"] += 1
        raw = fetch()
        payload = encode(raw)
        observed = utc_now()
        verified = str(pd.Timestamp(observed).tz_convert("America/New_York").date())
        self.database.put_snapshot(dataset, symbol, payload, verified, observed_at=observed)
        return {"payload": payload, "observed_at": observed, "verified_through": verified}

    @staticmethod
    def _encode_statement(frame):
        if frame is None or frame.empty:
            raise ValueError("No financial statement returned; request remains retryable")
        columns = []
        for date in frame.columns:
            values = {str(metric): float(value) if pd.notna(value) and np.isfinite(float(value)) else None
                      for metric, value in frame[date].items()}
            if any(value is not None for value in values.values()):
                columns.append({"period_end": str(pd.Timestamp(date).date()), "values": values})
        if not columns:
            raise ValueError("Statement has no numeric fields; request remains retryable")
        return columns

    @staticmethod
    def _encode_earnings(frame):
        if frame is None or frame.empty or "Reported EPS" not in frame:
            raise ValueError("No reported earnings dates returned")
        return [{"date": date.isoformat(), "reported_eps": float(row["Reported EPS"])}
                for date, row in frame.dropna(subset=["Reported EPS"]).sort_index().iterrows()]

    def has_sources(self, symbol, through):
        for dataset in ("yahoo_info", "earnings_dates", *("statement_{}_{}".format(*key) for key in SERIES)):
            saved = self.database.latest_snapshot(dataset, symbol)
            if saved is None or saved["verified_through"] < through:
                return False
            if dataset == "yahoo_info":
                try:
                    validate_yahoo_info(saved["payload"])
                except ValueError:
                    return False
        return True

    def ingest(self, symbol, statement, frequency, payload, earnings, calendar, currency, observed_at):
        """Store all fields; identical versions keep their first observation time."""
        releases = [(pd.Timestamp(row["date"]), row) for row in earnings]
        today = pd.Timestamp.now(tz="America/New_York").normalize().tz_localize(None)
        reports = []
        for column in payload:
            period = pd.Timestamp(column["period_end"])
            candidates = sorted(date for date, _ in releases
                                if period < date.tz_localize(None).normalize() <= period + pd.Timedelta(days=120))
            available = None
            if candidates:
                release_day = candidates[0].tz_localize(None).normalize()
                if release_day < today:
                    sessions = calendar(str(release_day.date()), str(today.date()))
                    later = sessions[sessions > release_day]
                    if len(later):
                        available = str(later[0].date())
            content = {"ticker": symbol, "statement": statement, "frequency": frequency,
                       "period_end": column["period_end"], "values": column["values"],
                       "available_from": available, "financial_currency": currency}
            identifier = hashlib.sha256(json.dumps(content, sort_keys=True, allow_nan=False).encode()).hexdigest()
            reports.append((identifier, content))
        with self.database.connection() as db:
            for identifier, content in reports:
                db.execute("INSERT OR IGNORE INTO fundamental_reports VALUES(?,?,?,?,?,?,?,?,?,?)", (
                    identifier, symbol, statement, frequency, content["period_end"], content["available_from"],
                    "next_session_after_earnings_proxy_within_120_days", currency, observed_at, SOURCE,
                ))
                db.executemany("INSERT OR IGNORE INTO fundamental_facts VALUES(?,?,?)", [
                    (identifier, metric, value) for metric, value in content["values"].items()
                ])

    def refresh(self, symbol, end_date, *, force=False, ticker=None, calendar=None):
        """Fetch missing statement endpoints only, retaining full source snapshots.

        A date request does not extend Yahoo's limited statement history.
        Failed endpoints are reported and never marked successfully fetched.
        """
        self.stats.clear()
        self.errors = []
        ticker = ticker or yf.Ticker(symbol)
        completed = pd.Timestamp.now(tz="America/New_York").normalize().tz_localize(None)
        through = str(min(pd.Timestamp(end_date), completed - pd.Timedelta(days=1)).date())
        if calendar is None:
            from .data import CachedYahooData
            calendar_source = CachedYahooData(self.database)
            calendar = calendar_source._calendar
        saved_info = self.database.latest_snapshot("yahoo_info", symbol)
        try:
            validate_yahoo_info(saved_info["payload"] if saved_info else None)
            incomplete_info = False
        except ValueError:
            incomplete_info = True
        try:
            info = self._source("yahoo_info", symbol, through, ticker.get_info,
                                validate_yahoo_info, force=force or incomplete_info)
        except Exception as exc:
            # An unavailable profile must not discard obtainable statement data.
            # Keep currency unknown until the source supplies it; scoring still
            # requires a separately validated company profile and USD financials.
            self.errors.append("{} company info: {}".format(symbol, exc))
            info = {"payload": {}, "observed_at": utc_now()}
        try:
            earnings = self._source("earnings_dates", symbol, through,
                                    lambda: ticker.get_earnings_dates(limit=100), self._encode_earnings, force=force)
        except Exception as exc:
            self.errors.append("{} earnings dates: {}".format(symbol, exc))
            earnings = {"payload": [], "observed_at": info["observed_at"]}
        frames = {}
        for statement, frequency in SERIES:
            dataset = "statement_{}_{}".format(statement, frequency)
            try:
                source = self._source(dataset, symbol, through,
                                      lambda: getattr(ticker, STATEMENTS[statement])(pretty=False, freq=frequency),
                                      self._encode_statement, force=force)
                self.ingest(symbol, statement, frequency, source["payload"], earnings["payload"],
                            calendar, info["payload"].get("financialCurrency"),
                            max(source["observed_at"], earnings["observed_at"]))
                frame = pd.DataFrame({pd.Timestamp(row["period_end"]): row["values"] for row in source["payload"]})
                frames[statement, frequency] = frame
            except Exception as exc:
                self.errors.append("{} {} {}: {}".format(symbol, statement, frequency, exc))
        calendar_source = locals().get("calendar_source")
        if calendar_source is not None:
            self.stats.update(calendar_source.stats)
        frames["earnings"] = pd.DataFrame(
            {"Reported EPS": [row["reported_eps"] for row in earnings["payload"]]},
            index=pd.to_datetime([row["date"] for row in earnings["payload"]], utc=True).tz_convert("America/New_York"),
        )
        frames["info"] = info["payload"]
        return frames

    def build_monthly(self, start_date, end_date, tickers=None):
        """Link months to the latest fiscal period with a usable availability date.

        Values use the latest downloaded version, not verified historical vintages.
        Current Yahoo info is deliberately absent from this historical table.
        """
        last = min(pd.Timestamp(end_date).to_period("M") - (0 if pd.Timestamp(end_date).is_month_end else 1),
                   pd.Timestamp.now(tz="America/New_York").tz_localize(None).to_period("M") - 1)
        months = pd.period_range(pd.Timestamp(start_date).to_period("M"), last, freq="M")
        if tickers is None:
            tickers = self.database.read("SELECT DISTINCT ticker FROM fundamental_reports ORDER BY ticker")["ticker"]
        output, states = [], []
        for symbol in tickers:
            reports = self.database.read("SELECT * FROM fundamental_reports WHERE ticker=? ORDER BY period_end,observed_at,rowid", (symbol,))
            # Filter publication eligibility at each cutoff before choosing a
            # version. A later filing's comparative column must not erase the
            # earlier original filing that was already public in 2021.
            digest = hashlib.sha256(("publication_cutoff_v2\n"+
                "\n".join(sorted(reports["report_id"]))).encode()).hexdigest()
            by_series = {}
            for record in reports.to_dict("records"):
                by_series.setdefault((record["statement"],record["frequency"]),[]).append(record)
            previous = self.database.latest_snapshot("monthly_fundamental_state", symbol)
            unchanged = previous is not None and previous["payload"]["source_digest"] == digest
            existing = self.database.read("SELECT month_end,statement,frequency FROM monthly_fundamentals WHERE ticker=?", (symbol,))
            saved = set(existing.itertuples(index=False, name=None))
            rebuild = set(months)
            if not unchanged:
                rebuild.update(pd.PeriodIndex(existing["month_end"], freq="M"))
            for month in sorted(rebuild):
                cutoff = str(month.end_time.date())
                for statement, frequency in SERIES:
                    if unchanged and (cutoff, statement, frequency) in saved:
                        continue
                    candidate = next((r for r in reversed(by_series.get((statement,frequency),[]))
                        if isinstance(r["available_from"],str) and r["available_from"]<=cutoff),None)
                    report = candidate["report_id"] if candidate is not None else None
                    output.append((symbol, cutoff, statement, frequency, report,
                                   "available_proxy" if report else "no_usable_report",
                                   "latest_downloaded_not_original_vintage", utc_now()))
            if not unchanged:
                states.append(("monthly_fundamental_state", symbol, utc_now(),
                               str(pd.Timestamp(end_date).date()), json.dumps({"source_digest": digest})))
        with self.database.connection() as db:
            db.executemany("INSERT OR REPLACE INTO monthly_fundamentals VALUES(?,?,?,?,?,?,?,?)", output)
            db.executemany("INSERT OR REPLACE INTO snapshots VALUES(?,?,?,?,?)", states)
        return len(output)

    def monthly_values(self, start_date, end_date, *, ticker=None, wide=False):
        query = "SELECT * FROM monthly_fundamental_values WHERE month_end>=? AND month_end<=?"
        parameters = [str(pd.Timestamp(start_date).date()), str(pd.Timestamp(end_date).date())]
        if ticker is not None:
            query += " AND ticker=?"
            parameters.append(ticker)
        frame = self.database.read(query + " ORDER BY month_end,ticker,statement,frequency,metric", parameters)
        if not wide:
            return frame
        values = frame.loc[frame["metric"].notna()].copy()
        values["field"] = values["frequency"] + "__" + values["statement"] + "__" + values["metric"]
        return values.pivot(index=["month_end", "ticker"], columns="field", values="value").reset_index()

    def info_at(self, ticker, as_of):
        """Read only info actually captured by this time; never invent older info."""
        boundary = pd.Timestamp(as_of)
        boundary = boundary.tz_localize("America/New_York") if boundary.tzinfo is None else boundary.tz_convert("America/New_York")
        boundary = boundary.normalize() + pd.DateOffset(days=1)
        snapshots = self.database.read("SELECT observed_at,payload FROM snapshots WHERE dataset='yahoo_info' AND ticker=? ORDER BY observed_at DESC", (ticker,))
        for row in snapshots.itertuples():
            if pd.Timestamp(row.observed_at) < boundary:
                return json.loads(row.payload)
        return None

    def prepare_universe(self, universe, start_date, end_date, *, workers=2):
        """Batch only the chosen universe; repeat runs reuse stored source endpoints."""
        self.errors, self.stats = [], Counter()
        from .data import CachedYahooData
        calendar_source = CachedYahooData(self.database)
        def load(symbol):
            store = FundamentalStore(self.database)
            try:
                store.refresh(symbol, end_date, calendar=calendar_source._calendar)
            except Exception as exc:
                store.errors.append("{}: {}".format(symbol, exc))
            store.build_monthly(start_date, end_date, [symbol])
            return store
        with ThreadPoolExecutor(max_workers=workers) as executor:
            jobs = {executor.submit(load, symbol): symbol for symbol in universe.tickers}
            for future in as_completed(jobs):
                store = future.result()
                self.stats.update(store.stats)
                self.errors.extend(store.errors)
        self.stats.update(calendar_source.stats)
        return self
