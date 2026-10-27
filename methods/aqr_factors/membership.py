"""Historical monthly IWB equity holdings, explicitly a Russell 1000 proxy."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import hashlib
import io
import json
from pathlib import Path
import re
import zlib

import pandas as pd
import requests

from .database import AQRDatabase, utc_now


PROVIDER = "iwb_holdings_proxy"
FUND_URL = "https://www.ishares.com/us/products/239707/ishares-russell-1000-etf"
DOCUMENT_URL = "https://www.blackrock.com/varnish-api/blk-one01-product-data/product-data/api/v1/get-fund-document"
ALIASES = {"BRKB": "BRK-B", "BRKA": "BRK-A", "BFB": "BF-B", "BFA": "BF-A",
           "HEIA": "HEI-A", "LENB": "LEN-B", "UHALB": "UHAL-B", "CWENA": "CWEN-A"}
ASSUMPTION = "Previous month-end holdings proxy; original publication timestamp is not verified"


class MonthlyMembership:
    def __init__(self, database: AQRDatabase):
        self.database = database
        self.network_requests = 0

    @staticmethod
    def parse(content, requested_date):
        text = content.decode("utf-8-sig")
        lines = text.splitlines()
        header = next((i for i, line in enumerate(lines) if line.startswith("Ticker,Name,")), None)
        if header is None:
            raise ValueError("Historical holdings header not returned")
        reported = next((row[1] for row in csv.reader(lines[:header])
                         if len(row) > 1 and row[0] == "Fund Holdings as of"), None)
        actual, requested = pd.Timestamp(reported), pd.Timestamp(requested_date)
        if (pd.isna(actual) or actual > requested or actual.to_period("M") != requested.to_period("M")
                or (requested - actual).days > 7):
            raise ValueError("Holdings date mismatch; refusing a fallback to another month")
        source = list(csv.DictReader(io.StringIO("\n".join(lines[header:]))))
        members = []
        for row in source:
            if row.get("Asset Class") != "Equity":
                continue
            original = row["Ticker"].strip()
            symbol = ALIASES.get(original, original.replace(" ", "-").replace(".", "-"))
            if not re.fullmatch(r"[A-Z][A-Z0-9\-]{0,19}", symbol):
                continue
            weight = row.get("Weight (%)", "").replace(",", "")
            members.append({
                "ticker": symbol, "provider_ticker": original, "name": row["Name"],
                "sector": row.get("Sector"), "weight": float(weight) if weight not in {"", "-"} else None,
                "location": row.get("Location"), "currency": row.get("Market Currency", row.get("Currency")),
            })
        if not 800 <= len(members) <= 1300:
            raise ValueError("Unexpected equity count: {}".format(len(members)))
        if len({row["ticker"] for row in members}) != len(members):
            raise ValueError("Duplicate normalized holding symbols")
        weight_sum = sum(row["weight"] or 0 for row in members)
        if not 90 <= weight_sum <= 102:
            raise ValueError("Unexpected equity weight total: {}".format(weight_sum))
        return str(actual.date()), members, sum(bool(row.get("Asset Class")) for row in source)

    def _fetch(self, target):
        cached = self.database.read("SELECT as_of FROM universe_snapshots WHERE provider=? AND requested_as_of=?", (PROVIDER, target))
        if not cached.empty:
            return cached.iloc[0, 0], False
        requested = pd.Timestamp(target)
        candidate = requested
        while candidate.weekday() >= 5:
            candidate -= pd.Timedelta(days=1)
        parameters = {
            "appSubType": "ISHARES", "appType": "PRODUCT_PAGE", "component": "holdings",
            "locale": "en_US", "portfolioId": "239707", "targetSite": "us-ishares",
            "userType": "individual",
        }
        last_error = None
        for _ in range(4):
            try:
                response = requests.get(DOCUMENT_URL, params={**parameters, "asOfDate": candidate.strftime("%Y%m%d")}, timeout=35)
                self.network_requests += 1
                response.raise_for_status()
                actual, members, count = self.parse(response.content, target)
                checksum, observed = hashlib.sha256(response.content).hexdigest(), utc_now()
                with self.database.connection() as db:
                    db.execute("INSERT OR IGNORE INTO source_documents VALUES(?,?,?,?)", (
                        checksum, response.url, observed, zlib.compress(response.content),
                    ))
                    db.execute("INSERT OR REPLACE INTO universe_snapshots VALUES(?,?,?,?,?,?,?,?,0)", (
                        PROVIDER, actual, target, response.url, checksum, observed, len(members), count,
                    ))
                    db.execute("DELETE FROM universe_members WHERE provider=? AND as_of=?", (PROVIDER, actual))
                    db.executemany("INSERT INTO universe_members VALUES(?,?,?,?,?,?,?,?,?)", [
                        (PROVIDER, actual, row["ticker"], row["provider_ticker"], row["name"], row["sector"],
                         row["weight"], row["location"], row["currency"]) for row in members
                    ])
                return actual, True
            except (requests.RequestException, ValueError) as exc:
                last_error = exc
                candidate -= pd.Timedelta(days=1)
        raise RuntimeError("Unable to verify IWB month {}: {}".format(target, last_error))

    def prepare(self, start_date, end_date):
        """Trade month M uses a verified holdings-as-of date in month M-1."""
        start = pd.Timestamp(start_date)
        today = pd.Timestamp.now(tz="America/New_York").normalize().tz_localize(None)
        end = min(pd.Timestamp(end_date), today)
        if start > end:
            raise ValueError("start_date must be no later than end_date")
        months = pd.period_range(start.to_period("M"), end.to_period("M"), freq="M")
        targets = {str(month): str((month - 1).end_time.date()) for month in months}
        finished = {}
        with ThreadPoolExecutor(max_workers=3) as executor:
            jobs = {executor.submit(self._fetch, target): month for month, target in targets.items()}
            for future in as_completed(jobs):
                month = jobs[future]
                actual, fetched = future.result()
                finished[month] = actual
                if fetched:
                    print("Stored IWB proxy {}: holdings {}".format(month, actual), flush=True)
        with self.database.connection() as db:
            db.executemany("INSERT OR REPLACE INTO monthly_universe VALUES(?,?,?,?,?)", [
                (PROVIDER, month, cutoff, finished[month], ASSUMPTION) for month, cutoff in targets.items()
            ])
        return self.audit(start_date, end_date)

    def audit(self, start_date, end_date):
        rows = self.database.read("""SELECT m.trade_month,m.decision_cutoff,m.snapshot_as_of,
            s.equity_count,s.source_url,s.source_sha256,s.observed_at,s.publication_time_verified
            FROM monthly_universe m JOIN universe_snapshots s
            ON s.provider=m.provider AND s.as_of=m.snapshot_as_of
            WHERE m.provider=? AND m.trade_month>=? AND m.trade_month<=? ORDER BY m.trade_month""",
            (PROVIDER, str(pd.Timestamp(start_date).to_period("M")), str(pd.Timestamp(end_date).to_period("M"))))
        previous = None
        added, removed = [], []
        for row in rows.itertuples():
            members = set(self.database.read("SELECT ticker FROM universe_members WHERE provider=? AND as_of=?", (PROVIDER, row.snapshot_as_of))["ticker"])
            added.append(None if previous is None else len(members - previous))
            removed.append(None if previous is None else len(previous - members))
            previous = members
        rows["added_since_previous_month"] = added
        rows["removed_since_previous_month"] = removed
        return rows

    def members_at_cutoff(self, cutoff):
        latest = self.database.read("SELECT MAX(as_of) AS as_of FROM universe_snapshots WHERE provider=? AND as_of<=?", (PROVIDER, str(pd.Timestamp(cutoff).date())))
        actual = latest.iloc[0, 0]
        if actual is None:
            return set()
        return set(self.database.read("SELECT ticker FROM universe_members WHERE provider=? AND as_of=?", (PROVIDER, actual))["ticker"])

    def sectors_at_cutoff(self, cutoff):
        """Use the same dated holdings snapshot for both members and sectors."""
        rows = self.database.read("""SELECT ticker,sector FROM universe_members
            WHERE provider=? AND as_of=(SELECT MAX(as_of) FROM universe_snapshots
                WHERE provider=? AND as_of<=?) ORDER BY ticker""",
            (PROVIDER, PROVIDER, str(pd.Timestamp(cutoff).date())))
        return rows.set_index("ticker")["sector"].replace({"": None, "-": None})

    def sector_summary(self, start_date, end_date):
        return self.database.read("""SELECT m.trade_month,m.snapshot_as_of,u.sector,COUNT(*) AS stock_count
            FROM monthly_universe m JOIN universe_members u
            ON u.provider=m.provider AND u.as_of=m.snapshot_as_of
            WHERE m.provider=? AND m.trade_month>=? AND m.trade_month<=?
            GROUP BY m.trade_month,m.snapshot_as_of,u.sector ORDER BY m.trade_month,u.sector""",
            (PROVIDER, str(pd.Timestamp(start_date).to_period("M")), str(pd.Timestamp(end_date).to_period("M"))))

    def write_html(self, path, start_date, end_date):
        audit = self.audit(start_date, end_date)
        data = {"rows": json.loads(audit.to_json(orient="records")), "members": {},
                "provider": PROVIDER, "source": FUND_URL,
                "database": "aqr_research/data/aqr.sqlite3"}
        for date in audit["snapshot_as_of"]:
            members = self.database.read("SELECT ticker,provider_ticker,sector FROM universe_members WHERE provider=? AND as_of=? ORDER BY ticker", (PROVIDER, date))
            data["members"][date] = members.to_numpy().tolist()
        template = Path(__file__).with_name("universe_template.html").read_text(encoding="utf-8")
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(template.replace("__DATA__", json.dumps(data, ensure_ascii=False, allow_nan=False).replace("</", "<\\/")), encoding="utf-8")
