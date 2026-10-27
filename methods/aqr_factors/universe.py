"""Fixed lists or dated complete membership snapshots, evaluated monthly."""
from pathlib import Path
import hashlib
import re

import pandas as pd




class Universe:
    def __init__(self, name, tickers=(), membership_file=None, *, database=None, start_date=None, end_date=None):
        self.name = name
        self.membership = None
        self.proxy = None
        if name not in {"custom", "russell1000", "russell1000_iwb_proxy"}:
            raise ValueError("Unknown universe: " + str(name))
        if name == "russell1000_iwb_proxy":
            from .membership import MonthlyMembership, PROVIDER
            if database is None or start_date is None or end_date is None:
                raise ValueError("The monthly IWB proxy requires the AQR database and date range")
            self.proxy = MonthlyMembership(database)
            # Include the next trading month's universe when the requested end
            # reaches a completed calendar month-end signal cutoff.
            upper = pd.Timestamp(end_date)
            if upper == upper + pd.offsets.MonthEnd(0):
                upper = upper + pd.offsets.MonthBegin(1)
            self.proxy.prepare(start_date, str(upper.date()))
            minimum = str(((pd.Timestamp(start_date).to_period("M") - 1).end_time - pd.Timedelta(days=7)).date())
            maximum = str(pd.Timestamp(end_date).date())
            records = database.read("SELECT DISTINCT ticker FROM universe_members WHERE provider=? AND as_of>=? AND as_of<=? ORDER BY ticker", (PROVIDER, minimum, maximum))
            self.tickers = tuple(records["ticker"])
        elif membership_file:
            if tickers:
                raise ValueError("Choose either tickers or membership_file")
            relative = Path(membership_file)
            if relative.is_absolute() or ".." in relative.parts or relative.suffix.lower() != ".csv":
                raise ValueError("membership_file must be a project-relative CSV path")
            source = Path(__file__).resolve().parents[2] / relative
            if not source.is_file():
                raise FileNotFoundError("Provide historical membership at " + str(relative))
            frame = pd.read_csv(source, dtype={"ticker": "string"})
            if not {"as_of", "ticker"}.issubset(frame.columns) or frame.empty:
                raise ValueError("Membership CSV requires as_of,ticker and at least one full snapshot")
            if frame[["as_of", "ticker"]].isna().any().any():
                raise ValueError("Membership dates and tickers cannot be missing")
            frame["as_of"] = pd.to_datetime(frame["as_of"], errors="raise").dt.normalize()
            frame["ticker"] = frame["ticker"].str.strip().str.upper()
            if frame.duplicated(["as_of", "ticker"]).any():
                raise ValueError("Duplicate ticker within a membership snapshot")
            self.membership = frame.sort_values(["as_of", "ticker"])
            self.tickers = tuple(self.membership["ticker"].drop_duplicates())
        elif name == "russell1000":
            raise ValueError("russell1000 requires membership_file with dated full constituent snapshots")
        else:
            if not tickers or isinstance(tickers, str):
                raise ValueError("custom requires a ticker sequence or membership_file")
            self.tickers = tuple(str(symbol).strip().upper() for symbol in tickers)
        if len(set(self.tickers)) != len(self.tickers):
            raise ValueError("Universe tickers must be unique")
        if not all(re.fullmatch(r"[A-Z][A-Z0-9.\-]{0,19}", symbol) for symbol in self.tickers):
            raise ValueError("Unsupported or blank universe ticker")

    def members(self, date):
        if self.proxy is not None:
            return self.proxy.members_at_cutoff(date)
        if self.membership is None:
            return set(self.tickers)
        available = self.membership.loc[self.membership["as_of"].le(pd.Timestamp(date))]
        if available.empty:
            return set()
        latest = available["as_of"].max()
        return set(available.loc[available["as_of"].eq(latest), "ticker"])

    def group_labels(self, date, group_by):
        """Return dated provider labels when they are part of membership data."""
        if self.proxy is not None and group_by == "sector":
            return self.proxy.sectors_at_cutoff(date)
        raise ValueError("This universe has no dated labels for " + str(group_by))

    def fingerprint(self, date, group_by=None):
        symbols = sorted(self.members(date))
        if self.proxy is not None and group_by == "sector":
            labels = self.group_labels(date, group_by)
            members = "\n".join(symbol + "\t" + str(labels.get(symbol, "")) for symbol in symbols)
        else:
            members = "\n".join(symbols)
        return hashlib.sha256(members.encode()).hexdigest()
