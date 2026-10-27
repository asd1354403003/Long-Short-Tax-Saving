"""Shared raw Open/Close/Volume downloads, cached only in memory."""
from __future__ import annotations
import re
import time
import numpy as np
import pandas as pd
import yfinance as yf
from spot_fx import SpotFX

def _normalize_tickers(tickers) -> list[str]:
    if isinstance(tickers, str):
        raise TypeError("Pass a list of tickers, not a single string")
    result = []
    for ticker in tickers:
        if not isinstance(ticker, str) or not ticker.strip():
            raise ValueError("Each ticker must be a nonempty string")
        ticker = ticker.strip().upper().replace(".", "-")
        if not re.fullmatch(r"[A-Z0-9^=\-]{1,24}", ticker):
            raise ValueError("Unsupported ticker: {}".format(ticker))
        if ticker not in result:
            result.append(ticker)
    return result

class YahooFinanceSource:
    """Download raw Open/Close/Volume in small batches; cache only in memory."""

    def __init__(self, config=None):
        self.batch_size = getattr(config, "batch_size", 20)
        self.events = []
        self._history_memo = {}

    def _request(self, call, context):
        for attempt in range(2):
            try:
                return call()
            except Exception as exc:
                if attempt == 0:
                    time.sleep(0.5)
                else:
                    message = "{}: {}: {}".format(context, type(exc).__name__, exc)
                    self.events.append(message)
                    raise RuntimeError(message) from exc

    def history(self, tickers, start, end_exclusive):
        tickers = _normalize_tickers(tickers)
        histories, pending = {}, []
        for ticker in tickers:
            memo_key = (ticker, start, end_exclusive)
            if memo_key in self._history_memo:
                if self._history_memo[memo_key] is not None:
                    histories[ticker] = self._history_memo[memo_key]
                continue
            pending.append(ticker)
        for offset in range(0, len(pending), self.batch_size):
            batch = pending[offset:offset + self.batch_size]
            source_batch = list(dict.fromkeys(SpotFX.source_ticker(ticker) for ticker in batch))
            try:
                raw = self._request(lambda: yf.download(
                    source_batch, start=start, end=end_exclusive, auto_adjust=False, actions=True,
                    group_by="ticker", progress=False, threads=min(5, len(batch)), timeout=20,
                ), "prices " + ",".join(batch))
            except RuntimeError:
                continue
            if raw is None or raw.empty:
                self.events.append("empty_history: " + ",".join(batch))
                continue
            for ticker in batch:
                try:
                    source_ticker = SpotFX.source_ticker(ticker)
                    if isinstance(raw.columns, pd.MultiIndex):
                        level = next(i for i in range(raw.columns.nlevels) if source_ticker in raw.columns.get_level_values(i))
                        frame = raw.xs(source_ticker, axis=1, level=level).copy()
                    elif len(batch) == 1:
                        frame = raw.copy()
                    else:
                        raise ValueError("Unexpected multi-ticker column format")
                    frame.index = pd.DatetimeIndex(pd.to_datetime(frame.index)).tz_localize(None).normalize()
                    frame = frame.loc[~frame.index.duplicated(keep="last")].sort_index()
                    frame = frame.loc[(frame.index >= pd.Timestamp(start)) & (frame.index < pd.Timestamp(end_exclusive))]
                    frame = frame.dropna(subset=[c for c in ("Open", "Close") if c in frame], how="all")
                    if frame.empty:
                        raise ValueError("No closing prices")
                    frame = frame.drop(columns=["Adj Close"], errors="ignore")
                    frame = SpotFX.normalize_prices(ticker, frame)
                    frame.index.name = "Date"
                    histories[ticker] = frame
                    self._history_memo[(ticker, start, end_exclusive)] = frame
                except (KeyError, ValueError, StopIteration) as exc:
                    self.events.append("missing_history {}: {}".format(ticker, exc))
        for ticker in tickers:
            self._history_memo.setdefault((ticker, start, end_exclusive), histories.get(ticker))
        return histories
