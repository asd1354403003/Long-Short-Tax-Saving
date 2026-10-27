"""Reusable factor definitions; all characteristics are oriented higher = better."""
from __future__ import annotations

import math
import pandas as pd


FACTOR_DEFINITIONS = {
    "value": "Latest available common stock equity / positive historical market cap; signed B/M retained unless require_positive_book_equity=True",
    "quality": "Latest available quarterly gross profit / latest available total assets",
    "momentum": "Close[t-21] / Close[t-252] - 1",
    "earnings_yield": "Sum of 4 consecutive available quarters of NetIncomeCommonStockholders / historical market cap; negative earnings retain their negative sign",
    "cashflow_yield": "Sum of 4 consecutive available quarters of OperatingCashFlow / historical market cap; negative cash flow retains its sign",
    "roa": "Sum of 4 consecutive available quarters of NetIncome / latest available total assets (ending-assets proxy, not average assets)",
    "low_volatility": "Negative annualized sample standard deviation of the last 126 daily Close returns",
}


def balance_ratios(frame):
    """Derive independent raw ratios without treating negative equity as missing."""
    result = {}
    if {"book_equity", "market_cap"}.issubset(frame):
        equity = pd.to_numeric(frame["book_equity"], errors="coerce")
        cap = pd.to_numeric(frame["market_cap"], errors="coerce")
        result["value"] = equity / cap.where(cap.gt(0))
    if {"gross_profit", "total_assets"}.issubset(frame):
        profit = pd.to_numeric(frame["gross_profit"], errors="coerce")
        assets = pd.to_numeric(frame["total_assets"], errors="coerce")
        result["quality"] = profit / assets.where(assets.gt(0))
    return result


def trailing_four(known_periods, values, metric):
    """Four consecutive fiscal quarters already available at the event date.

    known_periods maps fiscal period ends to the most recent report version
    known at that date. A missing quarter or value cannot be skipped or filled.
    Return the total and its start/end periods for audit exports.
    """
    periods = sorted(known_periods)[-4:]
    if len(periods) != 4:
        return None, pd.NaT, pd.NaT
    gaps = [(b-a).days for a, b in zip(periods, periods[1:])]
    if any(gap < 60 or gap > 120 for gap in gaps) or not 240 <= (periods[-1]-periods[0]).days <= 300:
        return None, pd.NaT, pd.NaT
    amounts = [values.get(known_periods[period]["report_id"], {}).get(metric) for period in periods]
    if any(amount is None or not math.isfinite(amount) for amount in amounts):
        return None, pd.NaT, pd.NaT
    return float(sum(amounts)), periods[0], periods[-1]
