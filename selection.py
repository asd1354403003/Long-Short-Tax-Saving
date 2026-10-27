"""Common selector output: independent names or method-owned pair models."""
from dataclasses import dataclass, field
import pandas as pd


@dataclass
class PoolSelection:
    kind: str
    pool: pd.DataFrame
    metadata: dict = field(default_factory=dict)
    source: object = None
    histories: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.kind not in ("independent", "pairs", "weights"):
            raise ValueError("Selector kind must be independent, pairs or weights.")


def independent_pool(frame):
    """Full dated snapshots; larger score means higher priority on either side.

    Required: as_of, ticker, side, score. Each snapshot contains the entire
    current candidate list. Optional entry_signal/exit_signal carry a future
    valuation model's decisions; defaults are True/False. No beta is required.
    """
    required = {"as_of", "ticker", "side", "score"}
    if not required.issubset(frame):
        raise ValueError("Independent pool requires: " + ", ".join(sorted(required)))
    pool = frame.copy()
    if pool.ticker.isna().any():
        raise ValueError("Ticker values cannot be missing.")
    pool["as_of"] = pd.to_datetime(pool.as_of, errors="raise").dt.normalize()
    pool["ticker"] = pool.ticker.astype(str).str.upper().str.strip().str.replace(".", "-", regex=False)
    pool["side"] = pool.side.astype(str).str.lower()
    pool["score"] = pd.to_numeric(pool.score, errors="raise")
    if (pool.as_of.isna().any() or not pool.ticker.str.fullmatch(r"[A-Z0-9^=\-]{1,24}").all()
            or not pool.side.isin(["long", "short"]).all()
            or not pool.score.map(lambda value: pd.notna(value) and abs(value) < float("inf")).all()
            or pool.duplicated(["as_of", "ticker"]).any()):
        raise ValueError("Use finite scores and one long or short row per ticker per snapshot.")
    for name, default in (("entry_signal", True), ("exit_signal", False)):
        if name not in pool:
            pool[name] = default
        if not pool[name].map(lambda value: isinstance(value, bool)).all():
            raise ValueError(name + " must contain booleans.")
    if "instrument_type" not in pool:
        pool["instrument_type"] = "EQUITY"
    if not pool.instrument_type.isin(["EQUITY", "ETF", "FX"]).all():
        raise ValueError("instrument_type must be EQUITY, ETF or FX.")
    if "unit_step" not in pool:
        pool["unit_step"] = 1
    pool["unit_step"] = pd.to_numeric(pool.unit_step, errors="raise")
    if not pool.unit_step.map(lambda value: pd.notna(value) and value >= 1 and value % 1 == 0).all():
        raise ValueError("unit_step must contain positive integers.")
    return pool.sort_values(["as_of", "score", "ticker"], ascending=[True, False, True], kind="stable").reset_index(drop=True)
