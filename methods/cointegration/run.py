"""Execute the retained frozen pair pool without refitting its formation models."""
from dataclasses import fields, is_dataclass, replace, asdict
from pathlib import Path
import json

import pandas as pd

from backtest import BacktestConfig
from reports import PerformanceReport
from risk import historical_risk_free
from .selector import CointegrationParameters, Selector

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _restore(template, values):
    replacements = {}
    for field in fields(template):
        if field.name not in values:
            continue
        old, value = getattr(template, field.name), values[field.name]
        if is_dataclass(old):
            value = _restore(old, value)
        elif isinstance(old, tuple):
            value = tuple(tuple(v) if isinstance(v, list) else v for v in value)
        elif isinstance(old, Path):
            value = Path(value)
        replacements[field.name] = value
    unknown = set(values) - {f.name for f in fields(template)}
    if unknown:
        raise ValueError(f"Unknown cointegration settings: {unknown}")
    return replace(template, **replacements)


def run(parameters, *, output=None, histories=None):
    """Reuse the saved pool. Missing history is downloaded only on an explicit run.

    The analysis notebook defaults to displaying the saved result without making
    network requests. Pass histories here for an entirely offline execution.
    """
    params = _restore(CointegrationParameters(), parameters["selection"])
    config = BacktestConfig(**parameters["backtest"])
    folder = PROJECT_ROOT / "result/cointegration/frozen2026"
    selector = Selector(params, config, folder)
    if not selector.pool_file.is_file() or not selector.metadata_file.is_file():
        raise FileNotFoundError("The frozen pool/selection metadata is missing; refusing to retrain implicitly")
    selection = selector.select(rebuild=False)
    result = selector.backtest(selection, histories=histories)
    rf = historical_risk_free(result["daily_summary"].index,
                             parameters.get("performance", {}).get("risk_free"))
    result["daily_risk_free"] = rf
    result["daily_summary"]["risk_free_return"] = rf
    result["daily_summary"]["excess_return"] = result["daily_summary"].daily_return - rf
    audit = PerformanceReport.audit(result).to_dict()
    daily = result["daily_summary"]
    last = daily.iloc[-1]
    document = dict(selection.metadata)
    document["backtest"] = dict(settings=asdict(config), start=str(daily.index[0].date()),
        end=str(daily.index[-1].date()), comparison=json.loads(PerformanceReport.compare(result).to_json(orient="index")),
        trades=json.loads(PerformanceReport.trades(result).to_json()),
        **{key: float(last[key]) for key in ("cumulative_pnl", "cumulative_net_realized_pnl",
                                            "net_unrealized_pnl", "cumulative_total_cost")})
    document["daily"] = json.loads(daily.reset_index().to_json(orient="records", date_format="iso", double_precision=15))
    document["annual_returns"] = json.loads(PerformanceReport.by_year(result).reset_index().to_json(orient="records"))
    document["verification"] = audit
    document["performance"] = dict(risk_free_settings=rf.attrs["settings"],
        risk_free_source=rf.attrs["provenance"], risk_free_basis=rf.attrs["basis"],
        cash_interest_in_nav=False,
        sharpe="sqrt(252) * mean(daily net return - RF) / sample_std(daily net return - RF)")
    document.pop("data_note", None)
    if output is not None:
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(document, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(f"Cointegration: {last.cumulative_pnl:,.2f} USD net P&L")
    return document
