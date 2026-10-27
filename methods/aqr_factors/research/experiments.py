"""Predeclared factor comparisons with both own and common monthly cohorts.

No data downloads, strategy tuning, or database writes. The normal selector's
baseline exports are not overwritten. Call from analysis.ipynb.
"""
from __future__ import annotations

from copy import copy, deepcopy
from dataclasses import asdict, replace
from pathlib import Path
import json

import numpy as np
import pandas as pd

from ..factors import FACTOR_DEFINITIONS
from ..signals import FactorSignals
from reports import PerformanceReport
from methods.aqr_factors.backtest import WeightedBacktester


SOURCES = {
    "earnings_price_research": "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/Data_Library/det_port_form_ep.html",
    "cashflow_price_research": "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/Data_Library/det_port_form_cfp.html",
    "quality_safety_research": "https://www.aqr.com/Insights/Research/Working-Paper/Quality-Minus-Junk",
}




from ..performance import _summary

def run_factor_comparison(settings, *, prepared=None,
                          output_dir="methods/aqr_factors/research/results/factor_comparison", progress=print):
    """Run settings.factor_comparison_sets on identical fees/execution rules.

    own: each specification drops only its own required missing factors.
    common: every specification ranks the same monthly set, requiring all
    factors in all declared specifications. No future sample membership is used.
    Every run starts with cash/NAV on the same configured start date; before
    eligibility begins the account stays in cash. Report Sharpe on active days.
    """
    if settings.missing_value_method != "drop":
        raise ValueError("These missing-data cohort controls require missing_value_method='drop'")
    sets = settings.factor_comparison_sets
    all_factors = tuple(dict.fromkeys(f for names in sets.values() for f in names))
    unknown = set(all_factors)-set(FACTOR_DEFINITIONS)
    if unknown:
        raise ValueError("Unsupported comparison factors: " + str(unknown))
    base = prepared or FactorSignals(settings.signal_config()).run(
        fetch_missing=False, workers=settings.download_workers,
        progress=lambda n,t,s,e: progress(f"Local sources: {n}/{t}") if n%250==0 or n==t else None)
    if any(f not in base.signals for f in all_factors):
        raise ValueError("Prepared signals lack new raw factors; recalculate local source signals once")
    raw = base.signals.copy()
    common = np.isfinite(raw[list(all_factors)]).all(axis=1)
    common &= ~raw.input_status.isin(["download_failed", "missing_month_end_price"])
    if base.config.require_positive_book_equity and "value" in all_factors:
        common &= raw.value_raw.gt(0)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    metadata = dict(signal_settings=asdict(base.config),
        backtest_settings=asdict(settings.backtest_config()),
        construction=asdict(settings.construction_config()),
        specifications=sets, definitions=FACTOR_DEFINITIONS, sources=SOURCES,
        source_metadata=base.metadata, common_cohort_rule="Same-month finite inputs for every declared factor, before sector minimum-size filter",
        primary_result="own: each strategy's independently eligible pool at each scheduled rebalance; common is auxiliary only",
        interpretation="Full configured historical backtest, no train/test split or model fitting. Adding factors is exploratory, with no parameter search or independent validation. Equal weight per factor; expanded changes style weights. This is not a replication of French or AQR portfolio definitions.")
    (out/"comparison_settings.json").write_text(json.dumps(metadata, ensure_ascii=False, default=str, indent=2), encoding="utf-8")
    raw.assign(common_factor_cohort=common).to_csv(out/"monthly_raw_factors.csv.gz", index=False)
    progress("Reading local price histories once for all comparisons")
    histories = WeightedBacktester.cached_histories(base.database,
        sorted(set(raw.ticker)|{"SPY"}), settings.start_date,
        str((pd.Timestamp(settings.end_date)+pd.Timedelta(days=1)).date()))
    rows, curves = [], {}
    cohorts = ("own", "common") if getattr(settings, "factor_comparison_include_common", False) else ("own",)
    for cohort in cohorts:
        for name, factors in sets.items():
            progress(f"Backtest: {cohort}/{name}")
            signals = copy(base)
            signals.metadata = deepcopy(base.metadata)
            signals.signals = raw.copy()
            cfg = replace(base.config, factor_operators={
                f: base.config.factor_operators.get(f, ("group_rank", "group_zscore")) for f in factors})
            signals.rescore(cfg, eligibility_mask=common if cohort=="common" else None)
            construction = replace(settings.construction_config(), factor_columns=tuple(factors))
            result = WeightedBacktester(settings.backtest_config(), construction).run(signals, histories=histories)
            PerformanceReport.audit(result)
            row = _summary(result, signals, name, cohort, settings.initial_nav)
            rows.append(row)
            folder = out/cohort/name
            folder.mkdir(parents=True, exist_ok=True)
            signals.coverage.to_csv(folder/"score_coverage.csv", index=False)
            signals.signals.to_csv(folder/"monthly_scores.csv.gz", index=False)
            for key in ("daily_summary", "target_weights", "rebalance_orders", "events"):
                result[key].to_csv(folder/(key+".csv"), index=key=="daily_summary")
            PerformanceReport.by_year(result).to_csv(folder/"performance_by_year.csv")
            curves[cohort+"/"+name] = result["daily_summary"].equity/settings.initial_nav-1
            progress(f"  Net {row['net_return']:.2%}; first trade {row['first_trade']}; latest stocks {row['latest_ranked']}")
            pd.DataFrame(rows).to_csv(out/"comparison.csv", index=False)
    table = pd.DataFrame(rows)
    pd.DataFrame(curves).to_csv(out/"net_return_curves.csv", index_label="Date")
    table.to_json(out/"comparison.json", orient="records", indent=2, force_ascii=False)
    return table
