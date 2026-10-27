"""Run a saved Market Neutral parameter dictionary against local data; no search or downloads.

The dated input snapshot lives beside the SQLite source database. It preserves
the exact inputs used to select the saved strategies, including publication-lag
assumptions. Every run recomputes ranking, targets and the transaction ledger.
"""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd

from reports import PerformanceReport
from risk import historical_risk_free
from .backtest import WeightedBacktestConfig, WeightedBacktester
from .construction import ConstructionConfig, PortfolioConstructor
from .factors import balance_ratios
from .performance import _summary, side_attribution
from .portfolio import (FACTOR_NAMES, FrozenTargetsConstructor, combine_targets,
                        decision_rows, smooth_targets, target_arrays, transform_targets, cap_targets)
from .signals import FactorSignals, SignalConfig

PROJECT_ROOT = Path(__file__).resolve().parents[2]
WEIGHTING_METHODS = {
    1: "Alpha demean / L1 (supplied scores)",
    2: "Factor scoring and configured model construction",
    3: "AQR reference construction (not implemented)",
}


def project_path(relative):
    """Resolve project-relative data paths without depending on the notebook CWD."""
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("Data paths must be relative to the project")
    return PROJECT_ROOT / path


def load_parameters(parameter_file=None):
    """Use an explicit preset, or the preset selected in AQRSettings when None.

    This does not replace the selected model with the separate baseline model.
    A misspelled explicit path raises instead of silently running another model.
    Execution and performance fields omitted by a preset use method defaults.
    """
    import runpy
    from .settings import AQRSettings
    chosen = AQRSettings.selected_parameter_file if parameter_file is None else parameter_file
    path = Path(chosen)
    path = path.resolve() if path.is_absolute() else project_path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Market Neutral parameter file does not exist: {path}")
    parameters = runpy.run_path(str(path))["PARAMETERS"]
    parameters["backtest"] = {**asdict(AQRSettings.backtest_config()), **parameters.get("backtest", {})}
    performance = {**AQRSettings.performance_config(), **parameters.get("performance", {})}
    performance["risk_free"] = {**AQRSettings.risk_free, **performance.get("risk_free", {})}
    parameters["performance"] = performance
    parameters.setdefault("weighting_method", AQRSettings.weighting_method)
    parameters.setdefault("alpha_column", AQRSettings.alpha_column)
    return parameters, path


def parameter_fingerprint(parameters):
    """Detect stale saved results after an execution, model or RF setting edit."""
    canonical = dict(parameters)
    choice = weighting_method(parameters)
    if choice == 2:
        # Explicit choice 2 is the unchanged historical runner. Ignore its
        # inactive alpha column and retain compatibility with existing results.
        canonical.pop("weighting_method", None)
        canonical.pop("alpha_column", None)
    else:
        from .settings import AQRSettings
        canonical["weighting_method"] = choice
        canonical["alpha_column"] = parameters.get("alpha_column", AQRSettings.alpha_column)
    return hashlib.sha256(json.dumps(canonical, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode("utf-8")).hexdigest()


def weighting_method(parameters):
    """Resolve the public three-way choice before any local data is loaded.

    The legacy internal aqr_style approximation is not the full reference
    implementation and is deliberately not substituted for reserved choice 3.
    """
    from .settings import AQRSettings
    choice = parameters.get("weighting_method", AQRSettings.weighting_method)
    if type(choice) is not int or choice not in WEIGHTING_METHODS:
        raise ValueError("weighting_method must be 1, 2 or 3")
    if choice == 3:
        raise NotImplementedError("Weighting method 3 (AQR reference) is reserved, not implemented")
    return choice


def portfolio_targets(parameters, raw, tickers, decisions, groups, volatility, beta):
    """Select actual target construction, not just a display label.

    Choice 1 applies the user's formula directly to a dated alpha column.
    It does not rank the supplied alpha, select tails, blend models, constrain
    beta, size by volatility or enforce the model's single-name cap. With the
    bundled alpha_column='score', alpha is the frozen baseline's three-factor
    score, NOT the selected complex model's alpha. Other dated factor columns
    may be selected explicitly. The constructor itself also accepts arbitrary
    caller-supplied alpha through build(..., score_column=...).

    Choice 2 preserves the saved model's scoring and all configured overlays.
    Both choices use the same subsequent execution, costs and RF accounting.
    """
    choice = weighting_method(parameters)
    if choice == 2:
        return _model_targets(parameters["model"], raw, tickers, decisions,
                              groups, volatility, beta, {})
    from .settings import AQRSettings
    column = parameters.get("alpha_column", AQRSettings.alpha_column)
    if column not in raw:
        raise ValueError(f"Unknown dated alpha column: {column}")
    config = ConstructionConfig(**parameters["portfolio"])
    if config.method != "demean_l1":
        raise ValueError("Choice 1 requires portfolio.method='demean_l1'")
    frame = raw.copy()
    frame["in_universe"] = (frame.in_universe.eq(True)
        & ~frame.input_status.isin(["download_failed", "missing_month_end_price"])
        & frame.close.gt(0) & frame.market_cap.gt(0)
        & np.isfinite(pd.to_numeric(frame[column], errors="coerce")))
    if config.group_by is not None:
        if config.group_by != parameters["signals"]["group_by"]:
            raise ValueError("Choice 1 grouping must match the frozen classification or be None")
        frame[config.group_by] = frame.score_group
        frame["in_universe"] &= frame.score_group.notna() & frame.score_group.ne("")
    keys = ["as_of"] + ([config.group_by] if config.group_by else [])
    eligible = frame.loc[frame.in_universe]
    sizes = eligible.groupby(keys)["ticker"].transform("size")
    frame.loc[eligible.index, "in_universe"] &= sizes.ge(config.min_group_size)
    targets = PortfolioConstructor(config).build(frame, score_column=column)
    return (targets.pivot(index="as_of", columns="ticker", values="target_weight")
            .reindex(index=decisions, columns=tickers).fillna(0).to_numpy())


def replay_file(parameter_file=None, *, output=None):
    """Replay the selected preset using only local data, without research outputs."""
    parameters, path = load_parameters(parameter_file)
    return run(parameters,output=Path(output) if output is not None else path.with_name("result.json"))


def records(frame, *, index=False):
    """JSON-safe dates and nulls, preserving enough precision to replay NAV."""
    if index:
        frame = frame.reset_index()
    return json.loads(frame.to_json(orient="records", date_format="iso", double_precision=15))


def model_factors(model):
    if "factor_weights" in model:
        return set(model["factor_weights"])
    return set().union(*(model_factors(child) for child in model["component_models"].values()))


def _model_targets(model, raw, tickers, decisions, groups, volatility, beta, cache):
    signature = hashlib.sha256(json.dumps(model, sort_keys=True).encode()).hexdigest()
    if signature in cache:
        return cache[signature]
    if "component_allocations" in model:
        names = list(model["component_allocations"])
        components = np.array([
            _model_targets(model["component_models"][name], raw, tickers,
                           decisions, groups, volatility, beta, cache) for name in names])
        weights = combine_targets(components, list(model["component_allocations"].values()))
        # Normalize at EVERY node, as in the selected model; do not flatten.
        weights = transform_targets(weights, groups, volatility, beta, beta_neutral=1.)
    else:
        factors = tuple(model["factor_weights"])
        allocations = tuple(model["factor_weights"].values())
        if not np.isfinite(allocations).all() or min(allocations) < 0 or not np.isclose(sum(allocations), 1):
            raise ValueError("Factor weights must be nonnegative and sum to one")
        spec = dict(variant=signature, factors=factors, weights=allocations)
        spec.update(model.get("scoring", {}))
        base, _ = target_arrays(raw, [spec], tickers, decisions, progress=lambda *_: None)
        weights = transform_targets(base[0], groups, volatility, beta, **model["construction"])
        if "smoothing" in model:
            weights = smooth_targets(weights, abs(base[0]) > 1e-12, groups, beta, **model["smoothing"])
    if model.get("final_risk_lookback", 252) != 252 or "risk_overlay" in model:
        raise ValueError("This saved-input runner supports the recorded 252-session risk model")
    if "sector_filter" in model:
        labels=sorted(raw.score_group.dropna().unique())
        unknown=set(model["sector_filter"])-set(labels)
        if unknown:
            raise ValueError(f"Unknown sectors: {unknown}")
        mask=[labels.index(label) for label in model["sector_filter"]]
        weights=transform_targets(weights,groups,volatility,beta,sector_mask=mask,beta_neutral=1.)
    if "weight_cap" in model:
        weights=cap_targets(weights,groups,beta,max_weight=model["weight_cap"])
    cache[signature] = weights
    return weights


def run(parameters, *, output=None, progress=print):
    """Return one compact result and optionally save only that result.json.

    PARAMETERS includes the entire model, costs and date settings. Factor weights
    and construction can be edited directly. The input snapshot has fixed source,
    universe and rank semantics; regenerate inputs before changing those rules.
    """
    choice = weighting_method(parameters)
    cfg = WeightedBacktestConfig(**parameters["backtest"])
    signal_settings = dict(parameters["signals"], database_path=parameters["database_path"])
    folder = project_path(parameters["input_folder"])
    metadata = json.loads((folder / "metadata.json").read_text(encoding="utf-8"))
    fixed = metadata["signal_settings"]
    for field in ("universe", "group_by", "combine_operator", "min_stocks", "factor_operators",
                  "financial_source", "financial_availability", "financial_reporting_lag_days",
                  "missing_value_method", "missing_value_scope", "require_positive_book_equity"):
        if signal_settings[field] != fixed[field]:
            raise ValueError(f"{field} differs from the frozen inputs; regenerate the dated input snapshot first")
    if cfg.start_date < fixed["start"] or pd.Timestamp(cfg.end_date) >= pd.Timestamp(fixed["end_exclusive"]):
        raise ValueError("Backtest dates exceed the frozen local input window")
    progress(f"{parameters['name']}: loading local dated inputs; no source requests", flush=True)
    raw = pd.read_csv(folder / "monthly_raw_factors.csv.gz", parse_dates=["as_of"], low_memory=False)
    raw = decision_rows(raw, cfg.rebalance_frequency)
    # Keep the preceding cutoff needed by the first execution session.
    first_cutoff = pd.Timestamp(cfg.start_date).to_period("M").start_time - pd.Timedelta(days=1)
    raw = raw.loc[raw.as_of.ge(first_cutoff) & raw.as_of.lt(pd.Timestamp(cfg.end_date))].copy()
    for name, values in balance_ratios(raw).items():
        raw[name] = values
    extra_columns = list(FACTOR_NAMES[7:]) + ["estimated_volatility", "estimated_beta"]
    extra = pd.read_csv(folder / "monthly_extra_inputs.csv.gz", parse_dates=["as_of"],
                        usecols=["as_of", "ticker"] + extra_columns)
    raw = raw.merge(extra, on=["as_of", "ticker"], how="left", validate="one_to_one")
    raw.loc[~raw.total_assets.gt(0), ["quality", "roa"]] = np.nan
    signals = FactorSignals(SignalConfig(**signal_settings))
    signals.metadata = metadata["source_metadata"].copy()
    signals.signals = raw.copy()
    profiles = {}
    for row in signals.database.read(
            "SELECT ticker,payload FROM snapshots WHERE dataset='company' ORDER BY observed_at").itertuples():
        profiles[row.ticker] = json.loads(row.payload)
    signals.companies = pd.DataFrame([dict(ticker=t, industry=i.get("industry"), sector=i.get("sector"))
                                     for t, i in profiles.items()])
    tickers = sorted(set(raw.ticker) | {"SPY"})
    decisions = pd.DatetimeIndex(sorted(raw.as_of.unique()))
    sectors = sorted(raw.score_group.dropna().unique())
    months = [raw.loc[raw.as_of.eq(date)].set_index("ticker").reindex(tickers) for date in decisions]
    groups = np.array([pd.Categorical(m.score_group, categories=sectors).codes for m in months])
    volatility = np.array([m.estimated_volatility.to_numpy(dtype=float) for m in months])
    beta = np.array([m.estimated_beta.to_numpy(dtype=float) for m in months])
    progress(f"Recalculating {len(decisions)} {cfg.rebalance_frequency} decisions", flush=True)
    portfolio = ConstructionConfig(**parameters["portfolio"])
    progress(f"Weighting {choice}: {WEIGHTING_METHODS[choice]}", flush=True)
    weights = portfolio_targets(parameters, raw, tickers, decisions, groups, volatility, beta)
    target_gross = portfolio.gross_exposure if choice == 1 else 1.
    if not np.allclose(abs(weights).sum(axis=1), target_gross, atol=1e-8):
        raise ValueError("The proposed model leaves an empty or unnormalized portfolio")
    # The constructor executes the dated weights; no selection on future completeness.
    dated = pd.DataFrame(weights, index=decisions, columns=tickers).stack()
    signals.signals["score"] = dated.reindex(pd.MultiIndex.from_frame(raw[["as_of", "ticker"]])).to_numpy()
    signals.coverage = pd.DataFrame(dict(as_of=decisions, ranked_stocks=(abs(weights) > 1e-10).sum(axis=1)))
    engine = WeightedBacktester(cfg, portfolio)
    engine.constructor = FrozenTargetsConstructor(portfolio, weights, tickers, decisions)
    histories = engine.cached_histories(signals.database, tickers, cfg.start_date,
                                       str((pd.Timestamp(cfg.end_date) + pd.Timedelta(days=1)).date()))
    progress("Executing integer-share ledger using cached Open/Close prices", flush=True)
    result = engine.run(signals, histories=histories)
    rf = historical_risk_free(result["daily_summary"].index,
                             parameters.get("performance", {}).get("risk_free"))
    result["daily_risk_free"] = rf
    result["daily_summary"]["risk_free_return"] = rf
    result["daily_summary"]["excess_return"] = result["daily_summary"].daily_return - rf
    audit = PerformanceReport.audit(result).to_dict()
    metrics = _summary(result, signals, parameters["variant"], "selected", cfg.initial_nav)
    if choice == 2:
        metrics["factors"] = " + ".join(sorted(model_factors(parameters["model"])))
    else:
        from .settings import AQRSettings
        column = parameters.get("alpha_column", AQRSettings.alpha_column)
        metrics["factors"] = (" + ".join(sorted(signal_settings["factor_operators"]))
                             if column == "score" else column)
    annual = PerformanceReport.by_year(result).reset_index()
    sides = side_attribution(result, cfg)
    constraints = []
    for t, date in enumerate(decisions):
        constraints.append(dict(as_of=date, long_weight=weights[t].clip(min=0).sum(),
            short_weight=weights[t].clip(max=0).sum(), gross=abs(weights[t]).sum(), net=weights[t].sum(),
            estimated_beta=weights[t] @ np.nan_to_num(beta[t]), max_weight=abs(weights[t]).max(),
            max_abs_sector_net=max(abs(weights[t, groups[t] == group].sum()) for group in np.unique(groups[t])),
            target_longs=int((weights[t] > 1e-10).sum()), target_shorts=int((weights[t] < -1e-10).sum())))
    document = dict(method="market_neutral", name=parameters["name"], parameters_sha256=parameter_fingerprint(parameters),
        weighting=dict(choice=choice, description=WEIGHTING_METHODS[choice],
            alpha_column=column if choice == 1 else None,
            risk_controls="none; dollar/group demeaning only" if choice == 1 else "as configured in model"),
        backtest=asdict(cfg), metrics=metrics,
        annual_returns=records(annual), annual_long_short=records(sides),
        daily=records(result["daily_summary"], index=True),
        rebalance_summary=records(pd.DataFrame(constraints)), verification=audit,
        performance=dict(risk_free_settings=rf.attrs["settings"], risk_free_source=rf.attrs["provenance"],
            risk_free_basis=rf.attrs["basis"], sharpe="sqrt(252) * mean(daily net return - RF) / sample_std(daily net return - RF)",
            cash_interest_in_nav=False, goal_sharpe=1.58, sharpe_goal_met=bool(metrics["sharpe"] >= 1.58),
            risk_free_cumulative_return=float((1 + rf).prod() - 1)),
        data=dict(database_path=parameters["database_path"], input_folder=parameters["input_folder"],
                  source_requests=0, source_metadata=signals.metadata),
        interpretation=parameters["interpretation"],
        limitations=["In-sample model selection; not independent out-of-sample evidence.",
            "Latest vendor financial versions; known dates or assumed period-end +90 days.",
            "Dividends, short dividend compensation, cash interest and tax excluded."],
        storage="Only summary, daily NAV and annual attribution are persisted; rerun for the full in-memory ledger.")
    # pandas handles timestamps, numpy scalars and nonfinite values consistently.
    document = json.loads(pd.Series(document).to_json(date_format="iso", double_precision=15))
    if output is not None:
        destination = Path(output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(document, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    progress(f"Sharpe {metrics['sharpe']:.4f}; cumulative net return {metrics['net_return']:.2%}", flush=True)
    return document
