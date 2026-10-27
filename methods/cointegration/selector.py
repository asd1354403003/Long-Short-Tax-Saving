"""One frozen formation period; no validation split or walk-forward experiments."""
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field, replace, asdict
import json
import logging
from pathlib import Path
import numpy as np
import pandas as pd
from selection import PoolSelection
from .builder import PairPoolBuilder, PairPoolConfig, PairPoolSource
from .training import PairSignalTrainer, SignalTrainingConfig, HalfLifeFilter
from .rules import PayoffRules, passes_payoff
from .backtest import PairBacktestConfig, PairPoolBacktester

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class CointegrationParameters:
    train_start: str = "2022-01-01"
    train_end: str = "2025-12-31"
    min_observations: int = 800
    max_half_life_years: float = 0.2
    min_train_trades: int = 8
    workers: int = 4
    pool: PairPoolConfig = field(default_factory=PairPoolConfig)
    rules: PayoffRules = field(default_factory=PayoffRules)
    entry_selection: str = "sharpe_then_distance"
    distance_fallback_after_days: int = 10
    require_reversion: bool = True
    max_fallback_distance_z: float = 0.5

    def __post_init__(self):
        if not pd.Timestamp(self.train_start) < pd.Timestamp(self.train_end):
            raise ValueError("Training start must precede its end.")
        for name in ("min_observations", "min_train_trades", "workers"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(name + " must be a positive integer.")
        if self.min_observations < 20 or self.workers > 4:
            raise ValueError("Use at least 20 observations and at most four workers.")
        HalfLifeFilter.from_limits(max_years=self.max_half_life_years)


def _train_chunk(formation, histories, parameters, training_config):
    trainer = PairSignalTrainer(training_config)
    start, end = pd.Timestamp(parameters.train_start), pd.Timestamp(parameters.train_end)
    assert all(frame.empty or frame.index.max() <= end for frame in histories.values())
    calendar = histories["SPY"].loc[start:end].index
    half_filter = HalfLifeFilter.from_limits(max_years=parameters.max_half_life_years)
    rules, rows, counts = parameters.rules, [], Counter()
    for pair in formation.to_dict("records"):
        try:
            y, x = pair["y_ticker"], pair["x_ticker"]
            prices = pd.concat([histories[y][["Open", "Close"]].add_suffix("_y"),
                                histories[x][["Open", "Close"]].add_suffix("_x")], axis=1)
            prices = prices.reindex(calendar).replace([np.inf, -np.inf], np.nan).dropna()
            if len(prices) < parameters.min_observations or not prices.gt(0).all().all():
                counts["insufficient_prices"] += 1
                continue
            alpha, beta = trainer._fit(prices)
            spread = prices.Close_y - alpha - beta * prices.Close_x
            ou = trainer.fit_legacy_scale(spread)
            try:
                half_life = HalfLifeFilter.estimate(spread.reindex(calendar))
            except ValueError:
                counts["invalid_half_life"] += 1
                continue
            if not half_filter.accepts(half_life["half_life_days"]):
                counts["half_life_rejected"] += 1
                continue
            z = trainer.z_score(spread, ou["mu_e"], ou["sigma_eq"])
            types = (pair["instrument_type_y"], pair["instrument_type_x"])
            steps = (int(pair["unit_step_y"]), int(pair["unit_step_x"]))
            choices = []
            for entry in rules.entries:
                for exit_ in rules.exits:
                    for gap in rules.stop_gaps:
                        risk = rules.model_fields(entry, gap, ou["half_life_days"])
                        upper_reward = entry + risk["entry_band_z"] - exit_
                        lower_risk = risk["stop_z"] - entry - risk["entry_band_z"] + risk["stop_overshoot_z"]
                        if upper_reward + 1e-12 < risk["min_entry_reward_risk"] * lower_risk:
                            continue
                        stats = trainer._simulate(prices, z, beta, entry, exit_, start, end,
                            instrument_types=types, unit_steps=steps, risk=risk, sigma_eq=ou["sigma_eq"])
                        if (stats["natural_trades"] >= parameters.min_train_trades and stats["pnl"] > 0
                                and np.isfinite(stats["sharpe"]) and stats["sharpe"] > 0
                                and passes_payoff(stats, rules)):
                            choices.append((stats, entry, exit_, risk))
            if not choices:
                counts["no_training_choice"] += 1
                continue
            stats, entry, exit_, risk = max(choices,
                key=lambda item: (item[0]["return_on_budget"], item[0]["sharpe"], -item[1]))
            pair.update(refit_alpha=alpha, refit_beta=beta, refit_mu_e=ou["mu_e"], refit_sigma_eq=ou["sigma_eq"],
                refit_status="ok", refit_end=str(end.date()), entry_z=entry, exit_z=exit_, stop_loss=.12,
                prevent_immediate_reentry=True, execution_timing=training_config.execution_timing,
                parameter_train_start=str(start.date()), parameter_train_end=str(end.date()),
                ranking_basis="formation_net_sharpe_no_validation_split", eligible=True,
                selection_sharpe=stats["sharpe"], selection_end=str(end.date()),
                training_min_win_rate=rules.min_win_rate, training_min_realized_payoff=rules.min_realized_payoff,
                training_stop_z=risk["stop_z"], training_half_life_days=ou["half_life_days"],
                max_half_life_years=parameters.max_half_life_years,
                training_transaction_cost_rate=training_config.transaction_cost_rate,
                training_borrow_fee_rate=training_config.borrow_fee_rate,
                training_fx_financing_rate=training_config.fx_financing_rate,
                training_leg_amount=training_config.long_amount, **risk)
            pair.update({"selection_" + key: value for key, value in half_life.items()})
            for key in ("pnl", "sharpe", "natural_trades", "payoff_ratio", "winners", "losers",
                        "profit_factor", "natural_win_rate"):
                pair["training_" + key] = stats[key]
            rows.append(pair)
        except (ValueError, np.linalg.LinAlgError, FloatingPointError):
            counts["model_error"] += 1
    return pd.DataFrame(rows), dict(counts)


class Selector:
    def __init__(self, parameters, config, result_dir):
        self.parameters, self.config = parameters, config
        self.result_dir = Path(__file__).resolve().parents[2] / Path(result_dir)
        self.pool_file = Path(__file__).with_name("data") / "pool.csv"
        self.metadata_file = self.result_dir / "result.json"
        self.training_config = SignalTrainingConfig(long_amount=config.long_amount/config.max_positions_per_side,
            borrow_fee_rate=config.borrow_fee_rate, transaction_cost_rate=config.transaction_cost_rate,
            fx_financing_rate=config.fx_financing_rate, execution_timing="close")
        if pd.Timestamp(parameters.train_end) >= pd.Timestamp(config.start_date):
            raise ValueError("Frozen pool training must finish before the backtest starts.")

    def select(self, *, rebuild=False):
        signature = json.loads(json.dumps(dict(parameters=asdict(self.parameters),
            training=asdict(self.training_config)), default=str))
        # Path separators describe the same files across notebook/CLI platforms;
        # they are not a change to the formation or trading rules.
        path_fields = ("relationship_file", "etf_catalog_file", "output_file")
        for key in path_fields:
            signature["parameters"]["pool"][key] = signature["parameters"]["pool"][key].replace("\\", "/")
        if not rebuild and self.pool_file.exists() and self.metadata_file.exists():
            metadata = json.loads(self.metadata_file.read_text(encoding="utf-8"))
            for key in path_fields:
                metadata["selection_settings"]["parameters"]["pool"][key] = (
                    metadata["selection_settings"]["parameters"]["pool"][key].replace("\\", "/"))
            if metadata.get("selection_settings") != signature:
                raise ValueError("Pool selection parameters/costs changed. Rebuild the pool explicitly.")
            return PoolSelection("pairs", pd.read_csv(self.pool_file), metadata)
        params = self.parameters
        settings = replace(params.pool, cointegration=replace(params.pool.cointegration,
            train_start=params.train_start, train_end=params.train_end,
            min_observations=params.min_observations, max_lag=1, check_i1=False), liquidity_window=60)
        source = PairPoolSource(settings)
        result = PairPoolBuilder(settings, source=source).build(save=False)
        formation = result["pairs"].copy()
        formation["candidate_order"] = np.arange(len(formation))
        tickers = list(dict.fromkeys(result["universe"].ticker.tolist()+["SPY"]))
        # Reuse exactly the builder's requested historical slice from its memory cache.
        histories = source.history(tickers, params.train_start,
            str((pd.Timestamp(params.train_end)+pd.Timedelta(days=1)).date()))
        histories = {ticker: frame.loc[params.train_start:params.train_end].copy() for ticker, frame in histories.items()}
        counts, chunks = Counter(), []
        pieces = [formation.iloc[index].copy() for index in np.array_split(np.arange(len(formation)), params.workers) if len(index)]
        if params.workers == 1:
            fitted = [_train_chunk(pieces[0], histories, params, self.training_config)]
        else:
            fitted = []
            with ProcessPoolExecutor(max_workers=params.workers) as executor:
                futures = [executor.submit(_train_chunk, piece, histories, params, self.training_config) for piece in pieces]
                for future in as_completed(futures):
                    fitted.append(future.result())
                    LOGGER.info("Cointegration training: %s/%s chunks complete", len(fitted), len(pieces))
        for frame, report in fitted:
            if not frame.empty:
                chunks.append(frame)
            counts.update(report)
        if not chunks:
            raise ValueError("No pair passed the configured training gates.")
        pool = pd.concat(chunks, ignore_index=True).sort_values("candidate_order", kind="stable")
        pool = pool.sort_values("selection_sharpe", ascending=False, kind="stable").reset_index(drop=True)
        metadata = dict(method="cointegration", train_start=params.train_start, train_end=params.train_end,
            validation_split=None, formation_pairs=len(formation), selected_pairs=len(pool),
            selection_settings=signature, counts=dict(counts),
            limitations=["Current universe metadata introduce survivorship bias.",
                         "The training spread-stop grid is retained; deployment exits are configured separately."])
        self.result_dir.mkdir(parents=True, exist_ok=True)
        self.pool_file.parent.mkdir(parents=True, exist_ok=True)
        pool.to_csv(self.pool_file, index=False)
        self.metadata_file.write_text(json.dumps(metadata, indent=2, allow_nan=False), encoding="utf-8")
        return PoolSelection("pairs", pool, metadata, source, histories)

    def backtest(self, selection, *, histories=None):
        cfg, params = self.config, self.parameters
        if cfg.short_amount != cfg.long_amount:
            raise ValueError("The retained pair adapter uses equal long/short entry caps.")
        if cfg.exit_on_pool_removal or not cfg.allow_reentry:
            raise ValueError("Pool-removal exits and a no-reentry rule belong to the independent engine.")
        engine_config = PairBacktestConfig(start_date=cfg.start_date, end_date=cfg.end_date,
            max_pairs=cfg.max_positions_per_side, long_amount=cfg.long_amount,
            borrow_fee_rate=cfg.borrow_fee_rate, transaction_cost_rate=cfg.transaction_cost_rate,
            fx_financing_rate=cfg.fx_financing_rate, execution_timing=cfg.execution_timing,
            entry_selection=params.entry_selection, distance_fallback_after_days=params.distance_fallback_after_days,
            require_reversion=params.require_reversion, max_fallback_distance_z=params.max_fallback_distance_z,
            enable_stop_loss=cfg.stop_loss_amount is not None, exit_stop_loss_amount=cfg.stop_loss_amount,
            exit_take_profit_amount=cfg.take_profit_amount, exit_on_mean_reversion=cfg.exit_on_signal)
        return PairPoolBacktester(selection.pool, engine_config, source=selection.source).run(histories=histories)
