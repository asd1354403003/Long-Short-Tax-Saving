"""All formation settings for the retained 2022-2025 cointegration method."""
from dataclasses import replace
from .builder import PairPoolConfig
from .selector import CointegrationParameters
from .rules import PayoffRules

PARAMETERS = CointegrationParameters(
    train_start="2022-01-01",
    train_end="2025-12-31",
    min_observations=800,
    max_half_life_years=0.2,
    min_train_trades=8,
    workers=4,
    pool=replace(PairPoolConfig(), min_market_cap=5_000_000_000, max_assets=2000,
        target_pairs=1000, target_commodity_pairs=150, target_other_etf_pairs=150,
        max_etf_pairs_per_group=15),
    rules=PayoffRules(min_realized_payoff=1.0, min_win_rate=0.5, stop_gaps=(0.9, 1.8)),
    entry_selection="sharpe_then_distance",
    distance_fallback_after_days=10,
    require_reversion=True,
    max_fallback_distance_z=0.5,
)
