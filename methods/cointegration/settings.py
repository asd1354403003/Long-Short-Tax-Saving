"""Frozen-pool cointegration selection and deployment settings."""
from backtest import BacktestConfig

class CointegrationSettings:
    """Retained pair method. Its formation settings remain in its parameter file."""
    selector_module = "methods.cointegration.selector"
    parameter_file = "methods/cointegration/parameters.py"
    result_dir = "result/cointegration/frozen2026"
    rebuild_pool = False

BACKTEST = BacktestConfig(
    start_date="2026-01-01",
    end_date="2026-10-22",
    long_amount=1_000_000,
    short_amount=1_000_000,
    max_positions_per_side=15,
    execution_timing="close",       # "close" or "next_open"
    take_profit_amount=2_000,        # Net liquidation P&L; None disables.
    stop_loss_amount=None,           # USD loss magnitude; None disables.
    exit_on_signal=True,             # Valuation exit flags, or pair mean exit.
    exit_on_pool_removal=False,
    allow_reentry=True,
    borrow_fee_rate=0.01,
    transaction_cost_rate=0.0005,     # 5 bps on every executed buy/sell.
    fx_financing_rate=0.0,
)
