"""Market Neutral source, scoring and baseline execution settings. Selected runs live in result/aqr/."""
from datetime import date, timedelta
from .signals import SignalConfig
from risk import RISK_FREE_SETTINGS

class AQRSettings:
    """Monthly factor data, scoring, construction and NAV execution settings."""
    start_date = "2022-01-01"  # Full historical backtest, no train/test split.
    end_date = "2026-10-22"  # Latest complete individual-stock price cache; inclusive.
    database_path = "aqr_research/data/aqr.sqlite3"
    initial_nav = 1_000_000
    transaction_cost_rate = 0.0005  # 5 bps on each actual buy/sell, including trims.
    borrow_fee_rate = 0.01  # Annual short market value, ACT/360.
    fetch_missing_sources = False  # This run uses local prices/shares and member-site CSVs.
    financial_source = "website_quarterly"  # website_quarterly or legacy_annual.
    financial_availability = "period_lag"  # Missing dates only: period end + lag, an explicit assumption.
    financial_reporting_lag_days = 90  # Known dates retained; raw publication dates are never changed.
    # existing_dates_only reproduces the conservative dated-report-only comparison.
    missing_value_method = "drop"  # drop, mean or median.
    missing_value_scope = "score_group"  # score_group or universe, within the same month only.
    require_positive_book_equity = False  # False retains finite negative/zero B/M in ranking.
    rebalance_frequency = "quarterly"  # Check eligible candidates at the quarter-end decision cutoff.
    download_workers = 10
    recovery_request_interval_seconds = 20
    recovery_start_date = "2022-01-01"  # Data recovery includes the 2021-12-31 decision cutoff.
    recovery_initial_cooldown_seconds = 900
    recovery_rate_limit_cooldown_seconds = 1800
    recovery_official_fallback_after_hours = 2
    universe = "russell1000_iwb_proxy"
    # Other choices: custom, russell1000 (official CSV).
    tickers = ()
    membership_file = None
    group_by = "sector"  # Historical IWB sector. Other choices: industry or None.
    factor_operators = {
        "value": ("group_rank", "group_zscore"),
        "quality": ("group_rank", "group_zscore"),
        "momentum": ("group_rank", "group_zscore"),
    }
    # Add/remove keys above to select factors. All are oriented higher = better.
    # earnings_yield = four available quarters' common net income / market cap (1/PE for profits).
    # cashflow_yield = four available quarters' operating cash flow / market cap.
    # roa = four available quarters' net income / ending total assets.
    # low_volatility = negative 126-session annualized daily-return volatility.
    # Example: "earnings_yield": ("group_rank", "group_zscore").
    # Predeclared comparisons, not a parameter search; the live default remains the original three.
    factor_comparison_sets = {
        "baseline": ("value", "quality", "momentum"),
        "plus_ep": ("value", "quality", "momentum", "earnings_yield"),
        "plus_cashflow": ("value", "quality", "momentum", "cashflow_yield"),
        "plus_roa": ("value", "quality", "momentum", "roa"),
        "plus_low_vol": ("value", "quality", "momentum", "low_volatility"),
        "expanded": ("value", "quality", "momentum", "earnings_yield", "cashflow_yield", "roa", "low_volatility"),
    }
    factor_comparison_include_common = False  # Main results use each method's own eligible pool per date.
    # Default notebook preset; its explicit fields override the baseline above.
    selected_parameter_file = "result/aqr/monthly_sharpe158/parameters.py"
    stable_monthly_candidate_file = selected_parameter_file
    # Replay choices: 1 = supplied alpha -> demean/L1; 2 = configured factor
    # model and overlays; 3 = AQR reference method, reserved and not implemented.
    # Explicit preset fields override these defaults.
    weighting_method = 2
    alpha_column = "score"  # Choice 1 only: cached baseline three-factor alpha.
    risk_free = dict(RISK_FREE_SETTINGS)  # Historical daily yields, never a zero-rate default.
    combine_operator = "vec_avg"  # vec_avg or vec_sum across each stock's factors.
    min_stocks = 10
    construction_method = "demean_l1"  # Low-level baseline, not the replay weighting selector.
    construction_group_by = "sector"  # None uses one group for the whole universe.
    construction_gross_exposure = 1.0  # NAV gross: full target for demean_l1; cap for aqr_style.
    aqr_construction = {
        "factor_columns": ("value", "quality", "momentum"),
        "risk_lookback": 252,
        "risk_min_observations": 120,
        "risk_lag_months": 1,
        "covariance_shrinkage": 0.1,
        "risk_aversion": 5.0,
        "alpha_scale": 1.0,
        "max_weight": 0.05,
        "net_exposure": 0.0,  # Default: neutral component. Nonzero needs benchmark_weights.
        "target_beta": 0.0,
    }

    @classmethod
    def performance_config(cls):
        return dict(risk_free=dict(cls.risk_free), cash_interest_in_nav=False,
                    annualization_sessions=252)

    @classmethod
    def construction_config(cls):
        from methods.aqr_factors.construction import ConstructionConfig
        return ConstructionConfig(method=cls.construction_method,
            group_by=cls.construction_group_by, gross_exposure=cls.construction_gross_exposure,
            min_group_size=cls.min_stocks, transaction_cost_rate=cls.transaction_cost_rate,
            borrow_fee_rate=cls.borrow_fee_rate, **cls.aqr_construction)

    @classmethod
    def backtest_config(cls):
        from methods.aqr_factors.backtest import WeightedBacktestConfig
        return WeightedBacktestConfig(start_date=cls.start_date,end_date=cls.end_date,
            initial_nav=cls.initial_nav, transaction_cost_rate=cls.transaction_cost_rate,
            borrow_fee_rate=cls.borrow_fee_rate, rebalance_frequency=cls.rebalance_frequency)

    @classmethod
    def signal_config(cls):
        return SignalConfig(
            universe=cls.universe, tickers=cls.tickers, membership_file=cls.membership_file,
            group_by=cls.group_by, factor_operators=cls.factor_operators,
            combine_operator=cls.combine_operator, min_stocks=cls.min_stocks,
            financial_source=cls.financial_source,
            financial_availability=cls.financial_availability,
            financial_reporting_lag_days=cls.financial_reporting_lag_days,
            missing_value_method=cls.missing_value_method,
            missing_value_scope=cls.missing_value_scope,
            require_positive_book_equity=cls.require_positive_book_equity,
            start=cls.start_date,
            end_exclusive=(date.fromisoformat(cls.end_date) + timedelta(days=1)).isoformat(),
            database_path=cls.database_path,
        )

    @classmethod
    def monthly_universe(cls):
        """Prepare only historical IWB membership; do not fetch all stock financials."""
        from methods.aqr_factors.database import AQRDatabase
        from methods.aqr_factors.membership import MonthlyMembership
        return MonthlyMembership(AQRDatabase(cls.database_path)).prepare(cls.start_date, cls.end_date)

    @classmethod
    def prepare_fundamentals(cls):
        """Cache full financial fields for all historical universe members; no backtest."""
        from methods.aqr_factors.fundamentals import FundamentalStore
        from methods.aqr_factors.universe import Universe
        from methods.aqr_factors.database import AQRDatabase
        database = AQRDatabase(cls.database_path)
        universe = Universe(cls.universe, cls.tickers, cls.membership_file, database=database,
                            start_date=cls.start_date, end_date=cls.end_date)
        return FundamentalStore(database).prepare_universe(universe, cls.start_date, cls.end_date)
