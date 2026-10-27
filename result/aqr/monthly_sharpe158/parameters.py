"""Complete runnable monthly Market Neutral research parameters using historical RF.
Fixed factor/sector model across all years; account gross=1, long/short targets .5 each.
A final projection enforces a 5% single-name cap, zero sector and estimated-beta exposure.
Run this file inside the project to recompute from local data, with no search/download.
"""

# Shared submodels are defined once; all parameters are self-contained.

# Factors: cashflow_yield, low_volatility, gross_profitability, average_asset_roa
MODEL_001 = {'factor_weights': {'cashflow_yield': 0.2,
                    'low_volatility': 0.1,
                    'gross_profitability': 0.2,
                    'average_asset_roa': 0.5},
 'construction': {'tail': 0.035,
                  'power': 0.0,
                  'inverse_vol': 0.0,
                  'sector_budget': 'signal',
                  'beta_neutral': 1.0}}

# Fixed component blend; sector: Communication
MODEL_002 = {'component_allocations': {'source': 1.0},
 'component_models': {'source': MODEL_001},
 'sector_filter': ['Communication']}

# Factors: cashflow_yield, low_volatility, gross_profitability, average_asset_roa
MODEL_003 = {'factor_weights': {'cashflow_yield': 0.2,
                    'low_volatility': 0.1,
                    'gross_profitability': 0.2,
                    'average_asset_roa': 0.5},
 'construction': {'tail': 0.05,
                  'power': 0.0,
                  'inverse_vol': 0.0,
                  'sector_budget': 'signal',
                  'beta_neutral': 1.0}}

# Fixed component blend; sector: Communication
MODEL_004 = {'component_allocations': {'source': 1.0},
 'component_models': {'source': MODEL_003},
 'sector_filter': ['Communication']}

# Fixed component blend; sector: Communication
MODEL_005 = {'component_allocations': {'source': 1.0},
 'component_models': {'source': MODEL_002},
 'sector_filter': ['Communication']}

# Fixed component blend; sector: Communication
MODEL_006 = {'component_allocations': {'source': 1.0},
 'component_models': {'source': MODEL_004},
 'sector_filter': ['Communication']}

# Factors: cashflow_yield, roa, low_volatility, gross_profitability
MODEL_007 = {'factor_weights': {'cashflow_yield': 0.3375,
                    'roa': 0.275,
                    'low_volatility': 0.1875,
                    'gross_profitability': 0.2},
 'construction': {'tail': 0.05,
                  'power': 0.0,
                  'inverse_vol': 0.0,
                  'sector_budget': 'signal',
                  'beta_neutral': 1.0}}

# Fixed component blend; sector: Communication
MODEL_008 = {'component_allocations': {'source': 1.0},
 'component_models': {'source': MODEL_007},
 'sector_filter': ['Communication']}

# Fixed component blend; sector: Communication
MODEL_009 = {'component_allocations': {'source': 1.0},
 'component_models': {'source': MODEL_008},
 'sector_filter': ['Communication']}

# Factors: cashflow_yield, quality, low_accruals, momentum_6_1
MODEL_010 = {'factor_weights': {'cashflow_yield': 0.4, 'quality': 0.1, 'low_accruals': 0.3, 'momentum_6_1': 0.2},
 'construction': {'tail': 0.035,
                  'power': 0.0,
                  'inverse_vol': 0.0,
                  'sector_budget': 'signal',
                  'beta_neutral': 1.0}}

# Fixed component blend
MODEL_011 = {'component_allocations': {'ccfa6bb91604447b9d69b547c94b1abf0ccf86f52440574cded165d98490446f': 0.45904618086234467,
                           'e9d8673bd7aaeb9f522e9db056174131dcabaf7e0061ca643f9c39882388b03e': 0.1703262364353153,
                           '6648c766ba6ee5bae9b1f72965a421bff2ec521bd841b56035e0a266d9489b56': 0.3706275827023401},
 'component_models': {'ccfa6bb91604447b9d69b547c94b1abf0ccf86f52440574cded165d98490446f': MODEL_007,
                      'e9d8673bd7aaeb9f522e9db056174131dcabaf7e0061ca643f9c39882388b03e': MODEL_001,
                      '6648c766ba6ee5bae9b1f72965a421bff2ec521bd841b56035e0a266d9489b56': MODEL_010}}

# Fixed component blend
MODEL_012 = {'component_allocations': {'ccfa6bb91604447b9d69b547c94b1abf0ccf86f52440574cded165d98490446f': 0.4590461808623446,
                           'e9d8673bd7aaeb9f522e9db056174131dcabaf7e0061ca643f9c39882388b03e': 0.17032623643531533,
                           '6648c766ba6ee5bae9b1f72965a421bff2ec521bd841b56035e0a266d9489b56': 0.37062758270234014},
 'component_models': {'ccfa6bb91604447b9d69b547c94b1abf0ccf86f52440574cded165d98490446f': MODEL_007,
                      'e9d8673bd7aaeb9f522e9db056174131dcabaf7e0061ca643f9c39882388b03e': MODEL_001,
                      '6648c766ba6ee5bae9b1f72965a421bff2ec521bd841b56035e0a266d9489b56': MODEL_010}}

# Fixed component blend
MODEL_013 = {'component_allocations': {'ccfa6bb91604447b9d69b547c94b1abf0ccf86f52440574cded165d98490446f': 0.4,
                           'e9d8673bd7aaeb9f522e9db056174131dcabaf7e0061ca643f9c39882388b03e': 0.2057595149333697,
                           '6648c766ba6ee5bae9b1f72965a421bff2ec521bd841b56035e0a266d9489b56': 0.39424048506663034},
 'component_models': {'ccfa6bb91604447b9d69b547c94b1abf0ccf86f52440574cded165d98490446f': MODEL_007,
                      'e9d8673bd7aaeb9f522e9db056174131dcabaf7e0061ca643f9c39882388b03e': MODEL_001,
                      '6648c766ba6ee5bae9b1f72965a421bff2ec521bd841b56035e0a266d9489b56': MODEL_010}}

# Fixed component blend
MODEL_014 = {'component_allocations': {'ccfa6bb91604447b9d69b547c94b1abf0ccf86f52440574cded165d98490446f': 0.6925726421591166,
                           'e9d8673bd7aaeb9f522e9db056174131dcabaf7e0061ca643f9c39882388b03e': 0.013127016135461581,
                           '6648c766ba6ee5bae9b1f72965a421bff2ec521bd841b56035e0a266d9489b56': 0.29430034170542174},
 'component_models': {'ccfa6bb91604447b9d69b547c94b1abf0ccf86f52440574cded165d98490446f': MODEL_007,
                      'e9d8673bd7aaeb9f522e9db056174131dcabaf7e0061ca643f9c39882388b03e': MODEL_001,
                      '6648c766ba6ee5bae9b1f72965a421bff2ec521bd841b56035e0a266d9489b56': MODEL_010}}

# Fixed component blend
MODEL_015 = {'component_allocations': {'ccfa6bb91604447b9d69b547c94b1abf0ccf86f52440574cded165d98490446f': 0.6925726421591171,
                           'e9d8673bd7aaeb9f522e9db056174131dcabaf7e0061ca643f9c39882388b03e': 0.013127016135461566,
                           '6648c766ba6ee5bae9b1f72965a421bff2ec521bd841b56035e0a266d9489b56': 0.29430034170542135},
 'component_models': {'ccfa6bb91604447b9d69b547c94b1abf0ccf86f52440574cded165d98490446f': MODEL_007,
                      'e9d8673bd7aaeb9f522e9db056174131dcabaf7e0061ca643f9c39882388b03e': MODEL_001,
                      '6648c766ba6ee5bae9b1f72965a421bff2ec521bd841b56035e0a266d9489b56': MODEL_010}}

# Fixed component blend
MODEL_016 = {'component_allocations': {'c6c57f6c79d13cefe6ac32189fb04eb0ced3a9eea845082ce00c9f7cd771a4f3': 0.21749755740567264,
                           'abd600a8cf924fd37b377740891da723e5d676552c1cc6bae74a0f7a2ecdc59b': 0.21749755740567267,
                           '1f87ff944470f0ad7f8d938d5fdd1ef4d40d242eb6029928c16d08e43685de25': 0.0785143350751865,
                           '5afdc4201d405062d92b0ab29a0835071065c0b308511279491584a15f89dc74': 0.2020693552344336,
                           '7dd1c5e8bffd2ac90bac880bcd20dfffc16ef486c6b578da3ada725b8bb497eb': 0.20206935523443384,
                           '6648c766ba6ee5bae9b1f72965a421bff2ec521bd841b56035e0a266d9489b56': 0.08235183964460069},
 'component_models': {'c6c57f6c79d13cefe6ac32189fb04eb0ced3a9eea845082ce00c9f7cd771a4f3': MODEL_011,
                      'abd600a8cf924fd37b377740891da723e5d676552c1cc6bae74a0f7a2ecdc59b': MODEL_012,
                      '1f87ff944470f0ad7f8d938d5fdd1ef4d40d242eb6029928c16d08e43685de25': MODEL_013,
                      '5afdc4201d405062d92b0ab29a0835071065c0b308511279491584a15f89dc74': MODEL_014,
                      '7dd1c5e8bffd2ac90bac880bcd20dfffc16ef486c6b578da3ada725b8bb497eb': MODEL_015,
                      '6648c766ba6ee5bae9b1f72965a421bff2ec521bd841b56035e0a266d9489b56': MODEL_010}}

# Fixed component blend; sector: Communication
MODEL_017 = {'component_allocations': {'source': 1.0},
 'component_models': {'source': MODEL_016},
 'sector_filter': ['Communication']}

# Fixed component blend; sector: Communication
MODEL_018 = {'component_allocations': {'source': 1.0},
 'component_models': {'source': MODEL_017},
 'sector_filter': ['Communication']}

# Fixed component blend; sector: Information Technology
MODEL_019 = {'component_allocations': {'source': 1.0},
 'component_models': {'source': MODEL_016},
 'sector_filter': ['Information Technology']}

# Fixed component blend; sector: Information Technology
MODEL_020 = {'component_allocations': {'source': 1.0},
 'component_models': {'source': MODEL_007},
 'sector_filter': ['Information Technology']}

# Factors: cashflow_yield, low_volatility, gross_profitability, average_asset_roa
MODEL_021 = {'factor_weights': {'cashflow_yield': 0.2,
                    'low_volatility': 0.1,
                    'gross_profitability': 0.2,
                    'average_asset_roa': 0.5},
 'construction': {'tail': 0.035,
                  'power': 0.0,
                  'inverse_vol': 0.0,
                  'sector_budget': 'signal',
                  'beta_neutral': 1.0},
 'smoothing': {'window': 2}}

# Fixed component blend; sector: Health Care
MODEL_022 = {'component_allocations': {'source': 1.0},
 'component_models': {'source': MODEL_021},
 'sector_filter': ['Health Care']}

# Fixed component blend; sector: Health Care
MODEL_023 = {'component_allocations': {'source': 1.0},
 'component_models': {'source': MODEL_016},
 'sector_filter': ['Health Care']}

# Factors: cashflow_yield, quality, momentum, earnings_yield
MODEL_024 = {'factor_weights': {'cashflow_yield': 0.25, 'quality': 0.25, 'momentum': 0.25, 'earnings_yield': 0.25},
 'construction': {'tail': 0.05,
                  'power': 0.0,
                  'inverse_vol': 0.0,
                  'sector_budget': 'signal',
                  'beta_neutral': 1.0}}

# Fixed component blend; sector: Health Care
MODEL_025 = {'component_allocations': {'source': 1.0},
 'component_models': {'source': MODEL_024},
 'sector_filter': ['Health Care']}

# Factors: cashflow_yield, quality, momentum, low_accruals
MODEL_026 = {'factor_weights': {'cashflow_yield': 0.25, 'quality': 0.25, 'momentum': 0.25, 'low_accruals': 0.25},
 'construction': {'tail': 0.025,
                  'power': 0.0,
                  'inverse_vol': 0.0,
                  'sector_budget': 'signal',
                  'beta_neutral': 1.0}}

# Fixed component blend; sector: Materials
MODEL_027 = {'component_allocations': {'source': 1.0},
 'component_models': {'source': MODEL_026},
 'sector_filter': ['Materials']}

# Factors: cashflow_yield, quality, low_accruals, momentum_6_1; sector: Consumer Staples
MODEL_028 = {'factor_weights': {'cashflow_yield': 0.4, 'quality': 0.1, 'low_accruals': 0.3, 'momentum_6_1': 0.2},
 'construction': {'tail': 0.15,
                  'power': 0.0,
                  'inverse_vol': 0.5,
                  'sector_budget': 'signal',
                  'beta_neutral': 1.0},
 'sector_filter': ['Consumer Staples']}

# Fixed component blend; sector: Materials
MODEL_029 = {'component_allocations': {'source': 1.0},
 'component_models': {'source': MODEL_024},
 'sector_filter': ['Materials']}

# Fixed component blend; final account weight cap 0.05
MODEL_030 = {'component_allocations': {'be3233b28ee33905b0af9963e4958b4a98a516759ea4eab0485c7f5f148c799d': 0.006272358780388416,
                           '71c886e624547be36f4963ef67e1adabbef86758f207c715b8b74bd0eb83d44c': 0.006272358780390958,
                           'dd80edfa7f467166691215baf0dab51838f94ac2104f645dbf9baf739e7c193c': 0.006272358780389233,
                           '9c4ddddd0fa2eecee9a806a0b635af2e35a6dba3a28d0306dd3e1589702b2ec6': 0.006272358780391793,
                           'f80c1899af620986fb511c8e288061e5484f50e937cd9ab0a2cf6159059f87ac': 0.006239643876395613,
                           '8f3198939fc3b959aeee67de840e92f6da5702bbb5994f60a1de85988e5594bb': 0.006239643876394529,
                           '80bcfac23815aa22e7b4e10a728b41b207bf0fce3779f4e4378cdcbf6c71ef76': 0.00041769817899858937,
                           '67f5008a2a7ef43f5ac69261f84d13aaddac1839eae826ae36b78b22f1701edc': 0.0004176981789986044,
                           '36177e97d29c9a0b89f47e0e61792c8e18fb535621a81ba6e1dd73d17f48fdcc': 0.33466385122773773,
                           '08043568b4b993c78a93fa3e8b73a272940df69c13b39212a6ca9a44499ad3a8': 0.004392099660832227,
                           '30a89d50d7028aac03fd50edc1ff15275f261119ff407ed70524ff2d674a330c': 0.009279289139700428,
                           '132771730c6c17117fb43de62d53e29f6b96af5e8c288ba12ab2abef9a6f33a1': 0.07682610779380644,
                           'dc181e2cbe66d972e1b6c6851112c54585ff7070bcf2712d63b478cdfef02354': 0.02390239101908301,
                           '9eb464d6d9619b7469bd896aeb3c72aa004cae6b5dd341b4310346721453a402': 0.032054322506154156,
                           '786f812da0feff33b0f796c533967216e9ddd4051f7a7fd1b9b0f3e0e307dec8': 0.4556655224157871,
                           'cb70bbe01f5e690055a549638cf53f3062b2d15bf5b54e479b30a1d2e32c78b3': 0.024812297004551353},
 'component_models': {'be3233b28ee33905b0af9963e4958b4a98a516759ea4eab0485c7f5f148c799d': MODEL_002,
                      '71c886e624547be36f4963ef67e1adabbef86758f207c715b8b74bd0eb83d44c': MODEL_004,
                      'dd80edfa7f467166691215baf0dab51838f94ac2104f645dbf9baf739e7c193c': MODEL_005,
                      '9c4ddddd0fa2eecee9a806a0b635af2e35a6dba3a28d0306dd3e1589702b2ec6': MODEL_006,
                      'f80c1899af620986fb511c8e288061e5484f50e937cd9ab0a2cf6159059f87ac': MODEL_009,
                      '8f3198939fc3b959aeee67de840e92f6da5702bbb5994f60a1de85988e5594bb': MODEL_008,
                      '80bcfac23815aa22e7b4e10a728b41b207bf0fce3779f4e4378cdcbf6c71ef76': MODEL_017,
                      '67f5008a2a7ef43f5ac69261f84d13aaddac1839eae826ae36b78b22f1701edc': MODEL_018,
                      '36177e97d29c9a0b89f47e0e61792c8e18fb535621a81ba6e1dd73d17f48fdcc': MODEL_019,
                      '08043568b4b993c78a93fa3e8b73a272940df69c13b39212a6ca9a44499ad3a8': MODEL_020,
                      '30a89d50d7028aac03fd50edc1ff15275f261119ff407ed70524ff2d674a330c': MODEL_022,
                      '132771730c6c17117fb43de62d53e29f6b96af5e8c288ba12ab2abef9a6f33a1': MODEL_023,
                      'dc181e2cbe66d972e1b6c6851112c54585ff7070bcf2712d63b478cdfef02354': MODEL_025,
                      '9eb464d6d9619b7469bd896aeb3c72aa004cae6b5dd341b4310346721453a402': MODEL_027,
                      '786f812da0feff33b0f796c533967216e9ddd4051f7a7fd1b9b0f3e0e307dec8': MODEL_028,
                      'cb70bbe01f5e690055a549638cf53f3062b2d15bf5b54e479b30a1d2e32c78b3': MODEL_029},
 'weight_cap': 0.05}

# Weighting choices: 1 = supplied alpha -> demean/L1; 2 = the configured model
# below; 3 = full AQR reference construction (reserved; raises, not implemented).
# Choice 1 uses alpha_column directly and bypasses MODEL_030 entirely. The
# cached 'score' column is the original value/quality/momentum baseline alpha,
# not this selected model. No tail, beta, volatility, blend or cap overlays are
# applied in choice 1. Set portfolio.group_by=None to center the whole pool,
# as in the five-stock example; 'sector' centers each historical sector.
PARAMETERS = {'name': 'monthly_sharpe158',
 'weighting_method': 2,
 'alpha_column': 'score',
 'variant': 'historical_rf_255dd5a853a0',
 'database_path': 'aqr_research/data/aqr.sqlite3',
 'input_folder': 'aqr_research/data/backtest_inputs',
 'backtest': {'start_date': '2022-01-01',
              'end_date': '2026-10-22',
              'initial_nav': 1000000,
              'transaction_cost_rate': 0.0005,
              'borrow_fee_rate': 0.01,
              'rebalance_frequency': 'monthly'},
 'signals': {'universe': 'russell1000_iwb_proxy',
             'tickers': [],
             'membership_file': None,
             'group_by': 'sector',
             'factor_operators': {'value': ['group_rank', 'group_zscore'],
                                  'quality': ['group_rank', 'group_zscore'],
                                  'momentum': ['group_rank', 'group_zscore']},
             'combine_operator': 'vec_avg',
             'database_path': 'aqr_research/data/aqr.sqlite3',
             'start': '2022-01-01',
             'end_exclusive': '2026-10-23',
             'min_stocks': 10,
             'financial_source': 'website_quarterly',
             'financial_availability': 'period_lag',
             'financial_reporting_lag_days': 90,
             'missing_value_method': 'drop',
             'missing_value_scope': 'score_group',
             'require_positive_book_equity': False},
 'portfolio': {'method': 'demean_l1',
               'group_by': 'sector',
               'gross_exposure': 1.0,
               'factor_columns': ['value', 'quality', 'momentum'],
               'risk_lookback': 252,
               'risk_min_observations': 120,
               'risk_lag_months': 1,
               'covariance_shrinkage': 0.1,
               'min_group_size': 10,
               'risk_aversion': 5.0,
               'alpha_scale': 1.0,
               'horizon_years': 0.08333333333333333,
               'net_exposure': 0.0,
               'target_beta': 0.0,
               'max_weight': 0.05,
               'transaction_cost_rate': 0.0005,
               'borrow_fee_rate': 0.01},
 'model': MODEL_030,
 'performance': {'risk_free': {'source': 'fred_dgs1mo',
                               'file': 'aqr_research/data/DGS1MO.csv',
                               'day_count': 365,
                               'max_quote_age_days': 7},
                 'cash_interest_in_nav': False,
                 'annualization_sessions': 252},
 'interpretation': 'Fixed model and sector allocations selected in-sample on inspected 2022-2026 returns '
                   'using historical daily RF. Selected for lower annual dispersion among qualifying '
                   'candidates; not independent out-of-sample validation.'}

def main():
    from pathlib import Path
    import sys
    roots = [*Path(__file__).resolve().parents, Path.cwd(), *Path.cwd().parents]
    root = next((p for p in roots if (p / "methods/aqr_factors/run.py").is_file()), None)
    if root is None:
        raise RuntimeError("Run this file inside the Long Short Tax Saving project.")
    sys.path.insert(0, str(root))
    from methods.aqr_factors.run import run
    run(PARAMETERS, output=Path(__file__).with_name("result.json"))

if __name__ == "__main__":
    main()
