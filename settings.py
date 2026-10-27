"""Shared method routing; edit each method settings.py or saved parameters.py for details."""
from dataclasses import dataclass
from methods.aqr_factors.settings import AQRSettings
from methods.cointegration.settings import CointegrationSettings, BACKTEST


@dataclass(frozen=True)
class MethodConfig:
    selector_module: str
    parameter_file: str | None = None
    result_dir: str | None = None


# AQR uses monthly target weights. The earlier cointegration and independent
# manual-list engines remain selectable with their own execution settings.
POOL_METHOD = "aqr_factors"
METHODS = {
    "aqr_factors": MethodConfig(selector_module="methods.aqr_factors.selector",
        result_dir="methods/aqr_factors/research/results"),
    "cointegration": MethodConfig(
        selector_module=CointegrationSettings.selector_module,
        parameter_file=CointegrationSettings.parameter_file,
        result_dir=CointegrationSettings.result_dir,
    ),
    "manual": MethodConfig(
        selector_module="methods.manual",
        parameter_file="methods/manual_parameters.py",
    ),
}
REBUILD_POOL = CointegrationSettings.rebuild_pool

# Compatibility alias. AQR settings live in methods/aqr_factors/settings.py.
AQR_SIGNALS = AQRSettings.signal_config()
