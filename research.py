"""Load the configured selector; optional selector interface for manual lists and baseline research."""
from dataclasses import asdict
import importlib
import json
from pathlib import Path
import numpy as np
import settings
from backtest import LongShortBacktester
from reports import PerformanceReport

PROJECT_ROOT = Path(__file__).resolve().parent


class ResearchProject:
    def __init__(self, settings_module=settings):
        self.settings = settings_module
        self.method_name = settings_module.POOL_METHOD
        if self.method_name not in settings_module.METHODS:
            raise ValueError("Unknown pool method: " + self.method_name)
        self.method = settings_module.METHODS[self.method_name]
        if self.method.parameter_file is None:
            parameters = settings_module.AQRSettings
        else:
            parameter_path = Path(self.method.parameter_file)
            if parameter_path.is_absolute() or parameter_path.suffix != ".py" or ".." in parameter_path.parts:
                raise ValueError("Use a project-relative Python parameter file.")
            parameter_module = importlib.import_module(".".join(parameter_path.with_suffix("").parts))
            parameter_module = importlib.reload(parameter_module)
            parameters = parameter_module.PARAMETERS
        selector_module = importlib.import_module(self.method.selector_module)
        self.selector = selector_module.Selector(parameters,
            settings_module.BACKTEST, PROJECT_ROOT / self.method.result_dir if self.method.result_dir else None)
        self.selection = self.result = None

    def select_pool(self, rebuild=None, **inputs):
        rebuild = self.settings.REBUILD_POOL if rebuild is None else rebuild
        self.selection = self.selector.select(rebuild=rebuild, **inputs)
        return self.selection

    def backtest(self, *, histories=None):
        if self.selection is None:
            self.select_pool()
        if self.selection.kind == "independent":
            self.result = LongShortBacktester(self.selection.pool, self.settings.BACKTEST,
                source=self.selection.source).run(histories=histories)
        else:
            self.result = self.selector.backtest(self.selection, histories=histories)
        PerformanceReport.audit(self.result)
        if self.method.result_dir is not None:
            path = PROJECT_ROOT / self.method.result_dir / "result.json"
            metadata = dict(self.selection.metadata)
            comparison = PerformanceReport.compare(self.result).replace([np.inf, -np.inf], np.nan)
            last = self.result["daily_summary"].iloc[-1]
            values = {key: float(last[key]) for key in ("cumulative_pnl", "cumulative_net_realized_pnl",
                "net_unrealized_pnl", "cumulative_total_cost")}
            trades = PerformanceReport.trades(self.result)
            config = getattr(self.selector,"backtest_config",self.settings.BACKTEST)
            metadata["backtest"] = dict(settings=asdict(config),
                start=str(self.result["daily_summary"].index[0].date()),
                end=str(self.result["daily_summary"].index[-1].date()),
                comparison=json.loads(comparison.to_json(orient="index")),
                trades=json.loads(trades.to_json()), **values)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(metadata, indent=2, allow_nan=False), encoding="utf-8")
        return self.result
