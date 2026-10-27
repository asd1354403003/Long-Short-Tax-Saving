"""AQR-style signals and scheduled target-weight execution through analysis.ipynb."""
from pathlib import Path
import json
import logging
import pandas as pd

from selection import PoolSelection
from methods.aqr_factors.backtest import WeightedBacktester
from reports import PerformanceReport
from .signals import FactorSignals


class Selector:
    def __init__(self, parameters, config=None, result_dir=None):
        self.parameters = parameters
        self.backtest_config = parameters.backtest_config()
        self.construction = parameters.construction_config()
        self.result_dir = Path(result_dir or "methods/aqr_factors/research/results")
        self.signals = None

    def select(self, *, rebuild=False, prepared_signals=None):
        # prepared_signals allows one source-preparation pass to feed the
        # notebook immediately. Ordinary runs recalculate scores from sources.
        if prepared_signals is not None:
            self.signals = prepared_signals
        else:
            def progress(done,total,ticker,error):
                if done%100==0 or done==total:
                    print("Source/cache checks: {}/{}".format(done,total),flush=True)
            logger=logging.getLogger("yfinance")
            level=logger.level
            try:
                logger.setLevel(logging.CRITICAL)
                self.signals = FactorSignals(self.parameters.signal_config()).run(
                    force_refresh=rebuild, workers=self.parameters.download_workers,
                    fetch_missing=self.parameters.fetch_missing_sources,progress=progress)
            finally:
                logger.setLevel(level)
        self.result_dir.mkdir(parents=True, exist_ok=True)
        self.signals.signals.to_csv(self.result_dir/"monthly_scores.csv",index=False)
        self.signals.coverage.to_csv(self.result_dir/"score_coverage.csv",index=False)
        active = self.signals.signals.loc[self.signals.signals.in_universe]
        active.groupby(["as_of","status"]).size().rename("stocks").reset_index().to_csv(
            self.result_dir/"score_status_counts.csv",index=False)
        outcomes = pd.DataFrame({"ticker":list(self.signals.universe.tickers)}).set_index("ticker")
        prices = self.signals.database.read("""SELECT ticker,COUNT(*) price_rows,MIN(date) price_first,
            MAX(date) price_last FROM prices WHERE close>0 GROUP BY ticker""").set_index("ticker")
        reports = self.signals.database.read("""SELECT ticker,COUNT(*) raw_reports,MIN(period_end) first_report,
            MAX(period_end) last_report FROM fundamental_reports GROUP BY ticker""").set_index("ticker")
        outcomes = outcomes.join(prices).join(reports)
        outcomes["active_months"] = active.groupby("ticker").size()
        outcomes["scored_months"] = active.loc[active.score.notna()].groupby("ticker").size()
        for name in ("price_rows","raw_reports","active_months","scored_months"):
            outcomes[name] = outcomes[name].fillna(0).astype(int)
        issues=self.signals.errors+self.signals.metadata.get("fundamental_source_errors",[])
        outcomes["source_issues"]=[" | ".join(e for e in issues if e.startswith(t+":") or e.startswith(t+" ")) for t in outcomes.index]
        outcomes.to_csv(self.result_dir/"source_outcomes.csv")
        self.signals.metadata["data_coverage"] = dict(requested_tickers=len(outcomes),
            cached_price_tickers=int(outcomes.price_rows.gt(0).sum()),
            raw_statement_tickers=int(outcomes.raw_reports.gt(0).sum()),
            ever_scored_tickers=int(outcomes.scored_months.gt(0).sum()))
        (self.result_dir/"source_errors.json").write_text(json.dumps(dict(errors=self.signals.errors,
            fundamental_errors=self.signals.metadata.get("fundamental_source_errors",[])),ensure_ascii=False,indent=2),encoding="utf-8")
        return PoolSelection("weights",self.signals.signals.copy(),
            metadata=dict(signal_config=self.signals.metadata,construction=self.construction.__dict__,
                          source_errors=self.signals.errors))

    def backtest(self, selection, *, histories=None):
        result = WeightedBacktester(self.backtest_config,self.construction).run(self.signals,histories=histories)
        for name in ("daily_summary","events","closed_trades","open_positions","target_weights","rebalance_orders","corporate_events"):
            result[name].to_csv(self.result_dir/(name+".csv"),index=name=="daily_summary")
        PerformanceReport.compare(result).to_csv(self.result_dir/"performance.csv")
        PerformanceReport.by_year(result).to_csv(self.result_dir/"performance_by_year.csv")
        return result
