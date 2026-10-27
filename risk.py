"""Historical risk-free benchmarks, market beta, and VaR/CVaR."""

from pathlib import Path
import hashlib
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import norm


RISK_FREE_SETTINGS = {
    "source": "fred_dgs1mo",
    "file": "aqr_research/data/DGS1MO.csv",
    "day_count": 365,
    "max_quote_age_days": 7,
}


def treasury_risk_free(dates, yields, *, day_count=365, max_quote_age_days=7):
    """Accrue a lagged 1-month Treasury yield proxy over each NAV interval.

    H.15 DGS1MO is an annual percent investment-basis YIELD, not a daily
    realized Treasury-index return. Use the latest observation strictly before
    the interval starts, simple annual/365 daily accrual and calendar-day gaps.
    The initial intraday NAV interval uses one day as an explicit approximation.
    Missing/stale observations raise; they never turn into zero or future fills.
    """
    dates = pd.DatetimeIndex(dates).tz_localize(None).normalize()
    if len(dates) == 0 or dates.has_duplicates or not dates.is_monotonic_increasing:
        raise ValueError("Risk-free dates must be nonempty, unique and increasing")
    if day_count != 365 or max_quote_age_days < 1:
        raise ValueError("Use ACT/365 and a positive maximum quote age")
    quotes = pd.Series(yields, dtype=float).copy()
    quotes.index = pd.DatetimeIndex(quotes.index).tz_localize(None).normalize()
    if quotes.index.has_duplicates:
        raise ValueError("Duplicate Treasury yield observations")
    quotes = quotes.dropna().sort_index()
    if not np.isfinite(quotes).all() or (1 + quotes / 100 / day_count <= 0).any():
        raise ValueError("Invalid Treasury yield")
    starts = pd.DatetimeIndex([dates[0] - pd.Timedelta(days=1), *dates[:-1]])
    matched = pd.merge_asof(pd.DataFrame({"start": starts}),
        pd.DataFrame({"quote_date": quotes.index, "annual_percent": quotes.values}),
        left_on="start", right_on="quote_date", direction="backward", allow_exact_matches=False)
    age = (matched.start - matched.quote_date).dt.days
    if matched.annual_percent.isna().any() or age.gt(max_quote_age_days).any():
        bad = dates[matched.annual_percent.isna().to_numpy() | age.gt(max_quote_age_days).to_numpy()]
        raise ValueError("Missing/stale Treasury benchmark for: " + ", ".join(bad.strftime("%Y-%m-%d")[:5]))
    days = (dates - starts).days.to_numpy()
    values = (1 + matched.annual_percent.to_numpy() / 100 / day_count) ** days - 1
    result = pd.Series(values, index=dates, name="risk_free_return")
    result.attrs["basis"] = "Lagged DGS1MO yield-accrual proxy, ACT/365; first NAV interval approximated by one day"
    return result


def historical_risk_free(dates, settings=None):
    """Read a cached, dated official-rate source; never download implicitly."""
    config = {**RISK_FREE_SETTINGS, **(settings or {})}
    if set(config) != set(RISK_FREE_SETTINGS) or config["source"] != "fred_dgs1mo":
        raise ValueError("Unsupported historical risk-free settings")
    path = Path(config["file"])
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("Use a project-relative risk-free source path")
    path = Path(__file__).resolve().parent / path
    frame = pd.read_csv(path, parse_dates=["observation_date"])
    values = pd.to_numeric(frame.DGS1MO, errors="coerce")
    rates = treasury_risk_free(dates, pd.Series(values.to_numpy(), index=frame.observation_date),
        day_count=config["day_count"], max_quote_age_days=config["max_quote_age_days"])
    metadata_path = path.with_suffix(".metadata.json")
    provenance = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if provenance.get("sha256", digest) != digest:
        raise ValueError("Risk-free source checksum does not match provenance")
    rates.attrs.update(settings=config, provenance=provenance, source_sha256=digest)
    return rates


def benchmark_returns(dates, *, daily_risk_free=None, annual_risk_free_rate=None):
    """Align a required benchmark; historical Treasury rates are the default."""
    dates = pd.DatetimeIndex(dates)
    if daily_risk_free is not None and annual_risk_free_rate is not None:
        raise ValueError("Specify daily or annual risk-free returns, not both")
    if annual_risk_free_rate is not None:
        if not np.isfinite(annual_risk_free_rate) or annual_risk_free_rate <= -1:
            raise ValueError("Invalid annual risk-free rate")
        return pd.Series((1 + annual_risk_free_rate) ** (1 / 252) - 1, index=dates)
    if daily_risk_free is None:
        return historical_risk_free(dates)
    if not isinstance(daily_risk_free, pd.Series) or daily_risk_free.index.has_duplicates:
        raise ValueError("Provide uniquely dated daily risk-free returns")
    aligned = daily_risk_free.reindex(dates)
    if not np.isfinite(aligned).all():
        raise ValueError("Risk-free returns must cover every NAV date; no zero filling")
    return aligned


def excess_sharpe(returns, risk_free):
    excess = returns - benchmark_returns(returns.index, daily_risk_free=risk_free)
    deviation = excess.std(ddof=1)
    return excess.mean() / deviation * np.sqrt(252) if deviation > 0 else np.nan


class BetaAnalysis:
    """Regress y on x and plot the fitted line and rolling beta."""

    def __init__(self, x, y):
        self.x_label = str(x.name) if x.name is not None else "x"
        self.y_label = str(y.name) if y.name is not None else "y"
        data = pd.concat([x.rename("x"), y.rename("y")], axis=1).dropna().sort_index()
        self.x, self.y = data["x"], data["y"]
        if len(data) < 2 or self.x.nunique() < 2:
            raise ValueError("Provide at least two paired observations with a nonconstant x.")
        self.model = sm.OLS(self.y, sm.add_constant(self.x)).fit()
        self.alpha = self.model.params["const"]
        self.beta = self.model.params["x"]

    def plot_regression(self):
        x = self.x.sort_values()
        plt.figure(figsize=(9, 4.5))
        plt.scatter(self.x, self.y, s=14, alpha=0.5)
        plt.plot(x, self.alpha + self.beta * x, color="red", label=f"Fitted line: beta = {self.beta:.3f}")
        plt.xlabel(self.x_label)
        plt.ylabel(self.y_label)
        plt.title(f"{self.y_label} versus {self.x_label}")
        plt.legend()
        plt.grid(alpha=0.25)
        plt.tight_layout()
        plt.show()
        print(f"Alpha: {self.alpha:.6f}; beta: {self.beta:.4f}")

    def plot_rolling_beta(self, window=60):
        if not 2 <= window <= len(self.x):
            raise ValueError("window must be between 2 and the number of paired observations.")
        rolling_beta = self.y.rolling(window).cov(self.x) / self.x.rolling(window).var()
        rolling_beta.plot(figsize=(9, 4), title=f"{self.y_label} versus {self.x_label}: {window}-day rolling beta")
        plt.axhline(self.beta, color="red", linestyle="--", label=f"Full-sample beta = {self.beta:.3f}")
        plt.xlabel("Date")
        plt.ylabel("Beta")
        plt.legend()
        plt.grid(alpha=0.25)
        plt.tight_layout()
        plt.show()
        print(f"Latest rolling beta: {rolling_beta.dropna().iloc[-1]:.4f}")
        return rolling_beta


class RiskAnalysis:
    """Estimate monetary VaR/CVaR directly from period P&L.

    Loss = -P&L: positive values mean losses, negative values mean gains.
    Historical estimates use an equal-weight empirical distribution. Parametric
    estimates assume normal P&L, using the sample mean and sample std.
    Results have the same currency units as P&L; no amount scaling is applied.
    """

    def __init__(self, pnl, confidence=0.95):
        self.pnl = pd.Series(pnl).dropna().astype(float)
        self.confidence = float(confidence)
        if len(self.pnl) < 2 or not np.isfinite(self.pnl).all():
            raise ValueError("Provide at least two finite P&L observations.")
        if not 0 < self.confidence < 1:
            raise ValueError("confidence must be strictly between 0 and 1.")

    def historical(self):
        """Empirical loss quantile and the mean of the worst tail mass."""
        losses = -self.pnl.to_numpy()
        var = np.quantile(losses, self.confidence, method="inverted_cdf")
        # Include only the required probability mass at a tied tail boundary.
        cvar = var + np.maximum(losses - var, 0).mean() / (1 - self.confidence)
        return pd.Series({"VaR": var, "CVaR": cvar})

    def parametric(self):
        """Normal-distribution VaR/CVaR, including the estimated mean."""
        mean = self.pnl.mean()
        std = self.pnl.std(ddof=1)
        z = norm.ppf(self.confidence)
        var = -mean + std * z
        cvar = -mean + std * norm.pdf(z) / (1 - self.confidence)
        return pd.Series({"VaR": var, "CVaR": cvar})

    def summary(self):
        """Compare historical and normal VaR/CVaR in the input P&L currency."""
        result = pd.DataFrame({
            "Historical": self.historical(),
            "Parametric (normal)": self.parametric(),
        }).T
        result.index.name = "Method"
        return result
