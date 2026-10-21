"""Market-beta plots and historical/normal VaR and CVaR."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import norm


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
