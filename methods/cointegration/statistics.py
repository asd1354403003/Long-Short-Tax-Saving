"""Cointegration class: ADF and explicit-OLS EG; date windows belong to research.py."""
from __future__ import annotations

import inspect
import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, coint
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant

_ADF_HAS_RESULT_OBJECT = "result_object" in inspect.signature(adfuller).parameters


class Cointegration:
    """ADF and explicit-OLS Engle-Granger tests with shared lag settings."""

    def __init__(self, max_lag: int = 1):
        if not isinstance(max_lag, int) or max_lag < 0:
            raise ValueError("max_lag must be a nonnegative integer")
        self.max_lag = max_lag

    def adf_test(self, series: pd.Series) -> dict:
        """ADF with an intercept, no time trend, and a fixed -2.86 threshold.

        H0: the series has a unit root.
        -2.86 is the approximate large-sample 5% critical value for this specific
        ADF specification. The p-value is reported but does not make the decision.
        This ordinary ADF critical value must not replace the EG critical value
        when testing an estimated cointegrating residual for cointegration.
        """
        values = pd.to_numeric(pd.Series(series), errors="coerce")
        values = values.replace([np.inf, -np.inf], np.nan).dropna().to_numpy()
        if len(values) < 20 or np.std(values) <= 1e-10:
            raise ValueError("ADF requires at least 20 valid, nonconstant observations")
        options = {"result_object": False} if _ADF_HAS_RESULT_OBJECT else {}
        result = adfuller(values, maxlag=self.max_lag, regression="c", autolag=None, **options)
        statistic = float(result[0])

        stationary_95 = statistic < -2.86

        return {
            "test_statistic": statistic,
            "critical_value_5pct": -2.86,
            "stationary_95": bool(stationary_95),
            "pvalue": float(result[1]),
            "used_lag": int(result[2]),
            "nobs": int(result[3]),
        }

    def engle_granger(self, y: pd.Series, x: pd.Series, *, confidence_level: float = 0.95) -> dict:
        """Fixed-direction EG test for y = intercept + hedge_ratio*x + residual.

        Uses the EG test's own 5% or 10% critical value, which accounts for estimating
        the cointegrating equation. Date cutoffs and I(1) checks are the caller's
        responsibility. The OLS slope is beta units of x per unit of y. The caller
        supplies Close prices consistently for formation and share sizing.
        The intercept is stored separately from the ratio.
        """
        if confidence_level not in (0.95, 0.90):
            raise ValueError("EG confidence_level must be 0.95 or 0.90")
        paired = pd.concat([
            pd.to_numeric(y, errors="coerce").rename("y"),
            pd.to_numeric(x, errors="coerce").rename("x"),
        ], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
        if len(paired) < 20 or min(paired["y"].std(), paired["x"].std()) <= 1e-10:
            raise ValueError("Engle-Granger requires at least 20 valid, nonconstant paired observations")
        yv, xv = paired["y"].to_numpy(), paired["x"].to_numpy()

        # Naive linear regression with an intercept: y = alpha + beta*x + error.
        regression = OLS(yv, add_constant(xv, has_constant="add")).fit()
        intercept = float(regression.params[0])
        beta = float(regression.params[1])
        residual = pd.Series(regression.resid, index=paired.index, name="residual")

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            statistic, pvalue, critical = coint(yv, xv, trend="c", maxlag=self.max_lag, autolag=None)
        if not np.isfinite(statistic) or any("collinear" in str(item.message).lower() for item in caught):
            raise ValueError("nearly_identical_or_collinear_series")
        residual_adf = self.adf_test(residual)
        critical_value_5pct = float(critical[1])  # coint order: 1%, 5%, 10%
        selected_critical_value = float(critical[1 if confidence_level == 0.95 else 2])
        return {
            "test_statistic": float(statistic),
            "critical_value_5pct": critical_value_5pct,
            "critical_value_10pct": float(critical[2]),
            "cointegrated_95": bool(statistic < critical_value_5pct),
            "cointegrated_90": bool(statistic < float(critical[2])),
            "confidence_level": confidence_level,
            "critical_value": selected_critical_value,
            "cointegrated": bool(statistic < selected_critical_value),
            "pvalue": float(pvalue),
            "intercept": float(intercept),
            "hedge_ratio": float(beta),
            "ols_r_squared": float(regression.rsquared),
            "residual": residual,
            "residual_adf": residual_adf,
            "nobs": len(paired),
            "used_lag": self.max_lag,
        }
