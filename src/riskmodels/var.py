"""Rolling one-day Value-at-Risk and Expected Shortfall estimators.

All functions return *positive* loss numbers for the next day, aligned so that
the forecast at index t is made with data up to t-1 (no look-ahead).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def historical(r: pd.Series, window=250, alpha=0.99) -> pd.DataFrame:
    q = r.rolling(window).quantile(1 - alpha)
    es = r.rolling(window).apply(lambda x: x[x <= np.quantile(x, 1 - alpha)].mean(), raw=True)
    return pd.DataFrame({"VaR": -q, "ES": -es}).shift(1)


def parametric_normal(r: pd.Series, window=250, alpha=0.99) -> pd.DataFrame:
    mu, sd = r.rolling(window).mean(), r.rolling(window).std()
    z = stats.norm.ppf(alpha)
    es = sd * stats.norm.pdf(z) / (1 - alpha) - mu
    return pd.DataFrame({"VaR": z * sd - mu, "ES": es}).shift(1)


def cornish_fisher(r: pd.Series, window=250, alpha=0.99) -> pd.DataFrame:
    """Modified VaR adjusting the normal quantile for rolling skewness and excess kurtosis."""
    mu, sd = r.rolling(window).mean(), r.rolling(window).std()
    s, k = r.rolling(window).skew(), r.rolling(window).kurt()
    z = stats.norm.ppf(1 - alpha)
    zcf = z + (z**2 - 1) * s / 6 + (z**3 - 3 * z) * k / 24 - (2 * z**3 - 5 * z) * s**2 / 36
    return pd.DataFrame({"VaR": -(mu + zcf * sd)}).shift(1)


def ewma_vol(r: pd.Series, lam=0.94) -> pd.Series:
    """RiskMetrics EWMA volatility; value at t uses returns up to t."""
    var = np.empty(len(r))
    var[0] = r.iloc[:30].var()
    x = r.values
    for t in range(1, len(r)):
        var[t] = lam * var[t - 1] + (1 - lam) * x[t - 1] ** 2
    # var[t] is the forecast for day t made with data to t-1
    return pd.Series(np.sqrt(var), index=r.index)


def filtered_hs(r: pd.Series, window=250, alpha=0.99, lam=0.94) -> pd.DataFrame:
    """Filtered historical simulation (Barone-Adesi et al., 1999).

    Standardise past returns by their EWMA volatility, take the empirical quantile of the
    standardised residuals, and rescale by today's volatility forecast.
    """
    sig = ewma_vol(r, lam)                    # forecast for t using data to t-1
    z = r / sig                               # standardised residuals
    zq = z.rolling(window).quantile(1 - alpha).shift(1)
    zes = z.rolling(window).apply(lambda x: x[x <= np.quantile(x, 1 - alpha)].mean(), raw=True).shift(1)
    return pd.DataFrame({"VaR": -zq * sig, "ES": -zes * sig})
