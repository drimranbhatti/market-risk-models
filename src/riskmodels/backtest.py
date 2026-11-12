"""Statistical VaR backtests."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def exceptions(r: pd.Series, var: pd.Series) -> pd.Series:
    d = pd.concat([r, var], axis=1).dropna()
    return (d.iloc[:, 0] < -d.iloc[:, 1]).astype(int)


def _xlogy(x, y):
    return 0.0 if x == 0 else x * np.log(y)


def kupiec_pof(hits: pd.Series, alpha=0.99) -> dict:
    """Kupiec (1995) proportion-of-failures likelihood-ratio test (chi2, 1 dof)."""
    n, x = len(hits), int(hits.sum())
    p, phat = 1 - alpha, x / n
    lr = -2 * (_xlogy(n - x, 1 - p) + _xlogy(x, p) - _xlogy(n - x, 1 - phat) - _xlogy(x, phat))
    return {"obs": n, "exceptions": x, "expected": n * p, "rate": phat, "LR_pof": lr, "p_pof": 1 - stats.chi2.cdf(lr, 1)}


def christoffersen(hits: pd.Series, alpha=0.99) -> dict:
    """Christoffersen (1998) independence and conditional-coverage tests."""
    h = hits.values
    n00 = int(((h[:-1] == 0) & (h[1:] == 0)).sum()); n01 = int(((h[:-1] == 0) & (h[1:] == 1)).sum())
    n10 = int(((h[:-1] == 1) & (h[1:] == 0)).sum()); n11 = int(((h[:-1] == 1) & (h[1:] == 1)).sum())
    pi0 = n01 / max(n00 + n01, 1); pi1 = n11 / max(n10 + n11, 1); pi = (n01 + n11) / max(n00 + n01 + n10 + n11, 1)
    l_ind = _xlogy(n00, 1 - pi0) + _xlogy(n01, pi0) + _xlogy(n10, 1 - pi1) + _xlogy(n11, pi1)
    l_null = _xlogy(n00 + n10, 1 - pi) + _xlogy(n01 + n11, pi)
    lr_ind = -2 * (l_null - l_ind)
    lr_cc = kupiec_pof(hits, alpha)["LR_pof"] + lr_ind
    return {"LR_ind": lr_ind, "p_ind": 1 - stats.chi2.cdf(lr_ind, 1), "LR_cc": lr_cc, "p_cc": 1 - stats.chi2.cdf(lr_cc, 2),
            "P(hit | hit yesterday)": pi1}


def basel_zone(hits_last_250: int) -> str:
    """Basel traffic light for 250 days of 99% VaR: green 0-4, yellow 5-9, red 10+."""
    return "green" if hits_last_250 <= 4 else ("yellow" if hits_last_250 <= 9 else "red")


def summary(r: pd.Series, var_table: dict[str, pd.Series], alpha=0.99) -> pd.DataFrame:
    rows = {}
    for name, v in var_table.items():
        h = exceptions(r, v)
        k, c = kupiec_pof(h, alpha), christoffersen(h, alpha)
        worst250 = int(h.rolling(250).sum().max())
        rows[name] = {**{key: k[key] for key in ("obs", "exceptions", "expected", "rate", "p_pof")},
                      "p_ind": c["p_ind"], "p_cc": c["p_cc"], "P(hit | hit yesterday)": c["P(hit | hit yesterday)"],
                      "worst 250d exceptions": worst250, "worst Basel zone": basel_zone(worst250)}
    return pd.DataFrame(rows).T
