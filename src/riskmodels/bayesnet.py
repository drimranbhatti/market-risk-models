"""Helpers for discrete Bayesian networks on macro/market drivers (uses pgmpy)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def tercile_states(s: pd.Series, labels=("low", "mid", "high"), fit_end=None) -> pd.Series:
    """Discretise into terciles. Cut-offs are estimated on data up to ``fit_end`` only (no look-ahead)."""
    ref = s.loc[:fit_end] if fit_end is not None else s
    q1, q2 = ref.quantile([1 / 3, 2 / 3])
    return pd.Series(np.select([s <= q1, s <= q2], labels[:2], labels[2]), index=s.index)


def macro_states(m: pd.DataFrame, fit_end=None) -> pd.DataFrame:
    """Month-level discrete states from FRED-style macro data.

    * rates:     level of the fed funds rate (terciles)
    * inflation: year-on-year CPI inflation (terciles)
    * growth:    year-on-year industrial production growth (terciles)
    * dollar:    3-month change in the broad dollar index (down / flat / up terciles)
    * equities:  3-month SPY return (terciles)
    * oil:       target - next-month WTI price change (down / up)

    ``fit_end``: last date of the training sample; bin edges use only data up to it.
    """
    df = pd.DataFrame(index=m.index)
    df["rates"] = m["fed_funds"]
    df["inflation"] = m["cpi"].pct_change(12)
    df["growth"] = m["industrial_production"].pct_change(12)
    df["dollar"] = m["usd_broad"].pct_change(3)
    df["equities"] = m["spy"].pct_change(3)
    df["oil_next"] = m["wti"].pct_change().shift(-1)
    df = df.dropna()
    out = pd.DataFrame(index=df.index)
    for c in ("rates", "inflation", "growth"):
        out[c] = tercile_states(df[c], fit_end=fit_end)
    out["dollar"] = tercile_states(df["dollar"], ("down", "flat", "up"), fit_end)
    out["equities"] = tercile_states(df["equities"], ("weak", "normal", "strong"), fit_end)
    out["oil_next"] = (df["oil_next"] > 0).map({True: "up", False: "down"})
    return out
