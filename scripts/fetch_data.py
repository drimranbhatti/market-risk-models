"""Download data used by the notebooks into data/.

* spy_daily.csv  - SPY adjusted close (Yahoo Finance), 2000-present
* macro_monthly.csv - FRED: fed funds rate, CPI, industrial production, WTI crude, broad dollar index;
                      plus SPY month-end close

    python scripts/fetch_data.py
"""
from io import StringIO
from pathlib import Path
from urllib.request import urlopen

import pandas as pd
import yfinance as yf

DATA = Path(__file__).resolve().parents[1] / "data"
FRED = {"FEDFUNDS": "fed_funds", "CPIAUCSL": "cpi", "INDPRO": "industrial_production",
        "MCOILWTICO": "wti", "TWEXBGSMTH": "usd_broad"}


def fred(series_id: str) -> pd.Series:
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    df = pd.read_csv(StringIO(urlopen(url, timeout=30).read().decode()), index_col=0, parse_dates=True)
    return pd.to_numeric(df.iloc[:, 0], errors="coerce").rename(FRED[series_id])


if __name__ == "__main__":
    DATA.mkdir(exist_ok=True)
    spy = yf.download("SPY", start="2000-01-01", auto_adjust=True, progress=False)["Close"].squeeze().rename("SPY")
    spy.round(4).to_csv(DATA / "spy_daily.csv")
    macro = pd.concat([fred(s) for s in FRED], axis=1, sort=True)
    macro.index = macro.index.to_period("M").to_timestamp("M")
    macro["spy"] = spy.resample("ME").last()
    macro.loc["2000-01-01":].round(4).to_csv(DATA / "macro_monthly.csv")
    print("written", DATA)
