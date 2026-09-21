"""
Data acquisition: user CSV uploads, Yahoo Finance price history (Swedish-market
tickers by default, but any Yahoo ticker works), and Kenneth French factor data.

Network calls happen only in this module, so the rest of the app (metrics,
factor_models, risk_decomposition) can be tested with synthetic data offline.
"""
from __future__ import annotations

import io
import zipfile
from functools import lru_cache

import pandas as pd
import requests
import yfinance as yf

# --- A starter list of well-known Nasdaq Stockholm large caps for the search
# box. This is NOT exhaustive -- users can type any Yahoo Finance ticker
# (e.g. "AAPL", "VOLV-B.ST", "^OMX") directly.
SWEDISH_TICKERS = {
    "Volvo B": "VOLV-B.ST",
    "Ericsson B": "ERIC-B.ST",
    "H&M B": "HM-B.ST",
    "Investor B": "INVE-B.ST",
    "Atlas Copco A": "ATCO-A.ST",
    "Atlas Copco B": "ATCO-B.ST",
    "SEB A": "SEB-A.ST",
    "Swedbank A": "SWED-A.ST",
    "Handelsbanken A": "SHB-A.ST",
    "Nordea Bank": "NDA-SE.ST",
    "Sandvik": "SAND.ST",
    "SKF B": "SKF-B.ST",
    "Telia Company": "TELIA.ST",
    "Essity B": "ESSITY-B.ST",
    "Assa Abloy B": "ASSA-B.ST",
    "Alfa Laval": "ALFA.ST",
    "Electrolux B": "ELUX-B.ST",
    "Boliden": "BOL.ST",
    "Hexagon B": "HEXA-B.ST",
    "Epiroc A": "EPI-A.ST",
    "EQT": "EQT.ST",
    "Evolution": "EVO.ST",
    "Getinge B": "GETI-B.ST",
    "Kinnevik B": "KINV-B.ST",
    "Skanska B": "SKA-B.ST",
    "Tele2 B": "TEL2-B.ST",
}

BENCHMARK_TICKERS = {
    "OMX Stockholm 30 (^OMX)": "^OMX",
    "OMX Stockholm All-Share (^OMXSPI)": "^OMXSPI",
    "S&P 500 (^GSPC)": "^GSPC",
    "MSCI World proxy - iShares URTH ETF": "URTH",
    "MSCI Sweden proxy - iShares EWD ETF": "EWD",
    "MSCI Europe proxy - iShares IEUR ETF": "IEUR",
    "MSCI Emerging Markets proxy - iShares EEM ETF": "EEM",
}

FF_EUROPE_URLS = {
    "FF3": "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/Europe_3_Factors_Daily_CSV.zip",
    "FF5": "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/Europe_5_Factors_Daily_CSV.zip",
}


def load_csv_returns(file, date_col: str | None = None, value_col: str | None = None,
                      is_prices: bool = False) -> pd.Series:
    """
    Parse a user-uploaded CSV into a clean returns Series.

    Accepts either:
      - two columns: a date column and a return-or-price column (auto-detected
        if date_col/value_col not given: first parseable-as-date column wins,
        first remaining numeric column wins).
    If `is_prices` is True the value column is treated as prices and converted
    to simple returns; otherwise it's treated as already-simple returns
    (fractions, e.g. 0.01 for 1%, not 1.0).
    """
    df = pd.read_csv(file)
    if date_col is None:
        date_col = df.columns[0]
    if value_col is None:
        remaining = [c for c in df.columns if c != date_col]
        value_col = remaining[0]

    df[date_col] = pd.to_datetime(df[date_col])
    df = df.set_index(date_col).sort_index()
    series = pd.to_numeric(df[value_col], errors="coerce").dropna()

    if is_prices:
        return series.pct_change().dropna()
    return series


@lru_cache(maxsize=64)
def fetch_price_history(ticker: str, start: str, end: str) -> pd.Series:
    """
    Fetch adjusted close prices for a Yahoo Finance ticker. `start`/`end` are
    ISO date strings so the result is cacheable.
    """
    data = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
    if data.empty:
        raise ValueError(f"No data returned for ticker '{ticker}'. Check the symbol.")
    close = data["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    close.name = ticker
    return close


def fetch_returns(ticker: str, start: str, end: str) -> pd.Series:
    prices = fetch_price_history(ticker, start, end)
    return prices.pct_change().dropna()


@lru_cache(maxsize=8)
def fetch_ff_factors(model: str = "FF3") -> pd.DataFrame:
    """
    Download and parse Kenneth French's daily Europe factor file (FF3 or FF5).
    Returns a DataFrame indexed by date with columns like Mkt-RF, SMB, HML,
    (RMW, CMA for FF5), RF -- all as simple daily fractions (already /100).

    Note: Kenneth French's library has no Sweden-specific factor set, only
    regional ones (Europe, Global, Developed). "Europe" is used here as the
    closest standard proxy for Swedish-market factor exposure.
    """
    if model not in FF_EUROPE_URLS:
        raise ValueError(f"model must be one of {list(FF_EUROPE_URLS)}")
    url = FF_EUROPE_URLS[model]
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        csv_name = [n for n in zf.namelist() if n.lower().endswith(".csv")][0]
        raw = zf.read(csv_name).decode("utf-8", errors="ignore")

    # Ken French CSVs have a few header lines of preamble, then a data table
    # with a date-string index (YYYYMMDD), then a blank line before annual data.
    lines = raw.splitlines()
    start_idx = next(i for i, l in enumerate(lines) if l.strip().split(",")[0].strip('"').isdigit())
    end_idx = next(
        (i for i in range(start_idx, len(lines)) if lines[i].strip() == ""), len(lines)
    )
    table_str = "\n".join(lines[start_idx:end_idx])
    header_line = lines[start_idx - 1] if lines[start_idx - 1].strip() else lines[start_idx - 2]
    columns = ["Date"] + [c.strip() for c in header_line.split(",") if c.strip()]

    df = pd.read_csv(io.StringIO(table_str), header=None, names=columns)
    df["Date"] = pd.to_datetime(df["Date"].astype(str), format="%Y%m%d")
    df = df.set_index("Date")
    df = df.apply(pd.to_numeric, errors="coerce") / 100.0
    return df.dropna(how="all")


def search_swedish_ticker(query: str) -> dict:
    """Fuzzy-ish substring search over the built-in Swedish ticker list."""
    q = query.strip().lower()
    return {name: tkr for name, tkr in SWEDISH_TICKERS.items() if q in name.lower() or q in tkr.lower()}
