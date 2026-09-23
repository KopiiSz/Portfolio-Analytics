"""
Data acquisition: user CSV uploads, Yahoo Finance price history (Swedish-market
tickers by default, but any Yahoo ticker works), and Kenneth French factor data.

Network calls happen only in this module, so the rest of the app (metrics,
factor_models, risk_decomposition) can be tested with synthetic data offline.
"""
from __future__ import annotations

from functools import lru_cache

import pandas as pd
import yfinance as yf
from pandas_datareader.famafrench import FamaFrenchReader

# --- Built-in ticker catalog for the quick-pick search box, organized by
# category. This is NOT exhaustive -- users can type ANY Yahoo Finance
# ticker directly (any market, any exchange): stocks, ETFs, index funds,
# indices, crypto, etc.
TICKER_CATALOG = {
    "Swedish Stocks": {
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
    },
    "Global Stocks": {
        "Apple": "AAPL",
        "Microsoft": "MSFT",
        "Alphabet (Google) A": "GOOGL",
        "Amazon": "AMZN",
        "Nvidia": "NVDA",
        "Meta Platforms": "META",
        "Tesla": "TSLA",
        "Berkshire Hathaway B": "BRK-B",
        "LVMH": "MC.PA",
        "Novo Nordisk": "NOVO-B.CO",
        "ASML": "ASML.AS",
        "Toyota Motor": "7203.T",
        "Nestle": "NESN.SW",
        "Samsung Electronics": "005930.KS",
    },
    "ETFs": {
        "SPDR S&P 500 (SPY)": "SPY",
        "Invesco QQQ (Nasdaq-100)": "QQQ",
        "Vanguard Total Stock Market (VTI)": "VTI",
        "iShares MSCI World (URTH)": "URTH",
        "iShares MSCI Sweden (EWD)": "EWD",
        "iShares MSCI Europe (IEUR)": "IEUR",
        "iShares MSCI Emerging Markets (EEM)": "EEM",
        "iShares 20+ Yr Treasury Bond (TLT)": "TLT",
        "SPDR Gold Shares (GLD)": "GLD",
        "Vanguard FTSE All-World (VWRL.L)": "VWRL.L",
        "Xtrackers MSCI Sweden (XSWE.DE)": "XSWE.DE",
    },
    "Index Funds / Indices": {
        "OMX Stockholm 30 (^OMX)": "^OMX",
        "OMX Stockholm All-Share (^OMXSPI)": "^OMXSPI",
        "S&P 500 (^GSPC)": "^GSPC",
        "Nasdaq Composite (^IXIC)": "^IXIC",
        "Dow Jones Industrial Average (^DJI)": "^DJI",
        "FTSE 100 (^FTSE)": "^FTSE",
        "Euro Stoxx 50 (^STOXX50E)": "^STOXX50E",
        "DAX (^GDAXI)": "^GDAXI",
        "Nikkei 225 (^N225)": "^N225",
        "MSCI World (via URTH ETF proxy)": "URTH",
    },
}

# Flat view, kept for backwards compatibility with older calls.
BENCHMARK_TICKERS = {**TICKER_CATALOG["Index Funds / Indices"], **TICKER_CATALOG["ETFs"]}

FF_DATASET_NAMES = {
    "FF3": "Europe_3_Factors_Daily",
    "FF5": "Europe_5_Factors_Daily",
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
    Download Kenneth French's daily Europe factor set (FF3 or FF5) via
    pandas-datareader's dedicated Fama-French reader (handles the file
    format, percent-to-fraction conversion, and edge cases for us, rather
    than hand-parsing the raw CSV).

    Note: Kenneth French's library has no Sweden-specific factor set, only
    regional ones (Europe, Global, Developed). "Europe" is used here as the
    closest standard proxy for Swedish-market factor exposure.
    """
    if model not in FF_DATASET_NAMES:
        raise ValueError(f"model must be one of {list(FF_DATASET_NAMES)}")
    dataset_name = FF_DATASET_NAMES[model]
    reader = FamaFrenchReader(dataset_name, start="1990-01-01")
    tables = reader.read()
    df = tables[0].copy()  # table 0 is always the main daily factor table

    df.columns = [str(c).strip() for c in df.columns]
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = df.index.to_timestamp()
    # FamaFrenchReader already divides by 100 (returns fractions, not percent).
    return df.dropna(how="all")


def search_tickers(query: str, category: str | None = None) -> dict:
    """
    Substring search over the built-in ticker catalog, across all categories
    unless `category` narrows it. Returns {display_name: ticker}. This is a
    fallback/quick-pick list — `live_ticker_search` below is the primary
    search and covers far more names (any fund, any exchange).
    """
    q = query.strip().lower()
    cats = [category] if category else list(TICKER_CATALOG.keys())
    matches = {}
    for cat in cats:
        for name, tkr in TICKER_CATALOG.get(cat, {}).items():
            if q in name.lower() or q in tkr.lower():
                matches[f"{name}  ·  {cat}"] = tkr
    return matches


@lru_cache(maxsize=128)
def live_ticker_search(query: str, max_results: int = 8) -> tuple[dict, ...]:
    """
    Live name -> ticker lookup against Yahoo Finance's own search endpoint
    (via yfinance's Search wrapper). This is what lets a query like
    "lynx dynamic" resolve straight to a real ticker, instead of being
    limited to the small hand-picked TICKER_CATALOG above -- it covers
    funds, small caps, and foreign listings that no static list would.

    Returns a tuple of dicts (tuple so the result is hashable for lru_cache):
    {symbol, name, exchange, type}. Empty tuple on no results or any error
    (e.g. offline, endpoint hiccup) -- callers should fall back gracefully.
    """
    query = query.strip()
    if not query:
        return tuple()
    try:
        results = yf.Search(query, max_results=max_results).quotes
    except Exception:
        return tuple()

    out = []
    for r in results:
        symbol = r.get("symbol")
        if not symbol:
            continue
        out.append({
            "symbol": symbol,
            "name": r.get("shortname") or r.get("longname") or symbol,
            "exchange": r.get("exchange", ""),
            "type": r.get("quoteType", ""),
        })
    return tuple(out)
