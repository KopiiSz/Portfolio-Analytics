# Portfolio Analytics

A from-scratch return/risk tearsheet tool: bring your own return stream (CSV)
or pick a ticker — Swedish market by default, but any Yahoo Finance symbol
works — compare it to a benchmark, and dig into performance, factor
decomposition, and multi-asset risk contribution.

## Features

**Single series vs. benchmark**
- CAGR, annualized volatility, Sharpe, Sortino, max drawdown
- Historical VaR and Expected Shortfall (CVaR)
- Equity curve, underwater (drawdown) plot, rolling Sharpe ratio
- CAPM / Fama-French 3 / Fama-French 5 return decomposition (alpha, betas, R²)
- Rolling factor betas over time

**Multi-asset portfolio risk decomposition**
- Enter tickers + weights
- See how *risk* is distributed across holdings via two methods:
  - **Euler / component risk contribution** — the standard closed-form method
  - **Shapley value decomposition** — game-theoretic, accounts for
    interaction effects between correlated assets
  - Both are exact: contributions sum to total portfolio volatility, so you
    can directly compare "% of capital" vs. "% of risk" per holding — this is
    where you'll see e.g. an equally-weighted portfolio with very unequally
    distributed risk.

## Project structure

```
app/
  app.py              # Streamlit UI — entry point
  lib/
    data_sources.py    # CSV parsing, Yahoo Finance fetch, Fama-French factor download
    metrics.py          # Sharpe, Sortino, CAGR, vol, drawdown, VaR, ES
    factor_models.py    # CAPM / FF3 / FF5 regressions, rolling betas
    risk_decomposition.py  # Euler + Shapley risk contribution
    plots.py             # Plotly chart builders
tests/
  test_math.py         # Offline smoke tests on synthetic data (no network needed)
sample_data/
  sample_returns.csv   # Example file for the CSV-upload path
requirements.txt
```

The analytics (`lib/`) is deliberately decoupled from the Streamlit UI
(`app.py`). If you outgrow Streamlit later and want a custom-designed website,
you can wrap the same `lib/` functions in a small FastAPI service and build
any frontend (React/Next.js, etc.) against it — the math doesn't need to
change.

## Data notes

- **Stock/index data**: pulled live from Yahoo Finance via `yfinance`. Swedish
  tickers use the `.ST` suffix (e.g. `VOLV-B.ST` for Volvo B). A short list of
  large-cap Swedish names is built in for the search box; you can type any
  other Yahoo ticker directly.
- **Benchmarks**: OMX Stockholm indices are available directly. True MSCI
  index *levels* aren't freely available, so MSCI exposure is approximated
  via liquid ETFs that track them (e.g. `URTH` for MSCI World, `EWD` for MSCI
  Sweden) — a very close proxy for return/risk analysis, though not identical
  to the official index (ETF returns include small tracking error and fees).
- **Fama-French factors**: downloaded live from Kenneth French's data
  library. There's no Sweden-specific factor set published there — only
  regional ones — so the **Europe** daily 3-factor and 5-factor sets are used
  as the closest standard proxy. Swap `FF_EUROPE_URLS` in
  `lib/data_sources.py` if you'd rather point at a different regional set, or
  build custom Swedish factors from a local stock universe later.

## Running locally

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app/app.py
```

Then open the URL it prints (usually `http://localhost:8501`).

## Deploying for free (GitHub + Streamlit Community Cloud)

1. Create a new **public** GitHub repository and push this folder to it:
   ```bash
   git init
   git add .
   git commit -m "Initial portfolio analytics app"
   git branch -M main
   git remote add origin https://github.com/<your-username>/<repo-name>.git
   git push -u origin main
   ```
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with
   GitHub.
3. Click **New app**, pick your repo/branch, and set the main file path to
   `app/app.py`.
4. Deploy. You'll get a free public URL like
   `https://<something>.streamlit.app`, and it auto-redeploys every time you
   push to `main`.

Free-tier notes: unlimited public apps, ~2.7 GB RAM / 2 CPU cores per app,
app goes to sleep after a period of inactivity (wakes up automatically on
the next visit — a few seconds' delay).

### Later: a fully custom website

If you want your own domain and design rather than Streamlit's look:
- Wrap `lib/` in a small FastAPI app exposing the same calculations as JSON
  endpoints.
- Build a frontend (React/Next.js, or a static site) that calls that API.
- Host the frontend anywhere (Vercel, Netlify, your own domain) and the API
  wherever's convenient (Render, Fly.io, etc.).

The reason `lib/` is written free of any Streamlit imports is specifically so
this migration doesn't require rewriting the analytics.

## Testing

The `lib/` modules are pure Python/pandas with no network calls, so they're
tested independently of live data:

```bash
python tests/test_math.py
```

This runs every metric, factor regression, and both risk-decomposition
methods against synthetic data and prints the results, so you can sanity
check them or extend the assertions.
