"""
Portfolio Analytics — a from-scratch tearsheet tool.

Two modes:
  1. Single series vs benchmark: bring your own CSV of returns, or pick ANY
     Yahoo Finance ticker — stocks, ETFs, or index funds, on any exchange
     worldwide (a built-in catalog covers popular Swedish + global names,
     but typing any other symbol works too). Get performance stats, equity
     curve, underwater plot, rolling Sharpe, and CAPM / FF3 / FF5 factor
     decomposition with rolling betas.
  2. Multi-asset portfolio: enter tickers + weights and see how risk is
     really distributed across holdings (Euler method + Shapley values),
     which can differ sharply from how *capital* is distributed.

Run locally:  streamlit run app/app.py
"""
from __future__ import annotations

import datetime as dt

import pandas as pd
import streamlit as st

from lib import data_sources, factor_models, metrics, plots, risk_decomposition, ui

st.set_page_config(page_title="Portfolio Analytics", layout="wide", page_icon="📊")
ui.inject_css()

# ---------------------------------------------------------------------------
# Sidebar — global settings
# ---------------------------------------------------------------------------
st.sidebar.title("Settings")
mode = st.sidebar.radio(
    "Mode",
    ["Single series vs benchmark", "Multi-asset portfolio risk decomposition"],
)
rf_annual = st.sidebar.number_input("Annual risk-free rate", value=0.0, step=0.001, format="%.4f")
var_confidence = st.sidebar.slider("VaR / ES confidence", 0.90, 0.99, 0.95, 0.01)
periods_per_year = st.sidebar.selectbox(
    "Data frequency", [252, 52, 12], format_func=lambda x: {252: "Daily", 52: "Weekly", 12: "Monthly"}[x]
)

st.title("📊 Portfolio Analytics")

# ---------------------------------------------------------------------------
# Helper: ticker picker used everywhere (any Yahoo symbol — stocks, ETFs,
# index funds, indices, any market)
# ---------------------------------------------------------------------------
def ticker_picker(label: str, key_prefix: str, default_ticker: str = "") -> str:
    st.caption(f"{label} — search by company/fund name (e.g. \"Lynx Dynamic\", \"Volvo\", \"S&P 500\") "
               "or type any Yahoo Finance symbol directly.")
    query = st.text_input("Search", key=f"{key_prefix}_search", placeholder="e.g. Lynx Dynamic, Apple, gold ETF...")
    ticker = default_ticker

    if query:
        live_results = data_sources.live_ticker_search(query)
        if live_results:
            options = {
                f"{r['name']}  ·  {r['symbol']}  ·  {r['exchange']}"
                + (f" ({r['type']})" if r["type"] else ""): r["symbol"]
                for r in live_results
            }
            chosen_name = st.selectbox("Matches", list(options.keys()), key=f"{key_prefix}_match")
            ticker = options[chosen_name]
        else:
            # Live search found nothing (offline, no hits, etc.) — fall back
            # to the built-in catalog, then to treating the query as a raw ticker.
            matches = data_sources.search_tickers(query)
            if matches:
                chosen_name = st.selectbox("Matches (built-in list)", list(matches.keys()), key=f"{key_prefix}_match")
                ticker = matches[chosen_name]
            else:
                st.caption("No name match found — treating your input as a ticker symbol directly.")
                ticker = query.strip()

    ticker = st.text_input(
        "Yahoo Finance ticker",
        value=ticker,
        key=f"{key_prefix}_ticker",
        help="Works for any global market: e.g. VOLV-B.ST, AAPL, SPY, ^GSPC, 7203.T, MC.PA",
    )
    return ticker


def get_return_series(label: str, key_prefix: str) -> pd.Series | None:
    st.subheader(label)
    source = st.radio("Source", ["Upload CSV", "Ticker"], key=f"{key_prefix}_source", horizontal=True)

    if source == "Upload CSV":
        f = st.file_uploader("CSV with a date column and a returns/price column",
                              type="csv", key=f"{key_prefix}_csv")
        is_prices = st.checkbox("Column is prices, not returns", key=f"{key_prefix}_isprice")
        if f is not None:
            try:
                return data_sources.load_csv_returns(f, is_prices=is_prices)
            except Exception as e:
                st.error(f"Could not parse CSV: {e}")
        return None

    ticker = ticker_picker("Search any stock, ETF, or index fund", key_prefix)
    d1, d2 = st.columns(2)
    with d1:
        start = st.date_input("Start", value=dt.date.today() - dt.timedelta(days=5 * 365),
                               key=f"{key_prefix}_start")
    with d2:
        end = st.date_input("End", value=dt.date.today(), key=f"{key_prefix}_end")

    if ticker:
        try:
            return data_sources.fetch_returns(ticker, str(start), str(end))
        except Exception as e:
            st.error(f"Could not fetch '{ticker}': {e}")
    return None


def get_benchmark_series(key_prefix: str) -> pd.Series | None:
    st.subheader("Benchmark (optional)")
    use_bench = st.checkbox("Compare against a benchmark", value=True, key=f"{key_prefix}_use")
    if not use_bench:
        return None
    ticker = ticker_picker("Search any index, ETF, or stock to use as benchmark", key_prefix, default_ticker="^OMX")

    d1, d2 = st.columns(2)
    with d1:
        start = st.date_input("Benchmark start", value=dt.date.today() - dt.timedelta(days=5 * 365),
                               key=f"{key_prefix}_bstart")
    with d2:
        end = st.date_input("Benchmark end", value=dt.date.today(), key=f"{key_prefix}_bend")
    if not ticker:
        return None
    try:
        return data_sources.fetch_returns(ticker, str(start), str(end))
    except Exception as e:
        st.error(f"Could not fetch benchmark '{ticker}': {e}")
        return None


def metric_cards_for(table: pd.Series) -> list[dict]:
    """Turn a metrics.summary_table Series into KPI-card dicts."""
    display_names = {
        "CAGR": ("CAGR", "Annualized growth"),
        "Annualized Volatility": ("Volatility", "Annualized std dev"),
        "Sharpe Ratio": ("Sharpe", "Risk-adjusted return"),
        "Sortino Ratio": ("Sortino", "Downside risk-adjusted"),
        "Max Drawdown": ("Max Drawdown", "Worst peak-to-trough"),
    }
    cards = []
    for key, (label, caption) in display_names.items():
        if key not in table.index:
            continue
        val = table[key]
        is_pct = key not in ("Sharpe Ratio", "Sortino Ratio")
        value_str = f"{val:.2%}" if is_pct else f"{val:.2f}"
        cards.append({
            "label": label, "value": value_str, "caption": caption,
            "sentiment": ui.sentiment_for(key, val),
        })
    for key in table.index:
        if key.startswith("Historical VaR") or key.startswith("Expected Shortfall"):
            cards.append({
                "label": key, "value": f"{table[key]:.2%}", "caption": "Single-period, historical",
                "sentiment": "negative" if table[key] < 0 else "neutral",
            })
    return cards


# ---------------------------------------------------------------------------
# MODE 1: Single series vs benchmark
# ---------------------------------------------------------------------------
if mode == "Single series vs benchmark":
    setup_tab, perf_tab, factor_tab = st.tabs(["Setup", "Performance", "Factor Decomposition"])

    with setup_tab:
        c1, c2 = st.columns(2)
        with c1:
            returns = get_return_series("Your return stream", "main")
        with c2:
            bench_returns = get_benchmark_series("bench")

    if "returns" in dir() and returns is not None:
        aligned = pd.concat([returns.rename("port"), bench_returns.rename("bench")], axis=1, join="inner") \
            if bench_returns is not None else pd.concat([returns.rename("port")], axis=1)
        r = aligned["port"]
        b = aligned["bench"] if "bench" in aligned.columns else None

        with perf_tab:
            table = metrics.summary_table(r, rf_annual, periods_per_year, var_confidence)
            ui.render_kpi_grid(metric_cards_for(table), n_cols=4)

            if b is not None:
                st.caption("Benchmark")
                table_b = metrics.summary_table(b, rf_annual, periods_per_year, var_confidence)
                ui.render_kpi_grid(metric_cards_for(table_b), n_cols=4)

            st.plotly_chart(plots.equity_curve_fig(r, b), use_container_width=True)
            st.plotly_chart(plots.underwater_fig(r, b), use_container_width=True)
            window = st.slider("Rolling window (periods)", 20, 252, 63, key="rollwin")
            st.plotly_chart(
                plots.rolling_sharpe_fig(r, window, rf_annual, periods_per_year, b),
                use_container_width=True,
            )

        with factor_tab:
            model = st.selectbox("Factor model", ["CAPM", "FF3", "FF5"])
            st.caption("Factor data: Kenneth French *Europe* daily factors (closest standard proxy — "
                       "no Sweden-specific Fama-French factor set is published).")
            try:
                factors = data_sources.fetch_ff_factors("FF3" if model == "CAPM" else model)
                result = factor_models.run_factor_regression(r, factors, model)

                cards = [
                    {"label": "Alpha (annualized)", "value": f"{result['alpha_annualized']:.2%}",
                     "caption": f"{model} intercept", "sentiment": ui.sentiment_for("_alpha", result["alpha_annualized"])},
                ]
                for f, beta in result["betas"].items():
                    cards.append({"label": f"Beta: {f}", "value": f"{beta:.3f}",
                                  "caption": "Factor exposure", "sentiment": "neutral"})
                cards.append({"label": "R²", "value": f"{result['r_squared']:.3f}",
                              "caption": "Variance explained", "sentiment": "neutral"})
                cards.append({"label": "Observations", "value": str(result["n_obs"]),
                              "caption": "Overlapping periods", "sentiment": "neutral"})
                ui.render_kpi_grid(cards, n_cols=4)

                st.plotly_chart(
                    plots.factor_contribution_fig(result["contribution"], result["residual_contribution"]),
                    use_container_width=True,
                )

                st.subheader("Rolling factor betas")
                rwindow = st.slider("Rolling window for betas (periods)", 40, 252, 126, key="betawin")
                rb = factor_models.rolling_factor_betas(r, factors, model, rwindow)
                st.plotly_chart(
                    plots.rolling_betas_fig(rb, factor_models.FACTOR_SETS[model]),
                    use_container_width=True,
                )
            except Exception as e:
                st.error(f"Factor regression failed: {e}")
                with st.expander("Full error details"):
                    st.exception(e)
    else:
        st.info("Provide a return series in the Setup tab to see analysis.")

# ---------------------------------------------------------------------------
# MODE 2: Multi-asset portfolio risk decomposition
# ---------------------------------------------------------------------------
else:
    st.subheader("Portfolio holdings")
    st.caption("Enter tickers and weights — any stock, ETF, or index fund on Yahoo Finance. "
               "Weights don't need to sum to 1 exactly.")

    n_assets = st.number_input("Number of holdings", min_value=2, max_value=18, value=4)
    tickers, weights = [], []
    cols = st.columns(min(n_assets, 4))
    for i in range(n_assets):
        with cols[i % len(cols)]:
            t = st.text_input(f"Ticker {i+1}", value="", key=f"tkr_{i}")
            w = st.number_input(f"Weight {i+1}", value=round(1 / n_assets, 3), key=f"w_{i}", step=0.01)
            tickers.append(t.strip())
            weights.append(w)

    d1, d2 = st.columns(2)
    with d1:
        start = st.date_input("Start", value=dt.date.today() - dt.timedelta(days=3 * 365))
    with d2:
        end = st.date_input("End", value=dt.date.today())

    if st.button("Run risk decomposition"):
        valid = [(t, w) for t, w in zip(tickers, weights) if t]
        if len(valid) < 2:
            st.error("Enter at least two tickers.")
        else:
            try:
                series = {}
                for t, _ in valid:
                    series[t] = data_sources.fetch_returns(t, str(start), str(end))
                asset_returns = pd.DataFrame(series).dropna()
                w_series = pd.Series({t: w for t, w in valid})

                port_returns = asset_returns.mul(w_series, axis=1).sum(axis=1)

                st.subheader("Portfolio-level metrics")
                port_table = metrics.summary_table(port_returns, rf_annual, periods_per_year, var_confidence)
                ui.render_kpi_grid(metric_cards_for(port_table), n_cols=4)

                st.plotly_chart(plots.equity_curve_fig(port_returns, title="Portfolio Equity Curve"),
                                 use_container_width=True)
                st.plotly_chart(plots.underwater_fig(port_returns), use_container_width=True)

                st.subheader("Risk contribution: weight vs. risk")
                euler = risk_decomposition.euler_risk_contribution(w_series, asset_returns)
                st.plotly_chart(
                    plots.risk_contribution_fig(euler, "pct_of_weight", "pct_of_risk",
                                                 "Euler method: % of capital vs. % of risk"),
                    use_container_width=True,
                )
                st.dataframe(euler.style.format("{:.2%}"), use_container_width=True)

                if len(valid) <= 15:
                    shap = risk_decomposition.shapley_risk_contribution(w_series, asset_returns)
                    st.plotly_chart(
                        plots.risk_contribution_fig(shap, "pct_of_weight", "pct_of_risk",
                                                     "Shapley method: % of capital vs. % of risk"),
                        use_container_width=True,
                    )
                    st.dataframe(shap.style.format("{:.2%}"), use_container_width=True)
                else:
                    st.info("Shapley decomposition skipped: more than 15 holdings (cost grows as 2^N).")
            except Exception as e:
                st.error(f"Something went wrong: {e}")
                with st.expander("Full error details"):
                    st.exception(e)
