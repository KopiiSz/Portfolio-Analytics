"""
Portfolio Analytics — a from-scratch tearsheet tool.

Two modes:
  1. Single series vs benchmark: bring your own CSV of returns, or pick a
     ticker (Swedish market focus, but any Yahoo Finance symbol works).
     Get performance stats, equity curve, underwater plot, rolling Sharpe,
     and CAPM / FF3 / FF5 factor decomposition with rolling betas.
  2. Multi-asset portfolio: enter tickers + weights and see how risk is
     really distributed across holdings (Euler method + Shapley values),
     which can differ sharply from how *capital* is distributed.

Run locally:  streamlit run app/app.py
"""
from __future__ import annotations

import datetime as dt

import pandas as pd
import streamlit as st

from lib import data_sources, factor_models, metrics, plots, risk_decomposition

st.set_page_config(page_title="Portfolio Analytics", layout="wide")

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
# Helper: get a single return series from CSV or ticker
# ---------------------------------------------------------------------------
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

    col1, col2 = st.columns([2, 1])
    with col1:
        query = st.text_input("Search Swedish tickers (or type any Yahoo ticker below)",
                               key=f"{key_prefix}_search")
        if query:
            matches = data_sources.search_swedish_ticker(query)
            if matches:
                chosen_name = st.selectbox("Matches", list(matches.keys()), key=f"{key_prefix}_match")
                default_ticker = matches[chosen_name]
            else:
                default_ticker = query
        else:
            default_ticker = ""
    ticker = st.text_input("Yahoo Finance ticker (e.g. VOLV-B.ST, AAPL, ^OMX)",
                            value=default_ticker, key=f"{key_prefix}_ticker")
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
    name = st.selectbox("Benchmark", ["Custom ticker..."] + list(data_sources.BENCHMARK_TICKERS.keys()),
                         key=f"{key_prefix}_choice")
    if name == "Custom ticker...":
        ticker = st.text_input("Benchmark ticker", value="^OMX", key=f"{key_prefix}_customticker")
    else:
        ticker = data_sources.BENCHMARK_TICKERS[name]

    d1, d2 = st.columns(2)
    with d1:
        start = st.date_input("Benchmark start", value=dt.date.today() - dt.timedelta(days=5 * 365),
                               key=f"{key_prefix}_bstart")
    with d2:
        end = st.date_input("Benchmark end", value=dt.date.today(), key=f"{key_prefix}_bend")
    try:
        return data_sources.fetch_returns(ticker, str(start), str(end))
    except Exception as e:
        st.error(f"Could not fetch benchmark '{ticker}': {e}")
        return None


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
            st.subheader("Headline metrics")
            table = metrics.summary_table(r, rf_annual, periods_per_year, var_confidence)
            if b is not None:
                table_b = metrics.summary_table(b, rf_annual, periods_per_year, var_confidence)
                display = pd.DataFrame({"Portfolio": table, "Benchmark": table_b})
            else:
                display = table.to_frame("Portfolio")
            pct_rows = [i for i in display.index if i != "Sharpe Ratio" and i != "Sortino Ratio"]
            fmt = {}
            for row in display.index:
                fmt[row] = "{:.2%}" if row in pct_rows else "{:.2f}"
            st.dataframe(display.style.format(fmt), use_container_width=True)

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

                c1, c2, c3 = st.columns(3)
                c1.metric("Alpha (annualized)", f"{result['alpha_annualized']:.2%}")
                c1.metric("R²", f"{result['r_squared']:.3f}")
                c2.metric("Observations", result["n_obs"])
                for i, (f, beta) in enumerate(result["betas"].items()):
                    (c3 if i == 0 else c2).metric(f"Beta: {f}", f"{beta:.3f}")

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
    else:
        st.info("Provide a return series in the Setup tab to see analysis.")

# ---------------------------------------------------------------------------
# MODE 2: Multi-asset portfolio risk decomposition
# ---------------------------------------------------------------------------
else:
    st.subheader("Portfolio holdings")
    st.caption("Enter tickers and weights. Weights don't need to sum to 1 exactly — "
               "they'll be shown as-given and as a share of total.")

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
                st.dataframe(
                    metrics.summary_table(port_returns, rf_annual, periods_per_year, var_confidence)
                    .to_frame("Portfolio").style.format("{:.2%}"),
                    use_container_width=True,
                )
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
