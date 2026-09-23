"""Plotly chart builders, styled to match a dark KPI-dashboard aesthetic.
Kept separate from Streamlit so they're reusable if the UI layer ever changes."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

from . import metrics

BG = "#0a0e17"
GRID = "#1f2937"
TEXT = "#c9d1d9"
GREEN = "#22c55e"
BLUE = "#60a5fa"
RED = "#ef4444"


def _base_layout(fig: go.Figure, title: str, yaxis_title: str = "", xaxis_title: str = "Date") -> go.Figure:
    fig.update_layout(
        title=dict(text=title, font=dict(size=16, color=TEXT)),
        paper_bgcolor=BG,
        plot_bgcolor=BG,
        font=dict(color=TEXT, family="-apple-system, Segoe UI, Roboto, sans-serif"),
        yaxis_title=yaxis_title,
        xaxis_title=xaxis_title,
        hovermode="x unified",
        legend=dict(orientation="h", y=1.08, bgcolor="rgba(0,0,0,0)"),
        margin=dict(l=10, r=10, t=50, b=10),
    )
    fig.update_xaxes(gridcolor=GRID, zerolinecolor=GRID, showgrid=True)
    fig.update_yaxes(gridcolor=GRID, zerolinecolor=GRID, showgrid=True)
    return fig


def equity_curve_fig(returns: pd.Series, benchmark_returns: pd.Series | None = None,
                      title: str = "Equity Curve — Growth of $1") -> go.Figure:
    curve = metrics.returns_to_cum_growth(returns)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=curve.index, y=curve.values, name="Portfolio", mode="lines",
                              line=dict(color=GREEN, width=2.5),
                              fill="tozeroy", fillcolor="rgba(34,197,94,0.08)"))
    if benchmark_returns is not None:
        bench_curve = metrics.returns_to_cum_growth(benchmark_returns)
        fig.add_trace(go.Scatter(x=bench_curve.index, y=bench_curve.values, name="Benchmark",
                                  mode="lines", line=dict(color=BLUE, width=1.5, dash="dot")))
    _base_layout(fig, title, yaxis_title="Growth of $1")
    fig.update_yaxes(tickprefix="$")
    return fig


def underwater_fig(returns: pd.Series, benchmark_returns: pd.Series | None = None,
                    title: str = "Drawdown from Peak (Underwater Plot)") -> go.Figure:
    uw = metrics.underwater_series(returns)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=uw.index, y=uw.values * 100, name="Portfolio", fill="tozeroy",
                              mode="lines", line=dict(color=RED, width=1.5),
                              fillcolor="rgba(239,68,68,0.15)"))
    if benchmark_returns is not None:
        uw_b = metrics.underwater_series(benchmark_returns)
        fig.add_trace(go.Scatter(x=uw_b.index, y=uw_b.values * 100, name="Benchmark",
                                  mode="lines", line=dict(color=BLUE, width=1.2, dash="dot")))
    _base_layout(fig, title, yaxis_title="Drawdown (%)")
    return fig


def rolling_sharpe_fig(returns: pd.Series, window: int = 63, rf: float = 0.0,
                        periods_per_year: int = 252,
                        benchmark_returns: pd.Series | None = None,
                        title: str | None = None) -> go.Figure:
    rs = metrics.rolling_sharpe(returns, window, rf, periods_per_year)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=rs.index, y=rs.values, name="Portfolio", mode="lines",
                              line=dict(color=GREEN, width=2)))
    if benchmark_returns is not None:
        rs_b = metrics.rolling_sharpe(benchmark_returns, window, rf, periods_per_year)
        fig.add_trace(go.Scatter(x=rs_b.index, y=rs_b.values, name="Benchmark",
                                  mode="lines", line=dict(color=BLUE, width=1.2, dash="dot")))
    fig.add_hline(y=0, line_dash="dash", line_color=GRID)
    _base_layout(fig, title or f"Rolling {window}-period Sharpe Ratio", yaxis_title="Sharpe Ratio")
    return fig


def rolling_betas_fig(rolling_betas_df: pd.DataFrame, factor_cols: list[str],
                       title: str = "Rolling Factor Betas") -> go.Figure:
    palette = [GREEN, BLUE, RED, "#f59e0b", "#a78bfa"]
    fig = go.Figure()
    for i, f in enumerate(factor_cols):
        fig.add_trace(go.Scatter(x=rolling_betas_df.index, y=rolling_betas_df[f], name=f,
                                  mode="lines", line=dict(color=palette[i % len(palette)], width=2)))
    fig.add_hline(y=0, line_dash="dash", line_color=GRID)
    _base_layout(fig, title, yaxis_title="Beta")
    return fig


def factor_contribution_fig(contribution: dict, residual: float,
                             title: str = "Return Decomposition") -> go.Figure:
    labels = list(contribution.keys()) + ["Alpha (unexplained)"]
    values = [v * 100 for v in contribution.values()] + [residual * 100]
    colors = [GREEN if v >= 0 else RED for v in values]
    fig = go.Figure(go.Bar(x=labels, y=values, marker_color=colors))
    _base_layout(fig, title, yaxis_title="Annualized contribution (%)")
    return fig


def risk_contribution_fig(contrib_df: pd.DataFrame, weight_col: str, risk_col: str,
                           title: str = "Weight vs. Risk Contribution") -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Bar(x=contrib_df.index, y=contrib_df[weight_col] * 100, name="% of Weight",
                          marker_color=BLUE))
    fig.add_trace(go.Bar(x=contrib_df.index, y=contrib_df[risk_col] * 100, name="% of Risk",
                          marker_color=GREEN))
    _base_layout(fig, title, yaxis_title="Percent (%)")
    fig.update_layout(barmode="group")
    return fig
