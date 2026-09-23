"""Custom styling layer for the Streamlit app: a dark theme + KPI card
component, since Streamlit's defaults don't give this level of visual control."""
from __future__ import annotations

import streamlit as st

ACCENT_GREEN = "#22c55e"
ACCENT_RED = "#ef4444"
ACCENT_BLUE = "#60a5fa"
TEXT_MUTED = "#8b93a7"
CARD_BG = "#0f1420"
CARD_BORDER = "#1f2937"

CUSTOM_CSS = f"""
<style>
    .stApp {{
        background-color: #0a0e17;
    }}
    /* Tighten default Streamlit padding so cards feel intentional */
    .block-container {{
        padding-top: 2rem;
        padding-bottom: 2rem;
    }}
    h1, h2, h3 {{
        font-family: -apple-system, "Segoe UI", Roboto, sans-serif;
        letter-spacing: -0.02em;
    }}
    /* KPI card grid */
    .kpi-card {{
        background-color: {CARD_BG};
        border: 1px solid {CARD_BORDER};
        border-radius: 12px;
        padding: 18px 20px;
        margin-bottom: 12px;
    }}
    .kpi-label {{
        color: {TEXT_MUTED};
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        margin-bottom: 6px;
    }}
    .kpi-value {{
        font-size: 1.9rem;
        font-weight: 700;
        font-family: "SF Mono", "Roboto Mono", monospace;
        line-height: 1.2;
    }}
    .kpi-caption {{
        color: {TEXT_MUTED};
        font-size: 0.78rem;
        margin-top: 4px;
    }}
    .kpi-value.positive {{ color: {ACCENT_GREEN}; }}
    .kpi-value.negative {{ color: {ACCENT_RED}; }}
    .kpi-value.neutral  {{ color: #e5e7eb; }}
    /* Streamlit tabs restyle */
    .stTabs [data-baseweb="tab"] {{
        font-weight: 600;
    }}
</style>
"""


def inject_css() -> None:
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def kpi_card(label: str, value: str, caption: str = "", sentiment: str = "neutral") -> str:
    """
    Build one KPI card's HTML. `sentiment` is 'positive', 'negative', or
    'neutral' and controls the value's color (green/red/white).
    """
    return f"""
    <div class="kpi-card">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value {sentiment}">{value}</div>
        <div class="kpi-caption">{caption}</div>
    </div>
    """


def render_kpi_grid(cards: list[dict], n_cols: int = 4) -> None:
    """
    cards: list of dicts with keys label, value, caption, sentiment.
    Renders them in a grid of n_cols columns using Streamlit columns +
    the kpi_card HTML snippet.
    """
    cols = st.columns(n_cols)
    for i, card in enumerate(cards):
        with cols[i % n_cols]:
            st.markdown(
                kpi_card(
                    card["label"],
                    card["value"],
                    card.get("caption", ""),
                    card.get("sentiment", "neutral"),
                ),
                unsafe_allow_html=True,
            )


def sentiment_for(metric_name: str, value: float) -> str:
    """Simple heuristic: is a higher or lower number 'good' for this metric?"""
    if value != value:  # NaN check
        return "neutral"
    lower_is_better = {"Max Drawdown", "Annualized Volatility"}
    if metric_name in lower_is_better:
        return "negative" if value < 0 else "positive"
    return "positive" if value >= 0 else "negative"
