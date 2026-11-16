"""Shared look-and-feel helpers for all dashboard pages."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
for sub in ("task1_stock_prediction", "task2_ai_dashboard", "task3_inventory", "task4_loan_approval"):
    p = str(ROOT / sub)
    if p not in sys.path:
        sys.path.insert(0, p)

COLORS = {"primary": "#2563eb", "green": "#059669", "red": "#dc2626", "amber": "#d97706",
          "violet": "#7c3aed", "slate": "#64748b", "cyan": "#0891b2"}
SEQ = ["#2563eb", "#059669", "#d97706", "#7c3aed", "#dc2626", "#0891b2", "#db2777", "#65a30d", "#ea580c", "#475569"]
# Title sits at the very top and the legend goes underneath the chart, so the two can never overlap.
PLOT = dict(template="plotly_white", margin=dict(l=10, r=10, t=50, b=10), font=dict(family="Segoe UI, Arial", size=13),
            title=dict(x=0, xanchor="left", y=0.98, yanchor="top", font=dict(size=16)),
            legend=dict(orientation="h", yanchor="top", y=-0.12, x=0, xanchor="left"), colorway=SEQ)

CSS = """
<style>
.block-container {padding-top: 2.2rem; max-width: 1400px;}
div[data-testid="stMetric"] {background: var(--secondary-background-color); border: 1px solid rgba(128,128,128,.18);
  border-radius: 12px; padding: 14px 16px;}
div[data-testid="stMetricLabel"] p {font-size: .82rem; opacity: .75;}
.pill {display:inline-block; padding:2px 10px; border-radius:99px; font-size:.72rem; font-weight:600; color:#fff;
  text-transform:uppercase; letter-spacing:.03em;}
.card {border:1px solid rgba(128,128,128,.22); border-radius:12px; padding:12px 16px; margin-bottom:10px;
  background: var(--secondary-background-color);}
.card h4 {margin:.35rem 0 .2rem; font-size:1rem;}
.card p {margin:.15rem 0; font-size:.9rem;}
.muted {opacity:.7; font-size:.86rem;}
</style>
"""


def page(title: str, subtitle: str = "") -> None:
    st.markdown(CSS, unsafe_allow_html=True)
    st.title(title)
    if subtitle:
        st.caption(subtitle)


def style(fig, height: int = 380, **kw):
    layout = dict(PLOT)
    text = fig.layout.title.text
    if text:                                              # keep the chart's own title text, apply our placement
        layout["title"] = dict(PLOT["title"], text=text)
    layout.update(kw)
    fig.update_layout(**layout, height=height)
    return fig


def show(fig, height: int = 380, **kw) -> None:
    st.plotly_chart(style(fig, height, **kw), width="stretch")


def pill(text: str, color: str) -> str:
    return f"<span class='pill' style='background:{color}'>{text}</span>"


def card(tag: str, color: str, title: str, body: str, action: str = "") -> None:
    extra = f"<p><b>Recommendation:</b> {action}</p>" if action else ""
    st.markdown(f"<div class='card'>{pill(tag, color)}<h4>{title}</h4><p>{body}</p>{extra}</div>", unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def load_json(path: str) -> dict:
    return json.loads(Path(path).read_text())


@st.cache_data(show_spinner=False)
def load_csv(path: str, **kw) -> pd.DataFrame:
    return pd.read_csv(path, **kw)


def need(path: Path, command: str) -> bool:
    """Tell the user how to produce a missing artifact. Returns True when it exists."""
    if path.exists():
        return True
    st.warning(f"This page needs results that have not been generated yet. Run:\n\n`python {command}`")
    return False


def money(x: float) -> str:
    ax = abs(x)
    if ax >= 1e9: return f"${x / 1e9:.2f}B"
    if ax >= 1e6: return f"${x / 1e6:.2f}M"
    if ax >= 1e3: return f"${x / 1e3:.1f}K"
    return f"${x:,.2f}"
