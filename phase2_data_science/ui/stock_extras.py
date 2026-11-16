"""Extra tabs of the stock page: news sentiment and portfolio analysis."""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from common import COLORS, card, show
import extras
import features as F

TONE = {"Positive": COLORS["green"], "Neutral": COLORS["slate"], "Negative": COLORS["red"]}


@st.cache_data(ttl=900, show_spinner="Fetching headlines...")
def headlines(symbol: str) -> pd.DataFrame:
    return extras.news_sentiment(symbol)


def news_tab(sym: str, names: dict, signal: str) -> None:
    st.subheader(f"News sentiment for {names[sym]}")
    st.write("Latest headlines from Yahoo Finance, scored from -1 (very negative) to +1 (very positive) with the VADER sentiment model.")
    if st.button("Analyse latest news", type="primary", key="news_btn"):
        st.session_state["news_for"] = sym
    if st.session_state.get("news_for") != sym:
        return
    try:
        nw = headlines(sym)
    except Exception as exc:
        st.error(f"News is unavailable right now ({str(exc)[:120]}).")
        return
    if nw.empty:
        st.info("Yahoo Finance returned no headlines for this company right now.")
        return
    avg = nw["Sentiment"].mean()
    mood = "positive" if avg > 0.1 else "negative" if avg < -0.1 else "neutral"
    c = st.columns(4)
    c[0].metric("Headlines", len(nw))
    c[1].metric("Average sentiment", f"{avg:+.2f}", mood, delta_color="off")
    c[2].metric("Positive", int((nw["Tone"] == "Positive").sum()))
    c[3].metric("Negative", int((nw["Tone"] == "Negative").sum()))
    a, b = st.columns([2, 3])
    with a:
        tone = nw["Tone"].value_counts().reset_index()
        tone.columns = ["Tone", "Headlines"]
        show(px.pie(tone, names="Tone", values="Headlines", hole=0.55, color="Tone", title="Tone of coverage", color_discrete_map=TONE), 320)
    with b:
        show(px.bar(nw.sort_values("Published"), x="Published", y="Sentiment", color="Tone", hover_name="Headline",
                    title="Sentiment by headline", color_discrete_map=TONE), 320)
    st.dataframe(nw, width="stretch", hide_index=True,
                 column_config={"Link": st.column_config.LinkColumn(display_text="open"),
                                "Published": st.column_config.DatetimeColumn(format="D MMM, HH:mm")})
    agree = (avg > 0.1 and signal == "BUY") or (avg < -0.1 and signal == "SELL")
    card("Reading", COLORS["primary"], f"News tone is {mood}; the price model says {signal}",
         "The two signals agree." if agree else "The two signals do not point the same way.",
         "Headline tone is a weak, short-lived signal. Use it as context for the price model, not instead of it.")


def portfolio_tab(names: dict, history) -> None:
    st.subheader("Portfolio performance analysis")
    a, b, c3 = st.columns([3, 1, 1])
    default = [s for s in ("AAPL", "MSFT", "JPM", "JNJ", "XOM") if s in names]
    held = a.multiselect("Holdings", list(names), default=default, format_func=lambda s: names[s], key="pf_hold")
    capital = b.number_input("Starting amount ($)", 1_000, 10_000_000, 10_000, 1_000)
    period = c3.selectbox("Period", ["1Y", "3Y", "5Y", "All"], index=2)
    if len(held) < 2:
        st.info("Choose at least two companies.")
        return
    st.caption("Weights (%). They are rescaled to add up to 100.")
    boxes = st.columns(len(held))
    weights = {s: box.number_input(s, 0.0, 100.0, round(100 / len(held), 1), 5.0, key=f"w_{s}") for s, box in zip(held, boxes)}
    if sum(weights.values()) <= 0:
        st.warning("Give at least one holding a weight above zero.")
        return
    n = {"1Y": 252, "3Y": 756, "5Y": 1260, "All": 10 ** 6}[period]
    closes = pd.DataFrame({s: history(s).set_index("Date")["Close"] for s in held}).dropna().tail(n)
    bench, _ = F.load_prices(F.MARKET)
    R = extras.portfolio(closes, weights, bench.set_index("Date")["Close"], capital)
    P, M = R["portfolio"], R["market"]
    c = st.columns(5)
    c[0].metric("Portfolio value", f"${P['Final value']:,.0f}", f"{P['Total return %']:+.1f}%")
    c[1].metric("S&P 500 same period", f"${M['Final value']:,.0f}", f"{M['Total return %']:+.1f}%")
    c[2].metric("Yearly return", f"{P['Yearly return % (CAGR)']:.1f}%",
                f"{P['Yearly return % (CAGR)'] - M['Yearly return % (CAGR)']:+.1f} pts vs market")
    c[3].metric("Sharpe ratio", f"{P['Sharpe ratio']:.2f}", f"market {M['Sharpe ratio']:.2f}", delta_color="off")
    c[4].metric("Worst fall", f"{P['Max drawdown %']:.1f}%", f"market {M['Max drawdown %']:.1f}%", delta_color="off")
    fig = go.Figure([go.Scatter(x=R["value"].index, y=R["value"], name="Portfolio", line=dict(color=COLORS["primary"], width=2.4)),
                     go.Scatter(x=R["benchmark"].index, y=R["benchmark"], name="S&P 500", line=dict(color=COLORS["slate"], width=1.6, dash="dot"))])
    fig.update_layout(title=f"Growth of ${capital:,.0f}", yaxis_title="Value ($)")
    show(fig, 400)
    a, b, c3 = st.columns(3)
    with a:
        we = (R["weights_end"] * 100).rename(index=names).reset_index()
        we.columns = ["Holding", "Weight %"]
        show(px.pie(we, names="Holding", values="Weight %", hole=0.5, title="Allocation today (after drift)"), 330)
    with b:
        ct = R["contribution"].rename(index=names).reset_index()
        ct.columns = ["Holding", "Contribution (pts)"]
        fig = px.bar(ct, x="Contribution (pts)", y="Holding", orientation="h", title="Where the return came from",
                     color=ct["Contribution (pts)"] > 0, color_discrete_map={True: COLORS["green"], False: COLORS["red"]})
        fig.update_yaxes(autorange="reversed")
        show(fig, 330, showlegend=False)
    with c3:
        dd = (R["drawdown"] * 100).reset_index()
        dd.columns = ["Date", "Drawdown %"]
        show(px.area(dd, x="Date", y="Drawdown %", title="Fall from the previous peak", color_discrete_sequence=[COLORS["red"]]), 330)
    st.dataframe(pd.DataFrame({"Portfolio": P, "S&P 500": M}).round(2), width="stretch")
    top = R["contribution"].index[0]
    ahead = P["Total return %"] > M["Total return %"]
    card("Summary", COLORS["green"] if ahead else COLORS["amber"],
         f"The portfolio {'beat' if ahead else 'trailed'} the S&P 500 by {abs(P['Total return %'] - M['Total return %']):.1f} points",
         f"Beta {R['beta']:.2f}: it moves about {R['beta']:.1f}x as much as the market. Biggest contributor: {names[top]} "
         f"({R['contribution'].iloc[0]:+.1f} points). It started at {R['weights_start'][top]:.0%} of the portfolio and is now {R['weights_end'][top]:.0%}.",
         "Rebalance if one holding has grown far past its starting weight. Past returns do not guarantee future ones.")
