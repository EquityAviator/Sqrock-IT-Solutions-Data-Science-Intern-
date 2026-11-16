"""Task 1 - Stock Market Prediction System."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from common import COLORS, ROOT, card, load_csv, load_json, need, page, show, style
import features as F

ART = ROOT / "task1_stock_prediction" / "artifacts"
page("Stock Market Prediction System", "Task 1 · Real Yahoo Finance prices · five models including a GPU-trained LSTM")
if not need(ART / "metrics.json", "task1_stock_prediction/train.py"):
    st.stop()

M = load_json(str(ART / "metrics.json"))
NAMES = {s: v["name"] for s, v in M["tickers"].items()}
NAIVE = "Naive (tomorrow = today)"


@st.cache_data(show_spinner=False)
def history(sym: str) -> pd.DataFrame:
    df, _ = F.build(sym)
    return df


with st.sidebar:
    st.subheader("Stock")
    sym = st.selectbox("Company", list(NAMES), format_func=lambda s: f"{NAMES[s]} ({s})")
    years = st.select_slider("History shown", ["6M", "1Y", "2Y", "5Y", "All"], "2Y")
    st.caption(f"Data: {M['date_range'][0]} → {M['date_range'][1]}")

df = history(sym)
T = M["tickers"][sym]
days = {"6M": 126, "1Y": 252, "2Y": 504, "5Y": 1260, "All": len(df)}[years]
view = df.tail(days)
last, prev = df["Close"].iloc[-1], df["Close"].iloc[-2]
fc = T["forecast"]

c = st.columns(5)
c[0].metric("Last close", f"${last:,.2f}", f"{(last / prev - 1) * 100:+.2f}% on the day")
c[1].metric("Predicted next close", f"${fc['next_day']:,.2f}", f"{(fc['next_day'] / last - 1) * 100:+.2f}%")
c[2].metric("30-day forecast", f"${fc['day_30']:,.2f}", f"{fc['expected_change_30d_pct']:+.2f}%")
c[3].metric("Model signal", fc["signal"], f"by {fc['model']}", delta_color="off")
c[4].metric("1-year return", f"{(last / df['Close'].iloc[-253] - 1) * 100:+.1f}%")

tabs = st.tabs(["Price history", "Market overview", "Actual vs predicted", "Forecast & signal", "Model performance", "Live prediction",
                "News sentiment", "Portfolio"])

# ----------------------------------------------------------------------------- price history
with tabs[0]:
    a, b, c2 = st.columns([2, 2, 3])
    chart = a.radio("Chart", ["Candlestick", "Line"], horizontal=True)
    mas = b.multiselect("Moving averages", [20, 50, 200], [20, 50])
    extra = c2.multiselect("Indicators", ["Bollinger bands", "RSI", "MACD"], ["Bollinger bands", "RSI"])
    rows = 2 + ("RSI" in extra) + ("MACD" in extra)
    heights = [0.55] + [0.45 / (rows - 1)] * (rows - 1)
    fig = make_subplots(rows=rows, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=heights)
    if chart == "Candlestick":
        fig.add_trace(go.Candlestick(x=view["Date"], open=view["Open"], high=view["High"], low=view["Low"], close=view["Close"],
                                     name="Price", increasing_line_color=COLORS["green"], decreasing_line_color=COLORS["red"]), 1, 1)
    else:
        fig.add_trace(go.Scatter(x=view["Date"], y=view["Close"], name="Close", line=dict(color=COLORS["primary"], width=2)), 1, 1)
    for n, col in zip((20, 50, 200), (COLORS["amber"], COLORS["violet"], COLORS["slate"])):
        if n in mas:
            fig.add_trace(go.Scatter(x=view["Date"], y=view[f"sma_{n}"], name=f"{n}-day average", line=dict(color=col, width=1.3)), 1, 1)
    if "Bollinger bands" in extra:
        fig.add_trace(go.Scatter(x=view["Date"], y=view["boll_up"], line=dict(width=0), showlegend=False, hoverinfo="skip"), 1, 1)
        fig.add_trace(go.Scatter(x=view["Date"], y=view["boll_lo"], line=dict(width=0), fill="tonexty", fillcolor="rgba(37,99,235,.10)",
                                 name="Bollinger band", hoverinfo="skip"), 1, 1)
    up = view["Close"] >= view["Open"]
    fig.add_trace(go.Bar(x=view["Date"], y=view["Volume"], name="Volume", marker_color=np.where(up, COLORS["green"], COLORS["red"]),
                         opacity=0.6, showlegend=False), 2, 1)
    fig.update_yaxes(title_text="Price ($)", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    r = 3
    if "RSI" in extra:
        fig.add_trace(go.Scatter(x=view["Date"], y=view["rsi_14"], name="RSI (14)", line=dict(color=COLORS["violet"], width=1.4)), r, 1)
        fig.add_hline(y=70, line_dash="dot", line_color=COLORS["red"], row=r, col=1)
        fig.add_hline(y=30, line_dash="dot", line_color=COLORS["green"], row=r, col=1)
        fig.update_yaxes(title_text="RSI", range=[0, 100], row=r, col=1)
        r += 1
    if "MACD" in extra:
        fig.add_trace(go.Bar(x=view["Date"], y=view["macd_hist"] * 100, name="MACD histogram", marker_color=COLORS["slate"]), r, 1)
        fig.add_trace(go.Scatter(x=view["Date"], y=view["macd"] * 100, name="MACD", line=dict(color=COLORS["primary"], width=1.2)), r, 1)
        fig.add_trace(go.Scatter(x=view["Date"], y=view["macd_signal"] * 100, name="Signal", line=dict(color=COLORS["amber"], width=1.2)), r, 1)
        fig.update_yaxes(title_text="MACD (% of price)", row=r, col=1)
    fig.update_layout(xaxis_rangeslider_visible=False, title=f"{NAMES[sym]} ({sym})")
    show(fig, 420 + 130 * (rows - 1))
    cl = T["cleaning"]
    st.caption(f"Cleaning: {cl['rows_raw']:,} rows read · {cl['duplicates']} duplicate dates removed · {cl['missing_values']} missing values filled · "
               f"{cl['invalid_prices']} invalid prices removed · prices adjusted for splits and dividends.")
    rsi_now = df["rsi_14"].iloc[-1]
    trend_up = df["sma_50"].iloc[-1] > df["sma_200"].iloc[-1]
    a, b, c2 = st.columns(3)
    with a:
        card("Trend", COLORS["green"] if trend_up else COLORS["red"], "Uptrend" if trend_up else "Downtrend",
             f"The 50-day average (${df['sma_50'].iloc[-1]:,.2f}) is {'above' if trend_up else 'below'} the 200-day average (${df['sma_200'].iloc[-1]:,.2f}).")
    with b:
        state = "Overbought" if rsi_now > 70 else "Oversold" if rsi_now < 30 else "Neutral"
        card("Momentum", COLORS["amber"] if state != "Neutral" else COLORS["primary"], f"RSI {rsi_now:.0f} · {state}",
             "Above 70 often precedes a pause; below 30 often precedes a bounce.")
    with c2:
        vol = df["vol_20"].iloc[-1] * np.sqrt(252) * 100
        card("Risk", COLORS["violet"], f"Volatility {vol:.0f}% a year", f"Typical daily move ±{df['vol_20'].iloc[-1] * 100:.1f}%. Beta to the S&P 500: {df['beta_60'].iloc[-1]:.2f}.")

# ----------------------------------------------------------------------------- market overview
with tabs[1]:
    pick = st.multiselect("Compare companies", list(NAMES), default=list(NAMES)[:6], format_func=lambda s: NAMES[s])
    if pick:
        closes = pd.DataFrame({s: history(s).set_index("Date")["Close"] for s in pick}).dropna().tail(days)
        norm = closes / closes.iloc[0] * 100
        fig = px.line(norm, labels={"value": "Value of $100 invested", "Date": "", "variable": ""},
                      title=f"Growth of $100 over the last {years}")
        fig.for_each_trace(lambda t: t.update(name=NAMES[t.name]))
        show(fig, 420)
        rets = np.log(closes).diff().dropna()
        perf = pd.DataFrame({
            "Company": [NAMES[s] for s in pick],
            "Last close": closes.iloc[-1].values,
            "1 month %": (closes.iloc[-1] / closes.iloc[-22] - 1).values * 100 if len(closes) > 22 else np.nan,
            "Period return %": (closes.iloc[-1] / closes.iloc[0] - 1).values * 100,
            "Volatility % (yearly)": rets.std().values * np.sqrt(252) * 100,
            "Worst day %": rets.min().values * 100,
            "Max drawdown %": ((closes / closes.cummax() - 1).min()).values * 100,
        }).round(2)
        a, b = st.columns([3, 2])
        with a:
            st.subheader("Performance table")
            st.dataframe(perf, width="stretch", hide_index=True)
            fig = px.scatter(perf, x="Volatility % (yearly)", y="Period return %", text="Company", size_max=18,
                             title="Risk vs return")
            fig.update_traces(textposition="top center", marker=dict(size=13, color=COLORS["primary"]))
            show(fig, 380)
        with b:
            corr = rets.corr().round(2)
            corr.index = corr.columns = [s for s in pick]
            show(px.imshow(corr, text_auto=True, color_continuous_scale="RdBu_r", zmin=-1, zmax=1, aspect="auto",
                           title="Correlation of daily returns"), 480, coloraxis_showscale=False)
            st.caption("Stocks that move together (values near 1) give little diversification when held together.")

# ----------------------------------------------------------------------------- actual vs predicted
with tabs[2]:
    p = load_csv(str(ART / "predictions" / f"{sym}.csv"), parse_dates=["Date"])
    models = [m for m in T["models"] if m != NAIVE]
    chosen = st.multiselect("Models", models, default=[T["best_model"]])
    window = st.slider("Days shown", 30, len(p), min(250, len(p)))
    pv = p.tail(window)
    fig = go.Figure(go.Scatter(x=pv["Date"], y=pv["Actual"], name="Actual close", line=dict(color="#111827", width=2.4)))
    for i, m in enumerate(chosen):
        fig.add_trace(go.Scatter(x=pv["Date"], y=pv[m], name=f"Predicted · {m}", line=dict(width=1.4, dash="dot")))
    fig.update_layout(title=f"{NAMES[sym]}: actual vs predicted next-day close (unseen test period, {len(p)} trading days)", yaxis_title="Price ($)")
    show(fig, 440)
    a, b = st.columns(2)
    with a:
        m0 = chosen[0] if chosen else T["best_model"]
        err = (p[m0] - p["Actual"]) / p["Actual"] * 100
        fig = px.histogram(err, nbins=50, title=f"Prediction error of {m0} (% of price)", labels={"value": "Error (%)"})
        fig.add_vline(x=0, line_dash="dot")
        show(fig, 330, showlegend=False)
    with b:
        fig = px.scatter(p, x="Actual", y=m0, opacity=0.5, title="Predicted vs actual price")
        lim = [p["Actual"].min(), p["Actual"].max()]
        fig.add_trace(go.Scatter(x=lim, y=lim, mode="lines", line=dict(dash="dot", color=COLORS["red"]), name="perfect"))
        show(fig, 330, showlegend=False)
    st.subheader("Evaluation metrics on the test period")
    tbl = pd.DataFrame(T["models"]).T.rename(columns={"MAPE_pct": "MAPE %", "Direction_acc_pct": "Direction accuracy %"})
    tbl.insert(0, "RMSE vs naive", (tbl["RMSE"] / tbl.loc[NAIVE, "RMSE"]).round(4))
    st.dataframe(tbl.style.highlight_min(subset=["MAE", "MSE", "RMSE", "MAPE %"], color="#bbf7d0")
                 .highlight_max(subset=["R2", "Direction accuracy %"], color="#bbf7d0"), width="stretch")
    st.caption(f"Best model for {sym}, chosen on the 2023 validation year: **{T['best_model']}**. "
               "The naive row predicts that tomorrow's close equals today's. A model only adds value where its RMSE vs naive is below 1.")

# ----------------------------------------------------------------------------- forecast
with tabs[3]:
    f = load_csv(str(ART / "forecast" / f"{sym}.csv"), parse_dates=["Date"])
    h = df.tail(120)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=h["Date"], y=h["Close"], name="History", line=dict(color="#111827", width=2)))
    fig.add_trace(go.Scatter(x=f["Date"], y=f["Upper_80"], line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=f["Date"], y=f["Lower_80"], line=dict(width=0), fill="tonexty", fillcolor="rgba(37,99,235,.15)",
                             name="80% range"))
    fig.add_trace(go.Scatter(x=f["Date"], y=f["Forecast"], name=f"Forecast · {fc['model']}", line=dict(color=COLORS["primary"], width=2.4)))
    fig.update_layout(title=f"{NAMES[sym]}: 30-trading-day forecast", yaxis_title="Price ($)")
    show(fig, 430)
    st.caption("The shaded range is where the price would land 8 times out of 10 if it keeps moving as much as it has over the last 60 days. "
               "It widens with time because uncertainty compounds.")
    bt = T["backtest"]
    st.subheader("Buy/sell signal, tested on the unseen period")
    c = st.columns(4)
    c[0].metric("Signal strategy return", f"{bt['strategy_return_pct']:+.1f}%")
    c[1].metric("Buy and hold return", f"{bt['buy_and_hold_return_pct']:+.1f}%")
    c[2].metric("Days invested", f"{bt['days_in_market_pct']:.0f}%")
    c[3].metric("Trades", bt["trades"])
    beat = bt["strategy_return_pct"] > bt["buy_and_hold_return_pct"]
    card("Result", COLORS["green"] if beat else COLORS["amber"],
         "The signal beat buy-and-hold on this stock" if beat else "The signal did not beat simply holding the stock",
         "The rule: hold the stock on days the model predicts a rise, otherwise hold cash. Each trade costs 0.05%.",
         "Treat the signal as one input, not as advice. A rule that wins on one stock over one period can easily lose on the next.")
    sig = p.tail(60).copy()
    sig["Predicted move %"] = ((sig[T["best_model"]] / sig["Prev_Close"] - 1) * 100).round(3)
    sig["Actual move %"] = ((sig["Actual"] / sig["Prev_Close"] - 1) * 100).round(3)
    st.dataframe(sig[["Date", "Prev_Close", "Actual", T["best_model"], "Predicted move %", "Actual move %", "Signal"]]
                 .sort_values("Date", ascending=False), width="stretch", hide_index=True, height=300)

# ----------------------------------------------------------------------------- model performance
with tabs[4]:
    S = pd.DataFrame(M["summary"]).T
    st.subheader("Average over all ten stocks (test period)")
    c = st.columns(4)
    best_overall = S["rmse_vs_naive"].drop(NAIVE).idxmin()
    c[0].metric("Best model overall", best_overall)
    c[1].metric("Its RMSE vs naive", f"{S.loc[best_overall, 'rmse_vs_naive']:.4f}")
    c[2].metric("Its mean R²", f"{S.loc[best_overall, 'mean_R2']:.4f}")
    c[3].metric("Its direction accuracy", f"{S.loc[best_overall, 'mean_direction_acc_pct']:.1f}%")
    a, b = st.columns(2)
    with a:
        d = S.reset_index(names="Model")
        fig = px.bar(d, x="Model", y="rmse_vs_naive", title="Error relative to the naive baseline (below 1.0 is better)",
                     color=d["rmse_vs_naive"] < 1, color_discrete_map={True: COLORS["green"], False: COLORS["red"]}, text_auto=".4f")
        fig.add_hline(y=1, line_dash="dot")
        fig.update_yaxes(range=[d["rmse_vs_naive"].min() - 0.02, d["rmse_vs_naive"].max() + 0.02])
        show(fig, 380, showlegend=False)
    with b:
        grid = pd.DataFrame({s: {m: v["RMSE"] / M["tickers"][s]["models"][NAIVE]["RMSE"] for m, v in M["tickers"][s]["models"].items() if m != NAIVE}
                             for s in NAMES}).T.round(3)
        show(px.imshow(grid, text_auto=True, color_continuous_scale="RdYlGn_r", zmin=0.97, zmax=1.03, aspect="auto",
                       title="RMSE vs naive, per stock and model"), 380, coloraxis_showscale=False)
    st.dataframe(S.rename(columns={"rmse_vs_naive": "RMSE vs naive", "mean_R2": "Mean R²", "mean_MAPE_pct": "Mean MAPE %",
                                   "mean_direction_acc_pct": "Direction accuracy %", "stocks_beating_naive": "Stocks beating naive (of 10)"}),
                 width="stretch")
    card("How to read this", COLORS["primary"], "R² near 1 does not mean the models can predict the market",
         "Tomorrow's price is almost always close to today's, so even the naive guess scores R² above 0.99. "
         "The honest test is whether a model's error is lower than the naive guess, and whether it calls the direction right more than half the time.",
         "Use these models for short-term expectation ranges and risk, not as a guarantee of direction.")
    a, b = st.columns(2)
    with a:
        hst = pd.DataFrame(M["lstm"]["history"])
        fig = px.line(hst, x="epoch", y=["train_loss", "val_loss"], title="LSTM training curve", labels={"value": "Loss", "variable": ""})
        show(fig, 320)
    with b:
        dv = M["device"]
        st.markdown(f"""
**Training setup**
- Split by time: train {M['split']['train']}, validate {M['split']['validation']}, test {M['split']['test']}
- {len(M['features'])} features: returns, moving-average gaps, MACD, RSI, Bollinger, volatility, volume, market context
- LSTM: {M['lstm']['architecture']}, {M['lstm']['sequence_length']}-day windows, trained on **{dv['lstm']}** in {dv['lstm_train_seconds']} s{' with mixed precision' if dv['mixed_precision'] else ''}
- XGBoost trained on **{dv['xgboost']}**
- Scalers fitted on training rows only; best model per stock chosen on validation, never on test
""")

# ----------------------------------------------------------------------------- live
with tabs[5]:
    st.subheader("Real-time prediction")
    st.write("Fetch the latest prices from Yahoo Finance and run the saved models on them.")
    if st.button(f"Get latest {sym} prices and predict", type="primary"):
        try:
            import predict as P
            with st.spinner("Contacting Yahoo Finance..."):
                prices, market = P.fetch_live(sym)
                res = P.next_day(sym, prices, market)
            st.success(f"Latest close {res['as_of']:%d %b %Y}: **${res['last_close']:,.2f}**")
            live = pd.DataFrame(res["models"]).T.rename(columns={"return_pct": "Expected move %", "price": "Predicted next close"})
            live["Signal"] = np.where(live["Expected move %"] > 0, "BUY", "SELL")
            c = st.columns(len(live))
            for box, (name, row) in zip(c, live.iterrows()):
                box.metric(name, f"${row['Predicted next close']:,.2f}", f"{row['Expected move %']:+.2f}%")
            st.dataframe(live.round(3), width="stretch")
            st.caption(f"Consensus: {int((live['Signal'] == 'BUY').sum())} of {len(live)} models expect a rise.")
        except Exception as exc:
            st.error(f"Live data is unavailable right now ({str(exc)[:120]}). The other tabs use the saved dataset and still work.")

# ----------------------------------------------------------------------------- bonus tabs
import stock_extras

with tabs[6]:
    stock_extras.news_tab(sym, NAMES, fc["signal"])
with tabs[7]:
    stock_extras.portfolio_tab(NAMES, history)
