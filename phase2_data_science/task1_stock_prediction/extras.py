"""Bonus analytics for the stock task: news sentiment and portfolio performance."""
from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def news_sentiment(symbol: str) -> pd.DataFrame:
    """Latest real headlines for a ticker from Yahoo Finance, scored with VADER (-1 .. +1)."""
    import yfinance as yf
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

    sia = SentimentIntensityAnalyzer()
    rows = []
    for item in yf.Ticker(symbol).news or []:
        c = item.get("content") or item
        title = c.get("title") or ""
        if not title:
            continue
        summary = c.get("summary") or c.get("description") or ""
        score = sia.polarity_scores(f"{title}. {summary}")["compound"]
        url = (c.get("canonicalUrl") or c.get("clickThroughUrl") or {}).get("url", "")
        rows.append({"Published": pd.to_datetime(c.get("pubDate"), errors="coerce", utc=True),
                     "Headline": title, "Source": (c.get("provider") or {}).get("displayName", ""),
                     "Sentiment": round(score, 3),
                     "Tone": "Positive" if score >= 0.2 else "Negative" if score <= -0.2 else "Neutral", "Link": url})
    df = pd.DataFrame(rows)
    return df.sort_values("Published", ascending=False).reset_index(drop=True) if len(df) else df


def portfolio(closes: pd.DataFrame, weights: dict[str, float], benchmark: pd.Series, capital: float = 10_000.0) -> dict:
    """Buy-and-hold portfolio (weights fixed at the start) against a benchmark.

    closes: one adjusted-close column per ticker. Returns curves and risk/return statistics.
    """
    w = pd.Series(weights, dtype=float)
    w = w / w.sum()
    px = closes[w.index].dropna()
    bench = benchmark.reindex(px.index).ffill()
    value = (px / px.iloc[0] * w * capital).sum(axis=1)
    bvalue = bench / bench.iloc[0] * capital

    def stats(v: pd.Series) -> dict:
        r = v.pct_change().dropna()
        years = max((v.index[-1] - v.index[0]).days / 365.25, 1e-9)
        cagr = (v.iloc[-1] / v.iloc[0]) ** (1 / years) - 1
        vol = r.std() * np.sqrt(TRADING_DAYS)
        return {"Final value": float(v.iloc[-1]), "Total return %": float((v.iloc[-1] / v.iloc[0] - 1) * 100),
                "Yearly return % (CAGR)": float(cagr * 100), "Volatility % (yearly)": float(vol * 100),
                "Sharpe ratio": float(r.mean() / r.std() * np.sqrt(TRADING_DAYS)) if r.std() > 0 else 0.0,
                "Max drawdown %": float((v / v.cummax() - 1).min() * 100),
                "Best day %": float(r.max() * 100), "Worst day %": float(r.min() * 100)}

    rp, rb = value.pct_change().dropna(), bvalue.pct_change().dropna()
    beta = float(rp.cov(rb) / rb.var()) if rb.var() > 0 else float("nan")
    contrib = (px.iloc[-1] / px.iloc[0] - 1) * w * 100         # each holding's share of the total return
    final_w = px.iloc[-1] / px.iloc[0] * w
    return {"value": value, "benchmark": bvalue, "portfolio": stats(value), "market": stats(bvalue), "beta": beta,
            "contribution": contrib.sort_values(ascending=False), "weights_start": w, "weights_end": final_w / final_w.sum(),
            "drawdown": value / value.cummax() - 1}
