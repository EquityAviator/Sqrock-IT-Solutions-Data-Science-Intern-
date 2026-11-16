"""Data loading, cleaning and feature engineering for the stock prediction task.

Shared by train.py and the dashboard so both always compute features the same way.
Every feature at day t uses only information available at the close of day t.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
STOCK_DIR = ROOT / "data" / "stocks"
MARKET = "GSPC"  # S&P 500 index, used as the market benchmark

FEATURES = [
    # recent returns (momentum / mean reversion)
    "ret_1", "ret_2", "ret_3", "ret_5", "ret_10", "ret_20",
    # distance of price from its moving averages
    "sma5_gap", "sma10_gap", "sma20_gap", "sma50_gap", "sma200_gap", "ema12_gap", "ema26_gap",
    # classic technical indicators
    "macd", "macd_signal", "macd_hist", "rsi_14", "boll_pct_b", "boll_width",
    # risk and activity
    "vol_10", "vol_20", "range_pct", "gap_pct", "volume_ratio", "volume_chg",
    # market context and calendar
    "mkt_ret_1", "mkt_ret_5", "mkt_vol_20", "beta_60", "dow",
]
TARGET = "target_ret"  # next-day log return; price forecast = close * exp(target_ret)


def tickers() -> dict[str, str]:
    names = json.loads((STOCK_DIR / "tickers.json").read_text())
    return {k.replace("^", ""): v for k, v in names.items()}


def load_prices(symbol: str) -> tuple[pd.DataFrame, dict]:
    """Read one ticker's CSV and clean it. Returns (frame, cleaning report)."""
    df = pd.read_csv(STOCK_DIR / f"{symbol}.csv")
    rep = {"rows_raw": len(df)}
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    rep["bad_dates"] = int(df["Date"].isna().sum())
    df = df.dropna(subset=["Date"])
    rep["duplicates"] = int(df.duplicated(subset="Date").sum())
    df = df.drop_duplicates(subset="Date").sort_values("Date").reset_index(drop=True)
    num = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
    df[num] = df[num].apply(pd.to_numeric, errors="coerce")
    rep["missing_values"] = int(df[num].isna().sum().sum())
    df[num] = df[num].ffill()  # a missing quote carries the last traded price forward
    df = df.dropna(subset=num)
    # non-positive prices are data errors
    bad = (df[["Open", "High", "Low", "Close"]] <= 0).any(axis=1)
    rep["invalid_prices"] = int(bad.sum())
    df = df[~bad].reset_index(drop=True)
    rep["rows_clean"] = len(df)
    # Work in split- and dividend-adjusted prices so that a 4:1 split is not a -75% "return".
    k = df["Adj Close"] / df["Close"]
    for c in ("Open", "High", "Low"):
        df[c] = df[c] * k
    df["Close"] = df["Adj Close"]
    return df.drop(columns="Adj Close"), rep


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def add_features(df: pd.DataFrame, market: pd.DataFrame | None = None) -> pd.DataFrame:
    """Add indicator columns (also used for charts) and the model features."""
    d = df.copy()
    c = d["Close"]
    lr = np.log(c).diff()
    for n in (1, 2, 3, 5, 10, 20):
        d[f"ret_{n}"] = np.log(c).diff(n)
    for n in (5, 10, 20, 50, 200):
        d[f"sma_{n}"] = c.rolling(n).mean()
        d[f"sma{n}_gap"] = c / d[f"sma_{n}"] - 1
    for n in (12, 26):
        d[f"ema_{n}"] = c.ewm(span=n, adjust=False).mean()
        d[f"ema{n}_gap"] = c / d[f"ema_{n}"] - 1
    macd = (d["ema_12"] - d["ema_26"]) / c  # scaled by price so it is comparable across stocks
    d["macd"] = macd
    d["macd_signal"] = macd.ewm(span=9, adjust=False).mean()
    d["macd_hist"] = d["macd"] - d["macd_signal"]
    d["rsi_14"] = rsi(c)
    sd20 = c.rolling(20).std()
    d["boll_up"] = d["sma_20"] + 2 * sd20
    d["boll_lo"] = d["sma_20"] - 2 * sd20
    d["boll_pct_b"] = (c - d["boll_lo"]) / (d["boll_up"] - d["boll_lo"])
    d["boll_width"] = (d["boll_up"] - d["boll_lo"]) / d["sma_20"]
    d["vol_10"] = lr.rolling(10).std()
    d["vol_20"] = lr.rolling(20).std()
    d["range_pct"] = (d["High"] - d["Low"]) / c
    d["gap_pct"] = d["Open"] / c.shift(1) - 1
    vol_ma = d["Volume"].rolling(20).mean()
    d["volume_ratio"] = d["Volume"] / vol_ma.replace(0, np.nan)
    d["volume_chg"] = np.log(d["Volume"].replace(0, np.nan)).diff()
    d["dow"] = d["Date"].dt.dayofweek

    if market is not None:
        m = market[["Date", "Close"]].rename(columns={"Close": "mkt_close"})
        d = d.merge(m, on="Date", how="left")
        d["mkt_close"] = d["mkt_close"].ffill()
        mlr = np.log(d["mkt_close"]).diff()
        d["mkt_ret_1"] = mlr
        d["mkt_ret_5"] = np.log(d["mkt_close"]).diff(5)
        d["mkt_vol_20"] = mlr.rolling(20).std()
        d["beta_60"] = lr.rolling(60).cov(mlr) / mlr.rolling(60).var()
    else:
        for col in ("mkt_ret_1", "mkt_ret_5", "mkt_vol_20", "beta_60"):
            d[col] = 0.0

    d[TARGET] = np.log(c).shift(-1) - np.log(c)   # tomorrow's return (unknown on the last row)
    d["next_close"] = c.shift(-1)
    return d


def build(symbol: str) -> tuple[pd.DataFrame, dict]:
    prices, rep = load_prices(symbol)
    market, _ = load_prices(MARKET)
    return add_features(prices, market), rep
