"""Use the trained models on fresh prices (the dashboard's live prediction).

PyTorch is optional here: without it the LSTM is skipped and the four other models still work.
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from features import FEATURES, MARKET, add_features, load_prices

ART = Path(__file__).resolve().parent / "artifacts"
CLASSIC = ["Linear Regression", "Decision Tree", "Random Forest", "XGBoost"]


def _lstm():
    try:
        import torch
        import torch.nn as nn
    except Exception:
        return None
    ck = torch.load(ART / "models" / "lstm.pt", map_location="cpu", weights_only=False)

    class PriceLSTM(nn.Module):
        def __init__(self, n_feat, n_sym, hidden):
            super().__init__()
            self.emb = nn.Embedding(n_sym, 4)
            self.lstm = nn.LSTM(n_feat + 4, hidden, num_layers=2, batch_first=True, dropout=0.2)
            self.head = nn.Sequential(nn.Linear(hidden, 32), nn.ReLU(), nn.Dropout(0.2), nn.Linear(32, 1))

        def forward(self, x, sym):
            e = self.emb(sym).unsqueeze(1).expand(-1, x.size(1), -1)
            out, _ = self.lstm(torch.cat([x, e], dim=2))
            return self.head(out[:, -1]).squeeze(-1)

    net = PriceLSTM(len(ck["features"]), len(ck["symbols"]), ck["hidden"])
    net.load_state_dict(ck["state"])
    net.eval()
    return torch, net, ck


def fetch_live(symbol: str, period: str = "2y") -> pd.DataFrame:
    """Latest adjusted daily prices from Yahoo Finance."""
    import yfinance as yf

    def one(sym):
        d = yf.download(sym, period=period, auto_adjust=True, progress=False, threads=False)
        if d is None or d.empty:
            raise RuntimeError(f"Yahoo Finance returned no data for {sym}")
        if isinstance(d.columns, pd.MultiIndex):
            d.columns = d.columns.get_level_values(0)
        d = d.reset_index()
        d = d.rename(columns={d.columns[0]: "Date"})
        return d[["Date", "Open", "High", "Low", "Close", "Volume"]].dropna()

    prices = one(symbol)
    try:
        market = one("^GSPC")
    except Exception:                                    # fall back to the saved index history
        market, _ = load_prices(MARKET)
    return prices, market


def next_day(symbol: str, prices: pd.DataFrame | None = None, market: pd.DataFrame | None = None) -> dict:
    """Predicted next close from every available model, given prices up to the last row."""
    if prices is None:
        prices, _ = load_prices(symbol)
    if market is None:
        market, _ = load_prices(MARKET)
    feats = add_features(prices, market).dropna(subset=FEATURES)
    bundle = joblib.load(ART / "models" / f"{symbol}.joblib")
    X = bundle["scaler"].transform(feats[FEATURES].tail(60)).astype(np.float32)
    last = float(feats["Close"].iloc[-1])
    out = {"as_of": feats["Date"].iloc[-1], "last_close": last, "models": {}}
    for name in CLASSIC:
        r = float(bundle[name].predict(X[-1:])[0])
        out["models"][name] = {"return_pct": r * 100, "price": last * float(np.exp(r))}
    lstm = _lstm()
    if lstm and len(X) == 60:
        torch, net, ck = lstm
        with torch.no_grad():
            r = float(net(torch.tensor(X[None]), torch.tensor([ck["symbols"].index(symbol)]))[0]) / 100
        out["models"]["LSTM"] = {"return_pct": r * 100, "price": last * float(np.exp(r))}
    return out
