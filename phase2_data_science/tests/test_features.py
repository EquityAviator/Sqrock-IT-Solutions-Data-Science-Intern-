"""Functional checks for the features that need a click or an upload.

    python tests/test_features.py

Uses a throwaway users file and random credentials; nothing here touches real accounts.
Network tests (live prices, news) are reported as SKIPPED when Yahoo Finance is unreachable.
"""
from __future__ import annotations

import io
import json
import os
import secrets
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for sub in ("ui", "task1_stock_prediction", "task2_ai_dashboard", "task3_inventory", "task4_loan_approval"):
    sys.path.insert(0, str(ROOT / sub))
os.chdir(ROOT)

import pandas as pd
from streamlit.testing.v1 import AppTest

results: list[tuple[str, str, str]] = []


def check(name: str):
    def deco(fn):
        try:
            note = fn() or ""
            results.append(("PASS", name, str(note)))
        except AssertionError as exc:
            results.append(("FAIL", name, str(exc)))
        except Exception as exc:                       # network or environment problem
            kind = "SKIP" if "network" in name.lower() else "FAIL"
            results.append((kind, name, f"{type(exc).__name__}: {str(exc)[:120]}"))
        return fn
    return deco


# ------------------------------------------------------------------ authentication
import auth

_tmp = Path(tempfile.mkdtemp()) / "users.json"
auth.USERS_FILE = _tmp
USER, PASSWORD = "tester_" + secrets.token_hex(3), secrets.token_urlsafe(9) + "a1"


@check("auth: weak passwords and bad usernames are rejected")
def _():
    assert not auth.register("ab", PASSWORD)[0]
    assert not auth.register(USER, "short1")[0]
    assert not auth.register(USER, "onlyletters")[0]


@check("auth: register, sign in, wrong password, duplicate user")
def _():
    ok, msg = auth.register(USER, PASSWORD, "Test User")
    assert ok, msg
    stored = json.loads(_tmp.read_text())[USER]
    assert PASSWORD not in json.dumps(stored), "password stored in plain text"
    assert stored["role"] == "admin", "first account should be admin"
    assert auth.verify(USER, PASSWORD)["name"] == "Test User"
    assert auth.verify(USER, PASSWORD + "x") is None
    assert auth.verify("nobody", PASSWORD) is None
    assert not auth.register(USER, PASSWORD)[0]
    return "PBKDF2 hash stored, plain password absent"


@check("auth: the app shows the sign-in screen, guest access opens the dashboard")
def _():
    os.environ["PHASE2_AUTH"] = "on"
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=120).run()
    assert not at.exception, at.exception[0].message
    labels = [b.label for b in at.button]
    assert "Continue as guest" in labels, labels
    assert len(at.metric) == 0, "dashboard content visible before signing in"
    next(b for b in at.button if b.label == "Continue as guest").click().run()
    assert not at.exception, at.exception[0].message
    assert len(at.metric) >= 3, "dashboard did not open for the guest"
    return f"gate hides content; guest sees {len(at.metric)} metrics"


@check("auth: PHASE2_AUTH=off skips the sign-in screen")
def _():
    os.environ["PHASE2_AUTH"] = "off"
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=120).run()
    assert not at.exception and len(at.metric) >= 3


# ------------------------------------------------------------------ task 2 uploads
import engine

SAMPLE = pd.read_csv(ROOT / "data" / "samples" / "bank_marketing.csv").head(1500)


@check("task 2: Excel upload is read and cleaned")
def _():
    buf = io.BytesIO()
    SAMPLE.to_excel(buf, index=False)
    df = engine.read_dataset(buf.getvalue(), "bank.xlsx")
    assert df.shape == SAMPLE.shape, df.shape
    res = engine.clean(df)
    assert len(engine.insights(res.df)) > 3
    return f"{df.shape[0]} rows x {df.shape[1]} columns"


@check("task 2: JSON upload is read (list and wrapped forms)")
def _():
    raw = SAMPLE.to_json(orient="records").encode()
    assert engine.read_dataset(raw, "bank.json").shape == SAMPLE.shape
    wrapped = json.dumps({"records": json.loads(raw)}).encode()
    assert engine.read_dataset(wrapped, "bank.json").shape == SAMPLE.shape


@check("task 2: semicolon CSV, and messy data cleaning")
def _():
    df = engine.read_dataset(SAMPLE.to_csv(index=False, sep=";").encode(), "bank.csv")
    assert df.shape == SAMPLE.shape
    messy = pd.DataFrame({" amount ": ["1,200", "$350", "n/a", "90", "90"], "when": ["2024-01-05", "2024-02-11", "2024-03-02", "2024-03-09", "2024-03-09"],
                          "city": ["Lahore", " Karachi", "?", "Lahore", "Lahore"], "empty": [None] * 5})
    res = engine.clean(messy)
    assert pd.api.types.is_numeric_dtype(res.df["amount"]), "currency text not converted"
    assert pd.api.types.is_datetime64_any_dtype(res.df["when"]), "dates not converted"
    assert "empty" not in res.df.columns and len(res.df) == 4, res.df.shape
    return f"{len(res.log)} cleaning operations logged"


@check("task 2: bad files are rejected with a clear message")
def _():
    for raw, name in ((b"", "x.csv"), (b"abc", "x.exe"), (b"only_one_column\n1\n2\n", "x.csv"), (b"not json", "x.json")):
        try:
            engine.read_dataset(raw, name)
        except engine.DatasetError:
            continue
        raise AssertionError(f"{name} with {raw[:12]!r} was accepted")


@check("task 2: HTML and PDF reports are produced")
def _():
    from pdf import Report
    res = engine.clean(SAMPLE)
    found = engine.insights(res.df)
    html = engine.build_report("bank", res.df, res.log, found)
    assert "<html" in html and len(html) > 5000
    rep = Report("Data analysis report", "bank")
    for f in found:
        rep.finding(f["kind"], f["title"], f["detail"], f["action"])
    pdf = rep.done()
    assert pdf[:4] == b"%PDF" and len(pdf) > 1500
    return f"PDF {len(pdf) / 1024:.0f} KB, HTML {len(html) / 1024:.0f} KB"


# ------------------------------------------------------------------ task 1
@check("task 1: portfolio analysis")
def _():
    import extras
    import features as F
    cl = pd.DataFrame({s: F.load_prices(s)[0].set_index("Date")["Close"] for s in ("AAPL", "JPM", "XOM")}).tail(756)
    r = extras.portfolio(cl, {"AAPL": 50, "JPM": 25, "XOM": 25}, F.load_prices(F.MARKET)[0].set_index("Date")["Close"])
    assert abs(r["weights_end"].sum() - 1) < 1e-9 and r["portfolio"]["Max drawdown %"] <= 0
    assert abs(r["contribution"].sum() - r["portfolio"]["Total return %"]) < 0.01, "contributions do not add up to the return"
    return f"3-year return {r['portfolio']['Total return %']:.1f}%"


@check("task 1 (network): live prices and next-day prediction")
def _():
    import predict as P
    prices, market = P.fetch_live("MSFT")
    res = P.next_day("MSFT", prices, market)
    assert len(res["models"]) >= 4 and all(abs(m["return_pct"]) < 10 for m in res["models"].values())
    return f"as of {res['as_of']:%Y-%m-%d}, {len(res['models'])} models"


@check("task 1 (network): news sentiment")
def _():
    import extras
    n = extras.news_sentiment("NVDA")
    assert len(n) > 0 and n["Sentiment"].between(-1, 1).all()
    return f"{len(n)} headlines, mean sentiment {n['Sentiment'].mean():+.2f}"


@check("task 1 (network): news and live-prediction buttons in the page")
def _():
    at = AppTest.from_file(str(ROOT / "ui" / "stock.py"), default_timeout=180).run()
    next(b for b in at.button if "Analyse latest news" in b.label).click().run()
    assert not at.exception, at.exception[0].message
    next(b for b in at.button if "prices and predict" in b.label).click().run()
    assert not at.exception, at.exception[0].message
    assert not at.error, at.error[0].value
    return "both buttons ran without errors"


# ------------------------------------------------------------------ task 3
@check("task 3: QR code, barcode and label sheet")
def _():
    import inventory_extras as X
    P = pd.read_csv(ROOT / "task3_inventory" / "artifacts" / "products.csv", dtype={"code": str})
    r = P.iloc[0]
    qr, bar = X.qr_png(X.label_text(r)), X.barcode_png(r["code"])
    assert qr[:8] == b"\x89PNG\r\n\x1a\n" and bar[:8] == b"\x89PNG\r\n\x1a\n"
    at = AppTest.from_file(str(ROOT / "ui" / "inventory.py"), default_timeout=180).run()
    next(b for b in at.button if "Build label sheet" in b.label).click().run()
    assert not at.exception, at.exception[0].message
    return f"QR {len(qr)} bytes, barcode {len(bar)} bytes, label sheet built"


@check("task 3: product search, status filter and Excel export")
def _():
    P = pd.read_csv(ROOT / "task3_inventory" / "artifacts" / "products.csv", dtype={"code": str})
    hit = P[P["name"].str.contains("tito", case=False)]
    assert len(hit) > 0
    buf = io.BytesIO()
    hit.to_excel(buf, index=False)
    assert len(pd.read_excel(io.BytesIO(buf.getvalue()))) == len(hit)
    at = AppTest.from_file(str(ROOT / "ui" / "inventory.py"), default_timeout=180).run()
    next(t for t in at.text_input if t.label == "Search product name or code").set_value("tito").run()
    assert not at.exception, at.exception[0].message
    assert any(f"{len(hit[hit['active']]):,} products" in c.value for c in at.caption), "filtered count not shown"
    return f"{len(hit)} products match 'tito'"


# ------------------------------------------------------------------ task 4
@check("task 4: a strong application is approved, a weak one rejected")
def _():
    at = AppTest.from_file(str(ROOT / "ui" / "loan.py"), default_timeout=180).run()
    assert not at.exception, at.exception[0].message
    assert any("APPROVED" in s.value for s in at.success), "default application should be approved"
    next(n for n in at.number_input if n.label == "Annual income ($)").set_value(30000)
    next(n for n in at.number_input if n.label == "Other monthly debt payments ($)").set_value(2500)
    next(b for b in at.button if b.label == "Predict approval").click().run()
    assert not at.exception, at.exception[0].message
    assert any("REJECTED" in e.value for e in at.error), "an over-indebted application should be rejected"
    return "default approved; $30K income with $2.5K monthly debts rejected"


# ------------------------------------------------------------------ summary
width = max(len(n) for _, n, _ in results)
for status, name, note in results:
    print(f"{status}  {name.ljust(width)}  {note}")
bad = [r for r in results if r[0] == "FAIL"]
print(f"\n{sum(r[0] == 'PASS' for r in results)} passed, {len(bad)} failed, {sum(r[0] == 'SKIP' for r in results)} skipped")
sys.exit(1 if bad else 0)
