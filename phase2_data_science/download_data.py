"""Download every dataset used in Phase 2. All sources are real and public.

    python download_data.py            # everything
    python download_data.py stocks     # one group: stocks | inventory | loans | samples

Nothing here is generated or simulated. Total size on disk is about 40 MB.
A provenance record (source URL, rows, download time) is written to data/SOURCES.json.
"""
from __future__ import annotations

import io
import json
import sys
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
UA = {"User-Agent": "Mozilla/5.0 (data-science-coursework)"}
SOURCES_FILE = DATA / "SOURCES.json"

# Task 1: ten large US companies across sectors + the S&P 500 index as market benchmark
TICKERS = {
    "AAPL": "Apple", "MSFT": "Microsoft", "GOOGL": "Alphabet", "AMZN": "Amazon",
    "NVDA": "NVIDIA", "TSLA": "Tesla", "META": "Meta Platforms", "JPM": "JPMorgan Chase",
    "JNJ": "Johnson & Johnson", "XOM": "Exxon Mobil", "^GSPC": "S&P 500 Index",
}
STOCK_START = "2015-01-01"

MOCO = "https://data.montgomerycountymd.gov/resource"
HMDA_STATE, HMDA_YEAR = "CO", 2023
HMDA_URL = ("https://ffiec.cfpb.gov/v2/data-browser-api/view/csv"
            f"?states={HMDA_STATE}&years={HMDA_YEAR}&actions_taken=1,2,3")
HMDA_KEEP = [
    "action_taken", "derived_sex", "loan_type", "loan_purpose", "lien_status", "occupancy_type",
    "loan_amount", "loan_term", "property_value", "income", "debt_to_income_ratio",
    "loan_to_value_ratio", "applicant_age", "co-applicant_age",
    "derived_race", "derived_ethnicity", "derived_loan_product_type", "derived_dwelling_category",
    "county_code", "tract_minority_population_percent", "tract_to_msa_income_percentage",
    "ffiec_msa_md_median_family_income", "total_units", "business_or_commercial_purpose",
    "open-end_line_of_credit", "preapproval", "denial_reason-1",
]


def record(key: str, **info) -> None:
    DATA.mkdir(exist_ok=True)
    src = json.loads(SOURCES_FILE.read_text()) if SOURCES_FILE.exists() else {}
    info["downloaded_utc"] = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")
    src[key] = info
    SOURCES_FILE.write_text(json.dumps(src, indent=2))


def stocks() -> None:
    import yfinance as yf

    out = DATA / "stocks"
    out.mkdir(parents=True, exist_ok=True)
    rows = {}
    for sym, name in TICKERS.items():
        df = yf.download(sym, start=STOCK_START, auto_adjust=False, progress=False, threads=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.reset_index()[["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"]]
        fname = sym.replace("^", "") + ".csv"
        df.to_csv(out / fname, index=False, float_format="%.4f")
        rows[sym] = len(df)
        print(f"  {sym:6s} {name:20s} {len(df):5d} days  {df['Date'].min():%Y-%m-%d} -> {df['Date'].max():%Y-%m-%d}")
        time.sleep(0.4)
    (out / "tickers.json").write_text(json.dumps(TICKERS, indent=2))
    record("task1_stocks", source="Yahoo Finance (via the yfinance library)",
           url="https://finance.yahoo.com", what="Daily OHLCV price history",
           tickers=TICKERS, rows=rows, start=STOCK_START)


def _socrata_csv(dataset: str, limit: int = 600_000) -> pd.DataFrame:
    r = requests.get(f"{MOCO}/{dataset}.csv", params={"$limit": limit}, headers=UA, timeout=600)
    r.raise_for_status()
    return pd.read_csv(io.BytesIO(r.content), dtype=str)


def inventory() -> None:
    out = DATA / "inventory"
    out.mkdir(parents=True, exist_ok=True)
    sets = {
        "store_inventory": ("ib5t-5ncy", "ABS Store Inventory and Sale Items (current stock on hand + shelf price, updated daily)"),
        "sales_history": ("v76h-r7br", "Warehouse and Retail Sales (monthly cases sold per item and supplier)"),
        "purchases": ("kpfd-c5d5", "Product Purchases (last month's purchase orders by vendor)"),
    }
    for name, (sid, what) in sets.items():
        df = _socrata_csv(sid)
        path = out / f"{name}.csv.gz" if name == "sales_history" else out / f"{name}.csv"
        df.to_csv(path, index=False)
        print(f"  {name:16s} {len(df):7,d} rows -> {path.name}")
        record(f"task3_{name}", source="Montgomery County (Maryland, USA) Alcohol Beverage Services - open data portal",
               url=f"https://data.montgomerycountymd.gov/d/{sid}", what=what, rows=len(df))


def loans() -> None:
    out = DATA / "loans"
    out.mkdir(parents=True, exist_ok=True)
    print(f"  downloading HMDA {HMDA_YEAR} {HMDA_STATE} (about 130 MB raw, kept as a ~6 MB extract) ...")
    raw = out / "_raw.csv"
    # The API builds the extract on first request and answers 403 until the file exists.
    for attempt in range(1, 13):
        r = requests.get(HMDA_URL, headers=UA, stream=True, timeout=1800)
        if r.status_code == 200:
            with open(raw, "wb") as f:
                for chunk in r.iter_content(1 << 20):
                    f.write(chunk)
            break
        print(f"    extract not ready (HTTP {r.status_code}), retry {attempt}/12 in 20s")
        time.sleep(20)
    else:
        raise RuntimeError("HMDA extract was not produced in time; run the command again")
    df = pd.read_csv(raw, usecols=HMDA_KEEP, dtype=str, low_memory=False)
    path = out / f"hmda_{HMDA_STATE.lower()}_{HMDA_YEAR}.csv.gz"
    df.to_csv(path, index=False)
    raw.unlink()
    print(f"  loans            {len(df):7,d} applications -> {path.name}")
    record("task4_loans", source="US Consumer Financial Protection Bureau - Home Mortgage Disclosure Act (HMDA) public data",
           url="https://ffiec.cfpb.gov/data-browser/", api=HMDA_URL,
           what=f"Every mortgage application decided in {HMDA_STATE} in {HMDA_YEAR} (originated, approved-not-accepted, denied)",
           rows=len(df), columns_kept=HMDA_KEEP)


def samples() -> None:
    """A real dataset to try the Task 2 dashboard with (any CSV/XLSX/JSON also works)."""
    out = DATA / "samples"
    out.mkdir(parents=True, exist_ok=True)
    url = "https://archive.ics.uci.edu/static/public/222/bank+marketing.zip"
    z = zipfile.ZipFile(io.BytesIO(requests.get(url, headers=UA, timeout=300).content))
    inner = zipfile.ZipFile(io.BytesIO(z.read("bank-additional.zip")))
    name = next(n for n in inner.namelist() if n.endswith("bank-additional-full.csv"))
    df = pd.read_csv(io.BytesIO(inner.read(name)), sep=";")
    df.to_csv(out / "bank_marketing.csv", index=False)
    print(f"  bank_marketing   {len(df):7,d} rows")
    record("task2_sample_bank_marketing", source="UCI Machine Learning Repository - Bank Marketing (Moro, Cortez & Rita, 2014)",
           url="https://archive.ics.uci.edu/dataset/222/bank+marketing",
           what="Real phone-campaign records of a Portuguese bank, 2008-2010", rows=len(df))


GROUPS = {"stocks": stocks, "inventory": inventory, "loans": loans, "samples": samples}

if __name__ == "__main__":
    for g in (sys.argv[1:] or list(GROUPS)):
        print(f"[{g}]")
        GROUPS[g]()
    print("done ->", DATA)
