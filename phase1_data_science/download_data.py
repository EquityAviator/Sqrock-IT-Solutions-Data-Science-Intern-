"""
Dataset downloader for the Phase 1 internship projects.
Re-fetches every dataset used, with source provenance, so the whole project is reproducible.

Run:  python download_data.py
"""
from __future__ import annotations

import io
import os
import zipfile

import requests

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
os.makedirs(DATA, exist_ok=True)

# name -> (url, human-readable source, rationale)
SOURCES = {
    "store_sales_train.csv": (
        "https://huggingface.co/datasets/t4tiana/store-sales-time-series-forecasting/resolve/main/train.csv",
        "Hugging Face - Store Sales (Kaggle 'Store Sales - Time Series Forecasting' mirror)",
        "3M real Favorita grocery transactions across 54 stores / 33 product families, 2013-2017."),
    "store_sales_stores.csv": (
        "https://huggingface.co/datasets/t4tiana/store-sales-time-series-forecasting/resolve/main/stores.csv",
        "Hugging Face - Store Sales metadata",
        "Store -> city / state / type / cluster lookup for optional region analysis."),
    "resume_dataset.csv": (
        "https://huggingface.co/datasets/AzharAli05/Resume-Screening-Dataset/resolve/main/dataset.csv",
        "Hugging Face - Resume Screening Dataset (AzharAli05)",
        "10,174 resumes across 45 roles with job descriptions and human select/reject labels."),
    "telco_churn_ibm.csv": (
        "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv",
        "IBM - Telco Customer Churn sample dataset",
        "7,043 real telecom customers, 21 attributes, Yes/No churn label (~26% churn)."),
    "student_uci.zip": (
        "https://archive.ics.uci.edu/static/public/320/student+performance.zip",
        "UCI Machine Learning Repository - Student Performance (Cortez & Silva)",
        "1,044 students (Math + Portuguese) with study time, absences, grades and family factors."),
}


def download(name: str, url: str) -> bool:
    dest = os.path.join(DATA, name)
    if os.path.exists(dest) and os.path.getsize(dest) > 1000:
        print(f"  skip  {name:32s} (already present, {os.path.getsize(dest)/1024:,.0f} KB)")
        return True
    print(f"  fetch {name:32s} <- {url[:70]}...")
    r = requests.get(url, timeout=300, stream=True)
    if r.status_code != 200:
        print(f"  FAIL  {name}: HTTP {r.status_code}")
        return False
    with open(dest, "wb") as f:
        for chunk in r.iter_content(1 << 16):
            f.write(chunk)
    print(f"  ok    {name:32s} ({os.path.getsize(dest)/1024:,.0f} KB)")
    return True


def extract_student() -> None:
    """The UCI archive is a zip inside a zip - unpack both."""
    outer = os.path.join(DATA, "student_uci.zip")
    if not os.path.exists(outer):
        return
    target = os.path.join(DATA, "student_uci")
    os.makedirs(target, exist_ok=True)
    with zipfile.ZipFile(outer) as z:
        z.extractall(target)
    inner = os.path.join(target, "student.zip")
    if os.path.exists(inner):
        with zipfile.ZipFile(inner) as z:
            z.extractall(target)
    print("  ok    student_uci/ extracted (student-mat.csv, student-por.csv)")


def main() -> None:
    print("Downloading Phase 1 datasets\n" + "-" * 62)
    for name, (url, source, why) in SOURCES.items():
        download(name, url)
        print(f"        source : {source}\n        why    : {why}")
    extract_student()
    print("-" * 62 + "\nAll datasets ready in ./data/")


if __name__ == "__main__":
    main()
