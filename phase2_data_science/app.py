"""Phase 2 dashboard: one Streamlit app, four projects.

    streamlit run app.py
"""
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "ui"))

st.set_page_config(page_title="Data Science Internship · Phase 2", page_icon=":bar_chart:", layout="wide")

import auth

auth.gate()             # sign-in screen (PHASE2_AUTH=off disables it)
auth.sidebar_badge()

pages = [
    st.Page("ui/home.py", title="Overview", icon=":material/home:", default=True),
    st.Page("ui/stock.py", title="Stock Market Prediction", icon=":material/candlestick_chart:", url_path="stocks"),
    st.Page("ui/dashboard.py", title="AI Data Analysis", icon=":material/insights:", url_path="analysis"),
    st.Page("ui/inventory.py", title="Smart Inventory", icon=":material/inventory_2:", url_path="inventory"),
    st.Page("ui/loan.py", title="Loan Approval", icon=":material/account_balance:", url_path="loans"),
]
st.navigation({"Phase 2": pages[:1], "Projects": pages[1:]}).run()
