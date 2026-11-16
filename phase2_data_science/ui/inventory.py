"""Task 3 - Smart Inventory Management System."""
from __future__ import annotations

import io

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from common import COLORS, ROOT, card, load_csv, load_json, money, need, page, show

ART = ROOT / "task3_inventory" / "artifacts"
page("Smart Inventory Management System",
     "Task 3 · Real stock, sales and supplier records of Montgomery County Alcohol Beverage Services (Maryland, USA)")
if not need(ART / "metrics.json", "task3_inventory/pipeline.py"):
    st.stop()

M = load_json(str(ART / "metrics.json"))
K = M["kpi"]
P = load_csv(str(ART / "products.csv"), dtype={"code": str}, parse_dates=["stock_entry_date", "last_sale_month"])
SUP = load_csv(str(ART / "supplier_performance.csv"))
CAT = load_csv(str(ART / "category_performance.csv"))
TREND = load_csv(str(ART / "monthly_trend.csv"), parse_dates=["month"])
HIST = load_csv(str(ART / "demand_history.csv.gz"), dtype={"code": str}).set_index("code")
STATUS_COLOR = {"Healthy": COLORS["green"], "Low Stock": COLORS["amber"], "Critical": "#ea580c", "Out of Stock": COLORS["red"],
                "Overstock": COLORS["violet"], "Inactive": "#94a3b8"}
ORDER = list(STATUS_COLOR)

with st.sidebar:
    st.subheader("Scope")
    depts = sorted(P["department"].unique())
    dept = st.multiselect("Department", depts, default=depts)
    show_inactive = st.checkbox("Include inactive products", False, help="Listed products with no stock and no sales this year")
    st.caption(f"Sales history: {M['history_months'][0]} → {M['history_months'][-1]} ({len(M['history_months'])} months on record)\n\n"
               f"Forecast month: **{M['forecast_month']}**")

D = P[P["department"].isin(dept)]
if not show_inactive:
    D = D[D["active"]]

# ----------------------------------------------------------------------------- alert banner
n_out, n_crit, n_low = [(D["stock_status"] == s).sum() for s in ("Out of Stock", "Critical", "Low Stock")]
if n_out + n_crit:
    st.error(f"**{n_out:,} products are out of stock and {n_crit:,} are at a critical level** (under two weeks of demand). "
             f"Sales at risk next month: {money(D['lost_sales_risk'].sum())}. See the Alerts tab.")
elif n_low:
    st.warning(f"{n_low:,} products are running low.")
else:
    st.success("No stock shortages in the current selection.")

c = st.columns(6)
c[0].metric("Total products", f"{len(D):,}")
c[1].metric("Categories", D["category"].nunique())
c[2].metric("Available stock", f"{D['stock_units'].sum() / 1e6:.2f}M", "units", delta_color="off")
c[3].metric("Low stock", f"{n_low + n_crit:,}", f"{n_crit:,} critical", delta_color="inverse")
c[4].metric("Out of stock", f"{n_out:,}")
c[5].metric("Inventory value", money(D["stock_value_retail"].sum()), f"cost {money(D['stock_value_cost'].sum())}", delta_color="off",
            help="Retail value of stock on hand. Cost is estimated: the source does not publish purchase prices.")

tabs = st.tabs(["Dashboard", "Alerts", "Demand forecast", "Insights", "Suppliers", "Products", "Report", "Labels (QR / barcode)", "Channels"])

# ----------------------------------------------------------------------------- dashboard
with tabs[0]:
    a, b = st.columns([2, 3])
    with a:
        vc = D["stock_status"].value_counts().reindex(ORDER).dropna().reset_index()
        vc.columns = ["Status", "Products"]
        fig = px.pie(vc, names="Status", values="Products", hole=0.58, color="Status", color_discrete_map=STATUS_COLOR,
                     title="Products by stock status")
        show(fig, 380)
    with b:
        g = D.groupby("category").agg(value=("stock_value_retail", "sum"), units=("stock_units", "sum")).nlargest(15, "value").reset_index()
        fig = px.bar(g, x="value", y="category", orientation="h", title="Inventory value by category (top 15)",
                     labels={"value": "Retail value ($)", "category": ""}, text_auto=".3s")
        fig.update_yaxes(autorange="reversed")
        show(fig, 380)
    a, b = st.columns([3, 2])
    with a:
        t = TREND[TREND["department"].isin(dept)].copy()
        t["Month"] = t["month"].dt.strftime("%Y-%m")
        fig = px.bar(t, x="Month", y="cases_sold", color="department", title="Monthly sales trend (cases sold from stores)",
                     labels={"cases_sold": "Cases", "Month": ""})
        fig.update_xaxes(type="category")
        show(fig, 380)
        st.caption("Only months published by the source are shown; it has gaps between 2018 and 2026.")
    with b:
        g = D.groupby("department").agg(units=("stock_units", "sum"), value=("stock_value_retail", "sum"), products=("code", "count")).reset_index()
        fig = px.treemap(D[D["stock_units"] > 0], path=["department", "category"], values="stock_value_retail",
                         title="Where the inventory value sits")
        show(fig, 380)
    a, b = st.columns(2)
    with a:
        top = D.nlargest(15, "revenue_recent")
        fig = px.bar(top, x="revenue_recent", y="name", orientation="h", color="stock_status", color_discrete_map=STATUS_COLOR,
                     title="Top-selling products (revenue, last 4 months on record)", labels={"revenue_recent": "Revenue ($)", "name": ""})
        fig.update_yaxes(autorange="reversed", tickfont=dict(size=10))
        show(fig, 470)
    with b:
        top = D.nlargest(15, "stock_units")
        fig = px.bar(top, x="stock_units", y="name", orientation="h", color="stock_status", color_discrete_map=STATUS_COLOR,
                     title="Product-wise stock: largest holdings", labels={"stock_units": "Units in stock", "name": ""})
        fig.update_yaxes(autorange="reversed", tickfont=dict(size=10))
        show(fig, 470)

# ----------------------------------------------------------------------------- alerts
with tabs[1]:
    al = D[D["stock_status"].isin(["Out of Stock", "Critical", "Low Stock"])].copy()
    c = st.columns(4)
    c[0].metric("Out of stock", f"{n_out:,}")
    c[1].metric("Critical (< 2 weeks)", f"{n_crit:,}")
    c[2].metric("Low (< 1 month)", f"{n_low:,}")
    c[3].metric("Cost to restock all", money(al["reorder_cost"].sum()), f"{int(al['reorder_cases'].sum()):,} cases", delta_color="off")
    level = st.multiselect("Alert level", ["Out of Stock", "Critical", "Low Stock"], default=["Out of Stock", "Critical"])
    av = al[al["stock_status"].isin(level)].sort_values(["lost_sales_risk", "forecast_units"], ascending=False)
    for _, r in av.head(5).iterrows():
        colr = STATUS_COLOR[r["stock_status"]]
        cover = "no stock" if r["stock_units"] == 0 else f"{r['months_cover'] * 30:.0f} days of cover"
        card(r["stock_status"], colr, f"{r['name']}  ·  {r['supplier']}",
             f"{int(r['stock_units']):,} units in stock ({cover}) against a forecast of {int(r['forecast_units']):,} units next month.",
             f"Order {int(r['reorder_cases']):,} cases (about {money(r['reorder_cost'])}).")
    st.dataframe(av[["code", "name", "category", "supplier", "stock_status", "stock_units", "forecast_units", "months_cover",
                     "reorder_cases", "reorder_cost", "lost_sales_risk"]]
                 .rename(columns={"forecast_units": "forecast (units/month)", "months_cover": "months of cover", "lost_sales_risk": "sales at risk ($)"}),
                 width="stretch", hide_index=True, height=380,
                 column_config={"reorder_cost": st.column_config.NumberColumn(format="$%.0f"), "sales at risk ($)": st.column_config.NumberColumn(format="$%.0f")})
    st.download_button("Download alert list (CSV)", av.to_csv(index=False).encode(), "stock_alerts.csv", "text/csv")

# ----------------------------------------------------------------------------- forecast
with tabs[2]:
    F = M["forecast"]
    ev = pd.DataFrame(F["models"]).T
    c = st.columns(4)
    c[0].metric("Best model", F["best_model"])
    c[1].metric("Forecast error (WAPE)", f"{ev.loc[F['best_model'], 'WAPE_pct']:.1f}%",
                f"{ev.loc[F['best_model'], 'WAPE_pct'] - ev.loc['Naive (last month)', 'WAPE_pct']:+.1f} pts vs naive", delta_color="inverse")
    c[2].metric("Forecast demand next month", f"{D['forecast_cases'].sum():,.0f} cases")
    c[3].metric("Tested on", " & ".join(F["test_months"]))
    a, b = st.columns(2)
    with a:
        d = ev.reset_index(names="Model")
        fig = px.bar(d, x="Model", y="WAPE_pct", title="Forecast error by model (lower is better)", text_auto=".1f",
                     color=d["Model"] == F["best_model"], color_discrete_map={True: COLORS["green"], False: "#94a3b8"},
                     labels={"WAPE_pct": "Weighted error (%)"})
        show(fig, 360, showlegend=False)
    with b:
        fi = pd.Series(F["feature_importance"]).head(10).iloc[::-1]
        show(px.bar(x=fi.values, y=fi.index, orientation="h", title="What the forecast model relies on",
                    labels={"x": "Importance", "y": ""}), 360)
    st.dataframe(ev.rename(columns={"MAE_cases": "MAE (cases)", "RMSE_cases": "RMSE (cases)", "WAPE_pct": "WAPE %", "Bias_pct": "Bias %"}), width="stretch")
    st.caption(f"Trained on {F['train_rows']:,} product-months, tested on {F['test_rows']:,}. XGBoost ran on {F['xgboost_device']}. "
               "WAPE = total absolute error ÷ total demand. Bias shows whether a model over- or under-forecasts in total.")
    st.subheader("Product forecast")
    opts = D.sort_values("revenue_recent", ascending=False)
    code = st.selectbox("Product", opts["code"], format_func=lambda k: f"{opts.set_index('code').loc[k, 'name']}  ({k})")
    r = D.set_index("code").loc[code]
    h = HIST.loc[code] * r["units_per_case"]
    fig = go.Figure(go.Bar(x=list(h.index), y=h.values, name="Units sold", marker_color=COLORS["cyan"]))
    fig.add_trace(go.Bar(x=[M["forecast_month"] + " (forecast)"], y=[r["forecast_units"]], name="Forecast", marker_color=COLORS["amber"]))
    fig.add_hline(y=r["stock_units"], line_dash="dot", line_color=COLORS["red"], annotation_text=f"stock on hand: {int(r['stock_units']):,}")
    fig.update_xaxes(type="category")
    fig.update_layout(title=f"{r['name']}: monthly demand and next-month forecast", yaxis_title="Units")
    show(fig, 400)
    c = st.columns(5)
    c[0].metric("Stock on hand", f"{int(r['stock_units']):,}")
    c[1].metric("Forecast next month", f"{int(r['forecast_units']):,} units")
    c[2].metric("Months of cover", "∞" if pd.isna(r["months_cover"]) else f"{r['months_cover']:.1f}")
    c[3].metric("Status", r["stock_status"])
    c[4].metric("Suggested order", f"{int(r['reorder_cases']):,} cases")

# ----------------------------------------------------------------------------- insights
with tabs[3]:
    act = D[D["active"]]
    a, b, c3 = st.columns(3)
    with a:
        abc = act.groupby("abc_class").agg(products=("code", "count"), revenue=("revenue_recent", "sum")).reset_index()
        abc["revenue share"] = abc["revenue"] / abc["revenue"].sum()
        fig = px.bar(abc, x="abc_class", y="products", text="products", title="ABC analysis: products per class",
                     color="abc_class", color_discrete_map={"A": COLORS["green"], "B": COLORS["amber"], "C": "#94a3b8"})
        show(fig, 320, showlegend=False)
    with b:
        fig = px.pie(abc, names="abc_class", values="revenue", hole=0.55, title="Revenue share per class", color="abc_class",
                     color_discrete_map={"A": COLORS["green"], "B": COLORS["amber"], "C": "#94a3b8"})
        show(fig, 320)
    with c3:
        mv = act["movement"].value_counts().reset_index()
        mv.columns = ["Movement", "Products"]
        show(px.bar(mv, x="Movement", y="Products", text="Products", title="Fast and slow movers"), 320)
    na = int((act["abc_class"] == "A").sum())
    over = act[act["stock_status"] == "Overstock"]
    a_out = act[(act["abc_class"] == "A") & act["stock_status"].isin(["Out of Stock", "Critical"])]
    slow_val = act.loc[act["movement"].isin(["Slow-moving", "No recent sales"]), "stock_value_retail"].sum()
    l, r2 = st.columns(2)
    with l:
        card("Concentration", COLORS["primary"], f"{na:,} class-A products ({na / max(len(act), 1):.0%} of the range) bring in 80% of revenue",
             "These are the products that must never run out.", "Count class-A stock weekly and give it priority in every order.")
        card("Risk", COLORS["red"], f"{len(a_out):,} class-A products are out of stock or critical",
             f"Next-month sales at risk on these alone: {money(a_out['lost_sales_risk'].sum() + (a_out['forecast_units'] * a_out['price']).where(a_out['stock_status'] == 'Critical', 0).sum())}.",
             "Expedite these orders first; the list is in the Alerts tab sorted by sales at risk.")
    with r2:
        card("Cash tied up", COLORS["violet"], f"{len(over):,} overstocked products hold {money(over['stock_value_retail'].sum())} of stock",
             f"Each has more than {M['assumptions']['overstock_cover_months']:.0f} months of demand on the shelf, or stock with no sales this year.",
             "Pause purchasing on these, and promote or mark down the largest ones.")
        card("Slow movers", COLORS["amber"], f"{money(slow_val)} of stock sits in slow-moving or non-selling products",
             "Slow movers are the bottom quarter of products by recent sales.", "Review for delisting; move the shelf space to fast movers.")
    a, b = st.columns(2)
    with a:
        st.subheader("Fast-moving products")
        st.dataframe(act[act["movement"] == "Fast-moving"].nlargest(50, "avg_cases_3m")[["name", "category", "avg_cases_3m", "stock_units", "months_cover", "stock_status"]],
                     width="stretch", hide_index=True, height=300)
    with b:
        st.subheader("Overstocked products")
        st.dataframe(over.nlargest(50, "stock_value_retail")[["name", "category", "stock_units", "forecast_units", "months_cover", "stock_value_retail"]],
                     width="stretch", hide_index=True, height=300)
    st.subheader("Category performance")
    cp = CAT[CAT["department"].isin(dept)].copy()
    cp["stock turn (months of revenue held)"] = (cp["stock_value"] / (cp["revenue_recent"] / 4).replace(0, np.nan)).round(1)
    fig = px.scatter(cp, x="revenue_recent", y="stock_value", size="products", color="department", hover_name="category", log_x=True, log_y=True,
                     title="Categories: revenue vs stock held (above the line = heavy stock for its sales)",
                     labels={"revenue_recent": "Revenue, last 4 months ($)", "stock_value": "Stock value ($)"})
    show(fig, 420)
    st.dataframe(cp, width="stretch", hide_index=True, height=300)

# ----------------------------------------------------------------------------- suppliers
with tabs[4]:
    s = SUP.copy()
    c = st.columns(4)
    c[0].metric("Suppliers", f"{len(s):,}")
    c[1].metric("Top supplier", s.iloc[0]["supplier"][:26], f"{s.iloc[0]['revenue_recent'] / s['revenue_recent'].sum():.0%} of revenue", delta_color="off")
    c[2].metric("Top 10 share of revenue", f"{s.head(10)['revenue_recent'].sum() / s['revenue_recent'].sum():.0%}")
    c[3].metric("Suppliers with a stock-out", f"{(s['out_of_stock'] > 0).sum():,}")
    a, b = st.columns(2)
    with a:
        fig = px.bar(s.head(15), x="revenue_recent", y="supplier", orientation="h", title="Top suppliers by revenue",
                     labels={"revenue_recent": "Revenue, last 4 months ($)", "supplier": ""}, text_auto=".3s")
        fig.update_yaxes(autorange="reversed", tickfont=dict(size=10))
        show(fig, 460)
    with b:
        big = s[s["products"] >= 10].nlargest(15, "stockout_rate_pct")
        fig = px.bar(big, x="stockout_rate_pct", y="supplier", orientation="h", title="Highest stock-out rate (suppliers with 10+ products)",
                     labels={"stockout_rate_pct": "Products out of stock (%)", "supplier": ""}, color_discrete_sequence=[COLORS["red"]])
        fig.update_yaxes(autorange="reversed", tickfont=dict(size=10))
        show(fig, 460)
    q = st.text_input("Search supplier")
    sv = s[s["supplier"].str.contains(q, case=False, na=False)] if q else s
    st.dataframe(sv, width="stretch", hide_index=True, height=340)
    st.caption("delivered_vs_sold_pct compares cases transferred into stores with cases sold over the last 4 months: "
               "well under 100% means supply is not keeping up with sales.")

# ----------------------------------------------------------------------------- products
with tabs[5]:
    a, b, c3, d4 = st.columns([3, 2, 2, 2])
    q = a.text_input("Search product name or code")
    fc = b.multiselect("Category", sorted(D["category"].unique()))
    fs = c3.multiselect("Stock status", ORDER)
    sort = d4.selectbox("Sort by", ["stock_value_retail", "stock_units", "forecast_units", "revenue_recent", "price", "months_cover", "name"])
    V = D
    if q:
        V = V[V["name"].str.contains(q, case=False, na=False, regex=False) | V["code"].str.contains(q, regex=False)]
    if fc:
        V = V[V["category"].isin(fc)]
    if fs:
        V = V[V["stock_status"].isin(fs)]
    V = V.sort_values(sort, ascending=(sort == "name") or st.toggle("Ascending", False))
    st.caption(f"{len(V):,} products")
    cols = ["code", "name", "category", "supplier", "stock_units", "stock_status", "purchase_price", "price", "sale_price",
            "stock_entry_date", "forecast_units", "months_cover", "stock_value_retail", "abc_class", "movement"]
    st.dataframe(V[cols], width="stretch", hide_index=True, height=480,
                 column_config={"price": st.column_config.NumberColumn("selling price", format="$%.2f"),
                                "purchase_price": st.column_config.NumberColumn("purchase price (est.)", format="$%.2f"),
                                "sale_price": st.column_config.NumberColumn("promo price", format="$%.2f"),
                                "stock_entry_date": st.column_config.DateColumn("last stock entry", format="MMM YYYY"),
                                "stock_value_retail": st.column_config.NumberColumn("stock value", format="$%.0f")})
    buf = io.BytesIO()
    V[cols].to_excel(buf, index=False)
    x, y = st.columns(2)
    x.download_button("Export to Excel", buf.getvalue(), "inventory.xlsx")
    y.download_button("Export to CSV", V[cols].to_csv(index=False).encode(), "inventory.csv", "text/csv")

# ----------------------------------------------------------------------------- report
with tabs[6]:
    al = D[D["stock_status"].isin(["Out of Stock", "Critical", "Low Stock"])]
    over = D[D["stock_status"] == "Overstock"]
    F = M["forecast"]
    top_cat = D.groupby("category")["revenue_recent"].sum().nlargest(5)
    lines = [
        "# Inventory report", f"_Scope: {', '.join(dept)} · forecast month {M['forecast_month']}_", "",
        "## Inventory summary",
        f"- Products: **{len(D):,}** in **{D['category'].nunique()}** categories from **{D['supplier'].nunique()}** suppliers",
        f"- Stock on hand: **{int(D['stock_units'].sum()):,} units**, worth **{money(D['stock_value_retail'].sum())}** at retail ({money(D['stock_value_cost'].sum())} at estimated cost)",
        f"- Status: " + ", ".join(f"{k} {int(v):,}" for k, v in D['stock_status'].value_counts().reindex(ORDER).dropna().items()), "",
        "## Product analysis",
        f"- Class A products ({int((D['abc_class'] == 'A').sum()):,}) generate 80% of revenue",
        f"- Fast-moving: {int((D['movement'] == 'Fast-moving').sum()):,} · Slow-moving: {int((D['movement'] == 'Slow-moving').sum()):,}",
        "- Top categories by revenue: " + ", ".join(f"{k} ({money(v)})" for k, v in top_cat.items()), "",
        "## Forecast results",
        f"- Model: **{F['best_model']}**, weighted error {F['models'][F['best_model']]['WAPE_pct']}% on {' and '.join(F['test_months'])} "
        f"(naive: {F['models']['Naive (last month)']['WAPE_pct']}%)",
        f"- Forecast demand next month: **{D['forecast_cases'].sum():,.0f} cases**", "",
        "## Stock alerts",
        f"- Out of stock: **{n_out:,}** · Critical: **{n_crit:,}** · Low: **{n_low:,}**",
        f"- Sales at risk next month from stock-outs: **{money(D['lost_sales_risk'].sum())}**", "",
        "| Product | Supplier | Status | Stock | Forecast | Order (cases) |", "|---|---|---|---|---|---|",
        *[f"| {r['name'][:48]} | {r['supplier'][:28]} | {r['stock_status']} | {int(r['stock_units']):,} | {int(r['forecast_units']):,} | {int(r['reorder_cases']):,} |"
          for _, r in al.sort_values('lost_sales_risk', ascending=False).head(15).iterrows()], "",
        "## Business recommendations",
        f"1. Place restock orders for {int((al['reorder_cases'] > 0).sum()):,} products: {int(al['reorder_cases'].sum()):,} cases, about {money(al['reorder_cost'].sum())}.",
        "2. Expedite class-A products that are out of stock or critical first.",
        f"3. Stop buying {len(over):,} overstocked products holding {money(over['stock_value_retail'].sum())}; promote the largest.",
        "4. Review suppliers with the highest stock-out rate (Suppliers tab).", "",
        f"_Purchase cost is estimated as selling price less a {M['assumptions']['gross_margin_for_cost_estimate']:.0%} margin; the source does not publish cost._",
    ]
    report = "\n".join(lines)
    from pdf import Report
    pdf = Report("Inventory report", f"{', '.join(dept)} | forecast month {M['forecast_month']}")
    pdf.heading("Inventory summary")
    pdf.facts({"Products": f"{len(D):,} in {D['category'].nunique()} categories, {D['supplier'].nunique()} suppliers",
               "Stock on hand": f"{int(D['stock_units'].sum()):,} units",
               "Inventory value": f"{money(D['stock_value_retail'].sum())} retail, {money(D['stock_value_cost'].sum())} est. cost",
               "Out of stock / critical / low": f"{n_out:,} / {n_crit:,} / {n_low:,}",
               "Overstocked": f"{len(over):,} products, {money(over['stock_value_retail'].sum())}",
               "Sales at risk next month": money(D['lost_sales_risk'].sum())})
    pdf.heading("Forecast results")
    pdf.text(f"Model: {F['best_model']}. Weighted error {F['models'][F['best_model']]['WAPE_pct']}% on {' and '.join(F['test_months'])} "
             f"(naive forecast: {F['models']['Naive (last month)']['WAPE_pct']}%). Forecast demand next month: {D['forecast_cases'].sum():,.0f} cases.")
    pdf.heading("Stock alerts: most urgent")
    pdf.table(al.sort_values("lost_sales_risk", ascending=False).head(22)[["name", "supplier", "stock_status", "stock_units", "forecast_units", "reorder_cases"]]
              .rename(columns={"name": "Product", "supplier": "Supplier", "stock_status": "Status", "stock_units": "Stock",
                               "forecast_units": "Forecast", "reorder_cases": "Order (cases)"}), widths=[62, 44, 22, 16, 18, 18])
    pdf.heading("Top categories by revenue")
    pdf.table(top_cat.reset_index().rename(columns={"category": "Category", "revenue_recent": "Revenue, last 4 months ($)"}), widths=[110, 70])
    pdf.heading("Business recommendations")
    pdf.bullets([f"Place restock orders for {int((al['reorder_cases'] > 0).sum()):,} products: {int(al['reorder_cases'].sum()):,} cases, about {money(al['reorder_cost'].sum())}.",
                 "Expedite class-A products that are out of stock or critical first.",
                 f"Stop buying {len(over):,} overstocked products holding {money(over['stock_value_retail'].sum())}; promote the largest.",
                 "Review suppliers with the highest stock-out rate.",
                 f"Purchase cost is estimated as selling price less a {M['assumptions']['gross_margin_for_cost_estimate']:.0%} margin; the source does not publish cost."])
    a, b = st.columns(2)
    a.download_button("Download report (PDF)", pdf.done(), "inventory_report.pdf", "application/pdf", type="primary")
    b.download_button("Download report (Markdown)", report.encode("utf-8"), "inventory_report.md", "text/markdown")
    st.markdown(report)

# ----------------------------------------------------------------------------- bonus tabs
import inventory_extras

with tabs[7]:
    inventory_extras.labels_tab(D, STATUS_COLOR)
with tabs[8]:
    inventory_extras.channels_tab(D, TREND, dept)
