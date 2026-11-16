"""Extra tabs of the inventory page: QR / barcode labels with code lookup, and sales channels."""
from __future__ import annotations

import io

import pandas as pd
import plotly.express as px
import streamlit as st

from common import COLORS, card, money, show


def qr_png(text: str) -> bytes:
    import qrcode
    img = qrcode.make(text, box_size=6, border=2)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def barcode_png(code: str) -> bytes:
    """Code 128 barcode, the standard for warehouse item labels."""
    from barcode import Code128
    from barcode.writer import ImageWriter
    buf = io.BytesIO()
    Code128(str(code), writer=ImageWriter()).write(buf, options={"module_height": 12, "font_size": 8, "text_distance": 4, "quiet_zone": 3})
    return buf.getvalue()


def label_text(r: pd.Series) -> str:
    """What a scanner reads from the QR code."""
    return (f"ITEM:{r['code']}|{r['name'][:60]}|CAT:{r['category']}|SIZE:{r['size']}|PRICE:{r['price']:.2f}|"
            f"SUPPLIER:{r['supplier'][:40]}")


def labels_tab(P: pd.DataFrame, status_color: dict) -> None:
    st.subheader("Barcode and QR code labels")
    st.write("Type or scan an item code to look a product up, and print its shelf label. "
             "A USB barcode scanner types the code and presses Enter, so it works with the box below as it is.")
    a, b = st.columns([2, 3])
    code = a.text_input("Scan or type an item code", placeholder="for example 53929", key="scan_code").strip()
    by_name = b.selectbox("...or pick a product", P.sort_values("revenue_recent", ascending=False)["code"],
                          format_func=lambda k: f"{P.set_index('code').loc[k, 'name']}  ({k})", key="label_pick")
    if code and "ITEM:" in code:                          # a scanned QR label carries the whole text
        code = code.split("ITEM:")[1].split("|")[0]
    code = code or by_name
    hit = P[P["code"] == code]
    if hit.empty:
        st.error(f"No product with item code {code}.")
        return
    r = hit.iloc[0]
    left, mid, right = st.columns([2, 2, 3])
    with left:
        st.image(qr_png(label_text(r)), caption="QR code (item, price, supplier)", width=230)
    with mid:
        st.image(barcode_png(r["code"]), caption="Code 128 barcode (item code)", width=260)
    with right:
        st.markdown(f"#### {r['name']}")
        st.markdown(f"**Item code** {r['code']} · **{r['category']}** · {r['size']}")
        c = st.columns(2)
        c[0].metric("In stock", f"{int(r['stock_units']):,}")
        c[1].metric("Price", f"${r['price']:.2f}")
        st.markdown(f"<span class='pill' style='background:{status_color.get(r['stock_status'], '#64748b')}'>{r['stock_status']}</span>",
                    unsafe_allow_html=True)
        st.caption(f"Supplier: {r['supplier']} · forecast {int(r['forecast_units']):,} units next month · "
                   f"suggested order {int(r['reorder_cases']):,} cases")
    x, y = st.columns(2)
    x.download_button("Download QR label (PNG)", qr_png(label_text(r)), f"qr_{r['code']}.png", "image/png")
    y.download_button("Download barcode (PNG)", barcode_png(r["code"]), f"barcode_{r['code']}.png", "image/png")

    st.subheader("Print a sheet of labels")
    which = st.selectbox("Products", ["Low and critical stock", "Out of stock", "Top 20 sellers"])
    sel = {"Low and critical stock": P[P["stock_status"].isin(["Low Stock", "Critical"])].nlargest(20, "forecast_units"),
           "Out of stock": P[P["stock_status"] == "Out of Stock"].nlargest(20, "lost_sales_risk"),
           "Top 20 sellers": P.nlargest(20, "revenue_recent")}[which]
    if st.button("Build label sheet (PDF)"):
        from pdf import Report
        rep = Report("Shelf labels", which)
        for _, p in sel.iterrows():
            if rep.get_y() > 245:
                rep.add_page()
            y0 = rep.get_y()
            rep.image(io.BytesIO(qr_png(label_text(p))), x=15, y=y0, w=24)
            rep.image(io.BytesIO(barcode_png(p["code"])), x=140, y=y0 + 2, w=52)
            rep.set_xy(43, y0 + 1)
            rep.set_font("Helvetica", "B", 10)
            rep.multi_cell(94, 5, __import__("pdf").clean(p["name"][:70]))
            rep.set_x(43)
            rep.set_font("Helvetica", "", 9)
            rep.multi_cell(94, 5, __import__("pdf").clean(f"Item {p['code']}  |  {p['size']}  |  ${p['price']:.2f}  |  {p['stock_status']}"))
            rep.set_y(y0 + 28)
        st.download_button("Download label sheet", rep.done(), "shelf_labels.pdf", "application/pdf", type="primary")


def channels_tab(P: pd.DataFrame, trend: pd.DataFrame, dept: list[str]) -> None:
    st.subheader("Sales channels: own stores vs wholesale")
    st.write("The business moves stock through two channels: its own retail stores, and wholesale to licensed bars, "
             "restaurants and shops. Stock is published as one total, so this view shows where demand on that stock comes from.")
    retail, whole = P["retail_cases_recent"].sum(), P["wholesale_cases_recent"].sum()
    moved = P["store_transfer_cases_recent"].sum()
    c = st.columns(4)
    c[0].metric("Retail stores", f"{retail:,.0f} cases", f"{retail / max(retail + whole, 1):.0%} of volume", delta_color="off")
    c[1].metric("Wholesale", f"{whole:,.0f} cases", f"{whole / max(retail + whole, 1):.0%} of volume", delta_color="off")
    c[2].metric("Warehouse to stores", f"{moved:,.0f} cases", f"{moved / max(retail, 1):.0%} of store sales", delta_color="off")
    only_w = int(((P["wholesale_cases_recent"] > 0) & (P["retail_cases_recent"] <= 0)).sum())
    c[3].metric("Wholesale-only products", f"{only_w:,}")
    st.caption("Figures cover the last 4 months on record.")
    a, b = st.columns(2)
    with a:
        g = P.groupby("department")[["retail_cases_recent", "wholesale_cases_recent"]].sum().reset_index()
        g = g.melt("department", var_name="Channel", value_name="Cases")
        g["Channel"] = g["Channel"].map({"retail_cases_recent": "Retail stores", "wholesale_cases_recent": "Wholesale"})
        show(px.bar(g, x="department", y="Cases", color="Channel", barmode="group", title="Volume by department and channel",
                    color_discrete_map={"Retail stores": COLORS["primary"], "Wholesale": COLORS["amber"]}), 380)
    with b:
        t = trend[trend["department"].isin(dept)].groupby("month")[["retail_cases", "wholesale_cases"]].sum().reset_index()
        t["Month"] = t["month"].dt.strftime("%Y-%m")
        m = t.melt("Month", ["retail_cases", "wholesale_cases"], var_name="Channel", value_name="Cases")
        m["Channel"] = m["Channel"].map({"retail_cases": "Retail stores", "wholesale_cases": "Wholesale"})
        fig = px.bar(m, x="Month", y="Cases", color="Channel", title="Monthly volume by channel",
                     color_discrete_map={"Retail stores": COLORS["primary"], "Wholesale": COLORS["amber"]})
        fig.update_xaxes(type="category")
        show(fig, 380)
    gap = P[(P["retail_cases_recent"] > 4) & (P["store_transfer_cases_recent"] < 0.8 * P["retail_cases_recent"])].copy()
    gap["shortfall_cases"] = (gap["retail_cases_recent"] - gap["store_transfer_cases_recent"]).round(1)
    l, r = st.columns(2)
    with l:
        share = whole / max(retail + whole, 1)
        card("Channel mix", COLORS["primary"], f"Wholesale is {share:.0%} of case volume",
             "Beer moves almost entirely through wholesale; spirits mostly through the stores.",
             "Plan warehouse stock on total demand, and store shelf stock on retail demand only.")
    with r:
        card("Replenishment", COLORS["amber"], f"{len(gap):,} products had less stock sent to stores than the stores sold",
             f"Total shortfall {gap['shortfall_cases'].sum():,.0f} cases over 4 months: these stores are running down their shelves.",
             "Raise the store transfer quantity for the products listed below.")
    st.dataframe(gap.nlargest(40, "shortfall_cases")[["code", "name", "category", "retail_cases_recent", "store_transfer_cases_recent",
                                                       "shortfall_cases", "stock_units", "stock_status"]],
                 width="stretch", hide_index=True, height=320)
