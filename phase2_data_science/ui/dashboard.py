"""Task 2 - AI Data Analysis Dashboard: upload any dataset, clean it, explore it, get insights."""
from __future__ import annotations

import io

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from common import COLORS, ROOT, card, page, show, style
import engine

page("AI Data Analysis Dashboard", "Task 2 · Upload a dataset, and the dashboard cleans it, profiles it and writes the findings for you")

SAMPLES = {
    "Bank marketing campaign (UCI, 41,188 real records)": ROOT / "data" / "samples" / "bank_marketing.csv",
    "Liquor inventory with forecasts (Task 3 output)": ROOT / "task3_inventory" / "artifacts" / "products.csv",
    "Apple daily share prices (Yahoo Finance)": ROOT / "data" / "stocks" / "AAPL.csv",
}
LEVEL = {"high": COLORS["red"], "medium": COLORS["amber"], "info": COLORS["primary"]}


@st.cache_data(show_spinner="Reading file...")
def load(raw: bytes, name: str) -> pd.DataFrame:
    return engine.read_dataset(raw, name)


@st.cache_data(show_spinner="Cleaning...")
def run_clean(raw: pd.DataFrame, missing: str, dedupe: bool, normalise: bool):
    r = engine.clean(raw, missing=missing, drop_duplicates=dedupe, normalise=normalise)
    return r.df, r.log


@st.cache_data(show_spinner="Looking for patterns...")
def run_insights(df: pd.DataFrame, measure, target):
    return engine.insights(df, measure=measure, target=target)


@st.cache_data(show_spinner=False)
def run_anomalies(df: pd.DataFrame, rate: float):
    return engine.anomalies(df, engine.profile(df), contamination=rate)


# ----------------------------------------------------------------------------- sidebar: dataset controls
with st.sidebar:
    st.subheader("Dataset")
    up = st.file_uploader("Upload CSV, Excel or JSON", type=["csv", "tsv", "txt", "xlsx", "xls", "json"])
    available = {k: v for k, v in SAMPLES.items() if v.exists()}
    sample = st.selectbox("...or use a real sample", list(available), disabled=up is not None)
    st.subheader("Preprocessing")
    missing = st.radio("Missing values", ["impute", "drop", "keep"], horizontal=True,
                       help="impute = median for numbers, most frequent value for text")
    dedupe = st.checkbox("Remove duplicate rows", True)
    normalise = st.checkbox("Normalise numeric columns (0-1)", False,
                            help="Only needed for comparing columns on different scales")

try:
    if up is not None:
        raw, name = load(up.getvalue(), up.name), up.name
    elif available:
        raw, name = load(available[sample].read_bytes(), available[sample].name), available[sample].name
    else:
        st.info("Upload a dataset to begin.")
        st.stop()
except engine.DatasetError as exc:
    st.error(f"File rejected: {exc}")
    st.stop()

df, clean_log = run_clean(raw, missing, dedupe, normalise)
roles = engine.profile(df)
ov = engine.overview(df)
st.success(f"**{name}** passed validation: {len(raw):,} rows × {raw.shape[1]} columns loaded, "
           f"{len(df):,} rows × {df.shape[1]} columns after cleaning.")

with st.sidebar:
    st.subheader("Analysis focus")
    auto_m, auto_t = engine.pick_measure(df, roles), engine.pick_target(df, roles)
    m_opts = ["(none)"] + roles["numeric"]
    measure = st.selectbox("Business measure", m_opts, index=m_opts.index(auto_m) if auto_m else 0,
                           help="The number you care about, such as sales or revenue")
    t_opts = ["(none)"] + roles["binary"]
    target = st.selectbox("Yes/No outcome", t_opts, index=t_opts.index(auto_t) if auto_t else 0,
                          help="An outcome to explain, such as churned or subscribed")
measure = None if measure == "(none)" else measure
target = None if target == "(none)" else target
found = run_insights(df, measure, target)

tabs = st.tabs(["Overview", "Preprocessing", "Visualise", "AI insights", "Analytics", "Anomalies", "Search & filter", "Report"])

# ----------------------------------------------------------------------------- overview
with tabs[0]:
    c = st.columns(5)
    c[0].metric("Rows", f"{ov['rows']:,}")
    c[1].metric("Columns", ov["columns"])
    c[2].metric("Missing cells", f"{ov['missing_pct']}%")
    c[3].metric("Duplicate rows", f"{ov['duplicates']:,}")
    c[4].metric("Memory", f"{ov['memory_mb']} MB")
    st.info(engine.executive_summary(df, roles, found))
    l, r = st.columns([3, 2])
    with l:
        st.subheader("Columns, data types and missing values")
        st.dataframe(ov["columns_table"], width="stretch", hide_index=True, height=360)
    with r:
        st.subheader("Column roles")
        kinds = pd.DataFrame({"Role": ["Numeric", "Categorical", "Date", "Identifier", "Free text"],
                              "Columns": [len(roles[k]) for k in ("numeric", "category", "date", "id", "text")]})
        fig = px.pie(kinds[kinds["Columns"] > 0], names="Role", values="Columns", hole=0.55)
        show(fig, 320, showlegend=True)
    st.subheader("Statistical summary")
    if len(ov["numeric_summary"]):
        st.dataframe(ov["numeric_summary"], width="stretch")
    if len(ov["category_summary"]):
        st.dataframe(ov["category_summary"], width="stretch", hide_index=True)
    st.subheader("First rows")
    st.dataframe(df.head(50), width="stretch", hide_index=True)

# ----------------------------------------------------------------------------- preprocessing
with tabs[1]:
    st.subheader("What was done to the data")
    log = pd.DataFrame(clean_log)
    c = st.columns(3)
    c[0].metric("Operations", len(log))
    c[1].metric("Rows before → after", f"{len(raw):,} → {len(df):,}")
    c[2].metric("Missing cells before → after", f"{int(raw.isna().sum().sum()):,} → {int(df.isna().sum().sum()):,}")
    st.dataframe(log, width="stretch", hide_index=True)
    miss = raw.isna().sum()
    miss = miss[miss > 0].sort_values()
    if len(miss):
        fig = px.bar(x=miss.values, y=miss.index.astype(str), orientation="h",
                     labels={"x": "Missing values in the uploaded file", "y": ""}, title="Missing values per column (before cleaning)")
        show(fig, max(260, 26 * len(miss) + 90))
    st.download_button("Download cleaned dataset (CSV)", df.to_csv(index=False).encode(), f"cleaned_{name.rsplit('.', 1)[0]}.csv", "text/csv")

# ----------------------------------------------------------------------------- visualise
with tabs[2]:
    kinds = ["Bar chart", "Line chart", "Pie chart", "Histogram", "Scatter plot", "Box plot", "Correlation heatmap"]
    left, right = st.columns([1, 3])
    with left:
        kind = st.selectbox("Chart type", kinds)
        cats, nums, dates = roles["category"], roles["numeric"], roles["date"]
        agg = "sum"
        if kind in ("Bar chart", "Pie chart"):
            x = st.selectbox("Category", cats or list(df.columns))
            y = st.selectbox("Value", ["(count of rows)"] + nums)
            agg = st.selectbox("Aggregate", ["sum", "mean", "median", "max"]) if y != "(count of rows)" else "count"
            top = st.slider("Show top", 3, 30, 10)
        elif kind == "Line chart":
            x = st.selectbox("X axis", dates + nums + cats)
            y = st.selectbox("Value", nums or ["(count of rows)"])
            agg = st.selectbox("Aggregate", ["sum", "mean", "median"])
            colour = st.selectbox("Split by", ["(none)"] + [c for c in cats if df[c].nunique() <= 8])
        elif kind == "Histogram":
            x = st.selectbox("Column", nums)
            bins = st.slider("Bins", 5, 100, 30)
            colour = st.selectbox("Split by", ["(none)"] + [c for c in cats if df[c].nunique() <= 6])
        elif kind == "Scatter plot":
            x = st.selectbox("X axis", nums)
            y = st.selectbox("Y axis", nums, index=min(1, len(nums) - 1))
            colour = st.selectbox("Colour by", ["(none)"] + [c for c in cats if df[c].nunique() <= 10])
        elif kind == "Box plot":
            y = st.selectbox("Value", nums)
            x = st.selectbox("Group by", [c for c in cats if df[c].nunique() <= 20] or cats)
    with right:
        try:
            if kind in ("Bar chart", "Pie chart"):
                g = (df.groupby(x).size() if agg == "count" else df.groupby(x)[y].agg(agg)).sort_values(ascending=False).head(top)
                label = "rows" if agg == "count" else f"{agg} of {y}"
                gd = g.reset_index(name=label)
                gd[x] = gd[x].astype(str)
                fig = (px.bar(gd, x=x, y=label, title=f"{label} by {x}", text_auto=".3s") if kind == "Bar chart"
                       else px.pie(gd, names=x, values=label, title=f"Share of {label} by {x}", hole=0.4))
            elif kind == "Line chart":
                d = df.copy()
                if x in dates:
                    span = (d[x].max() - d[x].min()).days
                    d[x] = d[x].dt.to_period("M" if span > 180 else "W" if span > 30 else "D").dt.start_time
                by = [x] + ([colour] if colour != "(none)" else [])
                g = (d.groupby(by).size().reset_index(name="rows") if y == "(count of rows)"
                     else d.groupby(by)[y].agg(agg).reset_index())
                fig = px.line(g.sort_values(x), x=x, y=g.columns[-1], color=None if colour == "(none)" else colour,
                              markers=len(g) < 80, title=f"{g.columns[-1]} over {x}")
            elif kind == "Histogram":
                fig = px.histogram(df, x=x, nbins=bins, color=None if colour == "(none)" else colour, marginal="box",
                                   barmode="overlay", opacity=0.75 if colour != "(none)" else 1, title=f"Distribution of {x}")
            elif kind == "Scatter plot":
                d = df.sample(min(len(df), 5000), random_state=1)
                fig = px.scatter(d, x=x, y=y, color=None if colour == "(none)" else colour, opacity=0.55,
                                 trendline="ols" if colour == "(none)" else None, trendline_color_override=COLORS["red"],
                                 title=f"{y} vs {x}" + (" (5,000-row sample)" if len(df) > 5000 else ""))
            elif kind == "Box plot":
                fig = px.box(df, x=x, y=y, color=x, title=f"{y} by {x}")
                fig.update_layout(showlegend=False)
            else:
                corr = df[nums].corr(numeric_only=True).round(2)
                fig = px.imshow(corr, text_auto=True, color_continuous_scale="RdBu_r", zmin=-1, zmax=1, aspect="auto",
                                title="Correlation between numeric columns")
            show(fig, 520)
            st.session_state.setdefault("report_figs", {})
            if st.button("Add this chart to the report"):
                st.session_state["report_figs"][fig.layout.title.text] = fig
                st.toast("Chart added to the report")
        except Exception as exc:
            st.warning(f"This combination cannot be charted: {exc}")

# ----------------------------------------------------------------------------- insights
with tabs[3]:
    st.subheader("Findings, ranked by importance")
    st.caption("Generated automatically with significance tests, effect sizes and an Isolation Forest. "
               "Change the business measure or outcome in the sidebar to refocus the analysis.")
    kinds_present = sorted({f["kind"] for f in found})
    chosen = st.multiselect("Show", kinds_present, default=kinds_present)
    levels = {"high": "Key finding", "medium": "Notable", "info": "FYI"}
    for f in [f for f in found if f["kind"] in chosen]:
        card(f"{levels[f['level']]} · {f['kind']}", LEVEL[f["level"]], f["title"], f["detail"], f["action"])

# ----------------------------------------------------------------------------- analytics
with tabs[4]:
    cats = [c for c in roles["category"] if 2 <= df[c].nunique() <= 40 and c != target]
    if not cats:
        st.info("No categorical columns to compare.")
    else:
        dim = st.selectbox("Compare by", cats)
        if target:
            y, pos = engine._as_binary(df[target])
            g = df.assign(_y=y).groupby(dim)["_y"].agg(rate="mean", records="count").sort_values("rate", ascending=False)
            base = y.mean()
            c = st.columns(4)
            c[0].metric(f"Overall {target} = {pos}", f"{base:.1%}")
            c[1].metric("Best group", str(g.index[0]), f"{g['rate'].iloc[0]:.1%}")
            c[2].metric("Weakest group", str(g.index[-1]), f"{g['rate'].iloc[-1]:.1%}", delta_color="inverse")
            c[3].metric("Groups", len(g))
            gd = g.reset_index()
            gd[dim] = gd[dim].astype(str)
            fig = px.bar(gd, x=dim, y="rate", color="rate", color_continuous_scale="Blues", text_auto=".1%",
                         title=f"Rate of {target} = {pos} by {dim}", hover_data=["records"])
            fig.add_hline(y=base, line_dash="dot", annotation_text="overall")
            fig.update_yaxes(tickformat=".0%")
            show(fig, 420, coloraxis_showscale=False)
        if measure:
            g = df.groupby(dim)[measure].agg(total="sum", average="mean", records="count").sort_values("total", ascending=False)
            c = st.columns(4)
            c[0].metric(f"Total {measure}", engine._fmt(g["total"].sum()))
            c[1].metric("Top performer", str(g.index[0]), f"{g['total'].iloc[0] / g['total'].sum():.0%} of total")
            c[2].metric("Highest average", str(g["average"].idxmax()), engine._fmt(g["average"].max()))
            c[3].metric("Lowest average", str(g["average"].idxmin()), engine._fmt(g["average"].min()), delta_color="inverse")
            gd = g.reset_index()
            gd[dim] = gd[dim].astype(str)
            a, b = st.columns(2)
            with a:
                show(px.bar(gd.head(15), x="total", y=dim, orientation="h", title=f"Top {dim} by total {measure}", text_auto=".3s")
                     .update_yaxes(autorange="reversed"), 420)
            with b:
                gd["cumulative share"] = gd["total"].cumsum() / gd["total"].sum()
                fig = px.line(gd, x=dim, y="cumulative share", markers=True, title="Concentration (Pareto curve)")
                fig.add_hline(y=0.8, line_dash="dot", annotation_text="80%")
                fig.update_yaxes(tickformat=".0%")
                show(fig, 420)
            if roles["date"]:
                d = roles["date"][0]
                ts = df.set_index(d)[measure].resample("MS").sum()
                growth = ts.pct_change()
                a, b = st.columns(2)
                with a:
                    show(px.area(ts.reset_index(), x=d, y=measure, title=f"Monthly {measure}"), 340)
                with b:
                    gdf = growth.dropna().reset_index(name="growth")
                    fig = px.bar(gdf, x=d, y="growth", title="Month-over-month growth",
                                 color=gdf["growth"] > 0, color_discrete_map={True: COLORS["green"], False: COLORS["red"]})
                    fig.update_yaxes(tickformat=".0%")
                    show(fig, 340, showlegend=False)
        if not target and not measure:
            g = df[dim].value_counts().reset_index()
            g.columns = [dim, "records"]
            g[dim] = g[dim].astype(str)
            show(px.bar(g.head(20), x=dim, y="records", title=f"Records by {dim}", text_auto=True), 420)
            st.caption("Pick a business measure or outcome in the sidebar for richer analytics.")
        if roles["numeric"]:
            st.subheader("Distribution analysis")
            num = st.selectbox("Numeric column", roles["numeric"], key="dist_col")
            s = df[num].dropna()
            c = st.columns(5)
            for box, (lab, val) in zip(c, [("Mean", s.mean()), ("Median", s.median()), ("Std dev", s.std()),
                                           ("5th percentile", s.quantile(.05)), ("95th percentile", s.quantile(.95))]):
                box.metric(lab, engine._fmt(val))
            show(px.violin(df, x=dim if df[dim].nunique() <= 12 else None, y=num, box=True, title=f"{num} distribution"), 360)

# ----------------------------------------------------------------------------- anomalies
with tabs[5]:
    rate = st.slider("Share of rows to flag as unusual", 0.5, 5.0, 1.0, 0.5, format="%.1f%%") / 100
    col_out, flagged = run_anomalies(df, rate)
    a, b = st.columns([2, 3])
    with a:
        st.subheader("Extreme values per column")
        st.caption("Values beyond 3 × the interquartile range.")
        st.dataframe(col_out, width="stretch", hide_index=True) if len(col_out) else st.success("No extreme values.")
    with b:
        st.subheader("Unusual records (Isolation Forest)")
        st.caption("Rows that are unlike the rest across all numeric columns together.")
        if len(flagged):
            st.dataframe(flagged, width="stretch", hide_index=True, height=360)
            st.download_button("Download flagged rows", flagged.to_csv(index=False).encode(), "anomalies.csv", "text/csv")
        else:
            st.info("Needs at least two numeric columns and 50 rows.")

# ----------------------------------------------------------------------------- search & filter
with tabs[6]:
    view = df
    q = st.text_input("Search all columns", placeholder="type any text or number")
    if q:
        mask = np.zeros(len(view), dtype=bool)
        for c in view.columns:
            mask |= view[c].astype(str).str.contains(q, case=False, na=False, regex=False).values
        view = view[mask]
    cols = st.multiselect("Filter by columns", list(df.columns))
    for c in cols:
        if pd.api.types.is_numeric_dtype(df[c]):
            lo, hi = float(df[c].min()), float(df[c].max())
            if lo < hi:
                a, b = st.slider(c, lo, hi, (lo, hi))
                view = view[view[c].between(a, b)]
        elif pd.api.types.is_datetime64_any_dtype(df[c]):
            a, b = st.date_input(c, (df[c].min().date(), df[c].max().date()))[:2] if True else (None, None)
            view = view[view[c].between(pd.Timestamp(a), pd.Timestamp(b) + pd.Timedelta(days=1))]
        else:
            opts = sorted(df[c].dropna().astype(str).unique())[:500]
            pick = st.multiselect(c, opts)
            if pick:
                view = view[view[c].astype(str).isin(pick)]
    cond = st.text_input("Custom condition (optional)", placeholder="example:  age > 40 and job == 'admin.'",
                         help="Uses pandas query syntax. Wrap column names with spaces or dots in `backticks`.")
    if cond:
        try:
            view = view.query(cond, engine="python")
        except Exception as exc:
            st.error(f"Condition not understood: {str(exc)[:150]}")
    st.caption(f"{len(view):,} of {len(df):,} rows match")
    sort = st.selectbox("Sort by", ["(original order)"] + list(df.columns))
    if sort != "(original order)":
        view = view.sort_values(sort, ascending=st.toggle("Ascending", False))
    st.dataframe(view.head(2000), width="stretch", hide_index=True, height=420)
    st.download_button("Download this subset (CSV)", view.to_csv(index=False).encode(), "subset.csv", "text/csv")

# ----------------------------------------------------------------------------- report
with tabs[7]:
    st.subheader("Summary report")
    figs = dict(st.session_state.get("report_figs", {}))
    if not figs:                                           # sensible defaults when the user picked nothing
        if len(roles["numeric"]) >= 2:
            figs["Correlation"] = style(px.imshow(df[roles["numeric"]].corr(numeric_only=True).round(2), text_auto=True,
                                                  color_continuous_scale="RdBu_r", zmin=-1, zmax=1, aspect="auto",
                                                  title="Correlation between numeric columns"), 480)
        if roles["numeric"]:
            n0 = measure or roles["numeric"][0]
            figs["Distribution"] = style(px.histogram(df, x=n0, nbins=40, title=f"Distribution of {n0}"), 360)
        if roles["category"]:
            c0 = roles["category"][0]
            vc = df[c0].astype(str).value_counts().head(12).reset_index()
            vc.columns = [c0, "records"]
            figs["Categories"] = style(px.bar(vc, x=c0, y="records", title=f"Records by {c0}"), 360)
    st.write(f"The report contains the dataset overview, the preprocessing log, **{len(found)} findings with recommendations** "
             f"and **{len(figs)} chart(s)**. Add your own charts from the Visualise tab.")
    report = engine.build_report(name, df, clean_log, found, list(figs.values()))
    from pdf import Report
    pdf = Report("Data analysis report", name)
    pdf.heading("Executive summary")
    pdf.text(engine.executive_summary(df, roles, found))
    pdf.heading("Dataset overview")
    pdf.facts({"Rows": f"{ov['rows']:,}", "Columns": str(ov["columns"]), "Missing cells": f"{ov['missing_pct']}%",
               "Duplicate rows": f"{ov['duplicates']:,}",
               "Column roles": f"{len(roles['numeric'])} numeric, {len(roles['category'])} categorical, {len(roles['date'])} date"})
    pdf.table(ov["columns_table"][["Column", "Type", "Unique", "Missing %", "Example"]], max_rows=40, widths=[48, 30, 22, 24, 56])
    pdf.heading("Preprocessing performed")
    pdf.table(pd.DataFrame(clean_log), max_rows=30, widths=[52, 104, 24])
    if len(ov["numeric_summary"]):
        pdf.heading("Statistical summary")
        ns = ov["numeric_summary"].reset_index(names="Column")[["Column", "mean", "std", "min", "50%", "max"]]
        pdf.table(ns.rename(columns={"50%": "median"}), max_rows=30, widths=[55, 25, 25, 25, 25, 25])
    pdf.heading("Key findings and recommendations")
    for f in found[:20]:
        pdf.finding(f"{f['level']} - {f['kind']}", f["title"], f["detail"], f["action"])
    a, b, c3 = st.columns(3)
    c3.download_button("Download report (PDF)", pdf.done(), f"report_{name.rsplit('.', 1)[0]}.pdf", "application/pdf")
    a.download_button("Download report (HTML, with interactive charts)", report.encode("utf-8"),
                      f"report_{name.rsplit('.', 1)[0]}.html", "text/html", type="primary")
    md = "\n".join(f"- **{f['title']}** {f['detail']} _Recommendation: {f['action']}_" for f in found)
    b.download_button("Download findings (Markdown)", md.encode("utf-8"), "findings.md", "text/markdown")
    st.markdown("#### Key findings and recommendations")
    for f in found[:8]:
        card(f["kind"], LEVEL[f["level"]], f["title"], f["detail"], f["action"])
