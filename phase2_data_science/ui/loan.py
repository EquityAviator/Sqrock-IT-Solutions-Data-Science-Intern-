"""Task 4 - Loan Approval Prediction System."""
from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from common import COLORS, ROOT, card, load_csv, load_json, need, page, show
import prep

ART = ROOT / "task4_loan_approval" / "artifacts"
page("Loan Approval Prediction System",
     "Task 4 · 174,000 real mortgage applications (HMDA 2023, Colorado, US Consumer Financial Protection Bureau)")
if not need(ART / "metrics.json", "task4_loan_approval/train.py"):
    st.stop()

M = load_json(str(ART / "metrics.json"))
E, BEST = M["eda"], M["best_model"]
R = M["models"][BEST]["test"]
NICE = {"dti": "Debt-to-income ratio", "ltv": "Loan-to-value ratio", "loan_amount": "Loan amount", "income": "Annual income",
        "property_value": "Property value", "loan_term": "Loan term", "loan_to_income": "Loan-to-income ratio",
        "payment_to_income": "Payment-to-income ratio", "loan_purpose": "Loan purpose", "loan_type": "Loan type", "lien": "Lien position",
        "occupancy": "Occupancy", "dwelling": "Dwelling type", "credit_line": "Credit line", "co_applicant": "Co-applicant",
        "tract_minority_pct": "Neighbourhood minority share", "tract_income_pct": "Neighbourhood income level",
        "area_median_income": "Area median income"}


@st.cache_resource(show_spinner="Loading model...")
def bundle():
    return joblib.load(ART / "loan_model.joblib")


def rate_chart(rows, title, height=330, horizontal=False):
    g = pd.DataFrame(rows)
    g["approval_rate"] *= 100
    fig = (px.bar(g, y="group", x="approval_rate", orientation="h", text_auto=".1f", hover_data=["applications"]) if horizontal
           else px.bar(g, x="group", y="approval_rate", text_auto=".1f", hover_data=["applications"]))
    line = dict(line_dash="dot", line_color=COLORS["red"])
    fig.add_vline(x=E["approval_rate"] * 100, **line) if horizontal else fig.add_hline(y=E["approval_rate"] * 100, **line)
    fig.update_layout(title=title, xaxis_title="Approved (%)" if horizontal else "", yaxis_title="" if horizontal else "Approved (%)")
    if horizontal:
        fig.update_yaxes(autorange="reversed")
    show(fig, height)


c = st.columns(5)
c[0].metric("Applications", f"{E['applications']:,}")
c[1].metric("Approval rate", f"{E['approval_rate']:.1%}")
c[2].metric("Denied", f"{E['denied']:,}")
c[3].metric("Best model", BEST, f"ROC-AUC {R['ROC_AUC']:.3f}", delta_color="off")
c[4].metric("Denials caught", f"{R['Denied_recall']:.1%}", f"accuracy {R['Accuracy']:.1%}", delta_color="off")

tabs = st.tabs(["Predict an application", "Approval statistics", "Applicants", "Model performance", "Insights", "Fairness audit"])

# ----------------------------------------------------------------------------- predict
with tabs[0]:
    B = bundle()
    med, opt = B["medians"], B["options"]
    st.subheader("Loan eligibility checker")
    with st.form("application"):
        a, b, c3 = st.columns(3)
        income = a.number_input("Annual income ($)", 10_000, 2_000_000, 95_000, 5_000)
        loan_amount = a.number_input("Loan amount ($)", 5_000, 3_000_000, 380_000, 5_000)
        property_value = a.number_input("Property value ($)", 20_000, 5_000_000, 475_000, 5_000)
        debts = b.number_input("Other monthly debt payments ($)", 0, 20_000, 600, 50, help="Car loans, cards, student loans")
        term_years = b.selectbox("Loan term (years)", [10, 15, 20, 25, 30, 40], index=4)
        loan_purpose = b.selectbox("Loan purpose", opt["loan_purpose"], index=opt["loan_purpose"].index("Home purchase"))
        loan_type = c3.selectbox("Loan type", opt["loan_type"])
        lien = c3.selectbox("Lien position", opt["lien"])
        occupancy = c3.selectbox("Occupancy", opt["occupancy"], index=opt["occupancy"].index("Main home"))
        d4, e5, f6 = st.columns(3)
        dwelling = d4.selectbox("Dwelling", opt["dwelling"])
        credit_line = e5.selectbox("Product", opt["credit_line"], index=opt["credit_line"].index("Fixed loan"))
        co_applicant = f6.selectbox("Application", opt["co_applicant"])
        go_btn = st.form_submit_button("Predict approval", type="primary")

    term = term_years * 12
    payment = float(prep.monthly_payment(loan_amount, term))
    dti = (payment + debts) * 12 / income * 100
    row = pd.DataFrame([{
        "loan_amount": loan_amount, "income": income, "property_value": property_value, "loan_term": term,
        "dti": min(dti, 100), "ltv": np.nan, "tract_minority_pct": med["tract_minority_pct"],
        "tract_income_pct": med["tract_income_pct"], "area_median_income": med["area_median_income"],
        "loan_type": loan_type, "loan_purpose": loan_purpose, "lien": lien, "occupancy": occupancy, "dwelling": dwelling,
        "credit_line": credit_line, "co_applicant": co_applicant}])
    row = prep.engineer(row)
    Z = B["preprocessor"].transform(row[B["features"]])
    proba = float(B["calibrator"].predict(B["model"].predict_proba(Z)[:, 1])[0])
    approved = proba >= B["threshold"]

    l, r2 = st.columns([2, 3])
    with l:
        fig = go.Figure(go.Indicator(mode="gauge+number", value=proba * 100, number={"suffix": "%", "font": {"size": 44}},
                                     title={"text": "Approval probability"},
                                     gauge={"axis": {"range": [0, 100]}, "bar": {"color": COLORS["green"] if approved else COLORS["red"]},
                                            "threshold": {"line": {"color": "#111827", "width": 3}, "value": B["threshold"] * 100},
                                            "steps": [{"range": [0, B["threshold"] * 100], "color": "rgba(220,38,38,.12)"},
                                                      {"range": [B["threshold"] * 100, 100], "color": "rgba(5,150,105,.12)"}]}))
        show(fig, 300)
        if approved:
            st.success(f"### Likely APPROVED\nProbability {proba:.1%} (decision line {B['threshold']:.0%})")
        else:
            st.error(f"### Likely REJECTED\nProbability {proba:.1%} (decision line {B['threshold']:.0%})")
        risk = (1 - proba) * 100
        band = "Low" if risk < 15 else "Moderate" if risk < 35 else "High" if risk < 60 else "Very high"
        m1, m2 = st.columns(2)
        m1.metric("Risk score (0-100)", f"{risk:.0f}", band, delta_color="off")
        m2.metric("Est. monthly payment", f"${payment:,.0f}", "at 6.8% interest", delta_color="off")
    with r2:
        k = st.columns(3)
        ltv = float(row["ltv"].iloc[0])
        k[0].metric("Debt-to-income", f"{dti:.0f}%", "healthy" if dti <= 43 else "above the 43% guideline", delta_color="normal" if dti <= 43 else "inverse")
        k[1].metric("Loan-to-value", f"{ltv:.0f}%", "healthy" if ltv <= 80 else "above 80%", delta_color="normal" if ltv <= 80 else "inverse")
        k[2].metric("Loan-to-income", f"{loan_amount / income:.1f}x")
        # explainable AI: SHAP contributions from the gradient-boosted model
        import xgboost as xgb
        contrib = B["xgb"].get_booster().predict(xgb.DMatrix(Z, feature_names=B["feature_names"]), pred_contribs=True)[0][:-1]
        s = pd.Series(contrib, index=B["feature_names"]).groupby(B["origin"]).sum()
        s.index = [NICE.get(i, i) for i in s.index]
        s = s[s.abs().sort_values(ascending=False).index].head(8).iloc[::-1]
        fig = go.Figure(go.Bar(x=s.values, y=s.index, orientation="h",
                               marker_color=[COLORS["green"] if v > 0 else COLORS["red"] for v in s.values]))
        fig.update_layout(title="Why: what pushed this application up (green) or down (red)", xaxis_title="Effect on the approval score")
        show(fig, 330)
    st.subheader("Recommendations for this applicant")
    tips = []
    if dti > 43:
        target_pay = 0.43 * income / 12 - debts
        tips.append(("Debt-to-income is the main obstacle", f"At {dti:.0f}% it is above the 43% most lenders accept. In this data only "
                     f"{next((g['approval_rate'] for g in E['by_dti'] if g['group'] == '>60%'), 0):.0%} of applications above 60% were approved.",
                     f"Reduce other debts, or lower the loan so the payment is about ${max(target_pay, 0):,.0f} a month."))
    if ltv > 95:
        tips.append(("Small deposit", f"The loan is {ltv:.0f}% of the property value.", f"A deposit of ${max(property_value * 0.2 - (property_value - loan_amount), 0):,.0f} more would bring it to 80%."))
    if co_applicant == "Single applicant":
        j = {g["group"]: g["approval_rate"] for g in E["by_co_applicant"]}
        tips.append(("Joint applications are approved more often", f"{j.get('Joint application', 0):.0%} vs {j.get('Single applicant', 0):.0%} for single applicants.",
                     "Adding a co-applicant with income strengthens the application."))
    if not tips:
        tips.append(("Strong application", "Debt-to-income and loan-to-value are both inside the usual guidelines.", "No change needed."))
    for t in tips:
        card("Advice", COLORS["primary"], *t)
    from pdf import Report
    pdf = Report("Loan application assessment", f"Model: {BEST} | HMDA 2023 Colorado")
    pdf.heading("Decision")
    pdf.text(f"{'LIKELY APPROVED' if approved else 'LIKELY REJECTED'}  -  approval probability {proba:.1%} (decision line {B['threshold']:.0%})", bold=True, size=13)
    pdf.facts({"Risk score (0-100)": f"{risk:.0f} ({band})", "Estimated monthly payment": f"${payment:,.0f} at 6.8% interest"})
    pdf.heading("Application")
    pdf.facts({"Annual income": f"${income:,.0f}", "Loan amount": f"${loan_amount:,.0f}", "Property value": f"${property_value:,.0f}",
               "Other monthly debts": f"${debts:,.0f}", "Loan term": f"{term_years} years", "Purpose": loan_purpose, "Loan type": loan_type,
               "Lien position": lien, "Occupancy": occupancy, "Dwelling": dwelling, "Product": credit_line, "Application": co_applicant})
    pdf.heading("Key ratios")
    pdf.facts({"Debt-to-income": f"{dti:.0f}%  (guideline: up to 43%)", "Loan-to-value": f"{ltv:.0f}%  (guideline: up to 80%)",
               "Loan-to-income": f"{loan_amount / income:.1f}x"})
    pdf.heading("Why the model decided this")
    pdf.table(pd.DataFrame({"Factor": s.index[::-1], "Effect on approval score": s.values[::-1].round(3),
                            "Direction": ["helps" if v > 0 else "hurts" for v in s.values[::-1]]}), widths=[90, 50, 40], size=9)
    pdf.heading("Recommendations")
    for t in tips:
        pdf.finding("advice", *t)
    pdf.text(f"Model performance on {M['split']['test']:,} unseen applications: ROC-AUC {R['ROC_AUC']:.3f}, accuracy {R['Accuracy']:.1%}, "
             f"denials caught {R['Denied_recall']:.1%}. This is a statistical estimate, not a lending decision. "
             "Sex, age, race and ethnicity are not used by the model.", size=8)
    st.download_button("Download assessment (PDF)", pdf.done(), "loan_assessment.pdf", "application/pdf")
    st.caption("Neighbourhood values are set to the state median. Sex, age, race and ethnicity are not asked: the model does not use them.")

# ----------------------------------------------------------------------------- approval statistics
with tabs[1]:
    a, b = st.columns([2, 3])
    with a:
        fig = px.pie(names=["Approved", "Denied"], values=[E["approved"], E["denied"]], hole=0.58, color=["Approved", "Denied"],
                     color_discrete_map={"Approved": COLORS["green"], "Denied": COLORS["red"]}, title="Loan decisions")
        show(fig, 330)
    with b:
        rate_chart(E["by_dti"], "Approval rate by debt-to-income ratio")
    a, b = st.columns(2)
    with a:
        rate_chart(E["by_loan_purpose"], "Approval rate by loan purpose", horizontal=True)
        rate_chart(E["by_income"], "Approval rate by annual income")
        rate_chart(E["by_ltv"], "Approval rate by loan-to-value ratio")
    with b:
        rate_chart(E["by_loan_type"], "Approval rate by loan type", horizontal=True)
        rate_chart(E["by_loan_amount"], "Approval rate by loan amount")
        dr = pd.Series(E["denial_reasons"]).mul(100).round(1).reset_index()
        dr.columns = ["Reason", "Share of denials (%)"]
        fig = px.bar(dr, x="Share of denials (%)", y="Reason", orientation="h", title="Why lenders said no (stated denial reason)",
                     color_discrete_sequence=[COLORS["red"]], text_auto=True)
        fig.update_yaxes(autorange="reversed")
        show(fig, 330)
    st.caption("The dotted red line is the overall approval rate. Hover a bar to see how many applications it represents.")
    corr = pd.DataFrame(E["correlation"]).rename(index=NICE, columns=NICE).rename(index={"approved": "Approved"}, columns={"approved": "Approved"})
    show(px.imshow(corr.round(2), text_auto=True, color_continuous_scale="RdBu_r", zmin=-1, zmax=1, aspect="auto", title="Correlation heatmap"), 560)

# ----------------------------------------------------------------------------- applicants
with tabs[2]:
    a, b = st.columns(2)
    for box, key, title, edge in ((a, "income_hist", "Income distribution", "Annual income ($)"), (b, "loan_hist", "Loan amount distribution", "Loan amount ($)")):
        h = E[key]
        x = [(h["edges"][i] + h["edges"][i + 1]) / 2 for i in range(len(h["approved"]))]
        fig = go.Figure([go.Bar(x=x, y=h["approved"], name="Approved", marker_color=COLORS["green"], opacity=0.75),
                         go.Bar(x=x, y=h["denied"], name="Denied", marker_color=COLORS["red"], opacity=0.75)])
        fig.update_layout(barmode="overlay", title=title, xaxis_title=edge, yaxis_title="Applications")
        with box:
            show(fig, 340)
    md = pd.DataFrame(E["medians"]).rename(index=NICE)
    md["difference %"] = ((md["denied"] / md["approved"] - 1) * 100).round(1)
    st.subheader("Typical approved vs denied applicant (medians)")
    st.dataframe(md.rename(columns={"approved": "Approved", "denied": "Denied"}), width="stretch")
    st.subheader("Demographics")
    a, b = st.columns(2)
    with a:
        rate_chart(E["by_age"], "Approval rate by applicant age")
        rate_chart(E["by_co_applicant"], "Single vs joint applications")
    with b:
        rate_chart(E["by_sex"], "Approval rate by sex of applicant(s)")
        rate_chart(E["by_occupancy"], "Approval rate by occupancy")
    st.caption("Demographics are shown for analysis only. They are not inputs to the model; see the Fairness audit tab.")

# ----------------------------------------------------------------------------- model performance
with tabs[3]:
    T = pd.DataFrame({k: v["test"] for k, v in M["models"].items()}).T
    show_cols = ["Accuracy", "Precision", "Recall", "F1", "Denied_recall", "Denied_precision", "Balanced_accuracy", "ROC_AUC"]
    tbl = T[show_cols].astype(float)
    tbl["Train seconds"] = [M["models"][k]["train_seconds"] for k in tbl.index]
    st.subheader("Multi-model comparison (unseen test set)")
    st.dataframe(tbl.style.highlight_max(subset=show_cols, color="#bbf7d0").format("{:.4f}", subset=show_cols), width="stretch")
    st.caption(f"Best model, chosen on validation: **{BEST}**. XGBoost trained on **{M['xgboost_device']}** ({M['xgboost_trees']} trees). "
               f"Always answering 'approved' would score {M['baseline_accuracy_always_approve']:.1%} accuracy while catching no denials, "
               "which is why balanced accuracy and denied-recall matter more than accuracy here.")
    a, b = st.columns(2)
    with a:
        fig = go.Figure()
        for name, rc in M["roc"].items():
            fig.add_trace(go.Scatter(x=rc["fpr"], y=rc["tpr"], name=f"{name} ({M['models'][name]['test']['ROC_AUC']:.3f})",
                                     line=dict(width=3 if name == BEST else 1.5)))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], line=dict(dash="dot", color="#94a3b8"), showlegend=False))
        fig.update_layout(title="ROC curves (AUC in brackets)", xaxis_title="False positive rate", yaxis_title="True positive rate")
        show(fig, 430)
    with b:
        cm = R["confusion"]
        z = [[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]]
        fig = px.imshow(z, text_auto=",", x=["Predicted denied", "Predicted approved"], y=["Actually denied", "Actually approved"],
                        color_continuous_scale="Blues", title=f"Confusion matrix · {BEST}")
        show(fig, 430, coloraxis_showscale=False)
    total = sum(cm.values())
    card("Interpretation", COLORS["primary"], f"{BEST} gets {R['Accuracy']:.1%} of {total:,} unseen applications right",
         f"It catches {R['Denied_recall']:.1%} of real denials ({cm['tn']:,} of {cm['tn'] + cm['fp']:,}) and correctly approves {R['Recall']:.1%} of good applications. "
         f"When it predicts approval it is right {R['Precision']:.1%} of the time. {cm['fp']:,} risky applications would slip through and "
         f"{cm['fn']:,} good applicants would be sent for manual review.",
         "Use it to fast-track clear approvals and to route predicted denials to a human underwriter, not to auto-reject.")
    a, b = st.columns(2)
    with a:
        tc = pd.DataFrame(M["threshold_curve"])
        fig = px.line(tc, x="threshold", y=["Accuracy", "Precision", "Recall", "Denied_recall"], title="Effect of the decision threshold",
                      labels={"value": "Score", "threshold": "Approve when probability is at least", "variable": ""})
        fig.add_vline(x=R["threshold"], line_dash="dot", annotation_text="chosen")
        show(fig, 360)
    with b:
        cal = pd.DataFrame(M["calibration"])
        fig = go.Figure([go.Scatter(x=cal["predicted"], y=cal["actual"], mode="lines+markers", name="Model"),
                         go.Scatter(x=[0, 1], y=[0, 1], line=dict(dash="dot", color="#94a3b8"), name="Perfect")])
        fig.update_layout(title="Are the probabilities trustworthy? (calibration)", xaxis_title="Predicted approval probability", yaxis_title="Actual approval rate")
        show(fig, 360)
    st.subheader("Prediction results on real test applications")
    S = load_csv(str(ART / "test_predictions_sample.csv"))
    only_wrong = st.toggle("Show only mistakes")
    Sv = S[S["actual"] != S["prediction"]] if only_wrong else S
    st.dataframe(Sv[["actual", "prediction", "approval_probability", "income", "loan_amount", "dti", "ltv", "loan_purpose", "loan_type", "co_applicant"]],
                 width="stretch", hide_index=True, height=320)

# ----------------------------------------------------------------------------- insights
with tabs[4]:
    imp = pd.Series(M["importance_permutation"]).rename(index=NICE).head(12)
    shp = pd.Series(M["importance_shap"]).rename(index=NICE).head(12)
    a, b = st.columns(2)
    with a:
        fig = px.bar(x=imp.values[::-1] * 100, y=imp.index[::-1], orientation="h", title="Feature importance (permutation)",
                     labels={"x": "Share of predictive power (%)", "y": ""}, color_discrete_sequence=[COLORS["violet"]])
        show(fig, 430)
    with b:
        fig = px.bar(x=shp.values[::-1] * 100, y=shp.index[::-1], orientation="h", title="Feature importance (SHAP, explainable AI)",
                     labels={"x": "Share of total effect (%)", "y": ""}, color_discrete_sequence=[COLORS["cyan"]])
        show(fig, 430)
    dti_hi = next(g for g in E["by_dti"] if g["group"] == ">60%")
    dti_ok = next(g for g in E["by_dti"] if g["group"] == "30-36%")
    pur = sorted(E["by_loan_purpose"], key=lambda g: g["approval_rate"])
    lien = {g["group"]: g["approval_rate"] for g in E["by_lien"]}
    top_reason = max(E["denial_reasons"], key=E["denial_reasons"].get)
    l, r2 = st.columns(2)
    with l:
        card("Factor", COLORS["red"], f"Debt-to-income decides most outcomes ({imp.iloc[0]:.0%} of the model's predictive power)" if "Debt" in imp.index[0]
             else f"{imp.index[0]} is the strongest factor ({imp.iloc[0]:.0%} of predictive power)",
             f"Above 60% debt-to-income only {dti_hi['approval_rate']:.0%} are approved, against {dti_ok['approval_rate']:.0%} at 30-36%. "
             f"It is also the most common stated denial reason: '{top_reason}' ({E['denial_reasons'][top_reason]:.0%} of denials).",
             "Pre-screen on debt-to-income before a full application; it saves both sides a hard credit check.")
        card("Trend", COLORS["amber"], f"Approval depends heavily on purpose: {pur[-1]['group']} {pur[-1]['approval_rate']:.0%} vs {pur[0]['group']} {pur[0]['approval_rate']:.0%}",
             "Purchase loans are backed by a fresh appraisal and usually a deposit; improvement and other-purpose loans are mostly second liens.",
             "Price and staff the product lines separately; they have very different risk.")
    with r2:
        card("Risk", COLORS["violet"], f"Second-lien loans are approved {lien.get('Second lien', 0):.0%} of the time vs {lien.get('First lien', 0):.0%} for first liens",
             "A second lien is repaid after the first mortgage if the borrower defaults, so lenders are stricter.",
             "Apply tighter debt-to-income limits to second liens.")
        card("Business", COLORS["green"], f"The model catches {R['Denied_recall']:.0%} of denials before underwriting",
             f"With ROC-AUC {R['ROC_AUC']:.3f}, it separates strong from weak applications well enough to prioritise underwriter time.",
             "Fast-track high-probability applications and offer the eligibility checker to applicants online.")

# ----------------------------------------------------------------------------- fairness
with tabs[5]:
    st.subheader("Fairness audit")
    st.write("Sex, age, race and ethnicity were **never given to the model**. This audit checks whether its decisions still differ between "
             "groups, which can happen through income or neighbourhood. It compares the model with the lenders' real decisions.")
    labels = {"sex": "Sex of applicant(s)", "age_band": "Age", "race": "Race", "ethnicity": "Ethnicity"}
    for key, title in labels.items():
        f = pd.DataFrame(M["fairness"][key])
        if f.empty:
            continue
        g = f.melt(id_vars="group", value_vars=["actual_approval_rate", "model_approval_rate"], var_name="Source", value_name="rate")
        g["Source"] = g["Source"].map({"actual_approval_rate": "Lenders (actual)", "model_approval_rate": "Model"})
        g["rate"] *= 100
        fig = px.bar(g, x="group", y="rate", color="Source", barmode="group", text_auto=".1f", title=f"Approval rate by {title.lower()}",
                     labels={"rate": "Approved (%)", "group": ""})
        a, b = st.columns([3, 2])
        with a:
            show(fig, 340)
        with b:
            ratio = f["model_approval_rate"].min() / f["model_approval_rate"].max()
            st.metric("Lowest ÷ highest model approval rate", f"{ratio:.2f}", "passes the 80% rule" if ratio >= 0.8 else "below the 80% rule",
                      delta_color="normal" if ratio >= 0.8 else "inverse")
            st.dataframe(f.rename(columns={"group": title, "actual_approval_rate": "actual", "model_approval_rate": "model",
                                           "wrongly_denied_rate": "wrongly denied"}), width="stretch", hide_index=True)
    st.caption("The 80% rule is a common screening test: the group with the lowest approval rate should be at least 80% of the highest. "
               "Gaps in the model mirror gaps already present in the lenders' decisions; they are a reason for review, not proof of bias.")
