"""Cleaning and feature engineering for the loan approval task (shared by train.py and the app)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "data" / "loans" / "hmda_co_2023.csv.gz"

LOAN_TYPE = {"1": "Conventional", "2": "FHA", "3": "VA", "4": "USDA / FSA"}
LOAN_PURPOSE = {"1": "Home purchase", "2": "Home improvement", "31": "Refinance", "32": "Cash-out refinance",
                "4": "Other purpose", "5": "Not applicable"}
LIEN = {"1": "First lien", "2": "Second lien"}
OCCUPANCY = {"1": "Main home", "2": "Second home", "3": "Investment property"}
DWELLING = {"Single Family (1-4 Units):Site-Built": "House (site-built)",
            "Single Family (1-4 Units):Manufactured": "Manufactured home"}
DENIAL = {"1": "Debt-to-income ratio", "2": "Employment history", "3": "Credit history", "4": "Collateral",
          "5": "Insufficient cash", "6": "Unverifiable information", "7": "Credit application incomplete",
          "8": "Mortgage insurance denied", "9": "Other"}
# HMDA publishes debt-to-income in bands; use each band's midpoint
DTI_BANDS = {"<20%": 15.0, "20%-<30%": 25.0, "30%-<36%": 33.0, "50%-60%": 55.0, ">60%": 65.0}
AGE_ORDER = ["<25", "25-34", "35-44", "45-54", "55-64", "65-74", ">74"]

NUMERIC = ["loan_amount", "income", "property_value", "loan_term", "dti", "ltv", "loan_to_income",
           "payment_to_income", "tract_minority_pct", "tract_income_pct", "area_median_income"]
CATEGORICAL = ["loan_type", "loan_purpose", "lien", "occupancy", "dwelling", "credit_line", "co_applicant"]
FEATURES = NUMERIC + CATEGORICAL
TARGET = "approved"
# Recorded for the fairness audit only. Never given to the model (fair-lending law forbids it).
PROTECTED = ["sex", "age_band", "race", "ethnicity"]


def monthly_payment(amount, term_months, annual_rate: float = 0.068):
    """Standard amortising payment. 6.8% was the average US 30-year mortgage rate in 2023."""
    r = annual_rate / 12
    n = np.where(pd.isna(term_months) | (term_months <= 0), 360, term_months)
    return amount * r / (1 - (1 + r) ** (-n))


def engineer(d: pd.DataFrame) -> pd.DataFrame:
    """Ratios a loan officer looks at. Works on a cleaned frame or a single application."""
    d = d.copy()
    d["loan_to_income"] = d["loan_amount"] / d["income"]
    d["payment_to_income"] = monthly_payment(d["loan_amount"], d["loan_term"]) * 12 / d["income"]
    implied = (d["loan_amount"] / d["property_value"] * 100).where(lambda v: (v > 0) & (v <= 150))
    d["ltv"] = d["ltv"].fillna(implied) if "ltv" in d else implied
    return d


def load_clean() -> tuple[pd.DataFrame, dict]:
    """Read the HMDA extract and return (clean frame, cleaning report)."""
    raw = pd.read_csv(DATA_FILE, dtype=str)
    rep = {"rows_raw": len(raw)}
    d = raw.drop_duplicates()
    rep["duplicates_removed"] = len(raw) - len(d)          # no application ID is published, so only exact repeats

    # scope: consumer mortgages on 1-4 unit homes, reported in full
    exempt = d["business_or_commercial_purpose"].eq("1111")
    business = d["business_or_commercial_purpose"].eq("1")
    multifamily = ~d["derived_dwelling_category"].isin(DWELLING)
    rep["removed_partially_exempt_reporters"] = int(exempt.sum())
    rep["removed_business_loans"] = int((business & ~exempt).sum())
    rep["removed_multifamily"] = int((multifamily & ~exempt & ~business).sum())
    d = d[~(exempt | business | multifamily)].copy()

    num = lambda c: pd.to_numeric(d[c], errors="coerce")
    out = pd.DataFrame(index=d.index)
    out[TARGET] = d["action_taken"].isin(["1", "2"]).astype(int)     # 1/2 = approved, 3 = denied
    out["loan_amount"] = num("loan_amount")
    out["income"] = num("income") * 1000                             # published in $ thousands
    out["property_value"] = num("property_value")
    out["loan_term"] = num("loan_term")
    out["dti"] = d["debt_to_income_ratio"].map(DTI_BANDS).fillna(num("debt_to_income_ratio"))
    out["ltv"] = num("loan_to_value_ratio")
    out["tract_minority_pct"] = num("tract_minority_population_percent")
    out["tract_income_pct"] = num("tract_to_msa_income_percentage")
    out["area_median_income"] = num("ffiec_msa_md_median_family_income")
    out["loan_type"] = d["loan_type"].map(LOAN_TYPE)
    out["loan_purpose"] = d["loan_purpose"].map(LOAN_PURPOSE)
    out["lien"] = d["lien_status"].map(LIEN)
    out["occupancy"] = d["occupancy_type"].map(OCCUPANCY)
    out["dwelling"] = d["derived_dwelling_category"].map(DWELLING)
    out["credit_line"] = np.where(d["open-end_line_of_credit"].eq("1"), "Credit line (HELOC)", "Fixed loan")
    out["co_applicant"] = np.where(d["co-applicant_age"].eq("9999"), "Single applicant", "Joint application")
    out["sex"] = d["derived_sex"].replace("Sex Not Available", np.nan)
    out["age_band"] = d["applicant_age"].where(d["applicant_age"].isin(AGE_ORDER))
    out["race"] = d["derived_race"].where(~d["derived_race"].isin(["Race Not Available", "Free Form Text Only"]))
    out["ethnicity"] = d["derived_ethnicity"].where(~d["derived_ethnicity"].isin(["Ethnicity Not Available", "Free Form Text Only"]))
    out["county"] = d["county_code"]
    out["denial_reason"] = d["denial_reason-1"].map(DENIAL)

    # validation: impossible values become missing rather than silently staying in
    fixes = {
        "income_non_positive": out["income"] <= 0,
        "ltv_out_of_range": (out["ltv"] <= 0) | (out["ltv"] > 150),
        "dti_out_of_range": (out["dti"] <= 0) | (out["dti"] > 100),
        "loan_term_out_of_range": (out["loan_term"] <= 0) | (out["loan_term"] > 480),
        "area_income_zero": out["area_median_income"] <= 0,
    }
    for name, mask in fixes.items():
        col = {"income_non_positive": "income", "ltv_out_of_range": "ltv", "dti_out_of_range": "dti",
               "loan_term_out_of_range": "loan_term", "area_income_zero": "area_median_income"}[name]
        rep[f"invalid_{name}"] = int(mask.sum())
        out.loc[mask, col] = np.nan
    bad = out["loan_amount"].isna() | (out["loan_amount"] <= 0)
    rep["removed_no_loan_amount"] = int(bad.sum())
    out = out[~bad]
    # extreme incomes / amounts are capped at the 99.9th percentile (reporting errors, not typical applicants)
    for c in ("income", "loan_amount", "property_value"):
        cap = out[c].quantile(0.999)
        rep[f"capped_{c}"] = int((out[c] > cap).sum())
        out[c] = out[c].clip(upper=cap)

    out = engineer(out)
    out.loc[~np.isfinite(out["loan_to_income"]), "loan_to_income"] = np.nan
    out.loc[~np.isfinite(out["payment_to_income"]), "payment_to_income"] = np.nan
    rep["rows_clean"] = len(out)
    rep["missing_by_feature_pct"] = (out[FEATURES].isna().mean() * 100).round(2).to_dict()
    return out.reset_index(drop=True), rep
