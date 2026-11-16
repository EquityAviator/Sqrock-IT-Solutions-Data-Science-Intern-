"""
TASK 2 - AI DATA ANALYSIS DASHBOARD : analysis engine
Sqrock IT Solutions - Data Science Internship - Phase 2

Everything the dashboard knows about a dataset is computed here, with no UI code,
so it can be tested and reused:

  read_dataset()   validate + load CSV / Excel / JSON
  clean()          type repair, duplicates, missing values, with a log of every operation
  profile()        role of each column (numeric, category, date, id, text)
  overview()       shape, dtypes, missing values, statistics
  insights()       automatic findings: trends, drivers, concentration, correlations,
                   anomalies (IQR rule + Isolation Forest), data-quality warnings,
                   each with a plain-language recommendation
  build_report()   self-contained HTML report

The "AI" is statistics and machine learning, not hand-written text per dataset:
significance tests decide what is worth reporting and an Isolation Forest finds
unusual rows across all numeric columns at once.
"""
from __future__ import annotations

import html
import io
import json
import re
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import IsolationForest

MAX_FILE_MB = 200
ALLOWED = (".csv", ".tsv", ".txt", ".xlsx", ".xls", ".json")
TRUE_WORDS, FALSE_WORDS = {"yes", "y", "true", "t", "1"}, {"no", "n", "false", "f", "0"}


# ----------------------------------------------------------------------------- loading
class DatasetError(ValueError):
    """Raised with a message that is safe to show to the user."""


def read_dataset(source, filename: str) -> pd.DataFrame:
    """Validate and load an uploaded file (path, bytes or file-like)."""
    name = filename.lower()
    if not name.endswith(ALLOWED):
        raise DatasetError(f"Unsupported file type. Please upload one of: {', '.join(ALLOWED)}")
    raw = source if isinstance(source, (bytes, bytearray)) else (
        open(source, "rb").read() if isinstance(source, str) else source.read())
    if len(raw) == 0:
        raise DatasetError("The file is empty.")
    if len(raw) > MAX_FILE_MB * 1024 * 1024:
        raise DatasetError(f"The file is larger than {MAX_FILE_MB} MB.")
    try:
        if name.endswith((".xlsx", ".xls")):
            df = pd.read_excel(io.BytesIO(raw))
        elif name.endswith(".json"):
            data = json.loads(raw.decode("utf-8-sig"))
            if isinstance(data, dict):                    # {"records": [...]} style wrappers
                lists = [v for v in data.values() if isinstance(v, list)]
                data = lists[0] if lists else [data]
            df = pd.json_normalize(data)
        else:
            text = None
            for enc in ("utf-8-sig", "cp1252", "latin-1"):
                try:
                    text = raw.decode(enc)
                    break
                except UnicodeDecodeError:
                    continue
            head = text[:20000]
            counts = {sep: min((ln.count(sep) for ln in head.splitlines()[:30] if ln.strip()), default=0)
                      for sep in (",", ";", "\t", "|")}
            sep = max(counts, key=counts.get) if max(counts.values()) > 0 else ","   # present on every line
            df = pd.read_csv(io.StringIO(text), sep=sep)
    except DatasetError:
        raise
    except Exception as exc:
        raise DatasetError(f"The file could not be read as a table: {str(exc)[:160]}") from exc
    if df.shape[1] < 2:
        raise DatasetError("Only one column was found. Check the file's separator.")
    if len(df) < 2:
        raise DatasetError("The file has fewer than two data rows.")
    return df


# ----------------------------------------------------------------------------- cleaning
@dataclass
class CleanResult:
    df: pd.DataFrame
    log: list[dict] = field(default_factory=list)       # one entry per operation performed

    def add(self, step: str, detail: str, affected: int) -> None:
        self.log.append({"Operation": step, "Detail": detail, "Affected": int(affected)})


def clean(raw: pd.DataFrame, missing: str = "impute", drop_duplicates: bool = True,
          normalise: bool = False) -> CleanResult:
    """Clean a dataset. `missing` is 'impute' (median / most frequent), 'drop' or 'keep'."""
    df = raw.copy()
    res = CleanResult(df)

    new_cols = [str(c).strip() for c in df.columns]
    if new_cols != list(map(str, df.columns)):
        res.add("Format column names", "trimmed spaces from headers", sum(a != str(b) for a, b in zip(new_cols, df.columns)))
    df.columns = new_cols
    unnamed = [c for c in df.columns if c.lower().startswith("unnamed:")]
    empty_cols = [c for c in df.columns if df[c].isna().all()]
    drop = sorted(set(empty_cols) | {c for c in unnamed if df[c].isna().mean() > 0.99
                                      or pd.api.types.is_integer_dtype(df[c]) and df[c].is_monotonic_increasing})
    if drop:
        df = df.drop(columns=drop)
        res.add("Remove empty / index columns", ", ".join(drop[:6]), len(drop))

    n = len(df)
    df = df.dropna(how="all")
    if len(df) < n:
        res.add("Remove empty rows", "rows with no values at all", n - len(df))

    # text clean-up and type repair
    for c in df.select_dtypes(include=["object", "string"]).columns:
        s = df[c].astype("string").str.strip()
        s = s.mask(s.str.lower().isin(["", "na", "n/a", "nan", "null", "none", "?", "-", "unknown"]))
        nonnull = s.dropna()
        if nonnull.empty:
            df[c] = s
            continue
        as_num = pd.to_numeric(nonnull.str.replace(r"[,$€£%\s]", "", regex=True), errors="coerce")
        if as_num.notna().mean() >= 0.95:
            df[c] = pd.to_numeric(s.str.replace(r"[,$€£%\s]", "", regex=True), errors="coerce")
            res.add("Convert text to number", c, int(nonnull.size))
            continue
        if nonnull.str.contains(r"\d", regex=True).mean() > 0.8 and nonnull.str.len().median() >= 6:
            as_date = pd.to_datetime(nonnull, errors="coerce", format="mixed")
            if as_date.notna().mean() >= 0.95:
                df[c] = pd.to_datetime(s, errors="coerce", format="mixed")
                res.add("Convert text to date", c, int(nonnull.size))
                continue
        df[c] = s.astype(object).where(s.notna(), np.nan)
    new_missing = int(df.isna().sum().sum() - raw.isna().sum().sum())
    if new_missing > 0:
        res.add("Standardise missing markers", "'unknown', 'n/a', '?', blanks -> missing", new_missing)

    if drop_duplicates:
        d = int(df.duplicated().sum())
        if d:
            df = df.drop_duplicates()
            res.add("Remove duplicate records", "exact duplicate rows", d)

    miss = df.isna().sum()
    miss = miss[miss > 0]
    if len(miss) and missing == "drop":
        n = len(df)
        df = df.dropna()
        res.add("Handle missing values", "dropped rows with any missing value", n - len(df))
    elif len(miss) and missing == "impute":
        for c, k in miss.items():
            if pd.api.types.is_numeric_dtype(df[c]):
                df[c] = df[c].fillna(df[c].median())
                res.add("Handle missing values", f"{c}: filled with median", k)
            elif pd.api.types.is_datetime64_any_dtype(df[c]):
                res.add("Handle missing values", f"{c}: dates left missing (not guessable)", 0)
            else:
                mode = df[c].mode(dropna=True)
                df[c] = df[c].fillna(mode.iat[0] if len(mode) else "Missing")
                res.add("Handle missing values", f"{c}: filled with most frequent value", k)

    if normalise:
        nums = [c for c in df.select_dtypes("number").columns if df[c].nunique() > 2]
        for c in nums:
            lo, hi = df[c].min(), df[c].max()
            if hi > lo:
                df[c] = (df[c] - lo) / (hi - lo)
        res.add("Normalise numeric columns", "min-max scaling to 0-1", len(nums))

    if not res.log:
        res.add("No cleaning needed", "the dataset was already tidy", 0)
    res.df = df.reset_index(drop=True)
    return res


# ----------------------------------------------------------------------------- profiling
def profile(df: pd.DataFrame) -> dict[str, list[str]]:
    """Sort columns into roles so charts and insights use them sensibly."""
    roles = {"numeric": [], "category": [], "date": [], "id": [], "text": [], "binary": []}
    n = len(df)
    for c in df.columns:
        s = df[c]
        u = s.nunique(dropna=True)
        if pd.api.types.is_datetime64_any_dtype(s):
            roles["date"].append(c)
        elif pd.api.types.is_bool_dtype(s):
            roles["binary"].append(c); roles["category"].append(c)
        elif pd.api.types.is_numeric_dtype(s):
            looks_id = u == n and pd.api.types.is_integer_dtype(s) and any(k in c.lower() for k in ("id", "code", "no", "number", "index"))
            if looks_id:
                roles["id"].append(c)
            elif u <= 2:
                roles["binary"].append(c); roles["category"].append(c)
            elif u <= 12 and pd.api.types.is_integer_dtype(s) and any(k in c.lower() for k in ("month", "year", "quarter", "day", "week", "class", "grade", "rating", "level")):
                roles["category"].append(c)
            else:
                roles["numeric"].append(c)
        else:
            if u == n and n > 50:
                roles["id"].append(c)
            elif u <= max(30, min(60, n // 20)):
                roles["category"].append(c)
                if u == 2:
                    roles["binary"].append(c)
            elif s.astype(str).str.len().mean() > 40:
                roles["text"].append(c)
            else:
                roles["category"].append(c) if u <= 200 else roles["text"].append(c)
    return roles


def overview(df: pd.DataFrame) -> dict:
    miss = df.isna().sum()
    cols = pd.DataFrame({
        "Column": df.columns, "Type": [str(t) for t in df.dtypes], "Unique": [df[c].nunique() for c in df.columns],
        "Missing": miss.values, "Missing %": (miss.values / max(len(df), 1) * 100).round(2),
        "Example": [str(df[c].dropna().iloc[0])[:40] if df[c].notna().any() else "" for c in df.columns],
    })
    num = df.select_dtypes("number")
    return {
        "rows": len(df), "columns": df.shape[1], "missing_cells": int(miss.sum()),
        "missing_pct": round(float(miss.sum() / max(df.size, 1) * 100), 2),
        "duplicates": int(df.duplicated().sum()), "memory_mb": round(df.memory_usage(deep=True).sum() / 1e6, 2),
        "columns_table": cols,
        "numeric_summary": num.describe().T.round(3) if num.shape[1] else pd.DataFrame(),
        "category_summary": pd.DataFrame([
            {"Column": c, "Unique": df[c].nunique(), "Most frequent": str(df[c].mode().iat[0]) if df[c].notna().any() else "",
             "Share %": round(float(df[c].value_counts(normalize=True).iloc[0] * 100), 1) if df[c].notna().any() else 0}
            for c in df.columns if not pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_datetime64_any_dtype(df[c])]),
    }


# ----------------------------------------------------------------------------- insights
def _fmt(x: float) -> str:
    ax = abs(x)
    if ax >= 1e9: return f"{x / 1e9:.2f}B"
    if ax >= 1e6: return f"{x / 1e6:.2f}M"
    if ax >= 1e4: return f"{x / 1e3:.1f}K"
    if ax >= 100: return f"{x:,.0f}"
    return f"{x:,.2f}"


def _as_binary(s: pd.Series) -> pd.Series | None:
    """Map a two-valued column to 0/1 with the 'positive' value as 1."""
    vals = s.dropna().unique()
    if len(vals) != 2:
        return None
    low = {str(v).strip().lower(): v for v in vals}
    pos = next((low[k] for k in low if k in TRUE_WORDS), None)
    if pos is None:
        pos = s.value_counts().index[-1]                  # the rarer outcome is usually the event of interest
    return (s == pos).astype(float).where(s.notna()), pos


def pick_measure(df: pd.DataFrame, roles: dict) -> str | None:
    """The numeric column that most looks like an additive business measure, if any."""
    words = ("sales", "revenue", "amount", "profit", "total", "price", "value", "income", "cost", "quantity", "qty", "units", "balance", "spend")
    # a measure varies from row to row; an index or rate that barely moves is not one
    nums = [c for c in roles["numeric"] if df[c].mean() != 0 and abs(df[c].std() / df[c].mean()) >= 0.15 and df[c].min() >= 0]
    for w in words:
        for c in nums:
            if w in re.split(r"[^a-z]+", c.lower()) or c.lower().endswith(w):
                return c
    return None


def pick_target(df: pd.DataFrame, roles: dict) -> str | None:
    """A yes/no outcome column, if the dataset has one."""
    words = ("target", "churn", "outcome", "result", "status", "approved", "response", "converted", "subscribed", "label", "class", "survived")
    # an outcome needs both values to be present in meaningful numbers
    cands = [c for c in roles["binary"] if df[c].value_counts(normalize=True).min() >= 0.01]
    for c in cands:
        if c.lower() in ("y", "target", "label", "outcome"):
            return c
    if cands and df.columns[-1] in cands:                 # by convention the outcome is the last column
        return df.columns[-1]
    for c in cands:
        if any(w in c.lower() for w in words):
            return c
    return None


def anomalies(df: pd.DataFrame, roles: dict, contamination: float = 0.01) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Column-wise outliers (IQR rule) and row-wise anomalies (Isolation Forest)."""
    rows = []
    for c in roles["numeric"]:
        s = df[c].dropna()
        q1, q3 = s.quantile([0.25, 0.75])
        iqr = q3 - q1
        if iqr <= 0:
            continue
        lo, hi = q1 - 3 * iqr, q3 + 3 * iqr              # 3 x IQR: extreme values only
        k = int(((s < lo) | (s > hi)).sum())
        if k:
            rows.append({"Column": c, "Outliers": k, "Share %": round(k / len(s) * 100, 2),
                         "Normal range": f"{_fmt(lo)} to {_fmt(hi)}", "Most extreme": _fmt(s.loc[(s - s.median()).abs().idxmax()])})
    col_out = pd.DataFrame(rows).sort_values("Outliers", ascending=False) if rows else pd.DataFrame()

    nums = [c for c in roles["numeric"] if df[c].notna().mean() > 0.9]
    flagged = pd.DataFrame()
    if len(nums) >= 2 and len(df) >= 50:
        X = df[nums].fillna(df[nums].median())
        X = (X - X.mean()) / X.std().replace(0, 1)
        iso = IsolationForest(n_estimators=200, contamination=contamination, random_state=42, n_jobs=2).fit(X)
        score = -iso.score_samples(X)
        idx = np.argsort(score)[::-1][: max(1, int(len(df) * contamination))]
        flagged = df.iloc[idx].copy()
        flagged.insert(0, "anomaly_score", score[idx].round(3))
        z = X.iloc[idx].abs()
        flagged.insert(1, "most_unusual_column", z.idxmax(axis=1).values)
    return col_out, flagged


def insights(df: pd.DataFrame, roles: dict | None = None, measure: str | None = "auto",
             target: str | None = "auto") -> list[dict]:
    """Automatic findings, most important first. Each has kind, level, title, detail, action.

    `measure` / `target` may name a column, be None (do not use one) or "auto" (detect).
    """
    roles = roles or profile(df)
    out: list[dict] = []
    add = lambda kind, level, title, detail, action, score: out.append(
        {"kind": kind, "level": level, "title": title, "detail": detail, "action": action, "score": float(score)})
    n = len(df)
    measure = pick_measure(df, roles) if measure == "auto" else measure
    target = pick_target(df, roles) if target == "auto" else target

    # --- outcome drivers (binary target) ------------------------------------------------
    if target:
        y, pos = _as_binary(df[target])
        base = y.mean()
        add("summary", "info", f"{base:.1%} of records have {target} = {pos}",
            f"{int(y.sum()):,} of {int(y.notna().sum()):,} records. This is the baseline every segment below is compared with.",
            "Use the baseline to judge any segment: only gaps well above or below it matter.", 95)
        for c in [c for c in roles["category"] if c != target and df[c].nunique() <= 25]:
            tab = pd.crosstab(df[c], y)
            if tab.shape[0] < 2 or tab.shape[1] < 2:
                continue
            chi2, p, _, _ = stats.chi2_contingency(tab)
            v = np.sqrt(chi2 / (tab.values.sum() * (min(tab.shape) - 1)))    # Cramer's V effect size
            g = df.assign(_y=y).groupby(c)["_y"].agg(["mean", "count"])
            g = g[g["count"] >= max(30, n * 0.005)]
            if p < 0.001 and v >= 0.08 and len(g) >= 2:
                hi, lo = g["mean"].idxmax(), g["mean"].idxmin()
                add("driver", "high" if v > 0.2 else "medium",
                    f"{c} strongly separates {target}: '{hi}' {g.loc[hi, 'mean']:.1%} vs '{lo}' {g.loc[lo, 'mean']:.1%}",
                    f"Baseline {base:.1%}. '{hi}' ({int(g.loc[hi, 'count']):,} records) is {g.loc[hi, 'mean'] / base:.1f}x the baseline. Association strength (Cramer's V) {v:.2f}.",
                    f"Prioritise the '{hi}' segment of {c} if {pos} is the goal, and investigate why '{lo}' underperforms.", 60 + v * 100)
        for c in roles["numeric"]:
            a, b = df.loc[y == 1, c].dropna(), df.loc[y == 0, c].dropna()
            if len(a) < 30 or len(b) < 30:
                continue
            pooled = np.sqrt((a.var() + b.var()) / 2)
            dcohen = (a.mean() - b.mean()) / pooled if pooled > 0 else 0
            if abs(dcohen) >= 0.3:
                add("driver", "high" if abs(dcohen) > 0.8 else "medium",
                    f"{c} is {'higher' if dcohen > 0 else 'lower'} when {target} = {pos} (average {_fmt(a.mean())} vs {_fmt(b.mean())})",
                    f"Effect size (Cohen's d) {dcohen:+.2f}; medians {_fmt(a.median())} vs {_fmt(b.median())}.",
                    f"{c} is a useful signal for predicting {target}. Check whether it is known before the outcome, otherwise it is a leak, not a lever.",
                    55 + min(abs(dcohen), 2) * 20)

    # --- time trends ---------------------------------------------------------------------
    for dcol in roles["date"][:1]:
        s = df[[dcol] + ([measure] if measure else [])].dropna(subset=[dcol])
        span = (s[dcol].max() - s[dcol].min()).days
        if span < 60:
            continue
        freq = "MS" if span > 180 else "W"
        ts = (s.set_index(dcol)[measure].resample(freq).sum() if measure else s.set_index(dcol).resample(freq).size())
        ts = ts.iloc[:-1] if len(ts) > 4 else ts          # the last period is usually incomplete
        if len(ts) >= 6:
            label = f"total {measure}" if measure else "record count"
            slope, _, r, p, _ = stats.linregress(np.arange(len(ts)), ts.values)
            k = max(1, len(ts) // 4)
            first, last = ts.iloc[:k].mean(), ts.iloc[-k:].mean()
            chg = (last / first - 1) if first else np.nan
            unit = "month" if freq == "MS" else "week"
            if p < 0.05 and abs(chg) > 0.05:
                add("trend", "high", f"{label.capitalize()} is {'growing' if slope > 0 else 'declining'}: {chg:+.0%} from the first to the last quarter of the period",
                    f"{len(ts)} {unit}s from {ts.index[0]:%b %Y} to {ts.index[-1]:%b %Y}. Average change {_fmt(slope)} per {unit} (trend fit R² {r * r:.2f}, p {p:.3f}).",
                    "Plan capacity and targets on the trend, not on the overall average." if slope > 0 else "Find what changed when the decline started; check the weakest categories first.", 85)
            else:
                add("trend", "info", f"{label.capitalize()} shows no reliable up or down trend",
                    f"{len(ts)} {unit}s analysed; first-to-last change {chg:+.0%}, not statistically significant (p {p:.2f}).",
                    "Treat period-to-period movements as noise unless they repeat.", 30)
            peak, low = ts.idxmax(), ts.idxmin()
            add("trend", "info", f"Best {unit}: {peak:%b %Y} ({_fmt(ts.max())}); weakest: {low:%b %Y} ({_fmt(ts.min())})",
                f"The best {unit} is {ts.max() / max(ts.min(), 1e-9):.1f}x the weakest.", "Compare what was different in the two periods (campaigns, stock, prices).", 40)
            if freq == "MS" and len(ts) >= 24:
                moy = ts.groupby(ts.index.month).mean()
                if moy.max() > 1.25 * moy.mean():
                    m = pd.Timestamp(2000, int(moy.idxmax()), 1).strftime("%B")
                    add("trend", "medium", f"Seasonal peak in {m}: {moy.max() / moy.mean() - 1:+.0%} above an average month",
                        "Based on the average of each calendar month across the years on record.", f"Build stock and staffing ahead of {m}.", 55)

    # --- concentration / top performers --------------------------------------------------
    if measure:
        for c in [c for c in roles["category"] if 3 <= df[c].nunique() <= 60 and c != target][:6]:
            g = df.groupby(c)[measure].sum().sort_values(ascending=False)
            if g.sum() <= 0:
                continue
            share = g / g.sum()
            k80 = int((share.cumsum() < 0.8).sum() + 1)
            add("performance", "medium" if share.iloc[0] > 0.3 else "info",
                f"Top {c}: '{g.index[0]}' with {share.iloc[0]:.0%} of total {measure}",
                f"{k80} of {len(g)} {c} values produce 80% of {measure}. Weakest: '{g.index[-1]}' ({share.iloc[-1]:.1%}).",
                (f"Dependence on '{g.index[0]}' is high; protect it and develop the next tier." if share.iloc[0] > 0.4
                 else f"Review the bottom {c} values: they add cost for little {measure}."), 45 + share.iloc[0] * 40)
            avg = df.groupby(c)[measure].agg(["mean", "count"])
            avg = avg[avg["count"] >= max(20, n * 0.01)]
            if len(avg) >= 3:
                groups = [df.loc[df[c] == k, measure].dropna() for k in avg.index]
                f, p = stats.f_oneway(*groups)
                if p < 0.001 and avg["mean"].max() > 1.3 * avg["mean"].min() > 0:
                    add("performance", "medium", f"Average {measure} differs by {c}: '{avg['mean'].idxmax()}' {_fmt(avg['mean'].max())} vs '{avg['mean'].idxmin()}' {_fmt(avg['mean'].min())}",
                        f"Differences are statistically significant (ANOVA p < 0.001) across {len(avg)} groups.",
                        f"Learn from '{avg['mean'].idxmax()}': what makes its average {measure} higher?", 50)

    else:
        for c in [c for c in roles["category"] if 3 <= df[c].nunique() <= 60 and c != target][:4]:
            share = df[c].value_counts(normalize=True)
            if share.iloc[0] >= 0.3:
                add("performance", "info", f"Largest {c} group: '{share.index[0]}' with {share.iloc[0]:.0%} of records",
                    f"{len(share)} groups in total; smallest is '{share.index[-1]}' ({share.iloc[-1]:.1%}).",
                    f"Results for small {c} groups rest on few records; read them with care.", 28 + share.iloc[0] * 20)

    # --- correlations ----------------------------------------------------------------------
    nums = roles["numeric"]
    if len(nums) >= 2:
        corr = df[nums].corr(numeric_only=True)
        pairs = [(a, b, corr.loc[a, b]) for i, a in enumerate(nums) for b in nums[i + 1:] if pd.notna(corr.loc[a, b])]
        for a, b, r in sorted(pairs, key=lambda t: -abs(t[2]))[:4]:
            if abs(r) >= 0.6:
                add("relationship", "medium" if abs(r) < 0.9 else "info",
                    f"{a} and {b} move {'together' if r > 0 else 'in opposite directions'} (correlation {r:+.2f})",
                    "Near-duplicate information." if abs(r) > 0.95 else f"About {r * r:.0%} of the variation in one is explained by the other.",
                    "Keep only one of the two in a predictive model." if abs(r) > 0.9 else f"Changes in {a} are a good early indicator for {b} (correlation is not proof of cause).",
                    35 + abs(r) * 25)

    # --- anomalies -------------------------------------------------------------------------
    col_out, flagged = anomalies(df, roles)
    for _, r in col_out.head(3).iterrows():
        add("anomaly", "medium" if r["Share %"] < 2 else "info", f"{r['Outliers']:,} extreme values in {r['Column']} ({r['Share %']}% of rows)",
            f"Outside the normal range {r['Normal range']}; most extreme value {r['Most extreme']}.",
            "Verify these records: they are either data-entry errors or your most important exceptions.", 42)
    if len(flagged):
        top_col = flagged["most_unusual_column"].mode().iat[0]
        add("anomaly", "medium", f"{len(flagged):,} records look unusual across several columns at once",
            f"Found by an Isolation Forest over {len([c for c in nums])} numeric columns; most often driven by {top_col}.",
            "Review the flagged rows in the Anomalies tab before trusting averages.", 44)

    # --- data quality ----------------------------------------------------------------------
    miss = (df.isna().mean() * 100).sort_values(ascending=False)
    for c, pct in miss[miss > 5].head(3).items():
        add("quality", "high" if pct > 30 else "medium", f"{c} is missing in {pct:.1f}% of rows",
            "Missing values this frequent can bias every statistic that uses the column.",
            f"Fix collection of {c} at the source, or exclude it from decisions.", 38 + pct / 3)
    for c in nums:
        sk = df[c].skew()
        if abs(sk) > 3 and df[c].min() >= 0:
            add("distribution", "info", f"{c} is highly skewed: a few very large values dominate",
                f"Mean {_fmt(df[c].mean())} vs median {_fmt(df[c].median())} (skewness {sk:.1f}).",
                f"Report the median of {c}, not the mean.", 25)
            break
    for c in roles["category"]:
        vc = df[c].value_counts(normalize=True)
        if len(vc) > 1 and vc.iloc[0] > 0.9:
            add("quality", "info", f"{c} is almost constant: {vc.iloc[0]:.0%} of rows are '{vc.index[0]}'",
                "A column with one dominant value carries little information.", f"Do not segment by {c}.", 15)

    if not out:
        add("summary", "info", "No strong patterns found", "The dataset has no significant trends, drivers or anomalies by the tests applied.",
            "Add more rows or more descriptive columns.", 1)
    return sorted(out, key=lambda d: -d["score"])


def executive_summary(df: pd.DataFrame, roles: dict, found: list[dict]) -> str:
    n_num, n_cat, n_date = len(roles["numeric"]), len(roles["category"]), len(roles["date"])
    parts = [f"The dataset has {len(df):,} records and {df.shape[1]} columns ({n_num} numeric, {n_cat} categorical, {n_date} date)."]
    high = [f for f in found if f["level"] == "high"]
    if high:
        parts.append("Most important: " + "; ".join(f["title"] for f in high[:3]) + ".")
    kinds = pd.Series([f["kind"] for f in found]).value_counts()
    parts.append("In total " + ", ".join(f"{v} {k} finding{'s' if v > 1 else ''}" for k, v in kinds.items()) + ".")
    return " ".join(parts)


# ----------------------------------------------------------------------------- report
def build_report(name: str, df: pd.DataFrame, clean_log: list[dict], found: list[dict], figures: list | None = None) -> str:
    """Self-contained HTML report (charts are embedded Plotly figures)."""
    roles = profile(df)
    ov = overview(df)
    e = html.escape
    badge = {"high": "#dc2626", "medium": "#d97706", "info": "#2563eb"}
    cards = "".join(
        f"<div class='f'><span class='b' style='background:{badge[f['level']]}'>{e(f['kind'])}</span>"
        f"<h3>{e(f['title'])}</h3><p>{e(f['detail'])}</p><p class='a'><b>Recommendation:</b> {e(f['action'])}</p></div>"
        for f in found)
    figs = ""
    for i, fig in enumerate(figures or []):
        figs += fig.to_html(full_html=False, include_plotlyjs="cdn" if i == 0 else False)
    table = lambda d: d.to_html(index=False, border=0, classes="t") if len(d) else "<p>None.</p>"
    return f"""<!doctype html><html><head><meta charset="utf-8"><title>Analysis report - {e(name)}</title>
<style>body{{font-family:Segoe UI,Arial,sans-serif;max-width:1000px;margin:32px auto;padding:0 20px;color:#0f172a;line-height:1.5}}
h1{{margin-bottom:0}}h2{{border-bottom:2px solid #e2e8f0;padding-bottom:6px;margin-top:36px}}.m{{color:#64748b}}
.k{{display:flex;gap:12px;flex-wrap:wrap}}.k div{{flex:1;min-width:140px;background:#f1f5f9;border-radius:10px;padding:12px 16px}}
.k b{{font-size:22px;display:block}}.f{{border:1px solid #e2e8f0;border-radius:10px;padding:12px 16px;margin:10px 0}}
.f h3{{margin:6px 0 4px;font-size:16px}}.f p{{margin:4px 0}}.a{{color:#334155}}.b{{color:#fff;font-size:11px;padding:2px 8px;border-radius:99px;text-transform:uppercase}}
.t{{border-collapse:collapse;font-size:13px;width:100%}}.t th,.t td{{border-bottom:1px solid #e2e8f0;padding:5px 8px;text-align:left}}</style></head><body>
<h1>Data analysis report</h1><p class="m">{e(name)} &middot; generated {datetime.now():%d %b %Y %H:%M}</p>
<h2>Executive summary</h2><p>{e(executive_summary(df, roles, found))}</p>
<h2>Dataset overview</h2><div class="k"><div><b>{ov['rows']:,}</b>rows</div><div><b>{ov['columns']}</b>columns</div>
<div><b>{ov['missing_pct']}%</b>missing cells</div><div><b>{ov['duplicates']:,}</b>duplicate rows</div></div>
<h3>Columns</h3>{table(ov['columns_table'])}<h3>Statistical summary</h3>{ov['numeric_summary'].reset_index(names='Column').to_html(index=False, border=0, classes='t') if len(ov['numeric_summary']) else '<p>No numeric columns.</p>'}
<h2>Preprocessing performed</h2>{table(pd.DataFrame(clean_log))}
<h2>Key findings and recommendations</h2>{cards}
<h2>Visualisations</h2>{figs or '<p>No charts selected.</p>'}
</body></html>"""
