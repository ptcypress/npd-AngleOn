# pages/Pile_Pull_Results.py
# ---------------------------------------------------------------
# Streamlit page: AngleOn™ Pile Pull Results
# - Load data from DOCX (tables) or CSV
# - Clean into a tidy DataFrame
# - Summary stats + outlier flags (IQR)
# - Plotly visuals (histogram, box, scatter)
# - Download cleaned CSV, expandable raw table
# ---------------------------------------------------------------

import io
import sys
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Pile Pull Results", layout="wide")

# ---------------------------
# Helpers
# ---------------------------
def _ensure_float(s):
    """Coerce to float, stripping units/brackets if needed."""
    if pd.isna(s):
        return np.nan
    if isinstance(s, (int, float)):
        return float(s)
    s = str(s).strip()
    # Remove common non-numeric bits
    for junk in ["[lbf]", "(lbf)", "[in]", "(in)", ","]:
        s = s.replace(junk, "")
    try:
        return float(s)
    except ValueError:
        return np.nan

def parse_docx(file_bytes: bytes) -> pd.DataFrame:
    """
    Parse a simple DOCX (like this report) that contains tables with:
      Specimen (index), Type (string), Maximum Load [lbf], Displacement at Maximum Load [in]
    Requires python-docx. If not installed, instruct user.
    """
    try:
        import docx  # python-docx
    except Exception:
        st.error(
            "To read .docx directly, install **python-docx** in your app environment:\n\n"
            "`pip install python-docx`"
        )
        return pd.DataFrame()

    doc = docx.Document(io.BytesIO(file_bytes))
    rows = []
    for tbl in doc.tables:
        # Heuristic: read all data-like rows (skip obvious header separators)
        for r in tbl.rows:
            vals = [c.text.strip() for c in r.cells]
            # Expect rows that look like: [Specimen#, Type, Max, Disp] (4 columns) OR
            # sometimes merged cells; we'll try to coerce when possible.
            if len(vals) >= 4:
                # Try to detect "Specimen" as the first cell being an integer-like index
                idx_raw, typ, max_raw, disp_raw = vals[0], vals[1], vals[2], vals[3]
                # Skip header-like lines
                headerish = {"specimen", "specimen type", "maximum load", "displacement", "displacement at maximum load"}
                if any(h.lower() in (idx_raw + " " + typ + " " + max_raw + " " + disp_raw).lower() for h in headerish):
                    continue
                # Coerce
                try:
                    specimen = int(str(idx_raw).strip().split()[0])
                except Exception:
                    # If it doesn't look like a specimen index, skip
                    continue
                max_lbf = _ensure_float(max_raw)
                disp_in = _ensure_float(disp_raw)
                rows.append(
                    {
                        "Specimen": specimen,
                        "Type": typ,
                        "Max Load (lbf)": max_lbf,
                        "Disp @ Max (in)": disp_in,
                    }
                )
    df = pd.DataFrame(rows).sort_values("Specimen").reset_index(drop=True)
    return df

def parse_csv(file_bytes: bytes) -> pd.DataFrame:
    df = pd.read_csv(io.BytesIO(file_bytes))
    # Try to map potential column name variants
    cols = {c.lower(): c for c in df.columns}
    def col_like(options):
        for o in options:
            if o in cols:
                return cols[o]
        return None

    spec_col = col_like(["specimen", "id", "index", "#"])
    type_col = col_like(["type", "specimen type"])
    max_col  = col_like(["maximum load", "max load", "max load (lbf)", "maximum load [lbf]"])
    disp_col = col_like(["displacement at maximum load", "disp @ max", "disp @ max (in)", "displacement at maximum load [in]"])

    # If not found, fall back to best-guess
    if max_col is None:
        max_col = df.columns[1] if df.shape[1] > 1 else df.columns[0]
    if disp_col is None and df.shape[1] > 2:
        disp_col = df.columns[2]
    if spec_col is None:
        # Create a 1..N specimen index if missing
        df["Specimen"] = np.arange(1, len(df) + 1)
        spec_col = "Specimen"
    if type_col is None:
        df["Type"] = ""
        type_col = "Type"

    out = pd.DataFrame(
        {
            "Specimen": df[spec_col],
            "Type": df[type_col],
            "Max Load (lbf)": df[max_col].map(_ensure_float),
            "Disp @ Max (in)": df[disp_col].map(_ensure_float) if disp_col in df.columns else np.nan,
        }
    )
    out = out.sort_values("Specimen").reset_index(drop=True)
    return out

def compute_summary(df: pd.DataFrame) -> pd.DataFrame:
    stats = {}
    for col in ["Max Load (lbf)", "Disp @ Max (in)"]:
        s = pd.to_numeric(df[col], errors="coerce")
        stats[col] = {
            "n": int(s.notna().sum()),
            "mean": float(s.mean()) if s.notna().any() else np.nan,
            "std": float(s.std(ddof=1)) if s.notna().sum() > 1 else np.nan,
            "min": float(s.min()) if s.notna().any() else np.nan,
            "p25": float(s.quantile(0.25)) if s.notna().any() else np.nan,
            "median": float(s.median()) if s.notna().any() else np.nan,
            "p75": float(s.quantile(0.75)) if s.notna().any() else np.nan,
            "max": float(s.max()) if s.notna().any() else np.nan,
        }
    return pd.DataFrame(stats).T

def flag_outliers_iqr(df: pd.DataFrame, col: str) -> pd.Series:
    x = pd.to_numeric(df[col], errors="coerce")
    q1, q3 = x.quantile(0.25), x.quantile(0.75)
    iqr = q3 - q1
    low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    return (x < low) | (x > high)

def make_hist(df: pd.DataFrame, col: str):
    fig = go.Figure()
    fig.add_histogram(x=df[col], nbinsx=12, name=col, opacity=0.9)
    fig.update_layout(
        title=f"Distribution — {col}",
        xaxis_title=col,
        yaxis_title="Count",
        bargap=0.05,
        height=420,
        margin=dict(l=40, r=20, t=60, b=40),
    )
    return fig

def make_box(df: pd.DataFrame, col: str, by_type: bool):
    if by_type and "Type" in df.columns and df["Type"].nunique() > 1:
        fig = go.Figure()
        for typ, g in df.groupby("Type"):
            fig.add_trace(go.Box(y=g[col], name=str(typ), boxmean=True))
        title = f"Box Plot by Type — {col}"
    else:
        fig = go.Figure(data=[go.Box(y=df[col], name=col, boxmean=True)])
        title = f"Box Plot — {col}"
    fig.update_layout(
        title=title,
        yaxis_title=col,
        height=420,
        margin=dict(l=40, r=20, t=60, b=40),
    )
    return fig

def make_scatter(df: pd.DataFrame):
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["Disp @ Max (in)"],
            y=df["Max Load (lbf)"],
            mode="markers+text",
            text=df["Specimen"],
            textposition="top center",
            name="Specimen",
        )
    )
    fig.update_layout(
        title="Max Load vs Displacement @ Max",
        xaxis_title="Disp @ Max (in)",
        yaxis_title="Max Load (lbf)",
        height=460,
        margin=dict(l=40, r=20, t=60, b=40),
    )
    return fig

# ---------------------------
# Header
# ---------------------------
st.title("Pile Pull Results — AngleOn™")
st.caption(
    "Upload a **DOCX** (report tables) or **CSV** with columns for *Specimen, Type, Max Load (lbf), Disp @ Max (in)*. "
    "This page summarizes results, flags outliers, and provides clean visuals + CSV export."
)

# ---------------------------
# Controls (left-aligned, no sidebar)
# ---------------------------
c1, c2, c3, c4 = st.columns([2, 2, 2, 3], vertical_alignment="center")
with c1:
    plot_kind = st.selectbox("Plot", ["Histogram", "Box", "Scatter (Load vs Disp)"])
with c2:
    metric_col = st.selectbox("Metric", ["Max Load (lbf)", "Disp @ Max (in)"])
with c3:
    by_type = st.checkbox("By Type (for Box)", value=False)
with c4:
    show_outliers = st.checkbox("Flag Outliers (IQR)", value=True)

st.markdown("---")

# ---------------------------
# Data Input
# ---------------------------
up = st.file_uploader("Upload DOCX or CSV", type=["docx", "csv"])

df = pd.DataFrame()
source_label = ""

if up is not None:
    file_bytes = up.read()
    if up.name.lower().endswith(".docx"):
        df = parse_docx(file_bytes)
        source_label = f"Loaded from DOCX: {up.name}"
    else:
        df = parse_csv(file_bytes)
        source_label = f"Loaded from CSV: {up.name}"
else:
    # Fallback: minimal sample extracted from your report to make the page immediately useful
    sample = [
        # Specimen, Type, Max, Disp
        (1,  "LC306064-42PN BK 100R", 34.28, 0.22),
        (2,  "LC306064-42PN BK 100R", 28.15, 0.25),
        (3,  "LC306064-42PN BK 100R", 40.83, 0.10),
        (4,  "LC306064-42PN BK 100R", 33.11, 0.07),
        (5,  "LC306064-42PN BK 100R", 26.07, 0.10),
        (6,  "LC306064-42PN BK 100R", 33.35, 0.09),
        (7,  "LC306064-42PN BK 100R", 38.23, 0.17),
        (8,  "LC306064-42PN BK 100R", 33.32, 0.11),
        (9,  "LC306064-42PN BK 100R", 40.61, 0.08),
        (10, "LC306064-42PN BK 100R", 26.18, 0.09),
        (11, "LC306064-42PN BK 100R", 33.67, 0.08),
        (12, "LC306064-42PN BK 100R", 36.89, 0.09),
        (13, "LC306064-42PN BK 100R", 30.44, 0.26),
        (14, "LC306064-42PN BK 100R", 34.77, 0.13),
        (15, "LC306064-42PN BK 100R", 29.99, 0.12),
        (16, "LC306064-42PN BK 100R", 29.25, 0.08),
        (17, "LC306064-42PN BK 100R", 34.98, 0.07),
        (18, "LC306064-42PN BK 100R", 26.10, 0.15),
        (19, "LC306064-42PN BK 100R", 34.29, 0.08),
        (20, "LC306064-42PN BK 100R", 35.80, 0.08),
    ]
    df = pd.DataFrame(sample, columns=["Specimen", "Type", "Max Load (lbf)", "Disp @ Max (in)"])
    source_label = "Sample embedded (replace by uploading your DOCX/CSV above)"

# Clean final types
df["Specimen"] = pd.to_numeric(df["Specimen"], errors="coerce").astype("Int64")
df["Max Load (lbf)"] = pd.to_numeric(df["Max Load (lbf)"], errors="coerce")
df["Disp @ Max (in)"] = pd.to_numeric(df["Disp @ Max (in)"], errors="coerce")

# Outlier flags (if enabled)
if show_outliers and len(df) > 3:
    df["Outlier (Max Load)"] = flag_outliers_iqr(df, "Max Load (lbf)")
    df["Outlier (Disp)"] = flag_outliers_iqr(df, "Disp @ Max (in)")

st.caption(source_label)

# ---------------------------
# Summary & Plots
# ---------------------------
sum_df = compute_summary(df)
m1, m2, m3, m4 = st.columns(4)
with m1: st.metric("n (Max Load)", int(sum_df.loc["Max Load (lbf)", "n"]))
with m2: st.metric("Mean Max Load (lbf)", f"{sum_df.loc['Max Load (lbf)', 'mean']:.2f}")
with m3: st.metric("Mean Disp @ Max (in)", f"{sum_df.loc['Disp @ Max (in)', 'mean']:.3f}")
with m4: st.metric("Max of Max Load (lbf)", f"{sum_df.loc['Max Load (lbf)', 'max']:.2f}")

st.markdown("")

# Choose and render plot
if plot_kind == "Histogram":
    fig = make_hist(df, metric_col)
elif plot_kind == "Box":
    fig = make_box(df, metric_col, by_type=by_type)
else:
    fig = make_scatter(df)

st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": True})

# ---------------------------
# Data Export & Details
# ---------------------------
clean_csv = df.to_csv(index=False).encode("utf-8")
st.download_button("Download cleaned CSV", data=clean_csv, file_name="pile_pull_results_clean.csv", mime="text/csv")

with st.expander("Show data table"):
    st.dataframe(df, use_container_width=True)
