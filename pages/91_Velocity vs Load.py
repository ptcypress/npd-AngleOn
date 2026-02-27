import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from scipy.integrate import quad

# =========================
# Streamlit Page Setup
# =========================
st.set_page_config(page_title="Velocity vs Load", layout="wide")
st.title("Object Velocity vs Load — AngleOn™ vs Competitor")

st.caption("""
This is a benchmarking time-study on a linear vibratory feeder. Objects of different weights are placed on each brush and timed over a fixed travel distance.
Measured time is converted to velocity, and object load is expressed as pressure (lb/in²). The purpose is not just “speed”, but understanding how stable/robust
each brush is as load increases (how gracefully performance degrades and where it transitions toward stall).
""")

# =========================
# Data Load
# =========================
csv_path = "data/velocity_data.csv"
poly_degree = 3  # cubic
line_width = 2   # fixed line width

if not os.path.exists(csv_path):
    st.error(f"No CSV found at `{csv_path}`. Please place your file there.")
    st.stop()

df = pd.read_csv(csv_path)
df.columns = df.columns.str.strip()

required_cols = ["Brush", "Pressure", "Velocity"]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    st.error(f"Missing required columns: {', '.join(missing)}")
    st.stop()

# =========================
# Sidebar Controls
# =========================
st.sidebar.header("Controls")

axis_choice = st.sidebar.radio(
    "X-axis",
    options=["Weight (lb)", "Pressure (lb/in²)"],
    index=1  # default to Pressure since bands are pressure-based
)

x_col = "Weight" if axis_choice.startswith("Weight") else "Pressure"
x_units = "lb" if x_col == "Weight" else "lb/in²"

if x_col not in df.columns:
    st.error(f"Column '{x_col}' not found in the CSV. Available columns: {list(df.columns)}")
    st.stop()

# =========================
# Load Bands (Pressure)
# =========================
st.sidebar.subheader("Load bands (Pressure, lb/in²)")
use_bands = st.sidebar.checkbox("Show load bands (Pressure plots only)", value=True)

P_normal_lo = st.sidebar.number_input("Normal band start", value=0.60, step=0.05)
P_normal_hi = st.sidebar.number_input("Normal band end", value=1.30, step=0.05)

P_trans_lo = st.sidebar.number_input("Transient band start", value=1.30, step=0.05)
P_trans_hi = st.sidebar.number_input("Transient band end", value=2.00, step=0.05)

P_avoid_lo = st.sidebar.number_input("Avoid band start", value=2.00, step=0.05)

# =========================
# Helper functions
# =========================
def fit_poly_model(x, y, degree):
    poly = PolynomialFeatures(degree=degree)
    X_poly = poly.fit_transform(x.reshape(-1, 1))
    model = LinearRegression().fit(X_poly, y)

    def f(x_scalar):
        return float(model.predict(poly.transform(np.array([[x_scalar]])))[0])

    return f

def safe_quad(func, a, b):
    if a == b:
        return 0.0
    lo, hi = min(a, b), max(a, b)
    val, _ = quad(func, lo, hi)
    return val

def common_domain(x1, x2):
    xmin = float(max(np.nanmin(x1), np.nanmin(x2)))
    xmax = float(min(np.nanmax(x1), np.nanmax(x2)))
    if xmax < xmin:
        raise ValueError("No overlapping domain between the two selected brushes.")
    return xmin, xmax

def clipped(func, lo, hi):
    def g(x):
        if x < lo:
            x = lo
        if x > hi:
            x = hi
        return func(x)
    return g

def add_pressure_bands(fig, x_col, show, x_max, Pn_lo, Pn_hi, Pt_lo, Pt_hi, Pa_lo):
    """
    Add shaded load bands to Plotly figure when x-axis is Pressure.
    x_max should be the max x shown on the plot so the 'Avoid' band can extend to the edge.
    """
    if (not show) or (x_col != "Pressure"):
        return fig

    # Normal band (green)
    fig.add_vrect(
        x0=Pn_lo, x1=Pn_hi,
        fillcolor="rgba(0, 200, 0, 0.14)",
        line_width=0,
        layer="below",
        annotation_text="Normal",
        annotation_position="top left",
        annotation_font_size=11
    )

    # Transient band (yellow)
    fig.add_vrect(
        x0=Pt_lo, x1=Pt_hi,
        fillcolor="rgba(255, 215, 0, 0.14)",
        line_width=0,
        layer="below",
        annotation_text="Transient",
        annotation_position="top left",
        annotation_font_size=11
    )

    # Avoid band (red) — extend to the plotted max
    fig.add_vrect(
        x0=Pa_lo, x1=float(x_max),
        fillcolor="rgba(255, 0, 0, 0.08)",
        line_width=0,
        layer="below",
        annotation_text="Avoid",
        annotation_position="top left",
        annotation_font_size=11
    )

    return fig

# =========================
# Prep data (AngleOn™ vs Competitor baseline)
# =========================
angleon = df[df["Brush"].str.strip() == "AngleOn™"].copy()
competitor = df[df["Brush"].str.strip() == "Competitor"].copy()

if angleon.empty or competitor.empty:
    st.error("Need rows for both 'AngleOn™' and 'Competitor' in the CSV.")
    st.stop()

x_angleon = angleon[x_col].to_numpy(dtype=float)
y_angleon = angleon["Velocity"].to_numpy(dtype=float)

x_comp = competitor[x_col].to_numpy(dtype=float)
y_comp = competitor["Velocity"].to_numpy(dtype=float)

f_angleon = fit_poly_model(x_angleon, y_angleon, degree=poly_degree)
f_comp = fit_poly_model(x_comp, y_comp, degree=poly_degree)

try:
    xmin_common, xmax_common = common_domain(x_angleon, x_comp)
except ValueError as e:
    st.error(str(e))
    st.stop()

f_angleon_c = clipped(f_angleon, xmin_common, xmax_common)
f_comp_c = clipped(f_comp, xmin_common, xmax_common)

def diff(x):
    return f_angleon_c(x) - f_comp_c(x)

# =========================
# Range selection
# =========================
st.sidebar.markdown("---")
full_range = float(xmax_common - xmin_common)
step_val = max(full_range / 200.0, 0.001)

low_val, high_val = st.sidebar.slider(
    f"Analysis range ({x_units})",
    min_value=float(xmin_common),
    max_value=float(xmax_common),
    value=(float(xmin_common), float(xmax_common)),
    step=float(step_val),
)

# =========================
# Relative / Point Advantage
# =========================
if abs(high_val - low_val) <= 1e-12:
    x0 = float(low_val)
    fa, fc = f_angleon_c(x0), f_comp_c(x0)
    point_advantage = (fa - fc) / fc * 100.0 if fc != 0 else np.nan
    st.metric(
        f"Point Advantage at {x0:.2f} {x_units}",
        f"{point_advantage:.1f}%" if np.isfinite(point_advantage) else "—"
    )
else:
    lo, hi = float(min(low_val, high_val)), float(max(low_val, high_val))
    area_diff = safe_quad(diff, lo, hi)
    area_comp = safe_quad(lambda _x: f_comp_c(_x), lo, hi)
    rel_adv = (area_diff / area_comp * 100.0) if area_comp != 0 else 0.0
    st.metric(f"Relative Advantage [{lo:.2f}–{hi:.2f}] {x_units}", f"{rel_adv:.1f}%")

# =========================
# Plot
# =========================
x_range = np.linspace(xmin_common, xmax_common, 400)
angleon_fit = np.array([f_angleon_c(x) for x in x_range])
comp_fit = np.array([f_comp_c(x) for x in x_range])

fig = go.Figure()

# Highlight range (if interval)
if abs(high_val - low_val) > 1e-12:
    mask = (x_range >= low_val) & (x_range <= high_val)
    if np.any(mask):
        xf = np.concatenate([x_range[mask], x_range[mask][::-1]])
        yf = np.concatenate([angleon_fit[mask], comp_fit[mask][::-1]])
        fig.add_trace(go.Scatter(
            x=xf, y=yf,
            fill="toself",
            fillcolor="rgba(150,150,150,0.25)",
            line=dict(color="rgba(0,0,0,0)"),
            hoverinfo="skip",
            name="Analysis window"
        ))
else:
    x0 = float(low_val)
    y_mid = (f_angleon_c(x0) + f_comp_c(x0)) / 2.0
    fig.add_trace(go.Scatter(
        x=[x0], y=[y_mid],
        mode="markers",
        name=f"Point @ {x0:.2f} {x_units}",
        marker=dict(size=10),
        hoverinfo="skip"
    ))

# Lines
fig.add_trace(go.Scatter(
    x=x_range, y=angleon_fit,
    mode="lines",
    name="AngleOn™ cubic fit",
    line=dict(width=line_width, color="blue"),
))
fig.add_trace(go.Scatter(
    x=x_range, y=comp_fit,
    mode="lines",
    name="Competitor cubic fit",
    line=dict(width=line_width, color="red"),
))

fig.update_layout(
    xaxis_title=f"{x_col} ({x_units})",
    yaxis_title="Velocity (in/sec)",
    height=650,
    hovermode="x",
    legend=dict(
        yanchor="top", y=0.99,
        xanchor="left", x=0.72
    ),
    xaxis=dict(
        showspikes=True,
        spikemode="across",
        spikesnap="cursor",
        showline=True,
        spikecolor="lightgray",
        spikethickness=0.7,
        spikedash="dot",
        showgrid=True,
        gridcolor="rgba(220,220,220,0.4)"
    ),
    yaxis=dict(
        showspikes=True,
        spikemode="across",
        spikesnap="cursor",
        showline=True,
        spikecolor="lightgray",
        spikethickness=0.7,
        spikedash="dot",
        showgrid=True,
        gridcolor="rgba(220,220,220,0.4)"
    ),
    hoverlabel=dict(
        bgcolor="rgba(0,0,0,0)",
        font_size=12,
        font_family="Arial"
    )
)

# Add shaded load bands if x-axis is Pressure
fig = add_pressure_bands(
    fig,
    x_col=x_col,
    show=use_bands,
    x_max=float(np.nanmax(x_range)),
    Pn_lo=P_normal_lo,
    Pn_hi=P_normal_hi,
    Pt_lo=P_trans_lo,
    Pt_hi=P_trans_hi,
    Pa_lo=P_avoid_lo,
)

st.plotly_chart(fig, use_container_width=True)

# Optional: show the band definitions under the plot (only meaningful for pressure view)
if x_col == "Pressure":
    st.caption(f"""
**Load band lens (Pressure):**
- **Normal:** {P_normal_lo:.2f} to {P_normal_hi:.2f} lb/in²
- **Transient:** {P_trans_lo:.2f} to {P_trans_hi:.2f} lb/in²
- **Avoid:** > {P_avoid_lo:.2f} lb/in² (brush-dependent; near-failure behavior)
""")

# =========================
# CSV Data (moved below chart)
# =========================
with st.expander("Show CSV data / object details", expanded=False):
    col_a, col_b, col_c = st.columns([1.2, 1, 1])
    brush_values = sorted(df["Brush"].dropna().unique().tolist())
    default_brushes = [b for b in brush_values if b in ["AngleOn™", "Competitor"]]

    with col_a:
        selected_brushes = st.multiselect(
            "Filter: Brush",
            options=brush_values,
            default=default_brushes
        )
    with col_b:
        sort_by = st.selectbox("Sort by", options=list(df.columns))
    with col_c:
        ascending = st.checkbox("Ascending sort", value=True)

    df_view = df[df["Brush"].isin(selected_brushes)].copy() if selected_brushes else df.copy()

    q = st.text_input("Quick search (matches text columns)", "")
    if q:
        text_cols = [c for c in df_view.columns if df_view[c].dtype == "object"]
        if text_cols:
            mask = False
            for c in text_cols:
                mask = mask | df_view[c].astype(str).str.contains(q, case=False, na=False)
            df_view = df_view[mask]

    df_view = df_view.sort_values(sort_by, ascending=ascending)
    st.dataframe(df_view, use_container_width=True, height=350)

    st.download_button(
        "Download filtered CSV",
        data=df_view.to_csv(index=False).encode("utf-8"),
        file_name="velocity_data_filtered.csv",
        mime="text/csv"
    )
