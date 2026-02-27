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
st.title("Object Velocity vs Load — Cubic Fits + Load Bands")

st.caption("""
Benchmarking time-study on a linear vibratory feeder: objects of different weights are placed on each brush and timed over a fixed travel distance.
Time is converted to velocity, and object load is expressed as pressure (lb/in²). This view emphasizes robustness as load increases (stability and how
gracefully performance degrades), not just peak speed.
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
    index=1  # default to Pressure (bands are pressure-based)
)

x_col = "Weight" if axis_choice.startswith("Weight") else "Pressure"
x_units = "lb" if x_col == "Weight" else "lb/in²"

if x_col not in df.columns:
    st.error(f"Column '{x_col}' not found in the CSV. Available columns: {list(df.columns)}")
    st.stop()

# Brush selection for plotting
st.sidebar.subheader("Brushes to plot")
brush_values = sorted(df["Brush"].dropna().unique().tolist())
default_plot = [b for b in ["AngleOn™", "Competitor"] if b in brush_values] or brush_values[:2]

selected_plot_brushes = st.sidebar.multiselect(
    "Select brushes",
    options=brush_values,
    default=default_plot
)

if not selected_plot_brushes:
    st.error("Select at least one brush to plot.")
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

def clipped(func, lo, hi):
    def g(x):
        if x < lo:
            x = lo
        if x > hi:
            x = hi
        return func(x)
    return g

def add_pressure_bands(fig, x_col, show, x_max, Pn_lo, Pn_hi, Pt_lo, Pt_hi, Pa_lo):
    """Add shaded load bands to Plotly figure when x-axis is Pressure."""
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

    # Avoid band (red) — extend to plotted max
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
# Fit models per selected brush (no extrapolation)
# =========================
df_plot = df[df["Brush"].isin(selected_plot_brushes)].copy()

models = {}
domains = {}

for b in selected_plot_brushes:
    sub = df_plot[df_plot["Brush"] == b]
    x = sub[x_col].to_numpy(dtype=float)
    y = sub["Velocity"].to_numpy(dtype=float)

    # Need enough unique x points for a meaningful cubic fit
    if len(np.unique(x)) < 4:
        st.warning(f"Not enough unique {x_col} points to fit cubic for: {b}")
        continue

    f = fit_poly_model(x, y, degree=poly_degree)
    models[b] = f
    domains[b] = (float(np.nanmin(x)), float(np.nanmax(x)))

if not models:
    st.error("No valid brushes to fit. Check selections or data.")
    st.stop()

# Common domain across ALL selected brushes (strict: prevents extrapolation)
xmin_common = max(d[0] for d in domains.values())
xmax_common = min(d[1] for d in domains.values())

if xmax_common <= xmin_common:
    st.error(
        "Selected brushes have no overlapping x-domain (no-extrapolation). "
        "Choose different brushes or switch x-axis."
    )
    st.stop()

models_c = {b: clipped(f, xmin_common, xmax_common) for b, f in models.items()}

# =========================
# Range selection (for optional advantage metric)
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
# Optional: Advantage metric (any two brushes)
# =========================
st.sidebar.subheader("Advantage comparison (optional)")
compare_mode = st.sidebar.checkbox("Show advantage metric", value=("AngleOn™" in models_c and "Competitor" in models_c))

if compare_mode:
    keys = list(models_c.keys())
    # Defaults: AngleOn vs Competitor if present
    default_base = keys.index("AngleOn™") if "AngleOn™" in keys else 0
    default_comp = keys.index("Competitor") if "Competitor" in keys else min(1, len(keys) - 1)

    baseline = st.sidebar.selectbox("Baseline", options=keys, index=default_base)
    compare_to = st.sidebar.selectbox("Compare to", options=keys, index=default_comp)

    f_base = models_c[baseline]
    f_comp = models_c[compare_to]

    def diff(x):
        return f_base(x) - f_comp(x)

    if abs(high_val - low_val) <= 1e-12:
        x0 = float(low_val)
        fb, fc = f_base(x0), f_comp(x0)
        point_advantage = (fb - fc) / fc * 100.0 if fc != 0 else np.nan
        st.metric(
            f"Point Advantage: {baseline} vs {compare_to} @ {x0:.2f} {x_units}",
            f"{point_advantage:.1f}%" if np.isfinite(point_advantage) else "—"
        )
    else:
        lo, hi = float(min(low_val, high_val)), float(max(low_val, high_val))
        area_diff = safe_quad(diff, lo, hi)
        area_comp = safe_quad(lambda _x: f_comp(_x), lo, hi)
        rel_adv = (area_diff / area_comp * 100.0) if area_comp != 0 else 0.0
        st.metric(
            f"Relative Advantage: {baseline} vs {compare_to} [{lo:.2f}–{hi:.2f}] {x_units}",
            f"{rel_adv:.1f}%"
        )

# =========================
# Plot: Cubic fits for all selected brushes
# =========================
x_range = np.linspace(xmin_common, xmax_common, 500)

fig = go.Figure()

# Plot all selected brush cubic fits
for b in selected_plot_brushes:
    if b not in models_c:
        continue
    y_fit = np.array([models_c[b](x) for x in x_range])
    fig.add_trace(go.Scatter(
        x=x_range, y=y_fit,
        mode="lines",
        name=f"{b} cubic fit",
        line=dict(width=line_width)
    ))

fig.update_layout(
    xaxis_title=f"{x_col} ({x_units})",
    yaxis_title="Velocity (in/sec)",
    height=650,
    hovermode="x",
    legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.72),
    xaxis=dict(showgrid=True, gridcolor="rgba(220,220,220,0.4)"),
    yaxis=dict(showgrid=True, gridcolor="rgba(220,220,220,0.4)")
)

# Add shaded load bands if x-axis is Pressure
fig = add_pressure_bands(
    fig,
    x_col=x_col,
    show=use_bands,
    x_max=float(np.nanmax(x_range)),
    Pn_lo=P_normal_lo, Pn_hi=P_normal_hi,
    Pt_lo=P_trans_lo, Pt_hi=P_trans_hi,
    Pa_lo=P_avoid_lo
)

st.plotly_chart(fig, use_container_width=True)

# Show band definitions under the plot (only meaningful for pressure view)
if x_col == "Pressure":
    st.caption(f"""
**Load band lens (Pressure):**
- **Normal:** {P_normal_lo:.2f} to {P_normal_hi:.2f} lb/in² (typical day-to-day operation)
- **Transient:** {P_trans_lo:.2f} to {P_trans_hi:.2f} lb/in² (orientation, stacking, tuning drift)
- **Avoid:** > {P_avoid_lo:.2f} lb/in² (unstable / near-failure region; brush-dependent)
""")

# =========================
# CSV Data (below chart)
# =========================
with st.expander("Show CSV data / object details", expanded=False):
    col_a, col_b, col_c = st.columns([1.2, 1, 1])

    with col_a:
        selected_table_brushes = st.multiselect(
            "Filter table: Brush",
            options=brush_values,
            default=default_plot
        )
    with col_b:
        sort_by = st.selectbox("Sort by", options=list(df.columns))
    with col_c:
        ascending = st.checkbox("Ascending sort", value=True)

    df_view = df[df["Brush"].isin(selected_table_brushes)].copy() if selected_table_brushes else df.copy()

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
