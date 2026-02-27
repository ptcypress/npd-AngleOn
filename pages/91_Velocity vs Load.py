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
st.title("Object Velocity vs Load — Cubic Fits + Derivatives + Load Bands")

st.caption("""
Benchmarking time-study on a linear vibratory feeder: objects of different weights are placed on each brush and timed over a fixed travel distance.
Time is converted to velocity, and object load is expressed as pressure (lb/in²).

Velocity vs pressure is the diagnostic signal: it helps reveal robustness (stability as load varies), how gracefully performance degrades, and where the system
transitions toward stall. Derivative views help quantify sensitivity and regime transitions.
""")

# =========================
# Data Load
# =========================
csv_path = "data/velocity_data.csv"
poly_degree = 3  # cubic
line_width = 2

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
    options=["Pressure (lb/in²)", "Weight (lb)"],
    index=0
)

x_col = "Pressure" if axis_choice.startswith("Pressure") else "Weight"
x_units = "lb/in²" if x_col == "Pressure" else "lb"

if x_col not in df.columns:
    st.error(f"Column '{x_col}' not found in the CSV. Available columns: {list(df.columns)}")
    st.stop()

# Brushes to plot
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

# Which views to show
st.sidebar.subheader("Views")
show_velocity = st.sidebar.checkbox("Velocity", value=True)
show_d1 = st.sidebar.checkbox("dV/dX (derivative)", value=True)
show_abs_d1 = st.sidebar.checkbox("|dV/dX| (sensitivity)", value=True)
show_d2 = st.sidebar.checkbox("d²V/dX² (2nd derivative)", value=True)

if not any([show_velocity, show_d1, show_abs_d1, show_d2]):
    st.warning("Select at least one view to display.")
    st.stop()

# =========================
# Load Bands (Pressure)
# =========================
st.sidebar.subheader("Load bands (Pressure only)")
use_bands = st.sidebar.checkbox("Show load bands", value=True)

P_normal_lo = st.sidebar.number_input("Normal band start", value=0.60, step=0.05)
P_normal_hi = st.sidebar.number_input("Normal band end", value=1.30, step=0.05)
P_trans_lo = st.sidebar.number_input("Transient band start", value=1.30, step=0.05)
P_trans_hi = st.sidebar.number_input("Transient band end", value=2.00, step=0.05)
P_avoid_lo = st.sidebar.number_input("Avoid band start", value=2.00, step=0.05)

# =========================
# Helper functions
# =========================
def safe_quad(func, a, b):
    if a == b:
        return 0.0
    lo, hi = min(a, b), max(a, b)
    val, _ = quad(func, lo, hi)
    return val

def add_pressure_bands(fig, x_col, show, x_max, Pn_lo, Pn_hi, Pt_lo, Pt_hi, Pa_lo):
    """Add shaded load bands to Plotly figure when x-axis is Pressure."""
    if (not show) or (x_col != "Pressure"):
        return fig

    # Normal
    fig.add_vrect(
        x0=Pn_lo, x1=Pn_hi,
        fillcolor="rgba(0, 200, 0, 0.14)",
        line_width=0, layer="below",
        annotation_text="Normal",
        annotation_position="top left",
        annotation_font_size=11
    )
    # Transient
    fig.add_vrect(
        x0=Pt_lo, x1=Pt_hi,
        fillcolor="rgba(255, 215, 0, 0.14)",
        line_width=0, layer="below",
        annotation_text="Transient",
        annotation_position="top left",
        annotation_font_size=11
    )
    # Avoid (to right edge)
    fig.add_vrect(
        x0=Pa_lo, x1=float(x_max),
        fillcolor="rgba(255, 0, 0, 0.08)",
        line_width=0, layer="below",
        annotation_text="Avoid",
        annotation_position="top left",
        annotation_font_size=11
    )
    return fig

def fit_cubic_coeffs(x, y):
    """
    Fit cubic using sklearn with PolynomialFeatures(include_bias=False) so we get:
    V(x) = a0 + a1*x + a2*x^2 + a3*x^3
    """
    poly = PolynomialFeatures(degree=3, include_bias=False)
    Xp = poly.fit_transform(x.reshape(-1, 1))  # columns: x, x^2, x^3
    model = LinearRegression().fit(Xp, y)

    a0 = float(model.intercept_)
    a1 = float(model.coef_[0])
    a2 = float(model.coef_[1])
    a3 = float(model.coef_[2])
    return a0, a1, a2, a3

def eval_cubic(a, x):
    a0, a1, a2, a3 = a
    return a0 + a1*x + a2*(x**2) + a3*(x**3)

def eval_d1(a, x):
    a0, a1, a2, a3 = a
    return a1 + 2*a2*x + 3*a3*(x**2)

def eval_d2(a, x):
    a0, a1, a2, a3 = a
    return 2*a2 + 6*a3*x

# =========================
# Fit models per selected brush (no extrapolation)
# =========================
df_plot = df[df["Brush"].isin(selected_plot_brushes)].copy()

coeffs = {}
domains = {}

for b in selected_plot_brushes:
    sub = df_plot[df_plot["Brush"] == b]
    x = sub[x_col].to_numpy(dtype=float)
    y = sub["Velocity"].to_numpy(dtype=float)

    if len(np.unique(x)) < 4:
        st.warning(f"Not enough unique {x_col} points to fit cubic for: {b}")
        continue

    a = fit_cubic_coeffs(x, y)
    coeffs[b] = a
    domains[b] = (float(np.nanmin(x)), float(np.nanmax(x)))

if not coeffs:
    st.error("No valid brushes to fit. Check selections or data.")
    st.stop()

# Common domain across ALL selected brushes (strict no-extrapolation)
xmin_common = max(d[0] for d in domains.values())
xmax_common = min(d[1] for d in domains.values())

if xmax_common <= xmin_common:
    st.error(
        "Selected brushes have no overlapping x-domain (no extrapolation). "
        "Choose different brushes or switch x-axis."
    )
    st.stop()

# Analysis range slider (within common domain)
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

# Optional advantage metric
st.sidebar.subheader("Advantage comparison (optional)")
compare_mode = st.sidebar.checkbox(
    "Show advantage metric",
    value=("AngleOn™" in coeffs and "Competitor" in coeffs)
)

if compare_mode:
    keys = list(coeffs.keys())
    default_base = keys.index("AngleOn™") if "AngleOn™" in keys else 0
    default_comp = keys.index("Competitor") if "Competitor" in keys else min(1, len(keys) - 1)

    baseline = st.sidebar.selectbox("Baseline", options=keys, index=default_base)
    compare_to = st.sidebar.selectbox("Compare to", options=keys, index=default_comp)

    a_base = coeffs[baseline]
    a_comp = coeffs[compare_to]

    def f_base(x): return float(eval_cubic(a_base, x))
    def f_comp(x): return float(eval_cubic(a_comp, x))
    def diff(x): return f_base(x) - f_comp(x)

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
# Plot builders
# =========================
def base_layout(fig, title, y_label):
    fig.update_layout(
        title=title,
        xaxis_title=f"{x_col} ({x_units})",
        yaxis_title=y_label,
        height=650,
        hovermode="x",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.72),
        xaxis=dict(showgrid=True, gridcolor="rgba(220,220,220,0.4)"),
        yaxis=dict(showgrid=True, gridcolor="rgba(220,220,220,0.4)")
    )
    return fig

def add_analysis_window(fig, x_range, y_a, y_b):
    # Shade between two curves only if range is interval
    if abs(high_val - low_val) <= 1e-12:
        return fig
    mask = (x_range >= low_val) & (x_range <= high_val)
    if not np.any(mask):
        return fig
    xf = np.concatenate([x_range[mask], x_range[mask][::-1]])
    yf = np.concatenate([y_a[mask], y_b[mask][::-1]])
    fig.add_trace(go.Scatter(
        x=xf, y=yf,
        fill="toself",
        fillcolor="rgba(150,150,150,0.22)",
        line=dict(color="rgba(0,0,0,0)"),
        hoverinfo="skip",
        name="Analysis window"
    ))
    return fig

# Common x sampling
x_range = np.linspace(xmin_common, xmax_common, 500)
x_max_plot = float(np.nanmax(x_range))

# Precompute curves for speed
vel_curves = {}
d1_curves = {}
abs_d1_curves = {}
d2_curves = {}

for b, a in coeffs.items():
    # only for selected brushes that successfully fit
    yv = eval_cubic(a, x_range)
    yd1 = eval_d1(a, x_range)
    yd2 = eval_d2(a, x_range)

    vel_curves[b] = yv
    d1_curves[b] = yd1
    abs_d1_curves[b] = np.abs(yd1)
    d2_curves[b] = yd2

# If compare mode and both brushes exist, precompute their curves for shading
shade_ok = False
if compare_mode and ("baseline" in locals()) and ("compare_to" in locals()):
    if baseline in vel_curves and compare_to in vel_curves:
        shade_ok = True

# =========================
# Display plots (Tabs)
# =========================
tabs = []
tab_names = []
if show_velocity: tab_names.append("Velocity")
if show_d1: tab_names.append("dV/dX")
if show_abs_d1: tab_names.append("|dV/dX|")
if show_d2: tab_names.append("d²V/dX²")

tabs = st.tabs(tab_names)

tab_i = 0

if show_velocity:
    with tabs[tab_i]:
        figV = go.Figure()
        for b in selected_plot_brushes:
            if b in vel_curves:
                figV.add_trace(go.Scatter(
                    x=x_range, y=vel_curves[b],
                    mode="lines",
                    name=b,
                    line=dict(width=line_width)
                ))

        # Optional shading window between baseline & compare_to
        if shade_ok:
            figV = add_analysis_window(figV, x_range, vel_curves[baseline], vel_curves[compare_to])

        figV = base_layout(figV, "Velocity vs X (cubic fit)", "Velocity (in/sec)")
        figV = add_pressure_bands(figV, x_col, use_bands, x_max_plot, P_normal_lo, P_normal_hi, P_trans_lo, P_trans_hi, P_avoid_lo)
        st.plotly_chart(figV, use_container_width=True)
    tab_i += 1

if show_d1:
    with tabs[tab_i]:
        figD1 = go.Figure()
        for b in selected_plot_brushes:
            if b in d1_curves:
                figD1.add_trace(go.Scatter(
                    x=x_range, y=d1_curves[b],
                    mode="lines",
                    name=b,
                    line=dict(width=line_width)
                ))
        if shade_ok:
            figD1 = add_analysis_window(figD1, x_range, d1_curves[baseline], d1_curves[compare_to])

        figD1 = base_layout(figD1, "First Derivative: dV/dX vs X", f"dV/d{x_col} (in/sec per {x_units})")
        figD1 = add_pressure_bands(figD1, x_col, use_bands, x_max_plot, P_normal_lo, P_normal_hi, P_trans_lo, P_trans_hi, P_avoid_lo)
        st.plotly_chart(figD1, use_container_width=True)
    tab_i += 1

if show_abs_d1:
    with tabs[tab_i]:
        figAbs = go.Figure()
        for b in selected_plot_brushes:
            if b in abs_d1_curves:
                figAbs.add_trace(go.Scatter(
                    x=x_range, y=abs_d1_curves[b],
                    mode="lines",
                    name=b,
                    line=dict(width=line_width)
                ))
        if shade_ok:
            figAbs = add_analysis_window(figAbs, x_range, abs_d1_curves[baseline], abs_d1_curves[compare_to])

        figAbs = base_layout(figAbs, "Sensitivity: |dV/dX| vs X", f"|dV/d{x_col}| (in/sec per {x_units})")
        figAbs = add_pressure_bands(figAbs, x_col, use_bands, x_max_plot, P_normal_lo, P_normal_hi, P_trans_lo, P_trans_hi, P_avoid_lo)
        st.plotly_chart(figAbs, use_container_width=True)
    tab_i += 1

if show_d2:
    with tabs[tab_i]:
        figD2 = go.Figure()
        for b in selected_plot_brushes:
            if b in d2_curves:
                figD2.add_trace(go.Scatter(
                    x=x_range, y=d2_curves[b],
                    mode="lines",
                    name=b,
                    line=dict(width=line_width)
                ))
        if shade_ok:
            figD2 = add_analysis_window(figD2, x_range, d2_curves[baseline], d2_curves[compare_to])

        figD2 = base_layout(figD2, "Second Derivative: d²V/dX² vs X", f"d²V/d{x_col}² (in/sec per {x_units}²)")
        figD2 = add_pressure_bands(figD2, x_col, use_bands, x_max_plot, P_normal_lo, P_normal_hi, P_trans_lo, P_trans_hi, P_avoid_lo)
        st.plotly_chart(figD2, use_container_width=True)
    tab_i += 1

# Band caption (only meaningful for pressure axis)
if x_col == "Pressure":
    st.caption(f"""
**Load band lens (Pressure):**
- **Normal:** {P_normal_lo:.2f} to {P_normal_hi:.2f} lb/in² (typical day-to-day operation)
- **Transient:** {P_trans_lo:.2f} to {P_trans_hi:.2f} lb/in² (orientation, stacking, tuning drift)
- **Avoid:** > {P_avoid_lo:.2f} lb/in² (unstable / near-failure region; brush-dependent)
""")

# =========================
# CSV Data (below charts)
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
