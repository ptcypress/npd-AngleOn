import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from scipy.integrate import quad
from scipy.optimize import fsolve


# =========================
# Streamlit Page Setup
# =========================
st.set_page_config(page_title="Velocity vs Pressure", layout="wide")
st.title("Object Velocity vs Pressure — AngleOn™ vs Competitor")

# -------------------------
# Sidebar Controls
# -------------------------
st.sidebar.header("Controls")

uploaded = st.sidebar.file_uploader("Upload velocity_data.csv", type=["csv"])
default_path = "data/velocity_data.csv"
use_default = st.sidebar.checkbox("Use default path", value=not bool(uploaded))

csv_path = default_path if use_default else None

poly_degree = st.sidebar.select_slider("Polynomial degree", options=[2, 3, 4], value=3)
area_to_pressure = st.sidebar.number_input(
    "Pressure window for relative advantage (psi)",
    min_value=0.00, max_value=100.00, value=0.50, step=0.05, format="%.2f"
)

show_points = st.sidebar.checkbox("Show raw data points", value=True)

# Theme-ish toggles
shade_alpha = st.sidebar.slider("Shaded area opacity", 0.0, 1.0, 0.30, 0.05)
line_width = st.sidebar.slider("Fit line width", 1, 6, 3, 1)

st.sidebar.markdown("---")
st.sidebar.caption("Tip: Set the window to **0.50 psi** to match your low-pressure comparison.")


# =========================
# Helpers
# =========================

def load_csv(_uploaded, _path: str) -> pd.DataFrame | None:
    if _uploaded is not None:
        return pd.read_csv(_uploaded)
    if _path and os.path.exists(_path):
        return pd.read_csv(_path)
    return None


def validate_columns(df: pd.DataFrame, required=("Brush", "Pressure", "Velocity")) -> tuple[bool, str]:
    missing = [c for c in required if c not in df.columns]
    if missing:
        return False, f"Missing required columns: {', '.join(missing)}"
    return True, ""


def fit_poly_model(x: np.ndarray, y: np.ndarray, degree: int):
    poly = PolynomialFeatures(degree=degree)
    X_poly = poly.fit_transform(x.reshape(-1, 1))
    model = LinearRegression().fit(X_poly, y)
    # return callable plus transformer so we can reuse easily
    def f(x_scalar: float) -> float:
        return float(model.predict(poly.transform(np.array([[x_scalar]])))[0])
    return f, poly, model


def safe_quad(func, a: float, b: float) -> float:
    if a == b:
        return 0.0
    lo, hi = min(a, b), max(a, b)
    val, _ = quad(func, lo, hi)
    return val


# =========================
# Data Load
# =========================
df = load_csv(uploaded, csv_path)

if df is None:
    st.error(
        "No CSV found. Upload a file in the sidebar or enable **Use default path** "
        f"(expects `{default_path}`)."
    )
    st.stop()

# Standardize columns
df.columns = df.columns.str.strip()

ok, msg = validate_columns(df)
if not ok:
    st.error(msg)
    st.stop()

# =========================
# Collapsible Data Table
# =========================
with st.expander("Show CSV data / object details", expanded=False):
    # Filters
    col_a, col_b, col_c = st.columns([1.2, 1, 1])
    brush_values = sorted(df["Brush"].dropna().unique().tolist())

    with col_a:
        selected_brushes = st.multiselect(
            "Filter: Brush", options=brush_values, default=brush_values
        )
    with col_b:
        sort_by = st.selectbox(
            "Sort by", options=list(df.columns),
            index=list(df.columns).index("Pressure") if "Pressure" in df.columns else 0
        )
    with col_c:
        ascending = st.checkbox("Ascending sort", value=True)

    # Apply
    df_view = df[df["Brush"].isin(selected_brushes)].copy() if selected_brushes else df.copy()

    # Quick text search across object-like columns
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

# =========================
# Prep Data & Fit Models
# =========================
# Keep just the two series we care about
angleon = df[df["Brush"].str.strip() == "AngleOn™"].copy()
competitor = df[df["Brush"].str.strip() == "Competitor"].copy()

if angleon.empty or competitor.empty:
    st.error("Need rows for both 'AngleOn™' and 'Competitor'.")
    st.stop()

x_angleon = angleon["Pressure"].to_numpy()
y_angleon = angleon["Velocity"].to_numpy()

x_comp = competitor["Pressure"].to_numpy()
y_comp = competitor["Velocity"].to_numpy()

# Fit
f_angleon, poly_obj, model_angleon = fit_poly_model(x_angleon, y_angleon, degree=poly_degree)
f_comp, _, model_comp = fit_poly_model(x_comp, y_comp, degree=poly_degree)

def diff(x: float) -> float:
    return f_angleon(x) - f_comp(x)

# Try to find the first intersection near the low-pressure region
# We attempt a bracket around the observed range to be robust
x_min = float(max(0.0, min(x_angleon.min(), x_comp.min()) - 0.2))
x_max = float(max(x_angleon.max(), x_comp.max()) + 0.2)
x0_guess = 1.0 if x_min <= 1.0 <= x_max else (x_min + x_max) / 2

try:
    x_intersect = float(fsolve(diff, x0=x0_guess)[0])
    # If it's wildly outside our observed region, ignore it
    if not (x_min - 1.0 <= x_intersect <= x_max + 1.0):
        raise ValueError("Intersection out of range")
except Exception:
    # Fallback: estimate by scanning a dense grid for sign change
    grid = np.linspace(x_min, x_max, 1000)
    vals = [diff(x) for x in grid]
    sign_changes = np.where(np.sign(vals[:-1]) != np.sign(vals[1:]))[0]
    if len(sign_changes) > 0:
        i = sign_changes[0]
        x_intersect = float((grid[i] + grid[i + 1]) / 2)
    else:
        # Final fallback: no intersection — set to max observed
        x_intersect = float(min(x_max, 3.08))  # keep your old fallback flavor

# =========================
# Areas & Metrics
# =========================
# 1) Relative advantage up to the user-chosen window (default 0.50 psi)
to_x = float(area_to_pressure)

area_diff_window = safe_quad(diff, 0.0, to_x)
area_comp_window = safe_quad(lambda _x: f_comp(_x), 0.0, to_x)
percent_improvement_window = (area_diff_window / area_comp_window * 100.0) if area_comp_window != 0 else 0.0

# 2) Also compute up to intersection (for context)
cap_x = float(max(0.0, min(x_intersect, x_max)))
area_diff_intersect = safe_quad(diff, 0.0, cap_x)
area_comp_intersect = safe_quad(lambda _x: f_comp(_x), 0.0, cap_x)
percent_improvement_intersect = (area_diff_intersect / area_comp_intersect * 100.0) if area_comp_intersect != 0 else 0.0

col_m1, col_m2, col_m3, col_m4 = st.columns(4)
col_m1.metric("Intersection (psi)", f"{x_intersect:.3f}")
col_m2.metric(f"Advantage Area 0–{to_x:.2f} psi", f"{area_diff_window:.3f} in/sec·psi")
col_m3.metric(f"Relative Advantage 0–{to_x:.2f} psi", f"{percent_improvement_window:.1f}%")
col_m4.metric(f"Relative Advantage 0–{cap_x:.2f} psi", f"{percent_improvement_intersect:.1f}%")

# =========================
# Plot
# =========================
# Smooth range that covers from 0 to a bit beyond the intersection and observed max
plot_hi = float(max(cap_x + 0.5, x_max))
pressure_range = np.linspace(0.0, plot_hi, 400)

angleon_fit = np.array([f_angleon(x) for x in pressure_range])
comp_fit = np.array([f_comp(x) for x in pressure_range])

# Build shaded polygons
def make_fill_between(x, y1, y2, limit):
    mask = x <= limit
    x_m = x[mask]
    y1_m = y1[mask]
    y2_m = y2[mask]
    xf = np.concatenate([x_m, x_m[::-1]])
    yf = np.concatenate([y1_m, y2_m[::-1]])
    return xf, yf

x_fill_window, y_fill_window = make_fill_between(pressure_range, angleon_fit, comp_fit, to_x)
x_fill_intersect, y_fill_intersect = make_fill_between(pressure_range, angleon_fit, comp_fit, cap_x)

fig = go.Figure()

# Shaded area 1: up to user window
fig.add_trace(go.Scatter(
    x=x_fill_window,
    y=y_fill_window,
    fill="toself",
    fillcolor=f"rgba(150,150,150,{shade_alpha})",
    line=dict(color="rgba(0,0,0,0)"),
    hoverinfo="skip",
    name=f"Advantage Area (0–{to_x:.2f} psi)"
))

# Optional: show the broader area to intersection in lighter shade (if larger than window)
if cap_x > to_x:
    fig.add_trace(go.Scatter(
        x=x_fill_intersect,
        y=y_fill_intersect,
        fill="toself",
        fillcolor=f"rgba(150,150,150,{max(0.05, shade_alpha/2)})",
        line=dict(color="rgba(0,0,0,0)"),
        hoverinfo="skip",
        name=f"Advantage Area (0–{cap_x:.2f} psi)"
    ))

# Fit lines
fig.add_trace(go.Scatter(
    x=pressure_range, y=angleon_fit,
    mode="lines",
    name="AngleOn™ cubic fit",
    line=dict(width=line_width),
    hovertemplate="Pressure: %{x:.2f} psi<br>Velocity: %{y:.2f} in/sec"
))

fig.add_trace(go.Scatter(
    x=pressure_range, y=comp_fit,
    mode="lines",
    name="Competitor cubic fit",
    line=dict(width=line_width),
    hovertemplate="Pressure: %{x:.2f} psi<br>Velocity: %{y:.2f} in/sec"
))

# Data points (optional)
if show_points:
    fig.add_trace(go.Scatter(
        x=x_angleon, y=y_angleon,
        mode="markers", name="AngleOn™ data",
        hovertemplate="Pressure: %{x:.2f} psi<br>Velocity: %{y:.2f} in/sec"
    ))
    fig.add_trace(go.Scatter(
        x=x_comp, y=y_comp,
        mode="markers", name="Competitor data",
        hovertemplate="Pressure: %{x:.2f} psi<br>Velocity: %{y:.2f} in/sec"
    ))

# Annotation
fig.add_annotation(
    x=np.clip(to_x, 0, plot_hi),
    y=(np.nanmax(angleon_fit) + np.nanmax(comp_fit)) / 2,
    text=(
        f"Advantage 0–{to_x:.2f} psi = {area_diff_window:.3f} in/sec·psi"
        f"<br>Relative = {percent_improvement_window:.1f}%"
    ),
    showarrow=False
)

fig.update_layout(
    xaxis_title="Pressure (psi)",
    yaxis_title="Velocity (in/sec)",
    height=650,
    hovermode="x",
    legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.72),
    xaxis=dict(
        showspikes=True, spikemode="across", spikesnap="cursor",
        showline=True, spikecolor="lightgray", spikethickness=0.7, spikedash="dot"
    ),
    yaxis=dict(
        showspikes=True, spikemode="across", spikesnap="cursor",
        showline=True, spikecolor="lightgray", spikethickness=0.7, spikedash="dot",
        range=[0, max(0.5, float(np.nanmax([angleon_fit.max(), comp_fit.max()])) + 0.5)]
    ),
    hoverlabel=dict(bgcolor="rgba(0,0,0,0)", font_size=12, font_family="Arial")
)

st.plotly_chart(fig, use_container_width=True)

st.caption(
    "This chart compares object velocity on AngleOn™ vs a competitor across applied pressure. "
    "AngleOn™ moves objects faster under identical conditions, particularly at lower pressures. "
    "Shaded region quantifies cumulative performance advantage over the chosen pressure window."
)
