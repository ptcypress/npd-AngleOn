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
st.set_page_config(page_title="Velocity vs Pressure", layout="wide")
st.title("Object Velocity vs Pressure — AngleOn™ vs Competitor")

st.caption("""
Key Points:
AngleOn™ moves objects faster than competitor product across all object weights/pressures tested with the largest advantage in the lower ranges.
Please reference the table for specifics on object details. Weight range tested 0.167lbs - 22.5lbs (feeder cannot physically move anything heavier).
""")

# Fixed file/path and cubic fit
default_path = "data/velocity_data.csv"
csv_path = default_path
poly_degree = 3  # cubic
line_width = 2   # fixed line width


# =========================
# Helpers
# =========================
def load_csv(_path: str) -> pd.DataFrame | None:
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
    def f(x_scalar: float) -> float:
        return float(model.predict(poly.transform(np.array([[x_scalar]])))[0])
    return f


def safe_quad(func, a: float, b: float) -> float:
    if a == b:
        return 0.0
    lo, hi = min(a, b), max(a, b)
    val, _ = quad(func, lo, hi)
    return val


def common_domain(x1: np.ndarray, x2: np.ndarray) -> tuple[float, float]:
    """Overlapping domain between two x arrays; raise if none."""
    xmin = float(max(np.nanmin(x1), np.nanmin(x2)))
    xmax = float(min(np.nanmax(x1), np.nanmax(x2)))
    if xmax < xmin:
        raise ValueError("No overlapping domain between AngleOn™ and Competitor.")
    return xmin, xmax


def clipped(func, lo: float, hi: float):
    """Clamp function evaluation to [lo, hi] to avoid extrapolation."""
    def g(x: float) -> float:
        if x < lo: x = lo
        if x > hi: x = hi
        return func(x)
    return g


# =========================
# Data Load
# =========================
df = load_csv(csv_path)

if df is None:
    st.error(f"No CSV found at `{csv_path}`. Please place your file there.")
    st.stop()

# Standardize columns
df.columns = df.columns.str.strip()

ok, msg = validate_columns(df)
if not ok:
    st.error(msg)
    st.stop()


# =========================
# Sidebar Controls (needs df)
# =========================
st.sidebar.header("Controls")

# Axis toggle (DEFAULT = Weight)
axis_choice = st.sidebar.radio(
    "X-axis",
    options=["Weight (lb)", "Pressure (psi)"],
    index=0
)

# Resolve x column & units
if axis_choice.startswith("Weight"):
    x_col = "Weight"
    x_units = "lb"
else:
    x_col = "Pressure"
    x_units = "psi"

# Ensure the chosen x-axis column exists
if x_col not in df.columns:
    st.error(f"Column '{x_col}' not found in the CSV. Available columns: {list(df.columns)}")
    st.stop()


# =========================
# Collapsible Data Table
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
        sort_by = st.selectbox(
            "Sort by",
            options=list(df.columns),
            index=list(df.columns).index("Brush") if "Brush" in df.columns else 0
        )
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


# =========================
# Prep Data & Fit Models (overlap only)
# =========================
angleon = df[df["Brush"].str.strip() == "AngleOn™"].copy()
competitor = df[df["Brush"].str.strip() == "Competitor"].copy()

if angleon.empty or competitor.empty:
    st.error("Need rows for both 'AngleOn™' and 'Competitor'.")
    st.stop()

x_angleon = angleon[x_col].to_numpy()
y_angleon = angleon["Velocity"].to_numpy()
x_comp = competitor[x_col].to_numpy()
y_comp = competitor["Velocity"].to_numpy()

# Fit cubics
f_angleon = fit_poly_model(x_angleon, y_angleon, degree=poly_degree)
f_comp    = fit_poly_model(x_comp,    y_comp,    degree=poly_degree)

# Overlapping domain only
try:
    xmin_common, xmax_common = common_domain(x_angleon, x_comp)
except ValueError as e:
    st.error(str(e))
    st.stop()

# Clamp fits to overlap
f_angleon_c = clipped(f_angleon, xmin_common, xmax_common)
f_comp_c    = clipped(f_comp,    xmin_common, xmax_common)

def diff(x: float) -> float:
    return f_angleon_c(x) - f_comp_c(x)


# =========================
# Free Range Selection (low & high) — allows single-point
# =========================
st.sidebar.markdown("---")
full_range = float(xmax_common - xmin_common)
step_val = max(full_range / 200.0, 0.001)  # smooth control

low_val, high_val = st.sidebar.slider(
    f"Analysis range ({x_units})",
    min_value=float(xmin_common),
    max_value=float(xmax_common),
    value=(float(xmin_common), float(xmax_common)),  # defaults to full overlap
    step=float(step_val)
)

show_points = st.sidebar.checkbox("Show raw data points", value=False)  # default OFF
shade_alpha = st.sidebar.slider("Shaded area opacity", 0.0, 1.0, 0.30, 0.05)


# =========================
# Relative / Point Advantage
# =========================
if abs(high_val - low_val) <= 1e-12:
    # Single-point advantage at x0
    x0 = float(low_val)
    fa = f_angleon_c(x0)
    fc = f_comp_c(x0)
    point_advantage = (fa - fc) / fc * 100.0 if fc != 0 else np.nan
    title = f"Point Advantage at {x0:.2f} {x_units}"
    value_str = f"{point_advantage:.1f}%" if np.isfinite(point_advantage) else "—"
    st.metric(title, value_str)
else:
    # Interval-based relative advantage
    lo, hi = float(min(low_val, high_val)), float(max(low_val, high_val))
    area_diff = safe_quad(diff, lo, hi)
    area_comp = safe_quad(lambda _x: f_comp_c(_x), lo, hi)
    rel_adv = (area_diff / area_comp * 100.0) if area_comp != 0 else 0.0
    st.metric(f"Relative Advantage [{lo:.2f}–{hi:.2f}] {x_units}", f"{rel_adv:.1f}%")


# =========================
# Plot (only within the common domain)
# =========================
x_range = np.linspace(xmin_common, xmax_common, 400)
angleon_fit = np.array([f_angleon_c(x) for x in x_range])
comp_fit    = np.array([f_comp_c(x)    for x in x_range])

def make_fill_between(x, y1, y2, a, b):
    mask = (x >= a) & (x <= b)
    if not np.any(mask):
        return np.array([]), np.array([])
    xm = x[mask]
    y1m, y2m = y1[mask], y2[mask]
    xf = np.concatenate([xm, xm[::-1]])
    yf = np.concatenate([y1m, y2m[::-1]])
    return xf, yf

fig = go.Figure()

# Shaded interval or point indicator
if abs(high_val - low_val) <= 1e-12:
    # draw a small marker at the mid value between the two fits
    x0 = float(low_val)
    y_mid = (f_angleon_c(x0) + f_comp_c(x0)) / 2.0
    fig.add_trace(go.Scatter(
        x=[x0], y=[y_mid],
        mode="markers",
        name=f"Point @ {x0:.2f} {x_units}",
        marker=dict(size=10),
        hoverinfo="skip"
    ))
else:
    xf, yf = make_fill_between(x_range, angleon_fit, comp_fit, float(min(low_val, high_val)), float(max(low_val, high_val)))
    if xf.size:
        fig.add_trace(go.Scatter(
            x=xf, y=yf,
            fill="toself",
            fillcolor=f"rgba(150,150,150,{shade_alpha})",
            line=dict(color="rgba(0,0,0,0)"),
            hoverinfo="skip",
            name=f"Analysis window"
        ))

# Model lines (hovertemplate braces fixed)
fig.add_trace(go.Scatter(
    x=x_range, y=angleon_fit,
    mode="lines",
    name="AngleOn™ cubic fit",
    line=dict(width=line_width, color="blue"),
    hovertemplate=f"{x_col}: %{{x:.2f}} {x_units}<br>Velocity: %{{y:.2f}} in/sec"
))
fig.add_trace(go.Scatter(
    x=x_range, y=comp_fit,
    mode="lines",
    name="Competitor cubic fit",
    line=dict(width=line_width, color="red"),
    hovertemplate=f"{x_col}: %{{x:.2f}} {x_units}<br>Velocity: %{{y:.2f}} in/sec"
))

# Raw points (optional) — inside overlap only
if show_points:
    mask_a = (x_angleon >= xmin_common) & (x_angleon <= xmax_common)
    mask_c = (x_comp     >= xmin_common) & (x_comp     <= xmax_common)
    if np.any(mask_a):
        fig.add_trace(go.Scatter(
            x=x_angleon[mask_a], y=y_angleon[mask_a],
            mode="markers", name="AngleOn™ data",
            marker=dict(size=8, color="blue")
        ))
    if np.any(mask_c):
        fig.add_trace(go.Scatter(
            x=x_comp[mask_c], y=y_comp[mask_c],
            mode="markers", name="Competitor data",
            marker=dict(size=8, color="red")
        ))

fig.update_layout(
    xaxis_title=f"{x_col} ({x_units})",
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
