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

st.caption("""
Key Points:    
AngleOn™ moves objects faster than competitor product across all object weights/pressures tested with the largest advantage in the lower pressure ranges. 
Please reference the table for specifics on object details. Weight range tested 0.167lbs - 22.5lbs (feeder cannot physically move anything heavier).
""")

# Fixed file/path and cubic fit (no upload, no degree slider)
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
# Sidebar Controls (after data load so we can bound controls)
# =========================
st.sidebar.header("Controls")

# Axis toggle (DEFAULT = Weight)
axis_choice = st.sidebar.radio(
    "X-axis",
    options=["Weight (lb)", "Pressure (psi)"],
    index=0
)

# Resolve x column, units, defaults
if axis_choice.startswith("Weight"):
    x_col = "Weight"
    x_units = "lb"
    window_label = f"{x_col} window for relative advantage ({x_units})"
    window_step = 0.5
    window_max = float(df[x_col].max()) if x_col in df.columns else 100.0
    window_default = min(5.0, window_max) if window_max > 0 else 5.0
else:
    x_col = "Pressure"
    x_units = "psi"
    window_label = f"{x_col} window for relative advantage ({x_units})"
    window_step = 0.05
    window_max = 100.0
    window_default = 0.50

# Ensure the chosen x-axis column exists
if x_col not in df.columns:
    st.error(f"Column '{x_col}' not found in the CSV. Available columns: {list(df.columns)}")
    st.stop()

area_to_var = st.sidebar.number_input(
    window_label,
    min_value=0.00, max_value=window_max,
    value=float(window_default), step=float(window_step), format="%.2f"
)

show_points = st.sidebar.checkbox("Show raw data points", value=False)  # default OFF
shade_alpha = st.sidebar.slider("Shaded area opacity", 0.0, 1.0, 0.30, 0.05)

st.sidebar.markdown("---")
st.sidebar.caption(
    "Tip: For Pressure view, set the window to **0.50 psi** for low-pressure comparisons. "
    "For Weight, **~5 lb** is a good starting window."
)


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
# Prep Data & Fit Models
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

f_angleon, _, _ = fit_poly_model(x_angleon, y_angleon, degree=poly_degree)
f_comp, _, _ = fit_poly_model(x_comp, y_comp, degree=poly_degree)

def diff(x: float) -> float:
    return f_angleon(x) - f_comp(x)


# =========================
# Intersection (where fits meet) & Range
# =========================
x_min = float(max(0.0, min(np.nanmin(x_angleon), np.nanmin(x_comp)) - 0.2))
x_max = float(max(np.nanmax(x_angleon), np.nanmax(x_comp)) + 0.2)
x0_guess = 1.0 if x_min <= 1.0 <= x_max else (x_min + x_max) / 2

try:
    x_intersect = float(fsolve(diff, x0=x0_guess)[0])
    if not (x_min - 1.0 <= x_intersect <= x_max + 1.0):
        raise ValueError("Intersection out of range")
except Exception:
    grid = np.linspace(x_min, x_max, 1000)
    vals = [diff(x) for x in grid]
    sign_changes = np.where(np.sign(vals[:-1]) != np.sign(vals[1:]))[0]
    if len(sign_changes) > 0:
        i = sign_changes[0]
        x_intersect = float((grid[i] + grid[i + 1]) / 2)
    else:
        # If no intersection, cap to a meaningful number within the observed range
        x_intersect = float(min(x_max, (x_min + x_max) / 2))


# =========================
# Areas & Metrics
# =========================
to_x = float(area_to_var)

area_diff_window = safe_quad(diff, 0.0, to_x)
area_comp_window = safe_quad(lambda _x: f_comp(_x), 0.0, to_x)
percent_improvement_window = (area_diff_window / area_comp_window * 100.0) if area_comp_window != 0 else 0.0

cap_x = float(max(0.0, min(x_intersect, x_max)))
area_diff_intersect = safe_quad(diff, 0.0, cap_x)
area_comp_intersect = safe_quad(lambda _x: f_comp(_x), 0.0, cap_x)
percent_improvement_intersect = (area_diff_intersect / area_comp_intersect * 100.0) if area_comp_intersect != 0 else 0.0

col_m1, col_m2, col_m3, col_m4 = st.columns(4)
col_m1.metric(f"Intersection ({x_units})", f"{x_intersect:.3f}")
col_m2.metric(f"Advantage Area 0–{to_x:.2f} {x_units}", f"{area_diff_window:.3f} in/sec·{x_units}")
col_m3.metric(f"Relative Advantage 0–{to_x:.2f} {x_units}", f"{percent_improvement_window:.1f}%")
col_m4.metric(f"Relative Advantage 0–{cap_x:.2f} {x_units}", f"{percent_improvement_intersect:.1f}%")


# =========================
# Plot
# =========================
plot_hi = float(max(cap_x + 0.5, x_max))
x_range = np.linspace(0.0, plot_hi, 400)

angleon_fit = np.array([f_angleon(x) for x in x_range])
comp_fit = np.array([f_comp(x) for x in x_range])

def make_fill_between(x, y1, y2, limit):
    mask = x <= limit
    x_m = x[mask]
    y1_m = y1[mask]
    y2_m = y2[mask]
    xf = np.concatenate([x_m, x_m[::-1]])
    yf = np.concatenate([y1_m, y2_m[::-1]])
    return xf, yf

x_fill_window, y_fill_window = make_fill_between(x_range, angleon_fit, comp_fit, to_x)
x_fill_intersect, y_fill_intersect = make_fill_between(x_range, angleon_fit, comp_fit, cap_x)

fig = go.Figure()

# Shaded advantage areas
fig.add_trace(go.Scatter(
    x=x_fill_window,
    y=y_fill_window,
    fill="toself",
    fillcolor=f"rgba(150,150,150,{shade_alpha})",
    line=dict(color="rgba(0,0,0,0)"),
    hoverinfo="skip",
    name=f"Advantage Area (0–{to_x:.2f} {x_units})"
))

if cap_x > to_x:
    fig.add_trace(go.Scatter(
        x=x_fill_intersect,
        y=y_fill_intersect,
        fill="toself",
        fillcolor=f"rgba(150,150,150,{max(0.05, shade_alpha/2)})",
        line=dict(color="rgba(0,0,0,0)"),
        hoverinfo="skip",
        name=f"Advantage Area (0–{cap_x:.2f} {x_units})"
    ))

# Model lines
fig.add_trace(go.Scatter(
    x=x_range, y=angleon_fit,
    mode="lines",
    name="AngleOn™ cubic fit",
    line=dict(width=line_width, color="blue"),
    hovertemplate=f"{x_col}: %{x:.2f} {x_units}<br>Velocity: %{y:.2f} in/sec"
))
fig.add_trace(go.Scatter(
    x=x_range, y=comp_fit,
    mode="lines",
    name="Competitor cubic fit",
    line=dict(width=line_width, color="red"),
    hovertemplate=f"{x_col}: %{x:.2f} {x_units}<br>Velocity: %{y:.2f} in/sec"
))

# Raw points (optional)
if show_points:
    fig.add_trace(go.Scatter(
        x=x_angleon, y=y_angleon,
        mode="markers", name="AngleOn™ data",
        marker=dict(size=8, color="blue")
    ))
    fig.add_trace(go.Scatter(
        x=x_comp, y=y_comp,
        mode="markers", name="Competitor data",
        marker=dict(size=8, color="red")
    ))

# Annotation
fig.add_annotation(
    x=float(np.clip(to_x, 0, plot_hi)),
    y=float((np.nanmax(angleon_fit) + np.nanmax(comp_fit)) / 2),
    text=(f"Advantage 0–{to_x:.2f} {x_units} = {area_diff_window:.3f} in/sec·{x_units}"
          f"<br>Relative = {percent_improvement_window:.1f}%"),
    showarrow=False
)

# Layout
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
