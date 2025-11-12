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
st.set_page_config(page_title="Velocity vs Mass & Pressure", layout="wide")
st.title("Object Velocity vs Mass & Pressure — AngleOn™ vs Competitor")

st.caption("""
Key Points:    
AngleOn™ moves objects (approx. 25%) faster than competitor product across all object weights/pressures tested with the largest advantage in the lower ranges.  
Please reference the table drop-down below for specifics on object details. Weight range tested 0.167lbs - 22.5lbs (feeder cannot physically move anything heavier).  
Adjust 'Analysis Range' slider in side panel to calculate speed advantage for any target or range of object mass or pressure.
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
    options=["Weight (lb)", "Pressure (psi)"],
    index=0
)

x_col = "Weight" if axis_choice.startswith("Weight") else "Pressure"
x_units = "lb" if x_col == "Weight" else "psi"

if x_col not in df.columns:
    st.error(f"Column '{x_col}' not found in the CSV. Available columns: {list(df.columns)}")
    st.stop()

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
        raise ValueError("No overlapping domain between AngleOn™ and Competitor.")
    return xmin, xmax

def clipped(func, lo, hi):
    def g(x):
        if x < lo: x = lo
        if x > hi: x = hi
        return func(x)
    return g

# =========================
# Prep data
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
    step=float(step_val)
)

# =========================
# Relative / Point Advantage
# =========================
if abs(high_val - low_val) <= 1e-12:
    x0 = float(low_val)
    fa, fc = f_angleon_c(x0), f_comp_c(x0)
    point_advantage = (fa - fc) / fc * 100.0 if fc != 0 else np.nan
    st.metric(f"Point Advantage at {x0:.2f} {x_units}",
              f"{point_advantage:.1f}%" if np.isfinite(point_advantage) else "—")
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
            fillcolor="rgba(150,150,150,0.3)",
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


st.plotly_chart(fig, use_container_width=True)

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
