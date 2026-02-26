import numpy as np
import streamlit as st
import plotly.graph_objects as go

# =========================
# Page Setup
# =========================
st.set_page_config(page_title="Velocity Curves & Derivatives", layout="wide")
st.title("Velocity Curves & Derivatives — Cubic Fits")
st.caption(
    "This page visualizes cubic velocity–pressure models and their derivatives. "
    "Domains are automatically extended from minimum valid pressure to stall (V=0) so curves naturally terminate at zero."
)

# =========================
# Curve Registry
# y = a0 + a1*x + a2*x^2 + a3*x^3
# x = Pressure (lb/in²)
# y = Velocity (in/s)
# domain_min = minimum valid x
# =========================
CURVES = {
    "Angle": {
        "AngleOn™": {
            "coeffs": {"a0": 4.5210, "a1": -3.9740, "a2": 1.9270, "a3": -0.3569},
            "domain_min": 0.167,
        },
        "Competitor": {
            "coeffs": {"a0": 3.0780, "a1": -2.4730, "a2": 1.3470, "a3": -0.2774},
            "domain_min": 0.166,
        },
    },
    "XT": {
        "XT10-250 Rev.1": {
            "coeffs": {"a0": 6.2710, "a1": -12.0900, "a2": 13.5200, "a3": -5.0130},
            "domain_min": 0.166,
        },
        "XT16-125 Rev.1": {
            "coeffs": {"a0": 3.3280, "a1": -2.1110, "a2": 1.1840, "a3": -0.2386},
            "domain_min": 0.166,
        },
        "XT16-125 Rev.2": {
            "coeffs": {"a0": 3.4984, "a1": -6.0848, "a2": 5.3119, "a3": -1.5377},
            "domain_min": 0.166,
        },
        "XT16-115 Rev.1": {
            "coeffs": {"a0": 3.3025, "a1": -5.8152, "a2": 4.8504, "a3": -1.3582},
            "domain_min": 0.166,
        },
        "XT16-105 Rev.1": {  # removed "Projected" per your earlier request
            "coeffs": {"a0": 3.1066, "a1": -5.5456, "a2": 4.3889, "a3": -1.1787},
            "domain_min": 0.166,
        },
        # Add additional curves here as needed:
        # "XT16-XXX Rev.X": {"coeffs": {...}, "domain_min": 0.166},
    },
}

# Flatten
CURVE_META = {}
for fam, d in CURVES.items():
    for nm, spec in d.items():
        CURVE_META[nm] = {"family": fam, **spec}

ALL_CURVES = sorted(CURVE_META.keys())

# =========================
# Math helpers
# =========================
def cubic_eval(x: float, c: dict) -> float:
    return c["a0"] + c["a1"] * x + c["a2"] * (x**2) + c["a3"] * (x**3)

def cubic_d1(x: float, c: dict) -> float:
    # dy/dx = a1 + 2*a2*x + 3*a3*x^2
    return c["a1"] + 2.0 * c["a2"] * x + 3.0 * c["a3"] * (x**2)

def cubic_d2(x: float, c: dict) -> float:
    # d2y/dx2 = 2*a2 + 6*a3*x
    return 2.0 * c["a2"] + 6.0 * c["a3"] * x

def stall_root(name: str) -> float | None:
    """
    Smallest real root >= domain_min where y=0.
    Returns None if no such root exists.
    """
    spec = CURVE_META[name]
    c = spec["coeffs"]
    dom_lo = float(spec["domain_min"])

    coeffs = [c["a3"], c["a2"], c["a1"], c["a0"]]
    roots = np.roots(coeffs)

    candidates = []
    for r in roots:
        if abs(r.imag) < 1e-8:
            xr = float(r.real)
            if xr >= dom_lo:
                candidates.append(xr)

    if not candidates:
        return None
    return float(min(candidates))

def effective_domain(name: str) -> tuple[float, float]:
    """
    Domain is [domain_min, stall] so curves terminate at V=0 (when a stall root exists).
    """
    lo = float(CURVE_META[name]["domain_min"])
    s = stall_root(name)
    if s is None or not np.isfinite(s) or s <= lo:
        # If a curve never crosses 0 mathematically, show a practical window.
        # (You can change 3.5 to whatever you consider "full range".)
        return lo, max(lo + 0.5, 3.5)
    return lo, float(s)

# =========================
# Sidebar controls
# =========================
st.sidebar.header("Controls")

st.sidebar.subheader("Load bands (Pressure, lb/in²)")

use_bands = st.sidebar.checkbox("Show load bands", value=True)

P_normal_lo = st.sidebar.number_input("Normal band start", value=0.60, step=0.05)
P_normal_hi = st.sidebar.number_input("Normal band end",   value=1.30, step=0.05)

P_trans_lo  = st.sidebar.number_input("Transient band start", value=1.30, step=0.05)
P_trans_hi  = st.sidebar.number_input("Transient band end",   value=2.00, step=0.05)

P_avoid_lo  = st.sidebar.number_input("Avoid band start", value=2.00, step=0.05)

families = st.sidebar.multiselect("Families", options=["Angle", "XT"], default=["Angle", "XT"])
available = [c for c in ALL_CURVES if CURVE_META[c]["family"] in families]
if not available:
    st.error("No curves selected. Choose at least one family.")
    st.stop()

curve_selection_mode = st.sidebar.radio("Curves to show", ["All in selected families", "Manual select"], index=0)
if curve_selection_mode.startswith("Manual"):
    curves_to_plot = st.sidebar.multiselect("Select curves", options=available, default=available)
else:
    curves_to_plot = available

if not curves_to_plot:
    st.error("No curves selected to plot.")
    st.stop()

st.sidebar.markdown("---")
resolution = st.sidebar.slider("Resolution (points per curve)", 200, 2500, 1000, 100)
show_stall_markers = st.sidebar.checkbox("Show stall markers", value=True)
clamp_velocity = st.sidebar.checkbox("Clamp velocity below 0 to 0", value=True)

st.sidebar.markdown("---")
st.sidebar.subheader("Derivative views")
show_abs = st.sidebar.checkbox("Include |dV/dP| view", value=True)
show_d2 = st.sidebar.checkbox("Include d²V/dP² view", value=False)

# =========================
# Build plots
# =========================
tab_names = ["Velocity (V) vs Pressure (P)", "Derivative (dV/dP) vs P"]
if show_abs:
    tab_names.append("Sensitivity |dV/dP| vs P")
if show_d2:
    tab_names.append("2nd Derivative (d²V/dP²) vs P")

tabs = st.tabs(tab_names)

def add_pressure_bands(fig, x_col, show, Pn_lo, Pn_hi, Pt_lo, Pt_hi, Pa_lo):
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

    # Avoid band (red) — extend to right edge of plot automatically
    xmax = fig.layout.xaxis.range[1] if fig.layout.xaxis.range else None
    if xmax is None:
        # if range isn't explicitly set, use current data max
        # (Plotly will still shade reasonably even if xmax isn't perfect)
        xmax = Pa_lo + 10

    fig.add_vrect(
        x0=Pa_lo, x1=xmax,
        fillcolor="rgba(255, 0, 0, 0.08)",
        line_width=0,
        layer="below",
        annotation_text="Avoid",
        annotation_position="top left",
        annotation_font_size=11
    )

    return fig

def make_fig(title: str, y_title: str):
    fig = go.Figure()
    fig.update_layout(
        title=title,
        xaxis_title="Pressure (lb/in²)",
        yaxis_title=y_title,
        height=720,
        hovermode="x",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.72),
        xaxis=dict(
            showspikes=True, spikemode="across", spikesnap="cursor",
            showline=True, spikecolor="lightgray", spikethickness=0.7, spikedash="dot",
            showgrid=True, gridcolor="rgba(220,220,220,0.4)"
        ),
        yaxis=dict(
            showspikes=True, spikemode="across", spikesnap="cursor",
            showline=True, spikecolor="lightgray", spikethickness=0.7, spikedash="dot",
            showgrid=True, gridcolor="rgba(220,220,220,0.4)"
        ),
    )
    # Add shaded load bands if x-axis is Pressure
        fig = add_pressure_bands(
            fig, x_col, use_bands,
            P_normal_lo, P_normal_hi,
            P_trans_lo, P_trans_hi,
            P_avoid_lo
        )
    return fig

def add_curve(fig, name: str, y_func, y_label: str):
    c = CURVE_META[name]["coeffs"]
    lo, hi = effective_domain(name)
    xs = np.linspace(lo, hi, int(resolution))

    ys = np.array([y_func(x, c) for x in xs])

    if y_label == "V" and clamp_velocity:
        ys = np.maximum(0.0, ys)

    fig.add_trace(go.Scatter(
        x=xs, y=ys, mode="lines", name=name,
        hovertemplate="P=%{x:.3f} lb/in²<br>"
                      f"{y_label}=%{{y:.4f}}"
                      "<extra>"+name+"</extra>"
    ))

    if show_stall_markers and y_label == "V":
        s = stall_root(name)
        if s is not None and np.isfinite(s):
            fig.add_trace(go.Scatter(
                x=[s], y=[0],
                mode="markers",
                marker=dict(size=9, symbol="x"),
                showlegend=False,
                hovertemplate=f"{name}<br>stall<br>P={s:.3f} lb/in²<br>V=0<extra></extra>",
            ))

# --- Velocity plot
with tabs[0]:
    figV = make_fig("Velocity vs Pressure (cubic fits, extended to stall)", "Velocity (in/s)")
    for nm in curves_to_plot:
        add_curve(figV, nm, cubic_eval, "V")
    st.plotly_chart(figV, use_container_width=True)

# --- First derivative plot
with tabs[1]:
    figD1 = make_fig("dV/dP vs Pressure (slope / sensitivity)", "dV/dP  (in/s) per (lb/in²)")
    for nm in curves_to_plot:
        add_curve(figD1, nm, cubic_d1, "dV/dP")
    st.plotly_chart(figD1, use_container_width=True)

# --- Absolute derivative plot
tab_idx = 2
if show_abs:
    with tabs[tab_idx]:
        figAbs = make_fig("|dV/dP| vs Pressure (robustness view)", "|dV/dP|  (in/s) per (lb/in²)")
        for nm in curves_to_plot:
            add_curve(figAbs, nm, lambda x, c: abs(cubic_d1(x, c)), "|dV/dP|")
        st.plotly_chart(figAbs, use_container_width=True)
    tab_idx += 1

# --- Second derivative plot (optional)
if show_d2:
    with tabs[tab_idx]:
        figD2 = make_fig("d²V/dP² vs Pressure (curvature / transitions)", "d²V/dP²")
        for nm in curves_to_plot:
            add_curve(figD2, nm, cubic_d2, "d²V/dP²")
        st.plotly_chart(figD2, use_container_width=True)

# =========================
# Summary table (stall + worst sensitivity)
# =========================
st.subheader("Quick Summary (per curve)")

rows = []
for nm in curves_to_plot:
    c = CURVE_META[nm]["coeffs"]
    lo, hi = effective_domain(nm)
    xs = np.linspace(lo, hi, 1200)
    v = np.maximum(0.0, np.array([cubic_eval(x, c) for x in xs])) if clamp_velocity else np.array([cubic_eval(x, c) for x in xs])
    d1 = np.array([cubic_d1(x, c) for x in xs])
    absd1 = np.abs(d1)

    s = stall_root(nm)
    rows.append({
        "Curve": nm,
        "Family": CURVE_META[nm]["family"],
        "Min P": round(lo, 3),
        "Stall P (V=0)": round(float(s), 3) if s is not None and np.isfinite(s) else "",
        "Max V": round(float(np.nanmax(v)), 3),
        "Min V": round(float(np.nanmin(v)), 3),
        "Worst |dV/dP|": round(float(np.nanmax(absd1)), 3),
        "Avg |dV/dP|": round(float(np.nanmean(absd1)), 3),
    })

st.dataframe(rows, use_container_width=True, height=360)
