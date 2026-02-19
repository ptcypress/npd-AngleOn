import numpy as np
import streamlit as st
import plotly.graph_objects as go
from scipy.integrate import quad

# =========================
# Page Setup
# =========================
st.set_page_config(page_title="Velocity vs Load & Pressure", layout="wide")
st.title("Velocity vs Pressure — Model Curves (Cubic Fits)")
st.caption(
    "Model-space comparison using validated cubic regression curves "
    "(no raw replicate points on this page)."
)

# =========================
# Curve Registry (Pressure-based)
# y = a0 + a1*x + a2*x^2 + a3*x^3
# x = Pressure (lb/in²)
# y = Velocity (in/s)
# domain = valid x-range for that curve
# =========================
CURVES = {
    "Angle": {
        "AngleOn™": {
            "coeffs": {"a0": 4.5210, "a1": -3.9740, "a2": 1.9270, "a3": -0.3569},
            "domain": (0.167, 3.000),
            "notes": "Validated production fit",
        },
        "Competitor": {
            "coeffs": {"a0": 3.0780, "a1": -2.4730, "a2": 1.3470, "a3": -0.2774},
            "domain": (0.166, 3.000),
            "notes": "Benchmark competitor product",
        },
    },
    "XT": {
        "XT10-250 Rev.1": {
            "coeffs": {"a0": 6.2710, "a1": -12.0900, "a2": 13.5200, "a3": -5.0130},
            "domain": (0.166, 1.70),
            "notes": "High coupling at low load; collapses at higher load",
        },
        "XT16-125 Rev.1": {
            "coeffs": {"a0": 3.3280, "a1": -2.1110, "a2": 1.1840, "a3": -0.2386},
            "domain": (0.166, 3.50),
            "notes": "Original XT16 architecture",
        },
        "XT16-125 Rev.2": {
            "coeffs": {"a0": 3.4984, "a1": -6.0848, "a2": 5.3119, "a3": -1.5377},
            "domain": (0.166, 2.076385),
            "notes": "Uniform height/angle – reduced margin",
        },
        "XT16-115 Rev.1": {
            "coeffs": {"a0": 3.3025, "a1": -5.8152, "a2": 4.8504, "a3": -1.3582},
            "domain": (0.166, 2.07051),
            "notes": "Intermediate EPI",
        },
        "XT16-105 Rev.1 (Projected)": {
            "coeffs": {"a0": 3.1066, "a1": -5.5456, "a2": 4.3889, "a3": -1.1787},
            "domain": (0.166, 2.061383),
            "notes": "Projected – confirmed by test (differences negligible)",
        },
    },
}

# Flatten curve names for selection while preserving family
CURVE_META = {}
for fam, items in CURVES.items():
    for name, spec in items.items():
        CURVE_META[name] = {"family": fam, **spec}

ALL_CURVES = sorted(CURVE_META.keys())

# =========================
# Helpers
# =========================
def cubic_eval(x: float, c: dict) -> float:
    return (
        c["a0"]
        + c["a1"] * x
        + c["a2"] * (x**2)
        + c["a3"] * (x**3)
    )

def safe_quad(func, a, b) -> float:
    if a == b:
        return 0.0
    lo, hi = min(a, b), max(a, b)
    val, _ = quad(func, lo, hi)
    return float(val)

def common_domain(domA, domB):
    xmin = max(domA[0], domB[0])
    xmax = min(domA[1], domB[1])
    if xmax <= xmin:
        raise ValueError("No overlapping domain between the selected curves.")
    return float(xmin), float(xmax)

def fmt_equation(name: str) -> str:
    c = CURVE_META[name]["coeffs"]
    # keep signs explicit and consistent
    return (
        f"V = {c['a0']:.4f} "
        f"+ ({c['a1']:.4f})·P "
        f"+ ({c['a2']:.4f})·P² "
        f"+ ({c['a3']:.4f})·P³"
    )

def stall_from_roots(name: str) -> float | None:
    """
    Returns the smallest real root >= domain_min where V(P)=0.
    If no such root exists, returns None.
    """
    spec = CURVE_META[name]
    c = spec["coeffs"]
    dom_lo, dom_hi = spec["domain"]

    # Polynomial: a3*x^3 + a2*x^2 + a1*x + a0 = 0
    coeffs = [c["a3"], c["a2"], c["a1"], c["a0"]]
    roots = np.roots(coeffs)

    real_roots = []
    for r in roots:
        if abs(r.imag) < 1e-8:
            xr = float(r.real)
            if xr >= dom_lo:
                real_roots.append(xr)

    if not real_roots:
        return None

    return min(real_roots)


# =========================
# Sidebar Controls
# =========================
st.sidebar.header("Controls")

family_filter = st.sidebar.radio(
    "Curve family",
    options=["All", "Angle", "XT"],
    index=0
)

if family_filter == "All":
    curve_options = ALL_CURVES
else:
    curve_options = sorted([n for n in ALL_CURVES if CURVE_META[n]["family"] == family_filter])

if len(curve_options) < 2:
    st.error("Need at least two curves available for comparison.")
    st.stop()

# Smart defaults
def pick_default(name: str) -> int:
    return curve_options.index(name) if name in curve_options else 0

default_a = "AngleOn™" if "AngleOn™" in curve_options else curve_options[0]
default_b = "Competitor" if "Competitor" in curve_options else (curve_options[1] if len(curve_options) > 1 else curve_options[0])

curve_a = st.sidebar.selectbox("Curve A", curve_options, index=pick_default(default_a))
curve_b = st.sidebar.selectbox("Curve B", curve_options, index=pick_default(default_b))

if curve_a == curve_b:
    st.sidebar.warning("Choose two different curves to compare.")

st.sidebar.markdown("---")
clamp_below_zero = st.sidebar.checkbox("Clamp velocities below 0 to 0", value=True)
show_equations = st.sidebar.checkbox("Show equation details", value=True)
show_stall_lines = st.sidebar.checkbox("Show stall markers (y=0)", value=True)

st.sidebar.markdown("---")
st.sidebar.subheader("Plot Sampling")
n_points = st.sidebar.slider("Curve resolution (points)", min_value=200, max_value=2000, value=800, step=100)

# =========================
# Prepare selected curves
# =========================
specA = CURVE_META[curve_a]
specB = CURVE_META[curve_b]

cA, domA = specA["coeffs"], specA["domain"]
cB, domB = specB["coeffs"], specB["domain"]

try:
    xmin_common, xmax_common = common_domain(domA, domB)
except ValueError as e:
    st.error(str(e))
    st.stop()

def fA(x: float) -> float:
    y = cubic_eval(float(x), cA)
    return max(0.0, y) if clamp_below_zero else y

def fB(x: float) -> float:
    y = cubic_eval(float(x), cB)
    return max(0.0, y) if clamp_below_zero else y

def diff(x: float) -> float:
    return fA(x) - fB(x)

# =========================
# Analysis window
# =========================
st.sidebar.markdown("---")
st.sidebar.subheader("Analysis Window")
full_range = float(xmax_common - xmin_common)
step_val = max(full_range / 200.0, 0.001)

low_val, high_val = st.sidebar.slider(
    "Pressure range (lb/in²)",
    min_value=float(xmin_common),
    max_value=float(xmax_common),
    value=(float(xmin_common), float(xmax_common)),
    step=float(step_val),
)

# =========================
# Advantage Metrics
# =========================
st.subheader("Advantage")

if abs(high_val - low_val) <= 1e-12:
    x0 = float(low_val)
    ya, yb = fA(x0), fB(x0)
    point_adv = (ya - yb) / yb * 100.0 if yb != 0 else np.nan
    st.metric(
        f"Point Advantage: {curve_a} vs {curve_b} @ {x0:.3f} lb/in²",
        f"{point_adv:.1f}%" if np.isfinite(point_adv) else "—",
    )
else:
    lo, hi = float(min(low_val, high_val)), float(max(low_val, high_val))
    area_diff = safe_quad(diff, lo, hi)
    area_b = safe_quad(lambda _x: fB(_x), lo, hi)
    rel_adv = (area_diff / area_b * 100.0) if area_b != 0 else 0.0
    st.metric(
        f"Relative Advantage: {curve_a} vs {curve_b} [{lo:.3f}–{hi:.3f}] lb/in²",
        f"{rel_adv:.1f}%",
    )

# =========================
# Plot
# =========================
st.subheader("Velocity vs Pressure (Cubic Fits)")

x = np.linspace(xmin_common, xmax_common, int(n_points))
yA = np.array([fA(v) for v in x])
yB = np.array([fB(v) for v in x])

fig = go.Figure()

# Shaded analysis window
if abs(high_val - low_val) > 1e-12:
    mask = (x >= low_val) & (x <= high_val)
    if np.any(mask):
        xf = np.concatenate([x[mask], x[mask][::-1]])
        yf = np.concatenate([yA[mask], yB[mask][::-1]])
        fig.add_trace(go.Scatter(
            x=xf, y=yf,
            fill="toself",
            fillcolor="rgba(150,150,150,0.22)",
            line=dict(color="rgba(0,0,0,0)"),
            hoverinfo="skip",
            name="Analysis window"
        ))

# Lines
fig.add_trace(go.Scatter(
    x=x, y=yA,
    mode="lines",
    name=f"{curve_a}",
    line=dict(width=2, color="blue"),
))
fig.add_trace(go.Scatter(
    x=x, y=yB,
    mode="lines",
    name=f"{curve_b}",
    line=dict(width=2, color="red"),
))

# Stall markers (first y<=0 crossing)
if show_stall_lines:
    stallA = stall_from_roots(curve_a)
    stallB = stall_from_roots(curve_b)

    if stallA is not None and xmin_common <= stallA <= xmax_common:
        fig.add_vline(
            x=stallA,
            line_width=1,
            line_dash="dot",
            line_color="blue",
            annotation_text=f"{curve_a} stall ~{stallA:.3f}",
            annotation_position="top left"
        )

    if stallB is not None and xmin_common <= stallB <= xmax_common:
        fig.add_vline(
            x=stallB,
            line_width=1,
            line_dash="dot",
            line_color="red",
            annotation_text=f"{curve_b} stall ~{stallB:.3f}",
            annotation_position="top right"
        )

fig.update_layout(
    xaxis_title="Pressure (lb/in²)",
    yaxis_title="Velocity (in/s)",
    height=680,
    hovermode="x",
    legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.72),
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
)

st.plotly_chart(fig, use_container_width=True)

# =========================
# Equation Details
# =========================
if show_equations:
    st.subheader("Curve Definitions")
    with st.expander("Show equations, domains, and notes", expanded=False):
        for nm in [curve_a, curve_b]:
            spec = CURVE_META[nm]
            dom = spec["domain"]
            notes = spec.get("notes", "")
            st.markdown(f"**{nm}**  ·  Family: `{spec['family']}`")
            st.code(fmt_equation(nm), language="text")
            st.write(f"Valid range: {dom[0]:.3f} to {dom[1]:.3f} lb/in²")
            if notes:
                st.caption(notes)
            st.markdown("---")

# =========================
# Export: Sampled datasets
# =========================
st.subheader("Exports")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Sampled points (selected curves only)**")
    export_step = st.number_input(
        "Pressure step for export (lb/in²)",
        min_value=0.001,
        max_value=0.200,
        value=0.010,
        step=0.001,
        help="Creates a sampled dataset from the model curves over the overlapping domain."
    )

with col2:
    st.markdown("**Include columns**")
    include_delta = st.checkbox("Include ΔV (A − B)", value=True)
    include_ratio = st.checkbox("Include Ratio (A / B)", value=True)

# Build export grid
grid = np.arange(xmin_common, xmax_common + export_step/2, export_step)
VA = np.array([fA(v) for v in grid])
VB = np.array([fB(v) for v in grid])

# CSV content
header = ["Pressure_lb_in2", f"V_{curve_a}", f"V_{curve_b}"]
cols = [grid, VA, VB]

if include_delta:
    header.append("Delta_V_A_minus_B")
    cols.append(VA - VB)

if include_ratio:
    header.append("Ratio_A_div_B")
    ratio = np.where(VB == 0, np.nan, VA / VB)
    cols.append(ratio)

arr = np.column_stack(cols)

# Format numeric precision
def to_csv_string(array, headers):
    lines = [",".join(headers)]
    for row in array:
        out = []
        for v in row:
            if isinstance(v, (float, np.floating)):
                if np.isnan(v):
                    out.append("")
                else:
                    out.append(f"{v:.6f}")
            else:
                out.append(str(v))
        lines.append(",".join(out))
    return "\n".join(lines)

csv_text = to_csv_string(arr, header)

st.download_button(
    label="Download sampled comparison CSV",
    data=csv_text.encode("utf-8"),
    file_name="velocity_model_curves_comparison.csv",
    mime="text/csv"
)

st.caption(
    "Note: Exports are generated from the cubic equations over the overlapping valid domain "
    "of the two selected curves."
)
