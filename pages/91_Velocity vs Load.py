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
    "Model-space comparison using validated cubic regression curves. "
    "Solid lines = validated domain. Dashed lines = extrapolated to stall (y=0) when stall occurs beyond domain."
)

# =========================
# Curve Registry
# y = a0 + a1*x + a2*x^2 + a3*x^3
# x = Pressure (lb/in²)
# y = Velocity (in/s)
# domain = validated x-range for that curve
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
            # You only had x>0.166 in your definition, so we treat domain_hi as "validated range"
            # and allow dashed extrapolation to show stall if it occurs later.
            "domain": (0.166, 1.700),
            "notes": "High coupling at low load; collapses at higher load",
        },
        "XT16-125 Rev.1": {
            "coeffs": {"a0": 3.3280, "a1": -2.1110, "a2": 1.1840, "a3": -0.2386},
            "domain": (0.166, 3.500),
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

# Flatten
CURVE_META = {}
for fam, d in CURVES.items():
    for nm, spec in d.items():
        CURVE_META[nm] = {"family": fam, **spec}

ALL_CURVES = sorted(CURVE_META.keys())

# =========================
# Helpers
# =========================
def cubic_eval(x: float, c: dict) -> float:
    return c["a0"] + c["a1"] * x + c["a2"] * (x**2) + c["a3"] * (x**3)

def fmt_equation(name: str) -> str:
    c = CURVE_META[name]["coeffs"]
    return (
        f"V = {c['a0']:.4f} + ({c['a1']:.4f})·P + ({c['a2']:.4f})·P² + ({c['a3']:.4f})·P³"
    )

def safe_quad(func, a, b) -> float:
    if a == b:
        return 0.0
    lo, hi = min(a, b), max(a, b)
    val, _ = quad(func, lo, hi)
    return float(val)

def stall_root(name: str) -> float | None:
    """
    Returns the smallest real root >= domain_lo where cubic crosses V=0.
    If none exist, returns None.
    """
    spec = CURVE_META[name]
    c = spec["coeffs"]
    dom_lo, _ = spec["domain"]

    # Polynomial coefficients for np.roots: a3*x^3 + a2*x^2 + a1*x + a0 = 0
    coeffs = [c["a3"], c["a2"], c["a1"], c["a0"]]
    roots = np.roots(coeffs)

    real_roots = []
    for r in roots:
        if abs(r.imag) < 1e-8:
            x = float(r.real)
            if x >= dom_lo:
                real_roots.append(x)

    if not real_roots:
        return None
    return float(min(real_roots))

def curve_segments(name: str, n_points: int, clamp_negative: bool):
    """
    Produces up to two segments for plotting:
    - solid: validated domain (dom_lo -> dom_hi) but trimmed to stall if stall is within domain
    - dashed: extrapolated (dom_hi -> stall) if stall occurs after domain_hi
    Also returns stall_x and whether it is extrapolated.
    """
    spec = CURVE_META[name]
    c = spec["coeffs"]
    dom_lo, dom_hi = spec["domain"]
    s = stall_root(name)

    # Define where solid segment ends
    solid_end = dom_hi
    if s is not None and s <= dom_hi:
        solid_end = s

    def yfun(x):
        y = cubic_eval(x, c)
        return max(0.0, y) if clamp_negative else y

    segs = []

    # Solid segment
    xs1 = np.linspace(dom_lo, solid_end, max(2, int(n_points)))
    ys1 = np.array([yfun(x) for x in xs1])
    segs.append(("solid", xs1, ys1))

    # Dashed extrapolation if stall beyond validated domain_hi
    extrapolated = False
    if s is not None and s > dom_hi:
        extrapolated = True
        xs2 = np.linspace(dom_hi, s, max(2, int(n_points * 0.6)))
        ys2 = np.array([yfun(x) for x in xs2])
        segs.append(("dash", xs2, ys2))

    return segs, s, extrapolated

# =========================
# Sidebar Controls
# =========================
st.sidebar.header("Controls")

mode = st.sidebar.radio("Mode", options=["Overview (all curves)", "Compare (A vs B)"], index=0)
family_filter = st.sidebar.multiselect("Families", options=["Angle", "XT"], default=["Angle", "XT"])
clamp_negative = st.sidebar.checkbox("Clamp velocities below 0 to 0", value=True)
show_equations = st.sidebar.checkbox("Show equation details", value=False)
show_stall_markers = st.sidebar.checkbox("Show stall markers", value=True)

st.sidebar.markdown("---")
st.sidebar.subheader("Plot Quality")
n_points = st.sidebar.slider("Resolution (points per curve)", min_value=200, max_value=1600, value=700, step=100)

selected_curves = [c for c in ALL_CURVES if CURVE_META[c]["family"] in family_filter]
if not selected_curves:
    st.error("No curves selected (choose at least one family).")
    st.stop()

# =========================
# Compare mode selections
# =========================
curve_a = curve_b = None
xmin_common = xmax_common = None

if mode.startswith("Compare"):
    st.sidebar.markdown("---")
    st.sidebar.subheader("Comparison")

    default_a = "AngleOn™" if "AngleOn™" in selected_curves else selected_curves[0]
    default_b = "Competitor" if "Competitor" in selected_curves else (selected_curves[1] if len(selected_curves) > 1 else selected_curves[0])

    curve_a = st.sidebar.selectbox("Curve A", options=selected_curves, index=selected_curves.index(default_a))
    curve_b = st.sidebar.selectbox("Curve B", options=selected_curves, index=selected_curves.index(default_b))

    if curve_a == curve_b:
        st.sidebar.warning("Choose two different curves to compare.")

    domA = CURVE_META[curve_a]["domain"]
    domB = CURVE_META[curve_b]["domain"]
    xmin_common = max(domA[0], domB[0])
    xmax_common = min(domA[1], domB[1])

    if xmax_common <= xmin_common:
        st.error("No overlapping validated domain between Curve A and Curve B.")
        st.stop()

    st.sidebar.markdown("---")
    st.sidebar.subheader("Analysis Window (validated overlap)")
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
# Advantage (Compare mode only)
# =========================
if mode.startswith("Compare"):
    st.subheader("Advantage (Model Curves)")
    cA = CURVE_META[curve_a]["coeffs"]
    cB = CURVE_META[curve_b]["coeffs"]

    def fA(x): 
        y = cubic_eval(float(x), cA)
        return max(0.0, y) if clamp_negative else y

    def fB(x):
        y = cubic_eval(float(x), cB)
        return max(0.0, y) if clamp_negative else y

    def diff(x):
        return fA(x) - fB(x)

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
# Plot (all curves or selected)
# =========================
st.subheader("Velocity vs Pressure (Cubic Fits)")

fig = go.Figure()

# In compare mode, shade analysis window on the plot (validated overlap only)
if mode.startswith("Compare") and abs(high_val - low_val) > 1e-12:
    # We'll build shading from A and B sampled over the overlap window
    xshade = np.linspace(xmin_common, xmax_common, 600)

    cA = CURVE_META[curve_a]["coeffs"]
    cB = CURVE_META[curve_b]["coeffs"]
    yA = np.array([max(0.0, cubic_eval(x, cA)) if clamp_negative else cubic_eval(x, cA) for x in xshade])
    yB = np.array([max(0.0, cubic_eval(x, cB)) if clamp_negative else cubic_eval(x, cB) for x in xshade])

    mask = (xshade >= low_val) & (xshade <= high_val)
    if np.any(mask):
        xf = np.concatenate([xshade[mask], xshade[mask][::-1]])
        yf = np.concatenate([yA[mask], yB[mask][::-1]])
        fig.add_trace(go.Scatter(
            x=xf, y=yf,
            fill="toself",
            fillcolor="rgba(150,150,150,0.22)",
            line=dict(color="rgba(0,0,0,0)"),
            hoverinfo="skip",
            name="Analysis window"
        ))

# Curves to plot
plot_curves = selected_curves if mode.startswith("Overview") else sorted(set(selected_curves) | {curve_a, curve_b})

# Add each curve in solid (validated) + dashed (extrapolated-to-stall) segments
for nm in plot_curves:
    segs, s, extrap = curve_segments(nm, n_points=n_points, clamp_negative=clamp_negative)

    # Solid segment
    kind, xs, ys = segs[0]
    fig.add_trace(go.Scatter(
        x=xs, y=ys,
        mode="lines",
        name=f"{nm}",
        line=dict(width=2),
        hovertemplate="P=%{x:.3f} lb/in²<br>V=%{y:.3f} in/s<extra>"+nm+"</extra>"
    ))

    # Dashed extrapolated segment (if present)
    if len(segs) > 1:
        _, xs2, ys2 = segs[1]
        fig.add_trace(go.Scatter(
            x=xs2, y=ys2,
            mode="lines",
            name=f"{nm} (extrapolated)",
            line=dict(width=2, dash="dash"),
            hovertemplate="P=%{x:.3f} lb/in²<br>V=%{y:.3f} in/s<extra>"+nm+" (extrapolated)</extra>",
            showlegend=False
        ))

    # Stall marker at (stall, 0)
    if show_stall_markers and s is not None and np.isfinite(s):
        # Label whether stall is beyond validated domain
        dom_hi = CURVE_META[nm]["domain"][1]
        tag = "extrapolated stall" if s > dom_hi else "stall"
        fig.add_trace(go.Scatter(
            x=[s], y=[0],
            mode="markers",
            marker=dict(size=9, symbol="x"),
            name=f"{nm} {tag}",
            hovertemplate=f"{nm}<br>{tag}<br>P={s:.3f} lb/in²<br>V=0<extra></extra>",
            showlegend=False
        ))

fig.update_layout(
    xaxis_title="Pressure (lb/in²)",
    yaxis_title="Velocity (in/s)",
    height=720,
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
    with st.expander("Show equations / domains / notes", expanded=True):
        for nm in plot_curves:
            spec = CURVE_META[nm]
            dom = spec["domain"]
            notes = spec.get("notes", "")
            st.markdown(f"**{nm}**  ·  Family: `{spec['family']}`")
            st.code(fmt_equation(nm), language="text")
            st.write(f"Validated domain: {dom[0]:.3f} to {dom[1]:.3f} lb/in²")
            s = stall_root(nm)
            if s is None:
                st.caption("No real stall (V=0) root found within cubic model.")
            else:
                extrap_txt = " (extrapolated beyond domain)" if s > dom[1] else ""
                st.caption(f"Estimated stall pressure: {s:.3f} lb/in²{extrap_txt}")
            if notes:
                st.caption(notes)
            st.markdown("---")
