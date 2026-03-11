import numpy as np
import streamlit as st
import plotly.graph_objects as go
from scipy.integrate import quad

# =========================
# Page Setup
# =========================
st.set_page_config(page_title="Velocity vs Load", layout="wide")
st.title("Velocity vs Load — AngleOn™ & AngleOn XT")
st.caption(
    "Objects of varying mass were conveyed along a linear vibratory feeder using different brush constructions. " 
    "Object velocity was measured as a function of applied normal load (pressure), and cubic regression models were used to "
    "describe velocity–load behavior up to stall (V = 0)."
)

# =========================
# Curve Registry (Pressure-based)
# y = a0 + a1*x + a2*x^2 + a3*x^3
# x = Pressure (lb/in²)
# y = Velocity (in/s)
# domain_min = minimum valid x (per your Desmos constraints)
# =========================
CURVES = {
    "Angle": {
        "AngleOn™": {
            "coeffs": {"a0": 4.5210, "a1": -3.9740, "a2": 1.9270, "a3": -0.3569},
            "domain_min": 0.167,
            "notes": "AngleOn™ cubic fit",
        },
        "Competitor": {
            "coeffs": {"a0": 3.0780, "a1": -2.4730, "a2": 1.3470, "a3": -0.2774},
            "domain_min": 0.166,
            "notes": "Competitor cubic fit",
        },
    },
    "XT": {
        "XT10250-625 Rev.1": {
            "coeffs": {"a0": 6.2710, "a1": -12.0900, "a2": 13.5200, "a3": -5.0130},
            "domain_min": 0.166,
            "notes": "XT10250-625 Rev.1 cubic fit",
        },
        "XT10250-250 Rev.1": {
            "coeffs": {"a0": 3.3280, "a1": -2.1110, "a2": 1.1840, "a3": -0.2386},
            "domain_min": 0.166,
            "notes": "XT10250-250 Rev.1 cubic fit",
        },
        "XT06250-250 Rev.1": {
            "coeffs": {"a0": 3.9049, "a1": -7.1959, "a2": 6.2119, "a3": -1.8053},
            "domain_min": 0.166,
            "notes": "XT06250-250 Rev.1 cubic fit",
        },
        "XT16125-625 Rev.1": {
            "coeffs": {"a0": 3.3280, "a1": -2.1110, "a2": 1.1840, "a3": -0.2386},
            "domain_min": 0.166,
            "notes": "XT16125-625 Rev.1 cubic fit",
        },
        "XT16125-625 Rev.2": {
            "coeffs": {"a0": 3.4984, "a1": -6.0848, "a2": 5.3119, "a3": -1.5377},
            "domain_min": 0.166,
            "notes": "XT16125-625 Rev.2 cubic fit",
        },
        "XT16115-625 Rev.1": {
            "coeffs": {"a0": 3.3025, "a1": -5.8152, "a2": 4.8504, "a3": -1.3582},
            "domain_min": 0.166,
            "notes": "XT16115-625 Rev.1 cubic fit",
        },
        # renamed: removed "(Projected)" per your request
        "XT16105-625 Rev.1": {
            "coeffs": {"a0": 3.1066, "a1": -5.5456, "a2": 4.3889, "a3": -1.1787},
            "domain_min": 0.166,
            "notes": "XT16105-625 Rev.1 cubic fit",
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

def stall_root(name: str) -> float | None:
    """
    Smallest real root >= domain_min where V(P)=0.
    Returns None if no such root exists.
    """
    spec = CURVE_META[name]
    c = spec["coeffs"]
    dom_lo = float(spec["domain_min"])

    # a3*x^3 + a2*x^2 + a1*x + a0 = 0
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
    Domain is [domain_min, stall], so curves always extend to y=0.
    If no stall root exists, we fall back to a reasonable max window.
    """
    dom_lo = float(CURVE_META[name]["domain_min"])
    s = stall_root(name)
    if s is None or not np.isfinite(s):
        # fallback: show a bit beyond typical range
        return dom_lo, max(dom_lo + 0.5, 3.5)
    # Guard: if numerical issues produce stall <= dom_lo, keep a tiny range
    if s <= dom_lo:
        return dom_lo, dom_lo + 0.05
    return dom_lo, float(s)

def safe_quad(func, a, b) -> float:
    if a == b:
        return 0.0
    lo, hi = min(a, b), max(a, b)
    val, _ = quad(func, lo, hi)
    return float(val)

def fmt_equation(name: str) -> str:
    c = CURVE_META[name]["coeffs"]
    return (
        f"V = {c['a0']:.4f} + ({c['a1']:.4f})·P + ({c['a2']:.4f})·P² + ({c['a3']:.4f})·P³"
    )

# =========================
# Sidebar Controls
# =========================
st.sidebar.header("Controls")

mode = st.sidebar.radio(
    "Mode",
    options=["Overview (many curves)", "Compare (A vs B)"],
    index=0
)

family_filter = st.sidebar.multiselect(
    "Families",
    options=["Angle", "XT"],
    default=["Angle", "XT"]
)

selected_curves = [c for c in ALL_CURVES if CURVE_META[c]["family"] in family_filter]
if not selected_curves:
    st.error("No curves selected — choose at least one family.")
    st.stop()

# plotting controls
st.sidebar.markdown("---")
st.sidebar.subheader("Plot Quality")
n_points = st.sidebar.slider("Resolution (points per curve)", 200, 2000, 900, 100)

show_stall_markers = st.sidebar.checkbox("Show stall markers (V=0)", value=True)
show_equations = st.sidebar.checkbox("Show equation details", value=False)

# =========================
# Compare controls
# =========================
curve_a = curve_b = None
low_val = high_val = None
xmin_common = xmax_common = None

if mode.startswith("Compare"):
    st.sidebar.markdown("---")
    st.sidebar.subheader("Comparison")

    # defaults
    default_a = "AngleOn™" if "AngleOn™" in selected_curves else selected_curves[0]
    default_b = "Competitor" if "Competitor" in selected_curves else (selected_curves[1] if len(selected_curves) > 1 else selected_curves[0])

    curve_a = st.sidebar.selectbox("Curve A", options=selected_curves, index=selected_curves.index(default_a))
    curve_b = st.sidebar.selectbox("Curve B", options=selected_curves, index=selected_curves.index(default_b))

    if curve_a == curve_b:
        st.sidebar.warning("Choose two different curves to compare.")

    # common domain for analysis: overlap of the two *effective* domains
    a_lo, a_hi = effective_domain(curve_a)
    b_lo, b_hi = effective_domain(curve_b)
    xmin_common = max(a_lo, b_lo)
    xmax_common = min(a_hi, b_hi)

    if xmax_common <= xmin_common:
        st.error("No overlapping domain between A and B (after extending to stall).")
        st.stop()

    st.sidebar.markdown("---")
    st.sidebar.subheader("Analysis Window (overlap)")
    step_val = max((xmax_common - xmin_common) / 200.0, 0.001)
    low_val, high_val = st.sidebar.slider(
        "Pressure range (lb/in²)",
        min_value=float(xmin_common),
        max_value=float(xmax_common),
        value=(float(xmin_common), float(xmax_common)),
        step=float(step_val),
    )

# =========================
# Advantage (Compare)
# =========================
if mode.startswith("Compare"):
    st.subheader("Advantage (Model Curves)")

    cA = CURVE_META[curve_a]["coeffs"]
    cB = CURVE_META[curve_b]["coeffs"]

    def fA(x): return max(0.0, cubic_eval(float(x), cA))
    def fB(x): return max(0.0, cubic_eval(float(x), cB))
    def diff(x): return fA(x) - fB(x)

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
st.subheader("Velocity vs Object Pressure")

fig = go.Figure()

# Analysis shading for compare mode (built from A & B over overlap)
if mode.startswith("Compare") and abs(high_val - low_val) > 1e-12:
    cA = CURVE_META[curve_a]["coeffs"]
    cB = CURVE_META[curve_b]["coeffs"]
    xs = np.linspace(xmin_common, xmax_common, 700)
    yA = np.array([max(0.0, cubic_eval(x, cA)) for x in xs])
    yB = np.array([max(0.0, cubic_eval(x, cB)) for x in xs])

    mask = (xs >= low_val) & (xs <= high_val)
    if np.any(mask):
        xf = np.concatenate([xs[mask], xs[mask][::-1]])
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
if mode.startswith("Overview"):
    plot_curves = selected_curves
else:
    plot_curves = sorted(set(selected_curves) | {curve_a, curve_b})

# Plot each curve from domain_min to stall
for nm in plot_curves:
    c = CURVE_META[nm]["coeffs"]
    lo, hi = effective_domain(nm)

    xs = np.linspace(lo, hi, int(n_points))
    ys = np.array([max(0.0, cubic_eval(x, c)) for x in xs])

    fig.add_trace(go.Scatter(
        x=xs, y=ys,
        mode="lines",
        name=nm,
        line=dict(width=2),
        hovertemplate="P=%{x:.3f} lb/in²<br>V=%{y:.3f} in/s<extra>"+nm+"</extra>"
    ))

    if show_stall_markers:
        s = stall_root(nm)
        if s is not None and np.isfinite(s):
            fig.add_trace(go.Scatter(
                x=[s], y=[0],
                mode="markers",
                marker=dict(size=9, symbol="x"),
                name=f"{nm} stall",
                hovertemplate=f"{nm}<br>stall<br>P={s:.3f} lb/in²<br>V=0<extra></extra>",
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
    with st.expander("Show equations and computed stall domains", expanded=True):
        for nm in plot_curves:
            spec = CURVE_META[nm]
            dom_lo = spec["domain_min"]
            s = stall_root(nm)
            lo, hi = effective_domain(nm)

            st.markdown(f"**{nm}**  ·  Family: `{spec['family']}`")
            st.code(fmt_equation(nm), language="text")
            st.write(f"Minimum pressure: {dom_lo:.3f} lb/in²")
            if s is None:
                st.caption("No real stall root found; curve may not reach V=0 under this cubic.")
            else:
                st.write(f"Computed stall pressure (V=0): {s:.6f} lb/in²")
            st.write(f"Plot domain used (no extrapolation): {lo:.3f} to {hi:.3f} lb/in²")
            if spec.get("notes"):
                st.caption(spec["notes"])
            st.markdown("---")
