import streamlit as st
import pandas as pd
import plotly.graph_objects as go

# Configure page BEFORE any other Streamlit calls
st.set_page_config(page_title="Decibel Measurements", layout="wide")

# --- Title ---
st.title("Decibel Measurements")
st.caption("Use the sidebar to add/remove brush types.")

# --- Read CSV ---
CSV_PATH = "data/decibel_measurements.csv"
try:
    df = pd.read_csv(CSV_PATH)
except FileNotFoundError:
    st.error(f"File not found: {CSV_PATH}")
    st.stop()
except Exception as e:
    st.error(f"Couldn't read {CSV_PATH}: {e}")
    st.stop()

# --- Identify time column (optional) ---
cols_lower = {c.lower(): c for c in df.columns}
time_col = None
if "timestamp" in cols_lower:
    time_col = cols_lower["timestamp"]
    try:
        df[time_col] = pd.to_datetime(df[time_col])
        x = df[time_col]
    except Exception:
        x = df.index
else:
    x = df.index

# --- Choose series to plot: all numeric columns except timestamp ---
# Try to coerce likely level columns to numeric if needed
likely_level_names = [c for c in df.columns if any(k in c.lower() for k in ["dba", "dbc", "db", "spl", "level", "value"])]
for c in likely_level_names:
    if not pd.api.types.is_numeric_dtype(df[c]):
        df[c] = pd.to_numeric(df[c], errors="coerce")

numeric_cols = df.select_dtypes(include="number").columns.tolist()
if time_col in numeric_cols:
    numeric_cols.remove(time_col)

if not numeric_cols:
    st.error("No numeric decibel columns found to plot. Add columns like dBA, dB, SPL, or Level.")
    st.stop()

# --- Select which columns to show ---
REQUIRED = ["Ambient", "Empty Feeder", "Full Feeder"]
# Only keep those that actually exist & are numeric
required_present = [c for c in REQUIRED if c in numeric_cols]

# Optional pool = all other numeric series
optional_pool = [c for c in numeric_cols if c not in REQUIRED]

# Sort options in a consistent, custom order
priority_order = ["AngleOn", "Competitor", "XT10", "XT16"]
def sort_key(name: str):
    try:
        idx = priority_order.index(name)
    except ValueError:
        idx = len(priority_order)
    return (idx, name)

optional_pool_sorted = sorted(optional_pool, key=sort_key)

# Define defaults: only show AngleOn and Competitor initially (if present)
default_names = {"AngleOn", "Competitor"}
default_optional = [c for c in optional_pool_sorted if c in default_names]

# Sidebar multiselect to choose optional series; default to AngleOn + Competitor only
selected_optional = st.sidebar.multiselect(
    "Optional series",
    options=optional_pool_sorted,
    default=default_optional,
)

# --- Build figure ---
fig = go.Figure()

# Add required series first (always shown)
for col in required_present:
    fig.add_trace(go.Scatter(x=x, y=df[col], mode="lines", name=col))

# Add user-selected optional series
for col in selected_optional:
    fig.add_trace(go.Scatter(x=x, y=df[col], mode="lines", name=col))

# OSHA TWA Action Level line at 85 dBA (legend-visible)
if len(df) > 0:
    if hasattr(x, "iloc"):
        x0, x1 = x.iloc[0], x.iloc[-1]
    else:
        x0, x1 = x[0], x[-1]
    fig.add_trace(
        go.Scatter(
            x=[x0, x1],
            y=[85, 85],
            mode="lines",
            name="OSHA TWA Action Level (85 dBA)",
            line=dict(dash="dash", width=2)
        )
    )

fig.update_layout(
    template="plotly_white",
    height=460,
    margin=dict(l=40, r=20, t=20, b=40),
    legend=dict(orientation="h", x=0, y=1.12, bgcolor="rgba(0,0,0,0)"),
    xaxis_title=None,
    yaxis_title="dBA",
)

st.plotly_chart(fig, use_container_width=True)
