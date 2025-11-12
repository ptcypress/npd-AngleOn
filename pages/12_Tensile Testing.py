# pages/N6_Tensile_Pulls.py
# ---------------------------------------------------------------
# Streamlit page: N6 Belt — Tensile Pulls (toggle metric)
# - Fixed dataset (from lab report)
# - Default visualization: Box Plot
# - Toggle between: Maximum Force, Displacement @ Max, Force @ 0.05 in
# ---------------------------------------------------------------

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ---------------------------
# Page Setup
# ---------------------------
st.set_page_config(page_title="N6 Belt — Tensile Pulls", layout="wide")
st.title("Tensile Testing — AngleOn™")
st.caption("""
Key Points: 
INSTRON tensile testing of AngleOn™ raw material.
-**Average Max Load:                   (Exceeded test parameters)**
-**Average Displacement @ Max Load:    0.937in** 
""")

# ---------------------------
# Fixed Dataset (from report)
# Columns: Sample, Max Force (lbf), Disp @ Max (in), Force @ 0.05in (lbf)
# ---------------------------
rows = [
    ( 1, 191.80, 0.96, 19.52),
    ( 2, 200.00, 1.02, 20.62),
    ( 3, 200.00, 0.98, 18.23),
    ( 4, 200.00, 0.88, 36.30),
    ( 5, 200.00, 0.87, 41.41),
    ( 6, 200.00, 0.87, 40.04),
    ( 7, 200.00, 0.85, 42.98),
    ( 8, 200.00, 0.92, 27.42),
    ( 9, 200.00, 0.88, 39.00),
    (10, 200.00, 0.86, 40.60),
    (11, 200.00, 0.93, 31.12),
    (12, 200.00, 0.89, 35.83),
    (13, 200.01, 0.88, 41.10),
    (14, 200.00, 0.85, 39.41),
    (15, 200.00, 0.91, 27.66),
    (16, 200.00, 0.90, 32.14),
    (17, 200.00, 0.87, 33.70),
    (18, 200.00, 0.88, 34.10),
    (19, 200.00, 0.89, 31.93),
    (20, 200.00, 0.90, 34.54),
    (21, 200.00, 0.95, 34.31),
    (22, 200.00, 0.93, 34.22),
    (23, 200.00, 0.91, 31.02),
    (24, 200.00, 1.14,  0.50),
    (25, 200.00, 1.15,  0.71),
    (26, 200.00, 1.15,  0.82),
    (27, 200.00, 0.93, 32.92),
    (28, 200.00, 0.89, 39.23),
    (29, 200.00, 0.93, 36.08),
    (30, 200.00, 1.00, 31.14),
    (31, 200.00, 0.94, 29.58),
    (32, 200.00, 0.97, 27.46),
    (33, 200.00, 0.92, 31.84),
    (34, 200.00, 0.94, 30.87),
    (35, 200.00, 0.98, 28.99),
    (36, 200.00, 0.97, 29.21),
    (37, 200.00, 0.94, 29.30),
    (38, 200.00, 0.96, 24.98),
    (39, 200.00, 0.91, 37.16),
    (40, 200.00, 0.97, 24.09),
]
df = pd.DataFrame(rows, columns=["Sample", "Maximum Force (lbf)", "Displacement @ Max (in)", "Force @ 0.05 in (lbf)"])

# ---------------------------
# Control: Metric toggle (left-aligned, no sidebar)
# ---------------------------
c1, = st.columns([3])
with c1:
    metric = st.selectbox(
        "Metric",
        ["Maximum Force (lbf)", "Displacement @ Max (in)", "Force @ 0.05 in (lbf)"],
        index=0
    )

# ---------------------------
# Summary Metrics (for selected metric)
# ---------------------------
series = df[metric]
mean_val = float(series.mean())
median_val = float(series.median())
min_val = float(series.min())
max_val = float(series.max())

m1, m2, m3, m4 = st.columns(4)
m1.metric("Samples", len(series))
m2.metric(f"Mean {metric}", f"{mean_val:.3f}")
m3.metric(f"Median {metric}", f"{median_val:.3f}")
m4.metric(f"Range {metric}", f"{min_val:.3f} – {max_val:.3f}")

# ---------------------------
# Box Plot (default visualization)
# ---------------------------
fig = go.Figure()
fig.add_trace(go.Box(y=series, name=metric, boxmean=True))
fig.update_layout(
    title=f"Distribution — {metric}",
    yaxis_title=metric,
    height=460,
    margin=dict(l=40, r=20, t=60, b=40),
)
st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": True})

# ---------------------------
# Data Table & Download
# ---------------------------
csv_data = df.to_csv(index=False).encode("utf-8")
st.download_button("Download Data (CSV)", data=csv_data, file_name="n6_tensile_pulls.csv", mime="text/csv")

with st.expander("Show data table"):
    st.dataframe(df, use_container_width=True)

