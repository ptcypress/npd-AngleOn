# pages/Pile_Pull_Results.py
# ---------------------------------------------------------------
# Streamlit page: AngleOn™ Pile Pull Results (toggle metric)
# - Fixed dataset (from your original test)
# - Default visualization: Box Plot
# - Toggle between "Max Load (lbf)" and "Disp @ Max (in)"
# ---------------------------------------------------------------

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ---------------------------
# Page Setup
# ---------------------------
st.set_page_config(page_title="Pile Pull Results", layout="wide")
st.title("Tensile Testing — AngleOn™")
st.caption("""
Key Points:    
INSTRON Tensile testing for AngleOn™ raw material.    
-**Average Max Load:                   33.02lbF**    
-**Average Displacement @ Max Load:    0.121in**  
""")

# ---------------------------
# Fixed Dataset (from your report)
# ---------------------------
sample = [
    (1,  "LC306064-42PN BK 100R", 34.28, 0.22),
    (2,  "LC306064-42PN BK 100R", 28.15, 0.25),
    (3,  "LC306064-42PN BK 100R", 40.83, 0.10),
    (4,  "LC306064-42PN BK 100R", 33.11, 0.07),
    (5,  "LC306064-42PN BK 100R", 26.07, 0.10),
    (6,  "LC306064-42PN BK 100R", 33.35, 0.09),
    (7,  "LC306064-42PN BK 100R", 38.23, 0.17),
    (8,  "LC306064-42PN BK 100R", 33.32, 0.11),
    (9,  "LC306064-42PN BK 100R", 40.61, 0.08),
    (10, "LC306064-42PN BK 100R", 26.18, 0.09),
    (11, "LC306064-42PN BK 100R", 33.67, 0.08),
    (12, "LC306064-42PN BK 100R", 36.89, 0.09),
    (13, "LC306064-42PN BK 100R", 30.44, 0.26),
    (14, "LC306064-42PN BK 100R", 34.77, 0.13),
    (15, "LC306064-42PN BK 100R", 29.99, 0.12),
    (16, "LC306064-42PN BK 100R", 29.25, 0.08),
    (17, "LC306064-42PN BK 100R", 34.98, 0.07),
    (18, "LC306064-42PN BK 100R", 26.10, 0.15),
    (19, "LC306064-42PN BK 100R", 34.29, 0.08),
    (20, "LC306064-42PN BK 100R", 35.80, 0.08),
]
df = pd.DataFrame(sample, columns=["Specimen", "Type", "Max Load (lbf)", "Disp @ Max (in)"])

# ---------------------------
# Controls (left-aligned)
# ---------------------------
c1, = st.columns([3])
with c1:
    metric = st.selectbox("Metric", ["Max Load (lbf)", "Disp @ Max (in)"], index=0)

# ---------------------------
# Summary Metrics (for selected metric)
# ---------------------------
series = df[metric]
mean_val = series.mean()
min_val = series.min()
max_val = series.max()
p25 = series.quantile(0.25)
median = series.median()
p75 = series.quantile(0.75)

m1, m2, m3, m4 = st.columns(4)
m1.metric("Sample Size", len(series))
m2.metric(f"Mean {metric}", f"{mean_val:.3f}")
m3.metric(f"Median {metric}", f"{median:.3f}")
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
st.download_button("Download Data (CSV)", data=csv_data, file_name="pile_pull_results.csv", mime="text/csv")

with st.expander("Show data table"):
    st.dataframe(df, use_container_width=True)
