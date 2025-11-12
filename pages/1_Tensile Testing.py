# pages/Pile_Pull_Results.py
# ---------------------------------------------------------------
# Streamlit page: AngleOn™ Pile Pull Results (simplified static version)
# - Displays fixed dataset (from your original test)
# - Defaults to Box Plot visualization
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
# Sample Dataset (from your report)
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
# Summary Metrics
# ---------------------------
mean_load = df["Max Load (lbf)"].mean()
mean_disp = df["Disp @ Max (in)"].mean()
max_load = df["Max Load (lbf)"].max()
min_load = df["Max Load (lbf)"].min()

m1, m2, m3, m4 = st.columns(4)
m1.metric("Sample Size", len(df))
m2.metric("Mean Max Load (lbf)", f"{mean_load:.2f}")
m3.metric("Mean Disp @ Max (in)", f"{mean_disp:.3f}")
m4.metric("Range (lbf)", f"{min_load:.1f} – {max_load:.1f}")

# ---------------------------
# Default Visualization: Box Plots
# ---------------------------
fig = go.Figure()
fig.add_trace(go.Box(y=df["Max Load (lbf)"], name="Max Load (lbf)", boxmean=True))
fig.add_trace(go.Box(y=df["Disp @ Max (in)"], name="Disp @ Max (in)", boxmean=True))
fig.update_layout(
    title="Distribution of Max Load & Displacement @ Max",
    yaxis_title="Value",
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
