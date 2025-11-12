import streamlit as st
import pandas as pd
import plotly.express as px
from plotly.colors import qualitative

# Set up the Streamlit page
st.set_page_config(page_title="Brush Velocity Boxplots", layout="wide")
st.title("Object Velocity Distribution by Brush Type")

st.caption("""
Key Points:    
This boxplot shows the distribution of velocity measurements of all objects tested across different brush types. Each box represents the spread of velocity 
measurements for that particular brush including individual data points. Use this to compare consistency and central tendency across brush types. AngleOn™ out performs Competitor product.
""")

# Load the CSV file
df = pd.read_csv("data/Velocity_Boxplots.csv")  # assumes file is in 'data/' folder

# Show the raw columns for debugging
#st.subheader("Raw CSV Columns")
#st.write(df.columns)
#st.write(f"Number of columns: {len(df.columns)}")

# Rename columns for clarity (optional)
df.columns = ["AngleOn™", "XT10", "XT16", "Competitor"]

# Convert wide to long format
df_long = df.melt(var_name="Brush Type", value_name="Velocity (in/sec)")

# Clean brush names (optional)
df_long["Brush Type"] = df_long["Brush Type"].str.replace("-Velocity", "")

# Define consistent color palette
color_sequence = qualitative.Plotly
brush_order = ["AngleOn™", "XT10", "XT16", "Competitor"]  # adjust as needed

# Plot
fig = px.box(
    df_long,
    x="Brush Type",
    y="Velocity (in/sec)",
    color="Brush Type",
    color_discrete_sequence=color_sequence,
    category_orders={"Brush Type": sorted(df_long["Brush Type"].unique())},  # optional consistent order
    points="all",  # show all data points
    template="plotly_white"
)

# Layout tweaks
fig.update_layout(
    title="Velocity Distributions by Brush Type",
    xaxis_title="Brush Type",
    yaxis_title="Velocity (in/sec)",
    legend_title="Brush Type",
    height=600,
    showlegend=False
)

# Display the plot
st.plotly_chart(fig, use_container_width=True)
