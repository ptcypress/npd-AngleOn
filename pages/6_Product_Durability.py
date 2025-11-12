import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.interpolate import interp1d
from scipy.integrate import simpson
import os

# Streamlit page config
st.set_page_config(page_title="Material Loss Over Time", layout="wide")
st.title("Product Durability - AngleOn™ vs. Competitor")

# Load data
csv_path = "data/product_durability.csv"
if not os.path.exists(csv_path):
    st.error(f"File not found: {csv_path}")
else:
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()

    expected_cols = [
        "AngleOn™ Run Time (hrs)",
        "Competitor Product Run Time (hrs)",
        "Mat'l Loss (%)"
    ]

    if not all(col in df.columns for col in expected_cols):
        st.error("Missing required columns in CSV.")
        st.write("Available columns:", df.columns.tolist())
    else:
        # Extract relevant data
        y_loss = df["Mat'l Loss (%)"]
        x_angleon = df["AngleOn™ Run Time (hrs)"]
        x_comp = df["Competitor Product Run Time (hrs)"]

        # Interpolate data for smooth plotting
        angleon_interp = interp1d(x_angleon, y_loss, kind='linear', fill_value='extrapolate')
        competitor_interp = interp1d(x_comp, y_loss, kind='linear', fill_value='extrapolate')

        x_smooth = np.linspace(
            min(x_angleon.min(), x_comp.min()),
            max(x_angleon.max(), x_comp.max()),
            500
        )
        y_angleon_smooth = angleon_interp(x_smooth)
        y_comp_smooth = competitor_interp(x_smooth)

        # Calculate advantage and percent improvement
        area_diff = simpson(y_comp_smooth - y_angleon_smooth, x_smooth)
        baseline_area = simpson(y_comp_smooth, x_smooth)
        percent_improvement = (area_diff / baseline_area) * 100 if baseline_area > 0 else 0

        # Create shaded area coordinates
        x_fill = np.concatenate([x_smooth, x_smooth[::-1]])
        y_fill = np.concatenate([y_angleon_smooth, y_comp_smooth[::-1]])

        # Plot
        fig = go.Figure()

        # Shaded area
        fig.add_trace(go.Scatter(
            x=x_fill, y=y_fill,
            fill='toself',
            fillcolor='rgba(100,100,100,0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            hoverinfo='skip',
            name='Difference Area'
        ))

        # AngleOn™ line
        fig.add_trace(go.Scatter(
            x=x_smooth, y=y_angleon_smooth,
            mode='lines',
            name='AngleOn™',
            line=dict(color='blue', width=3),
            hovertemplate='Hour: %{x:.2f}<br>Loss: %{y:.1f}%'
        ))

        # Competitor line
        fig.add_trace(go.Scatter(
            x=x_smooth, y=y_comp_smooth,
            mode='lines',
            name='Competitor',
            line=dict(color='red', width=3),
            hovertemplate='Hour: %{x:.2f}<br>Loss: %{y:.1f}%'
        ))

        # Annotation
        fig.add_annotation(
            x=x_smooth[len(x_smooth)//3],
            y=max(y_comp_smooth) * 0.7,
            text=(f"Advantage area = {area_diff:.6f}<br>"
                  f"Relative advantage = {percent_improvement:.2f}%"),
            showarrow=False,
            font=dict(size=13)
        )

        # Layout and style
        fig.update_layout(
            xaxis_title="Runtime (Hours)",
            yaxis_title="Material Loss (%)",
            height=650,
            hovermode='x',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="center",
                x=0.5
            ),
            xaxis=dict(
                showspikes=True,
                spikemode="across",
                spikesnap="cursor",
                showline=True,
                spikecolor="lightgray",
                spikethickness=0.7,
                spikedash="dot"
            ),
            yaxis=dict(
                showspikes=True,
                spikemode="across",
                spikesnap="cursor",
                showline=True,
                spikecolor="lightgray",
                spikethickness=0.7,
                spikedash="dot",
                tickformat=".1f"
            ),
            hoverlabel=dict(
                bgcolor="rgba(0,0,0,0)",
                font_size=12,
                font_family="Arial"
            )
        )

        # Text below chart
        st.caption("""
        Key Points:    
        This chart shows estimated material loss (%) of AngleOn™ and Competitor products over time (hrs). AngleOn™ demonstrates superior durability under identical wear conditions. 
        Calculated runtime hours is an extrapolation of data gathered from accelerated wear test combined with real-time longevity material loss measurements. Assumption: AngleOn™ demonstrates 
        greater durability simply due to the fact there is more material per unit area to be removed in comparison to Competitor product (Reference Monofilament Density page for more detail).
        """)
        st.plotly_chart(fig, use_container_width=True)
