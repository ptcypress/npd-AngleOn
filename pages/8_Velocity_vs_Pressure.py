import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from scipy.integrate import quad
from scipy.optimize import fsolve
import os

# Streamlit page setup
st.set_page_config(page_title="Velocity vs Pressure", layout="wide")
st.title("Object Velocity vs Pressure - AngleOn™ vs Competitor")
#st.subheader("Velocity vs Pressure")

# Load data
csv_path = "data/velocity_data.csv"
if not os.path.exists(csv_path):
    st.error(f"File not found: {csv_path}")
else:
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()

    # Ensure correct columns exist
    if 'Brush' not in df.columns or 'Pressure' not in df.columns or 'Velocity' not in df.columns:
        st.error("Missing required columns in the CSV.")
    else:
        # Filter data
        angleon = df[df['Brush'] == 'AngleOn™']
        competitor = df[df['Brush'] == 'Competitor']

        x_angleon = angleon['Pressure'].values.reshape(-1, 1)
        y_angleon = angleon['Velocity'].values

        x_comp = competitor['Pressure'].values.reshape(-1, 1)
        y_comp = competitor['Velocity'].values

        # Fit cubic models
        poly = PolynomialFeatures(degree=3)
        X_angleon_poly = poly.fit_transform(x_angleon)
        X_comp_poly = poly.fit_transform(x_comp)

        model1 = LinearRegression().fit(X_angleon_poly, y_angleon)
        model2 = LinearRegression().fit(X_comp_poly, y_comp)

        def f(x): return model1.predict(poly.transform(np.array([[x]])))[0]
        def g(x): return model2.predict(poly.transform(np.array([[x]])))[0]
        def diff(x): return f(x) - g(x)

        # Solve for intersection
        try:
            x_intersect = fsolve(diff, x0=1.0)[0]
        except:
            x_intersect = 3.08  # fallback

        # Calculate area between curves (advantage area)
        area_diff, _ = quad(diff, 0, x_intersect)
        area_comp, _ = quad(g, 0, x_intersect)
        percent_improvement = (area_diff / area_comp) * 100 if area_comp != 0 else 0

        # Generate smooth range
        pressure_range = np.linspace(0, x_intersect + 0.5, 300)
        angleon_fit = [f(x) for x in pressure_range]
        competitor_fit = [g(x) for x in pressure_range]

        # Shaded area
        x_fill = np.concatenate([pressure_range, pressure_range[::-1]])
        y_fill = np.concatenate([angleon_fit, competitor_fit[::-1]])

        # Plot
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=x_fill,
            y=y_fill,
            fill='toself',
            fillcolor='rgba(150,150,150,0.3)',
            line=dict(color='rgba(255,255,255,0)'),
            hoverinfo='skip',
            name='Performance Advantage Area'
        ))

        fig.add_trace(go.Scatter(
            x=pressure_range, y=angleon_fit,
            mode='lines',
            name='AngleOn™ Cubic Fit',
            line=dict(color='blue', width=3),
            hovertemplate='Pressure: %{x:.2f} psi<br>Velocity: %{y:.2f} in/sec'
        ))

        fig.add_trace(go.Scatter(
            x=pressure_range, y=competitor_fit,
            mode='lines',
            name='Competitor Cubic Fit',
            line=dict(color='red', width=3),
            hovertemplate='Pressure: %{x:.2f} psi<br>Velocity: %{y:.2f} in/sec'
        ))

        # Annotate
        fig.add_annotation(
            x=pressure_range[len(pressure_range)//2],
            y=(max(angleon_fit) + max(competitor_fit)) / 2,
            text=f"Advantage Area = {area_diff:.3f} in/sec·psi<br>Relative Advantage = {percent_improvement:.1f}%",
            showarrow=False,
            font=dict(size=13)
        )

        fig.update_layout(
            xaxis_title="Pressure (psi)",
            yaxis_title="Velocity (in/sec)",
            height=650,
            hovermode='x',
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.75),
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
                range=[0, max(max(angleon_fit), max(competitor_fit)) + 0.5]
            ),
            hoverlabel=dict(
                bgcolor="rgba(0,0,0,0)",
                font_size=12,
                font_family="Arial"
            )
        )

        st.caption("""This chart compares the velocity output of AngleOn™ and Competitor product for a range of pressures exerted by different objects on each brush type. Key take away is AngleOn™
        out performs competitor product throughout especially at lower pressures (lower object weight ex. Nuts, Bolts, Washers, etc.). For both products, once objects exerted approx. 3 lbs/in², the 
        feeder coil could no longer generate the force required for movement.
        The shaded area quantifies cumulative performance advantage.""")
        st.plotly_chart(fig, use_container_width=True)
