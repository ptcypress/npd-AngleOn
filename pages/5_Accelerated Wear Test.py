import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from scipy.integrate import quad

# Load durability data
csv_path = "data/durability_data.csv"
df = pd.read_csv(csv_path)
df.columns = df.columns.str.strip()

# Extract data for each material
x = df['Belt Speed'].values.reshape(-1, 1)
y_angleon = df['AngleOn'].values
y_competitor = df['Competitor'].values

# Apply log transformation for modeling
log_x = np.log(x)

# Fit cubic polynomial models on log(x)
poly = PolynomialFeatures(degree=3)
X_poly = poly.fit_transform(log_x)
model_angleon = LinearRegression().fit(X_poly, y_angleon)
model_competitor = LinearRegression().fit(X_poly, y_competitor)

# Define prediction functions
predict_angleon = lambda x_val: model_angleon.predict(poly.transform(np.log(np.array([[x_val]]))))[0]
predict_competitor = lambda x_val: model_competitor.predict(poly.transform(np.log(np.array([[x_val]]))))[0]

# Area between curves and % improvement
x_start, x_end = 6.3, 50
area, _ = quad(lambda x: predict_angleon(x) - predict_competitor(x), x_start, x_end)
baseline_area, _ = quad(lambda x: predict_competitor(x), x_start, x_end)
percent_improvement = abs((area / baseline_area) * 100)

# Plotting range
x_range = np.linspace(x_start, x_end, 300)
y_pred_angleon = [predict_angleon(xi) for xi in x_range]
y_pred_competitor = [predict_competitor(xi) for xi in x_range]

# Axis limits
y_max = max(max(y_pred_angleon), max(y_pred_competitor)) * 1.05

# Streamlit layout
st.set_page_config(page_title="Durability vs Belt Speed", layout="wide")
st.title("Accelerated Wear Test - AngleOn™ vs. Competitor")
#st.subheader("Rate of Material Loss vs Belt Speed")

# Explanatory text
st.caption("""
Key Points:    
This chart represents material loss for both AngleOn™ and Competitor product using Ultrafab's accelerated wear test stand. The test 
stand is designed to simulate material wear given an applied force per unit area with variable speed belt sander (220-grit). The shaded area between the curves, 
from 6.3 to 50 in/sec, represents the cumulative durability advantage of one material over the other. Lower material loss indicates superior wear resistance under 
accelerated wear conditions. AngleOn™ out performs Competitor product.
""")

# Plot
fig = go.Figure()

# Lines
fig.add_trace(go.Scatter(x=x_range, y=y_pred_angleon, mode='lines', name='AngleOn™', line=dict(color='blue', width=3)))
fig.add_trace(go.Scatter(x=x_range, y=y_pred_competitor, mode='lines', name='Competitor', line=dict(color='red', width=3)))

# Shaded area between curves
fig.add_trace(go.Scatter(
    x=np.concatenate([x_range, x_range[::-1]]),
    y=np.concatenate([y_pred_angleon, y_pred_competitor[::-1]]),
    fill='toself',
    fillcolor='rgba(100,100,100,0.2)',
    line=dict(color='rgba(255,255,255,0)'),
    hoverinfo='skip',
    name='Advantage Area'
))

# Annotation
fig.add_annotation(
    x=15,
    y=y_max * 0.25,
    text=(f"Advantage Area = {area:.6e} in²/min·sec<br>"
          f"Relative Advantage = {percent_improvement:.2f}%"),
    showarrow=False,
    font=dict(size=13)
)

# Layout
fig.update_layout(
    xaxis_title="Belt Speed (in/sec)",
    yaxis_title="Rate of Material Loss (in/min)",
    height=650,
    hovermode='x',
    xaxis=dict(
        range=[x_start, x_end],
        showspikes=True,
        spikemode="across",
        spikesnap="cursor",
        showline=True,
        spikecolor="lightgray",
        spikethickness=0.7,
        spikedash="dot"
    ),
    yaxis=dict(
        range=[0, y_max],
        tickformat='e',
        showspikes=True,
        spikemode="across",
        spikesnap="cursor",
        showline=True,
        spikecolor="lightgray",
        spikethickness=0.7,
        spikedash="dot"
    ),
    hoverlabel=dict(
        bgcolor="rgba(0,0,0,0)",
        font_size=12,
        font_family="Arial"
    ),
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="center",
        x=0.5
    )
)

st.plotly_chart(fig, use_container_width=True)

