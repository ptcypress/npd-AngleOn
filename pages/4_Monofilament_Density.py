import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------
# Preset filament configurations
# ---------------------------
PRESETS = {
    "AngleOn™ (6910 epi², 0.006\")": (6910, 0.006),
    "Competitor (9750 epi², 0.0045\")": (9750, 0.0045),
    "XT10 (2275 epi², 0.010\")": (2275, 0.010),
    "XT16 (1135 epi², 0.016\")": (1135, 0.016),
    "Custom": None
}

# ---------------------------
# Calculate monofilament positions
# ---------------------------
@st.cache_resource
def generate_monofilament_data(density, diameter, pattern="hex"):
    area_each = np.pi * (diameter / 2) ** 2
    total_area = area_each * density  # in²
    percent_coverage = total_area * 100  # % of 1in² box

    positions = []

    if pattern == "hex":
        spacing = np.sqrt(2 / (np.sqrt(3) * density))
        y = 0
        row = 0
        while y < 1.0:
            x_offset = 0 if row % 2 == 0 else spacing / 2
            x = x_offset
            while x < 1.0:
                positions.append((x, y))
                x += spacing
            y += spacing * np.sqrt(3) / 2
            row += 1

    elif pattern == "grid":
        n_filaments = int(np.sqrt(density))
        spacing = 1.0 / n_filaments
        for i in range(n_filaments):
            for j in range(n_filaments):
                x = (i + 0.5) * spacing
                y = (j + 0.5) * spacing
                positions.append((x, y))

    return np.array(positions), total_area, percent_coverage

# ---------------------------
# Draw the visual pattern
# ---------------------------
def draw_monofilament(positions, diameter, title, density, total_area, percent_coverage):
    fig, ax = plt.subplots(figsize=(6, 6), dpi=150)

    if len(positions) > 0:
        pt_per_inch = 72
        diameter_pt = diameter * pt_per_inch
        size_pt_squared = (diameter_pt / 2) ** 2 * np.pi

        ax.scatter(positions[:, 0], positions[:, 1],
                   s=size_pt_squared, edgecolor='black', facecolors='gray', linewidth=0.2)

    ax.plot([0, 1, 1, 0, 0], [0, 0, 1, 1, 0], 'k-', lw=1)
    ax.set_aspect('equal')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([0, 0.5, 1])
    ax.set_yticks([0, 0.5, 1])
    ax.set_xlabel("inches")
    ax.set_ylabel("inches")
    ax.set_title(f"{title}\n{density} ends/in², {diameter}\" dia")
    
    fig.text(0.5, 0.01, f"Monofilament area = {total_area:.4f} in²  •  Coverage = {percent_coverage:.1f}%",
             ha='center', fontsize=9)
    plt.tight_layout()
    return fig

# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(page_title="Monofilament Density Viewer", layout="centered")
st.title("Monofilament Density Visualizer")
st.caption("""This is a scaled visualizer showing coverage area of different brush types given 
their density (epi²) and monofilament diameter (in) within one square inch of brush. 
Presets use actual densities and diameters of brushes tested to date.""")

# Layout
col1, col2, col3 = st.columns(3)
with col1:
    pattern = st.selectbox("Pattern Type", ["hex", "grid"])

with col2:
    preset = st.selectbox("Choose Preset", list(PRESETS.keys()))

if preset == "Custom":
    with col3:
        density = st.slider("Filament Density (ends/in²)", 1000, 12000, 6912, step=10)
        diameter = st.slider("", 0.002, 0.02, 0.006, step=0.0005)
else:
    density, diameter = PRESETS[preset]

positions, total_area, percent_coverage = generate_monofilament_data(density, diameter, pattern)
fig = draw_monofilament(positions, diameter, preset, density, total_area, percent_coverage)
st.pyplot(fig)
