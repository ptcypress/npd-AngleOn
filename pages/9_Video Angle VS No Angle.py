# pages/Video_Angle_vs_No_Angle.py
import streamlit as st

st.set_page_config(page_title="Video — Angle vs No Angle", layout="wide")

# --- Header ---
st.title("AngleOn™ vs UltraNylon — Feeder Demonstration")
st.caption("""
Key Points:    
Video demonstration of difference in object movement using straight Ultra Nylon vs AngleOn™ product. **Angled monofilament forces object movement** that aligns with linear feeder impulse direction. 
Straight monofilament allows object to move with and in opposition to linear feeder impulse direction.
""")

# --- Sidebar Controls ---
st.sidebar.header("Video Controls")
video_url = st.sidebar.text_input(
    "Video URL",
    value="https://youtu.be/Eerrp0QNPqk",
    help="Paste a YouTube or Vimeo link here."
)

size_label = st.sidebar.select_slider(
    "Display size",
    options=["Compact (55%)", "Medium (65%)", "Wide (75%)"],
    value="Medium (65%)",
    help="Adjust the visual width of the embedded video."
)

# Map size label to column weights (smaller middle column → smaller video)
size_to_cols = {
    "Compact (55%)": [3, 2.4, 3],   # ~40% of page width
    "Medium (65%)":  [2, 2.6, 2],   # ~46% of page width
    "Wide (75%)":    [2, 3.2, 2],   # ~53% of page width
}
col_weights = size_to_cols[size_label]

# --- Description / Notes ---
with st.expander("Test setup & notes", expanded=True):
    st.markdown(
        """
- **System:** 36” linear vibratory feeder @ 120 VAC  
- **Brushes:** AngleOn™ vs. Non-angled configuration  
- **Observation:** Angle forces movement direction  
        """
    )

# --- Video Display (Streamlit-native; sized via columns) ---
left, mid, right = st.columns(col_weights)
with mid:
    # st.video() handles YouTube/Vimeo/MP4 and stays within the column width
    st.video(video_url)

# --- Footer (optional) ---
# st.caption("All testing performed on the same platform as the Velocity vs Pressure study.")
