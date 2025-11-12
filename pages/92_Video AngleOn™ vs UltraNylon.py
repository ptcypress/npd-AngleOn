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

# --- Fixed video settings ---
VIDEO_URL = "https://youtu.be/Eerrp0QNPqk"
AUTOPLAY = True
LOOP = True
MUTED = False

# --- Description / Notes ---
with st.expander("Test setup & notes", expanded=True):
    st.markdown(
        """
- **System:** 36” linear vibratory feeder @ 120 VAC  
- **Brushes:** AngleOn™ vs. UltraNylon Non-angled configuration  
- **Observation:** Angle forces movement direction  
        """
    )

# --- Video Display (balanced width) ---
# Wider middle column gives the video about 70–75 % of page width
col_left, col_video, col_right = st.columns([1, 3, 1])
with col_video:
    st.video(VIDEO_URL, autoplay=AUTOPLAY, muted=MUTED, loop=LOOP)

# --- Optional footer ---
# st.caption("All testing performed on the same platform as the Velocity vs Pressure study.")
