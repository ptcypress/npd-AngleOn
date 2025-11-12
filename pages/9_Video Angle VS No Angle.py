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
autoplay = st.sidebar.checkbox("Autoplay", value=True)
loop = st.sidebar.checkbox("Loop", value=True)
muted = st.sidebar.checkbox("Muted", value=False)

# --- Description / Notes ---
with st.expander("Test setup & notes", expanded=True):
    st.markdown("""
- **System:** 36” linear vibratory feeder @ 120 VAC  
- **Brushes:** AngleOn™ vs. Non-angled configuration  
- **Observation:** Angle forces movement direction  
    """)

# --- Custom CSS to resize video ---
st.markdown(
    """
    <style>
    .video-container {
        display: flex;
        justify-content: center;
        align-items: center;
    }
    .video-container iframe {
        width: 70%;  /* Adjust this to make video larger/smaller (e.g., 60%, 80%) */
        height: auto;
        aspect-ratio: 16 / 9;
        border-radius: 12px;
        box-shadow: 0 0 15px rgba(0,0,0,0.25);
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --- Video Display ---
st.markdown('<div class="video-container">', unsafe_allow_html=True)
st.video(video_url, autoplay=autoplay, muted=muted, loop=loop)
st.markdown('</div>', unsafe_allow_html=True)
