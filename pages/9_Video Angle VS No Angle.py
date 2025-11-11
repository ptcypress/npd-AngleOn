# pages/Video_Angle_vs_No_Angle.py
import streamlit as st

st.set_page_config(page_title="Video — Angle vs No Angle", layout="wide")

# --- Header ---
st.title("Angle vs No Angle — Feeder Demonstration")
st.caption(
    "Clip showing object transport behavior under varying applied pressure. "
    "Pairs with the **Velocity vs Pressure** analysis page."
)

# --- Sidebar Controls ---
st.sidebar.header("Video Controls")
video_url = st.sidebar.text_input(
    "Video URL",
    value="https://youtu.be/Eerrp0QNPqk",
    help="Paste a YouTube or Vimeo link here."
)
autoplay = st.sidebar.checkbox("Autoplay (browser-dependent)", value=False)
loop = st.sidebar.checkbox("Loop", value=False)
muted = st.sidebar.checkbox("Muted", value=False)

# --- Description / Notes ---
with st.expander("Test setup & notes", expanded=False):
    st.markdown(
        """
- **System:** Older 24” linear vibratory feeder @ 120 VAC  
- **Brushes:** AngleOn™ vs. Competitor Product 
- **Observation:** Angled Monofilaments force object direction    
        """
    )

# --- Video Display ---
st.video(video_url, autoplay=autoplay, muted=muted, loop=loop)

# --- Footer ---
st.caption(
    "All testing performed on the same platform as the Velocity vs Pressure study."
)
st.page_link("Velocity_vs_Pressure.py", label="⬅️ Back to Velocity vs Pressure", icon="📊")
