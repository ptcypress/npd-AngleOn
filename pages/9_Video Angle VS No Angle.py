# pages/Video_Angle_vs_No_Angle.py
import os
import streamlit as st

st.set_page_config(page_title="Video — Angle vs No Angle", layout="wide")

# --- Header (matches tone of main page) ---
st.title("Angle vs No Angle — Feeder Demonstration")
st.caption(
    "Clip showing object transport behavior under varying applied pressure. "
    "Pairs with the **Velocity vs Pressure** analysis page."
)

# --- Sidebar controls kept minimal (same style as your main app) ---
st.sidebar.header("Video Controls")
video_path = st.sidebar.text_input(
    "Local MP4 path",
    value="assets/angle_vs_no_angle.mp4",
    help="Put your video in an /assets folder and set its filename here."
)
video_url = st.sidebar.text_input(
    "…or paste a video URL (YouTube/Vimeo)",
    value="",
    help="If provided, this URL will be used instead of the local file."
)
autoplay = st.sidebar.checkbox("Autoplay (browser-dependent)", value=False)
loop = st.sidebar.checkbox("Loop", value=False)
muted = st.sidebar.checkbox("Muted", value=False)

# --- Main content area ---
# Keep description + notes in an expander to mirror your data-table expander pattern
with st.expander("Test setup & notes", expanded=False):
    st.markdown(
        """
- **System:** 36” linear vibratory feeder @ 120 VAC  
- **Brush:** AngleOn™ vs. non-angled configuration  
- **Observation:** Motion cease point aligns with ~3 lbs/in² seen in the chart  
- **Use:** This video is a qualitative companion to the cubic-fit model
        """
    )

# Prefer URL if provided; otherwise try local path
if video_url.strip():
    st.video(video_url, autoplay=autoplay, muted=muted, loop=loop)
else:
    if not os.path.exists(video_path):
        st.error(
            f"Video not found at `{video_path}`. "
            "Place your MP4 in `/assets` (e.g., `assets/angle_vs_no_angle.mp4`) "
            "or paste a URL in the sidebar."
        )
    else:
        st.video(video_path, autoplay=autoplay, muted=muted, loop=loop)

# A small footer note to visually match your other page’s caption style
st.caption("All testing performed on the same platform as the Velocity vs Pressure study.")
