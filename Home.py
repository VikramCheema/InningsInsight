import streamlit as st
from PIL import Image
import os

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="InningsInsight",
    page_icon="🏏",
    layout="centered"
)

# --- PATH ANCHOR ---
current_dir = os.path.dirname(os.path.abspath(__file__))
logo_path = os.path.join(current_dir, "logo.png")

# --- LOAD ASSETS ---
if not os.path.exists(logo_path):
    # Optional: simpler warning or just pass
    logo = None
else:
    logo = Image.open(logo_path)

# --- MAIN LAYOUT ---
col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    if logo:
        st.image(logo, use_container_width=True)

st.markdown(
    """
    <h1 style='text-align: center; color: #4CAF50;'>InningsInsight</h1>
    <p style='text-align: center; font-size: 1.2rem;'>
        Advanced Cricket Analytics & Match Intelligence
    </p>
    """,
    unsafe_allow_html=True
)

st.divider()

# --- NAVIGATION HUB ---
st.subheader("Available Modules")

col_a, col_b, col_c = st.columns(3)

with col_a:
    with st.container(border=True):
        st.markdown("### ⚔️ Match Center")
        st.write("Head-to-Head rivalries, venue stats, and player matchups.")
        # Link to Match Center
        st.page_link("pages/1_Match_Center.py", label="Launch Match Center", icon="🚀")

with col_b:
    with st.container(border=True):
        st.markdown("### 📈 Player Trajectory")
        st.write("Career graphs, form analysis, and milestone tracking.")
        # UPDATED: Link to Player Trajectory
        st.page_link("pages/2_Player_Trajectory.py", label="View Player Stats", icon="📈")
with col_c:
    with st.container(border=True):
        st.markdown("### 📊 Player Comparison")
        st.write("Comparison graphs, stats and analysis.")
        st.page_link("pages/3_Player_Comparison.py", label = "Launch Player Comparison", icon = "📊")

st.divider()
st.caption("© 2026 InningsInsight AI | Powered by Streamlit")