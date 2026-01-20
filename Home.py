import streamlit as st
from PIL import Image
import os

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="InningsInsight",
    page_icon="🏏",
    layout="wide",  # Changed to wide for better dashboard feel
    initial_sidebar_state="expanded"
)

# --- ASSET LOADING ---
# Robust path handling compatible with different OS
current_dir = os.path.dirname(os.path.abspath(__file__))
logo_path = os.path.join(current_dir, "assets", "logo.png") # Assuming you might put images in an assets folder
if not os.path.exists(logo_path):
    # Fallback to current dir if assets folder doesn't exist
    logo_path = os.path.join(current_dir, "logo.png")

# --- CUSTOM CSS ---
st.markdown("""
    <style>
    .main-header {
        text-align: center;
        padding-bottom: 20px;
    }
    .main-header h1 {
        color: #4CAF50;
        font-size: 3rem;
        margin-bottom: 0px;
    }
    .main-header p {
        font-size: 1.2rem;
        color: #666;
    }
    /* Hover effect for containers (subtle) */
    div[data-testid="stContainer"] {
        transition: transform 0.2s;
    }
    </style>
    """, unsafe_allow_html=True)

# --- HERO SECTION ---
col_L, col_M, col_R = st.columns([3, 2, 3])

with col_M:
    if os.path.exists(logo_path):
        st.image(logo_path, use_container_width=True)
    else:
        # Fallback emoji if logo missing
        st.markdown("<h1 style='text-align: center; font-size: 5rem;'>🏏</h1>", unsafe_allow_html=True)

st.markdown(
    """
    <div class='main-header'>
        <h1>InningsInsight</h1>
        <p>Advanced Cricket Analytics & Match Intelligence</p>
    </div>
    """,
    unsafe_allow_html=True
)

st.divider()

# --- NAVIGATION GRID ---
st.subheader("📍 Dashboard Modules")

# Row 1: The Core Analytics (3 Columns)
row1_col1, row1_col2, row1_col3 = st.columns(3, gap="medium")

with row1_col1:
    with st.container(border=True):
        st.markdown("### ⚔️ Match Center")
        st.write("Deep dive into Head-to-Head rivalries, venue history, and predictive matchup modeling.")
        st.page_link("pages/1_Match_Center.py", label="Open Match Center", icon="🚀", use_container_width=True)

with row1_col2:
    with st.container(border=True):
        st.markdown("### 📈 Player Trajectory")
        st.write("Track career arcs, analyze current form, and visualize milestone progressions.")
        st.page_link("pages/2_Player_Trajectory.py", label="Analyze Players", icon="📈", use_container_width=True)

with row1_col3:
    with st.container(border=True):
        st.markdown("### 👥 Player Comparison")
        st.write("Head-to-head stat comparisons across different phases of play and opposition.")
        st.page_link("pages/3_Player_Comparison.py", label="Compare Players", icon="🆚", use_container_width=True)

# Row 2: Tournament & Teams (2 Columns, Centered)
st.write("") # Spacer
_, row2_col1, row2_col2, _ = st.columns([1, 3, 3, 1], gap="medium")

with row2_col1:
    with st.container(border=True):
        st.markdown("### 🌍 Global Rankings")
        st.write("Current team standings, rating trends, and historical dominance analysis.")
        st.page_link("pages/4_Team_Ranking.py", label="View Rankings", icon="🏆", use_container_width=True)

with row2_col2:
    with st.container(border=True):
        st.markdown("### 🏆 World Cup 2026")
        st.write("Exclusive HQ for the 2026 tournament. Points table, NRR scenarios, and MVP race.")
        st.page_link("pages/5_World_Cup_2026.py", label="Enter Tournament Mode", icon="🔥", use_container_width=True)

# --- FOOTER ---
st.divider()
st.markdown(
    """
    <div style='text-align: center; color: grey; font-size: 0.8rem;'>
        © 2026 InningsInsight
    </div>
    """, 
    unsafe_allow_html=True
)