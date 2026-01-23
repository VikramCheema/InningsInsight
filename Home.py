import streamlit as st
from PIL import Image
import os

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="InningsInsight",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- ASSET LOADING ---
current_dir = os.path.dirname(os.path.abspath(__file__))
logo_path = os.path.join(current_dir, "assets", "logo.png")

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
    /* Hover effect for containers */
    div[data-testid="stContainer"] {
        transition: transform 0.2s;
    }
    </style>
    """, unsafe_allow_html=True)

# --- HERO SECTION ---
# Create two columns: Left for Identity (Logo+Text), Right for the Video
col_hero_text, col_hero_video = st.columns([1.5, 1], gap="medium")

# LEFT COLUMN: Logo & Title
with col_hero_text:
    # 1. The Logo
    if os.path.exists(logo_path):
        st.image(logo_path, width=130) # Adjusted width for side-by-side look
    else:
        st.markdown("<h1 style='font-size: 4rem;'>🏏</h1>", unsafe_allow_html=True)
    
    # 2. The Title & Subtitle
    # Note: We align text left here to balance with the video on the right
    st.markdown(
        """
        <div>
            <h1 style="color: #4CAF50; font-size: 3.5rem; margin-bottom: 0;">InningsInsight</h1>
            <p style="font-size: 1.3rem; color: #666; margin-top: 5px;">Advanced Cricket Analytics & Match Intelligence</p>
        </div>
        """,
        unsafe_allow_html=True
    )

# RIGHT COLUMN: The Loop Video
with col_hero_video:
    video_path = os.path.join(current_dir, "assets", "intro_video.mp4")
    
    if os.path.exists(video_path):
        # autoplay + loop + muted is required for auto-start
        st.video(video_path, format="video/mp4", autoplay=True, loop=True, muted=True)

st.divider()

# --- NAVIGATION GRID ---
st.subheader("📍 Dashboard Modules")

# Row 1: The Core Analytics
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

# Row 2: Tournament & Rankings
st.write("") # Spacer
row2_col1, row2_col2, row2_col3 = st.columns(3, gap="medium")

with row2_col1:
    with st.container(border=True):
        st.markdown("### 🌍 Team Rankings")
        st.write("Current team standings, rating trends, and historical dominance analysis.")
        st.page_link("pages/4_Team_Ranking.py", label="View Rankings", icon="🏆", use_container_width=True)

with row2_col2:
    with st.container(border=True):
        st.markdown("### 🏆 World Cup 2026")
        st.write("Exclusive HQ for the 2026 tournament. Points table, NRR scenarios, and MVP race.")
        st.page_link("pages/5_World_Cup_2026.py", label="Enter Tournament Mode", icon="🔥", use_container_width=True)

with row2_col3:
    with st.container(border=True):
        st.markdown("### 💪 Global Player Ranking")
        st.write("Complete ICC player ranking lists across all formats and disciplines.")
        st.page_link("pages/6_Global_Ranking.py", label="View Rankings", icon="📊", use_container_width=True)

# Row 3: Advanced Intelligence
st.write("") # Spacer
row3_col1, row3_col2, row3_col3 = st.columns(3, gap="medium")

# 1. Venue Atlas
with row3_col1:
    with st.container(border=True):
        st.markdown("### 🏟️ Venue Atlas")
        st.write("Interactive global map with stadium stats, pitch history, and host intelligence.")
        st.page_link("pages/7_Venues.py", label="Explore Venues", icon="🗺️", use_container_width=True)

# 2. AI Analyst (New Addition)
with row3_col2:
    with st.container(border=True):
        st.markdown("### 🤖 AI Data Analyst")
        st.write("Ask questions in plain English. Powered by Llama-3 to generate SQL and insights instantly.")
        st.page_link("pages/8_AI_Agent.py", label="Ask the AI", icon="✨", use_container_width=True)

# 3. Empty Placeholder (Optional - keeps the grid alignment if you add a 9th page later)
with row3_col3:
    st.empty()
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