import streamlit as st
from PIL import Image
import os
import base64
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
col_hero_text, col_hero_video = st.columns([1.25, 1], gap="medium")

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
    
    # Check if file exists AND has content
    if os.path.exists(video_path) and os.path.getsize(video_path) > 0:
        
        # 1. READ & ENCODE
        try:
            with open(video_path, "rb") as f:
                video_bytes = f.read()
                # standard b64encode results in no newlines, which is safer for HTML
                video_b64 = base64.b64encode(video_bytes).decode()
                
            # 2. EMBED HTML (Updated structure)
            st.markdown(
                f"""
                <video 
                    width="100%" 
                    autoplay 
                    loop 
                    muted 
                    playsinline 
                    style="border-radius: 10px; pointer-events: none;" 
                    src="data:video/mp4;base64,{video_b64}">
                </video>
                """,
                unsafe_allow_html=True
            )
        except Exception as e:
            st.error(f"Error loading video: {e}")

    elif os.path.exists(video_path) and os.path.getsize(video_path) == 0:
        st.error("File found but it is empty (0 KB).")
    else:
        # Fallback if file is missing
        st.info("Video file not found in 'assets/intro_video.mp4'")

    # 3. CAPTION
    st.markdown(
        """
        <div style="
            font-family: 'New York', 'Times New Roman', serif;
            font-style: italic;
            font-weight: bold;
            text-align: center;
            margin-top: 8px;
            font-size: 1.1rem;
            opacity: 0.8; 
        ">
            Innings Insight Cricket™
        </div>
        """,
        unsafe_allow_html=True
    )
st.divider()

# --- NAVIGATION GRID ---
st.subheader("📍 Dashboard Modules")

# Row 1: The Core Analytics
row1_col1, row1_col2, row1_col3 = st.columns(3, gap="medium")

with row1_col1:
    with st.container(border=True):
        st.markdown("### ⚔️ Match Center")
        st.write("Deep dive into Head-to-Head rivalries, venue history, and predictive matchup modeling.")
        st.page_link("pages/01_Match_Center.py", label="Open Match Center", icon="🚀", use_container_width=True)

with row1_col2:
    with st.container(border=True):
        st.markdown("### 📈 Player Trajectory")
        st.write("Track career arcs, analyze current form, and visualize milestone progressions.")
        st.page_link("pages/02_Player_Trajectory.py", label="Analyze Players", icon="📈", use_container_width=True)

with row1_col3:
    with st.container(border=True):
        st.markdown("### 👥 Player Comparison")
        st.write("Head-to-head stat comparisons across different phases of play and opposition.")
        st.page_link("pages/03_Player_Comparison.py", label="Compare Players", icon="🆚", use_container_width=True)

# Row 2: Tournament & Rankings
st.write("") # Spacer
row2_col1, row2_col2, row2_col3 = st.columns(3, gap="medium")

with row2_col1:
    with st.container(border=True):
        st.markdown("### 🌍 Team Rankings")
        st.write("Current team standings, rating trends, and historical dominance analysis.")
        st.page_link("pages/04_Team_Ranking.py", label="View Rankings", icon="🏆", use_container_width=True)

with row2_col2:
    with st.container(border=True):
        st.markdown("### 🏆 World Cup 2026")
        st.write("Exclusive HQ for the 2026 tournament. Points table, NRR scenarios, and MVP race.")
        st.page_link("pages/05_World_Cup_2026.py", label="Enter Tournament Mode", icon="🔥", use_container_width=True)

with row2_col3:
    with st.container(border=True):
        st.markdown("### 💪 Global Player Ranking")
        st.write("Complete ICC player ranking lists across all formats and disciplines.")
        st.page_link("pages/06_Global_Ranking.py", label="View Rankings", icon="📊", use_container_width=True)

# Row 3: Advanced Intelligence
st.write("") # Spacer
row3_col1, row3_col2, row3_col3 = st.columns(3, gap="medium")

# 1. Venue Atlas
with row3_col1:
    with st.container(border=True):
        st.markdown("### 🏟️ Venue Atlas")
        st.write("Interactive global map with stadium stats, pitch history, and host intelligence.")
        st.page_link("pages/07_Venues.py", label="Explore Venues", icon="🗺️", use_container_width=True)

# 2. AI Analyst (New Addition)
with row3_col2:
    with st.container(border=True):
        st.markdown("### 🤖 AI Data Analyst")
        st.write("Ask questions in plain English. Powered by Llama-3 to generate SQL and insights instantly.")
        st.page_link("pages/08_AI_Agent.py", label="Ask the AI", icon="✨", use_container_width=True)

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