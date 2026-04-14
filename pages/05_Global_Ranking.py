import streamlit as st
import pandas as pd
import sqlite3
import numpy as np
import os

# --- PAGE CONFIG ---
st.set_page_config(page_title="Global Rankings", page_icon="🌍", layout="wide")

st.title("🌍 Global Player Rankings")
st.markdown("### Performance Leaderboards")

# --- 1. ROBUST PATH FINDER ---
def get_db_path():
    """Locates the database file in the root directory relative to the script."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(script_dir)
    # The schema specifies 'cricket data.db' but code uses 'cricket_data.db'
    # Checking for the version that works with your Deep Dive app
    db_path = os.path.join(root_dir, "cricket_data.db")
    if os.path.exists(db_path):
        return db_path
    return "cricket_data.db" 

DB_FILE = get_db_path()

def make_trajectory_link(player_name):
    """Generates a query parameter link for the Deep Dive page."""
    return f"/Player_Trajectory?player={player_name.replace(' ', '+')}"

# --- 2. LOGIC: ROLE ASSIGNMENT ---
def determine_primary_role(row):
    """Assigns the primary role based on the highest impact points."""
    bat, bowl, ar = row['Pts_Batting'], row['Pts_Bowling'], row['Pts_AllRounder']
    if bat >= bowl and bat >= ar: return "Batsman"
    elif bowl >= bat and bowl >= ar: return "Bowler"
    return "All-Rounder"

# --- 3. DATA ENGINE ---
@st.cache_data
def get_rankings_data(tournament_filter="All", post_wc_only=False):
    if not os.path.exists(DB_FILE): return pd.DataFrame()
    conn = sqlite3.connect(DB_FILE)
    
    # CAST Tournament_ID to INT for numerical comparison (>= 29)
    if post_wc_only:
        # Strictly Tournament 29 and above for advanced metrics 
        where_clause = "WHERE CAST(Tournament_ID AS INT) >= 29"
    else:
        where_clause = "" if tournament_filter == "All" else f"WHERE Tournament_ID = '{tournament_filter}'"
    
    query = f"""
    SELECT 
        Player_Name,
        MAX(Team_Name) as Team_Name,
        SUM(Runs_Scored) as Total_Runs,
        SUM(Balls_Faced) as Total_Balls_Faced,
        SUM(Innings_Out) as Total_Outs,
        SUM(Wickets_Taken) as Total_Wickets,
        SUM(Runs_Conceded) as Total_Runs_Conceded,
        SUM(CASE WHEN Is_MoM = 1 THEN 1 ELSE 0 END) as Total_MoMs,
        SUM(Fours_Hit) as Total_4s,
        SUM(Sixes_Hit) as Total_6s,
        SUM(Dot_Balls_Bowled) as Total_Dots,
        SUM(CAST(Overs_Balled AS INT) * 6 + CAST(ROUND((Overs_Balled - CAST(Overs_Balled AS INT)) * 10) AS INT)) as Total_Balls_Bowled,
        SUM(CASE WHEN Runs_Scored >= 100 THEN 1 ELSE 0 END) as Count_100s,
        SUM(CASE WHEN Runs_Scored >= 50 AND Runs_Scored < 100 THEN 1 ELSE 0 END) as Count_50s,
        SUM(CASE WHEN Runs_Scored = 0 AND Innings_Out = 1 THEN 1 ELSE 0 END) as Count_Ducks,
        COUNT(DISTINCT Match_ID) as Matches_Played
    FROM player_stats
    {where_clause}
    GROUP BY Player_Name
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    if df.empty: return pd.DataFrame()

    # Shared Calculations
    df['Bat_Avg'] = np.where(df['Total_Outs'] > 0, df['Total_Runs'] / df['Total_Outs'], df['Total_Runs'])
    df['Bat_SR'] = np.where(df['Total_Balls_Faced'] > 0, (df['Total_Runs'] / df['Total_Balls_Faced']) * 100, 0.0)
    df['Total_Overs'] = df['Total_Balls_Bowled'] / 6.0
    df['Bowl_Econ'] = np.where(df['Total_Overs'] > 0, df['Total_Runs_Conceded'] / df['Total_Overs'], 0.0)

    # Point Logic (Career/Standard)
    df['Pts_Batting'] = (df['Total_Runs'] * 0.5) + (df['Bat_Avg'] * 0.5) + (df['Total_MoMs'] * 10)
    df['Pts_Bowling'] = (df['Total_Wickets'] * 15) + (df['Total_Overs'] * 1.0)
    df['Pts_AllRounder'] = (df['Total_Runs'] * 0.8) + (df['Total_Wickets'] * 12)
    
    # All-Rounder Mask: Must have wickets and runs to be considered an AR
    mask_not_ar = (df['Total_Wickets'] == 0) | (df['Total_Runs'] < 50)
    df.loc[mask_not_ar, 'Pts_AllRounder'] = -1.0

    # Modern Metrics (Power/Pressure)
    df['Boundary_Runs'] = (df['Total_4s'] * 4) + (df['Total_6s'] * 6)
    df['Boundary_Pct'] = np.where(df['Total_Runs'] > 0, (df['Boundary_Runs'] / df['Total_Runs']) * 100, 0.0)
    df['Dot_Ball_Pct'] = np.where(df['Total_Balls_Bowled'] > 0, (df['Total_Dots'] / df['Total_Balls_Bowled']) * 100, 0.0)

    df['Role'] = df.apply(determine_primary_role, axis=1)
    return df

# --- 4. UI LOGIC ---
if not os.path.exists(DB_FILE):
    st.error(f"🚨 Database not found at: {DB_FILE}")
    st.stop()

with st.sidebar:
    st.header("⚙️ Ranking Filters")
    conn = sqlite3.connect(DB_FILE)
    tours = pd.read_sql("SELECT DISTINCT Tournament_ID FROM player_stats", conn)['Tournament_ID'].tolist()
    conn.close()
    tour_select = st.selectbox("Select Tournament (Career Tabs)", ["All"] + sorted(tours))

# Fetch Datasets
df_career = get_rankings_data(tournament_filter=tour_select)
df_modern = get_rankings_data(post_wc_only=True)

tab1, tab2, tab3, tab4, tab5 = st.tabs(["🏏 Batsmen", "⚾ Bowlers", "⭐ All-Rounders", "🚀 Power Hitters", "🎯 Hard to Play"])

# Unified config for LinkColumns to ensure proper name display
link_config = st.column_config.LinkColumn("Player Name", display_text=r"player=(.*)$")

# Standard Career Tabs (1-3)
for t, role, pts_col, cols in [
    (tab1, 'Batsman', 'Pts_Batting', ['Player_Link', 'Team_Name', 'Matches_Played', 'Total_Runs', 'Bat_Avg', 'Pts_Batting']),
    (tab2, 'Bowler', 'Pts_Bowling', ['Player_Link', 'Team_Name', 'Matches_Played', 'Total_Wickets', 'Bowl_Econ', 'Pts_Bowling']),
    (tab3, 'All-Rounder', 'Pts_AllRounder', ['Player_Link', 'Team_Name', 'Matches_Played', 'Total_Runs', 'Total_Wickets', 'Pts_AllRounder'])
]:
    with t:
        if not df_career.empty:
            df_role = df_career[df_career['Role'] == role].sort_values(pts_col, ascending=False).head(50)
            df_role['Player_Link'] = df_role['Player_Name'].apply(make_trajectory_link)
            st.dataframe(
                df_role[cols], 
                column_config={"Player_Link": link_config}, 
                use_container_width=True, 
                hide_index=True
            )

# --- TAB 4: POWER HITTERS (Post-WC 2025 ONLY) ---
with tab4:
    st.subheader("🚀 Power Hitters (Post-WorldCup 2025)")
    st.caption("Boundary dominance metrics since Tournament 29.")
    if not df_modern.empty:
        # FILTER: Show only players who have hit at least one boundary (4 or 6)
        df_p = df_modern[(df_modern['Total_4s'] > 0) | (df_modern['Total_6s'] > 0)].copy()
        
        if not df_p.empty:
            df_p = df_p.sort_values(['Total_6s', 'Boundary_Pct'], ascending=False).head(50)
            df_p['Player_Link'] = df_p['Player_Name'].apply(make_trajectory_link)
            st.dataframe(
                df_p[['Player_Link', 'Team_Name', 'Total_4s', 'Total_6s', 'Boundary_Pct']], 
                column_config={
                    "Player_Link": link_config,
                    "Boundary_Pct": st.column_config.NumberColumn("Boundary %", format="%.1f%%")
                }, 
                use_container_width=True, 
                hide_index=True
            )
        else:
            st.info("No players with boundaries found in this period.")

# --- TAB 5: HARD TO PLAY (Post-WC 2025 ONLY) ---
with tab5:
    st.subheader("🎯 Hard to Play (Post-WorldCup 2025)")
    st.caption("Bowling pressure and dot ball metrics since Tournament 29.")
    if not df_modern.empty:
        # FILTER: Show only players with a Dot Ball % greater than 0
        df_h = df_modern[df_modern['Dot_Ball_Pct'] > 0].copy()
        
        if not df_h.empty:
            df_h = df_h.sort_values(['Dot_Ball_Pct', 'Total_Dots'], ascending=False).head(50)
            df_h['Player_Link'] = df_h['Player_Name'].apply(make_trajectory_link)
            st.dataframe(
                df_h[['Player_Link', 'Team_Name', 'Dot_Ball_Pct', 'Total_Dots']], 
                column_config={
                    "Player_Link": link_config,
                    "Dot_Ball_Pct": st.column_config.NumberColumn("Dot %", format="%.1f%%")
                }, 
                use_container_width=True, 
                hide_index=True
            )
        else:
            st.info("No players with recorded dot balls found in this period.")