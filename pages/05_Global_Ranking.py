import streamlit as st
import pandas as pd
import sqlite3
import numpy as np
import os

# --- PAGE CONFIG ---
st.set_page_config(page_title="Global Rankings", page_icon="🌍", layout="wide")

st.title("🌍 Global Player Rankings")
st.markdown("### All-Time Performance Leaderboards")
st.caption("Players are assigned a **single primary role** (Batsman, Bowler, or All-Rounder) based on where they have the highest Impact Points.")

# --- 1. DB CONNECTION ---
def get_db_path():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_path = os.path.join(current_dir, "../cricket_data.db")
    if os.path.exists(parent_path): return parent_path
    return "cricket_data.db" 

DB_FILE = get_db_path()

# --- HELPER FUNCTION ---

def make_trajectory_link(player_name):
    return f"/Player_Trajectory?player={player_name.replace(' ', '+')}"

# --- 2. LOGIC: ROLE ASSIGNMENT ---
def determine_primary_role(row):
    """
    Assigns the role based on which category has the highest points.
    Crucial: Pts_AllRounder is already penalized to -1 inside the main function
    if the player doesn't meet the minimum criteria (runs + wickets).
    """
    bat = row['Pts_Batting']
    bowl = row['Pts_Bowling']
    ar = row['Pts_AllRounder']
    
    if bat >= bowl and bat >= ar:
        return "Batsman"
    elif bowl >= bat and bowl >= ar:
        return "Bowler"
    else:
        return "All-Rounder"

# --- 3. DATA ENGINE ---
@st.cache_data
def get_global_rankings():
    if not os.path.exists(DB_FILE):
        return pd.DataFrame()
        
    conn = sqlite3.connect(DB_FILE)
    
    query = """
    SELECT 
        Player_Name,
        MAX(Team_Name) as Team_Name,
        SUM(Runs_Scored) as Total_Runs,
        SUM(Balls_Faced) as Total_Balls_Faced,
        SUM(Innings_Out) as Total_Outs,
        SUM(Wickets_Taken) as Total_Wickets,
        SUM(Runs_Conceded) as Total_Runs_Conceded,
        SUM(CASE WHEN Is_MoM = 1 THEN 1 ELSE 0 END) as Total_MoMs,
        SUM(
            CAST(Overs_Balled AS INT) * 6 + 
            CAST(ROUND((Overs_Balled - CAST(Overs_Balled AS INT)) * 10) AS INT)
        ) as Total_Balls_Bowled,
        
        -- BAT: EXCLUSIVE RANGES
        SUM(CASE WHEN Runs_Scored >= 100 THEN 1 ELSE 0 END) as Count_100s,
        SUM(CASE WHEN Runs_Scored >= 50 AND Runs_Scored < 100 THEN 1 ELSE 0 END) as Count_50s,
        SUM(CASE WHEN Runs_Scored >= 30 AND Runs_Scored < 50 THEN 1 ELSE 0 END) as Count_30s,
        SUM(CASE WHEN Runs_Scored = 0 AND Innings_Out = 1 THEN 1 ELSE 0 END) as Count_Ducks,
        
        -- BOWL: EXCLUSIVE RANGES
        SUM(CASE WHEN Wickets_Taken >= 5 THEN 1 ELSE 0 END) as Count_5W,
        SUM(CASE WHEN Wickets_Taken = 4 THEN 1 ELSE 0 END) as Count_4W,
        SUM(CASE WHEN Wickets_Taken = 3 THEN 1 ELSE 0 END) as Count_3W,
        SUM(CASE WHEN Wickets_Taken = 0 AND Overs_Balled > 0 THEN 1 ELSE 0 END) as Count_Zero_Wkt,
        
        COUNT(DISTINCT Match_ID) as Matches_Played

    FROM player_stats
    GROUP BY Player_Name
    """
    
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    if df.empty: return pd.DataFrame()

    # --- CALCULATIONS ---
    
    # Basic Metrics
    df['Bat_Avg'] = np.where(df['Total_Outs'] > 0, df['Total_Runs'] / df['Total_Outs'], df['Total_Runs'])
    df['Bat_SR'] = np.where(df['Total_Balls_Faced'] > 0, (df['Total_Runs'] / df['Total_Balls_Faced']) * 100, 0.0)
    
    df['Total_Overs_Precise'] = df['Total_Balls_Bowled'] / 6.0
    df['Bowl_Econ'] = np.where(df['Total_Overs_Precise'] > 0, df['Total_Runs_Conceded'] / df['Total_Overs_Precise'], 0.0)
    df['Bowl_Avg'] = np.where(df['Total_Wickets'] > 0, df['Total_Runs_Conceded'] / df['Total_Wickets'], 0.0)
    df['Bowl_SR'] = np.where(df['Total_Wickets'] > 0, df['Total_Balls_Bowled'] / df['Total_Wickets'], 0.0)

    # 1. Batting Points
    b_sr_pts = np.where(df['Bat_SR'] > 100, (df['Bat_SR'] - 100)/5, 0)
    df['Pts_Batting'] = (
        (df['Total_Runs'] * 0.5) + (df['Bat_Avg'] * 0.5) + b_sr_pts +
        (df['Count_100s']*50 + df['Count_50s']*20 + df['Count_30s']*10) +
        (df['Total_MoMs']*10) - (df['Count_Ducks']*10)
    )

    # 2. Bowling Points
    w_econ_pts = np.where(df['Bowl_Econ'] < 12.0, (12.0 - df['Bowl_Econ']) * 2, 0.0)
    df['Pts_Bowling'] = (
        (df['Total_Wickets'] * 15) + (df['Total_Overs_Precise'] * 1.0) + w_econ_pts +
        (df['Count_5W']*30 + df['Count_4W']*20 + df['Count_3W']*10) +
        (df['Total_MoMs']*10) - (df['Count_Zero_Wkt']*5)
    )

    # 3. All-Rounder Points
    df['Pts_AllRounder'] = (
        (df['Total_Runs'] * 1) + (df['Total_Wickets'] * 10.0) +
        (df['Total_MoMs'] * 10.0) - (df['Count_Ducks'] * 1.1) - (df['Count_Zero_Wkt'] * 1.1)
    )
    
    # --- CRITICAL FIX: DISQUALIFY ONE-DIMENSIONAL PLAYERS ---
    # If they haven't taken wickets OR haven't scored meaningful runs globally,
    # their All-Rounder points are voided (-1) so they default to Bat/Bowl roles.
    # Global thresholds: >0 Wickets AND >50 Runs
    mask_not_ar = (df['Total_Wickets'] == 0) | (df['Total_Runs'] < 50)
    df.loc[mask_not_ar, 'Pts_AllRounder'] = -1.0

    # 4. Determine Primary Role
    df['Role'] = df.apply(determine_primary_role, axis=1)

    return df

# --- MAIN APP LOGIC ---
df = get_global_rankings()

if df.empty:
    st.error("No data found in database.")
    st.stop()

# --- FILTERS ---
with st.sidebar:
    st.header("⚙️ Filter Options")
    min_matches = st.slider("Minimum Matches Played", 1, int(df['Matches_Played'].max()), 5)
    
    # Filter DataFrame
    df_filtered = df[df['Matches_Played'] >= min_matches].copy()

# --- TABS ---
tab1, tab2, tab3 = st.tabs(["🏏 Batsmen", "⚾ Bowlers", "⭐ All-Rounders"])

# --- TAB 1: BATSMEN ---
with tab1:
    st.subheader("Global Batting Rankings")
    df_bat = df_filtered[df_filtered['Role'] == 'Batsman'].sort_values('Pts_Batting', ascending=False).head(50)
    
    # We overwrite the Player_Name column to contain the URL string
    df_bat['Player_Link'] = df_bat['Player_Name'].apply(make_trajectory_link)
    
    cols_bat = ['Player_Link', 'Team_Name', 'Matches_Played', 'Total_Runs', 'Bat_Avg', 'Bat_SR', 'Pts_Batting']
    
    st.dataframe(
        df_bat[cols_bat],
        column_config={
            # The 'Player_Link' column holds the URL, but displays the 'Player_Name'
            "Player_Link": st.column_config.LinkColumn(
                "Player Name", 
                display_text=r"^.*player=(.*)$" # This regex trick pulls the name back out of the URL for display
            ),
            "Team_Name":"Team",
            "Matches_Played": "Matches",
            "Bat_Avg": st.column_config.NumberColumn("Avg", format="%.2f"),
            "Bat_SR": st.column_config.NumberColumn("SR", format="%.2f"),
            "Pts_Batting": st.column_config.NumberColumn("Points", format="%.0f")
        },
        use_container_width=True, hide_index=True
    )

# --- TAB 2: BOWLERS ---
with tab2:
    st.subheader("Global Bowling Rankings")
    df_bowl = df_filtered[df_filtered['Role'] == 'Bowler'].sort_values('Pts_Bowling', ascending=False).head(50)
    
    df_bowl['Player_Link'] = df_bowl['Player_Name'].apply(make_trajectory_link)
    
    cols_bowl = ['Player_Link', 'Team_Name', 'Matches_Played', 'Total_Wickets', 'Bowl_Avg', 'Bowl_Econ', 'Pts_Bowling']
    
    st.dataframe(
        df_bowl[cols_bowl],
        column_config={
            "Player_Link": st.column_config.LinkColumn("Player Name", display_text=r"^.*player=(.*)$"),
            "Total Wickets": "Wickets", "Team_Name":"Team", "Matches_Played":"Matches",
            "Bowl_Avg": st.column_config.NumberColumn("Avg", format="%.2f"),
            "Bowl_Econ": st.column_config.NumberColumn("Econ", format="%.2f"),
            "Pts_Bowling": st.column_config.NumberColumn("Points", format="%.0f")
        },
        use_container_width=True, hide_index=True
    )

# --- TAB 3: ALL-ROUNDERS ---
with tab3:
    st.subheader("Global All-Rounder Rankings")
    df_ar = df_filtered[df_filtered['Role'] == 'All-Rounder'].sort_values('Pts_AllRounder', ascending=False).head(50)
    
    df_ar['Player_Link'] = df_ar['Player_Name'].apply(make_trajectory_link)
    
    cols_ar = ['Player_Link', 'Team_Name', 'Matches_Played', 'Total_Runs', 'Total_Wickets', 'Pts_AllRounder']
    
    st.dataframe(
        df_ar[cols_ar],
        column_config={
            "Player_Link": st.column_config.LinkColumn("Player Name", display_text=r"^.*player=(.*)$"),
            "Pts_AllRounder": st.column_config.NumberColumn("Points", format="%.0f"),
            "Total Wickets": "Wickets", "Team_Name":"Team", "Matches_Played":"Matches", "Total_Runs":"Runs"
        },
        use_container_width=True, hide_index=True
    )