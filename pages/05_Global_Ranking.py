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

def highlight_podium(row):
    gold, silver, bronze = 'background-color: rgba(255, 215, 0, 0.3)', 'background-color: rgba(192, 192, 192, 0.3)', 'background-color: rgba(205, 127, 50, 0.3)'
    if row['Rank'] == 1: return [gold] * len(row)
    if row['Rank'] == 2: return [silver] * len(row)
    if row['Rank'] == 3: return [bronze] * len(row)
    return [''] * len(row)


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
            SUM(CASE WHEN Runs_Scored >= 30 AND Runs_Scored < 50 THEN 1 ELSE 0 END) as Count_30s,
            SUM(CASE WHEN Runs_Scored = 0 AND Innings_Out = 1 THEN 1 ELSE 0 END) as Count_Ducks,
            SUM(CASE WHEN Wickets_Taken = 3 THEN 1 ELSE 0 END) as Count_3W,
            SUM(CASE WHEN Wickets_Taken = 4 THEN 1 ELSE 0 END) as Count_4W,
            SUM(CASE WHEN Wickets_Taken >= 5 THEN 1 ELSE 0 END) as Count_5W,
            SUM(CASE WHEN Wickets_Taken = 0 THEN 1 ELSE 0 END) as Count_ZeroW_Matches,
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
    sr_bonus = np.where(df['Bat_SR'] > 100, (df['Bat_SR'] - 100) / 5.0, 0.0)
    milestone_pts = (df['Count_100s'] * 50) + (df['Count_50s'] * 20) + (df['Count_30s'] * 10)
    duck_penalty = df['Count_Ducks'] * 10
    df['Pts_Batting'] = ((df['Total_Runs'] * 0.5) + (df['Bat_Avg'] * 0.5) + sr_bonus + milestone_pts + (df['Total_MoMs'] * 10) - duck_penalty)

    econ_bonus = np.where(df['Total_Overs'] > 0, (12.0 - df['Bowl_Econ']) * 2, 0.0)
    wicket_haul_pts = (df['Count_3W'] * 10) + (df['Count_4W'] * 20) + (df['Count_5W'] * 30)
    zero_w_penalty = df['Count_ZeroW_Matches'] * 5
    df['Pts_Bowling'] = ((df['Total_Wickets'] * 15) + (df['Total_Overs'] * 1.0) + econ_bonus + wicket_haul_pts + (df['Total_MoMs'] * 10) - zero_w_penalty)

    ar_duck_penalty = df['Count_Ducks'] * 1.1
    ar_zero_w_penalty = df['Count_ZeroW_Matches'] * 1.1
    df['Pts_AllRounder'] = ((df['Total_Runs'] * 1.0) + (df['Total_Wickets'] * 10) + (df['Total_MoMs'] * 10) -ar_duck_penalty - ar_zero_w_penalty)
    
    # All-Rounder Mask: Must have wickets and runs to be considered an AR
    mask_not_ar = (df['Total_Wickets'] <= 5) | (df['Total_Runs'] < 50)
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

column_configuration = {
    "Rank": st.column_config.NumberColumn("Rank", width="small", format="%d"),
    "Player_Link": st.column_config.LinkColumn("Player Name", display_text=r"player=(.*)$", width="large"),
    "Team_Name": st.column_config.Column("Team", width="small"),
    "Matches_Played": st.column_config.NumberColumn("Mat", width="small"),
    "Total_Runs": st.column_config.NumberColumn("Runs", width="small"),
    "Total_Wickets": st.column_config.NumberColumn("Wkts", width="small"),
    "Bat_Avg": st.column_config.NumberColumn("Avg", width="small", format="%.1f"),
    "Bat_SR": st.column_config.NumberColumn("Bat SR", width="small", format="%.1f"),
    "Bowl_Avg": st.column_config.NumberColumn("Bowl Avg", width="small", format="%.1f", help="Bowling Average (Runs/Wicket)"),
    "Bowl_Econ": st.column_config.NumberColumn("Econ", width="small", format="%.1f"),
    "Bowl_SR": st.column_config.NumberColumn("Bowl SR", width="small", format="%.1f", help="Runs conceded per wicket"),
    "Total_MoMs": st.column_config.NumberColumn("MoM", width="small", help="Man of the Match Awards"),
    "Count_30s": st.column_config.NumberColumn("30s", width="small"),
    "Count_50s": st.column_config.NumberColumn("50s", width="small"),
    "Count_3W": st.column_config.NumberColumn("3W", width="small"),
    "Count_4W": st.column_config.NumberColumn("4W", width="small"),
    "Count_5W": st.column_config.NumberColumn("5W", width="small"),
    "Total_Balls_Bowled": st.column_config.NumberColumn("Total Balls Bowled", width="small", help="Total balls bowled (Post-WC 2026 only)",format="%d"),
    "Total_Dots": st.column_config.NumberColumn("Dot Balls", width="small", help="Total dot balls bowled (Post-WC 2026 only)"),
    "Dot_Ball_Pct":st.column_config.NumberColumn("Dot Balls Percentage", width= 'small', help='Percentage of dot balls post World Cup 2026'),
    "Boundary_Runs": st.column_config.NumberColumn("Bndry Runs", width="small",help="Runs scored through 4s and 6s (Post-WC 2026)",format="%d"),
    "Balls_per_Boundary": st.column_config.NumberColumn("Balls/Bndry", width="small", help="Balls played per boundary hit (Lower is better)",format="%.1f"),
    "Points_Visual": st.column_config.ProgressColumn(
        "Rel. Performance", 
        help="Performance relative to the Rank 1 player (100%)",
        format="%.1f%%", # Displays the percentage on the bar
        min_value=0,
        max_value=100,
    ),
    "Points_Value": st.column_config.NumberColumn("Points", width="small", format="%.1f"),
}

for t, role, pts_col, cols in [
    (tab1, 'Batsman', 'Pts_Batting',['Rank', 'Player_Link', 'Team_Name', 'Matches_Played', 'Total_Runs', 'Bat_Avg', 'Bat_SR', 'Count_30s', 'Count_50s', 'Total_MoMs', 'Points_Value', 'Points_Visual']),
    
    (tab2, 'Bowler', 'Pts_Bowling', ['Rank', 'Player_Link', 'Team_Name', 'Matches_Played', 'Total_Wickets', 'Bowl_Avg', 'Bowl_SR', 'Bowl_Econ', 'Count_3W', 'Count_4W', 'Count_5W', 'Total_MoMs', 'Points_Value', 'Points_Visual']),
    
    (tab3, 'All-Rounder', 'Pts_AllRounder', ['Rank', 'Player_Link', 'Team_Name', 'Matches_Played', 'Total_Runs', 'Total_Wickets', 'Bat_Avg', 'Bowl_Avg', 'Bowl_Econ', 'Total_MoMs', 'Points_Value', 'Points_Visual'])
]:

    with t:
        if not df_career.empty:
            # Filter and Sort
            df_role = df_career[df_career['Role'] == role].sort_values(pts_col, ascending=False).head(50).copy()
            df_role['Rank'] = range(1, len(df_role) + 1)
            df_role['Player_Link'] = df_role['Player_Name'].apply(make_trajectory_link)

            df_role['Bowl_SR'] = np.where(df_role['Total_Wickets'] > 0, 
                                          df_role['Total_Runs_Conceded'] / df_role['Total_Wickets'], 
                                          0.0)
            df_role['Bowl_Avg'] = np.where(
                df_role['Total_Wickets'] > 0, 
                df_role['Total_Runs_Conceded'] / df_role['Total_Wickets'], 
                0.0
            )
            
            # --- Sparkline Logic ---
            top_score = df_role[pts_col].max()
            if top_score > 0:
                df_role['Points_Visual'] = (df_role[pts_col] / top_score) * 100
            else:
                df_role['Points_Visual'] = 0
                
            df_role['Points_Value'] = df_role[pts_col]
            
            styled_df = (df_role[cols].style
                         .apply(highlight_podium, axis=1))
            
            st.dataframe(
                styled_df, 
                column_config=column_configuration, 
                use_container_width=True, 
                hide_index=True
            )

# --- TAB 4: POWER HITTERS (Post-WC 2025 ONLY) ---
with tab4:
    st.subheader("🚀 Power Hitters (Post-WorldCup 2026)")
    st.caption("Boundary dominance metrics post World Cup 2026.")
    if not df_modern.empty:
        # FILTER: Show only players who have hit at least one boundary (4 or 6)
        df_p = df_modern[(df_modern['Total_4s'] > 0) | (df_modern['Total_6s'] > 0)].copy()
        
        if not df_p.empty:

            df_p['Balls_per_Boundary'] = np.where(
                (df_p['Total_4s'] + df_p['Total_6s']) > 0,
                df_p['Total_Balls_Faced'] / (df_p['Total_4s'] + df_p['Total_6s']),
                0.0
            )
            df_p = df_p.sort_values(['Boundary_Pct'], ascending=False).head(50)

            df_p['Rank'] = range(1, len(df_p) + 1)
            df_p['Player_Link'] = df_p['Player_Name'].apply(make_trajectory_link)
            

            p_cols = ['Rank', 'Player_Link', 'Team_Name','Total_Runs', 'Total_4s', 'Total_6s', 'Boundary_Pct', 'Balls_per_Boundary']
            
            # 3. Apply Olympic Highlight Styling
            styled_p = (df_p[p_cols].style
                        .apply(highlight_podium, axis=1)
                        .format({'Boundary_Pct': "{:.1f}%"}))
            
            st.dataframe(
                styled_p, 
                column_config=column_configuration, # Uses global config for Rank/Player_Link
                use_container_width=True, 
                hide_index=True
            )
        else:
            st.info("No players with boundaries found in this period.")

# --- TAB 5: HARD TO PLAY (Post-WC 2025 ONLY) ---
with tab5:
    st.subheader("🎯 Hard to Play (Post-WorldCup 2026)")
    st.caption("Bowling pressure and dot ball metrics post World Cup 2026.")
    if not df_modern.empty:
        # FILTER: Show only players with a Dot Ball % greater than 0
        df_h = df_modern[df_modern['Dot_Ball_Pct'] > 0].copy()
        
        if not df_h.empty:
            df_h = df_h.sort_values(['Total_Dots'], ascending=False).head(50)
            df_h['Rank'] = range(1, len(df_h) + 1)
            df_h['Player_Link'] = df_h['Player_Name'].apply(make_trajectory_link)
            h_cols = ['Rank', 'Player_Link', 'Team_Name', 'Total_Balls_Bowled', 'Total_Dots', 'Dot_Ball_Pct']
                
            # 3. Apply Olympic Highlight Styling
            styled_h = (df_h[h_cols].style
                        .apply(highlight_podium, axis=1)
                        .format({'Dot_Ball_Pct': "{:.1f}%",'Total_Balls_Bowled': "{:.0f}",'Total_Dots': "{:.0f}"}))
            
            st.dataframe(
                styled_h, 
                column_config=column_configuration, 
                use_container_width=True, 
                hide_index=True
            )
        else:
            st.info("No players with recorded dot balls found in this period.")