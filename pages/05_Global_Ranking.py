import streamlit as st
import pandas as pd
import sqlite3
import numpy as np
import os, base64
from cricket_utils import get_db_path, make_trajectory_link,highlight_podium,determine_primary_role
from cricket_utils import TEAM_ALIASES
script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(script_dir)
ASSETS_DIR = os.path.join(root_dir, "assets")


def get_base64_asset(name, type="player", extension="png"):
    """Converts local image to Base64. extension can be 'png' or 'gif'."""
    if type == "player":
        file_name = f"{str(name).replace(' ', '').lower()}.{extension}"
        default_file = "default_player.jpg"
    else:
        full_name = TEAM_ALIASES.get(str(name).upper(), name)
        file_name = f"{full_name}.png"
        default_file = "default_team.png"

    file_path = os.path.join(ASSETS_DIR, file_name)
    
    # Fallback if the specific extension doesn't exist
    if not os.path.exists(file_path):
        # Try falling back to .png if the .gif was requested but missing
        if extension == "gif":
            file_name_png = f"{str(name).replace(' ', '').lower()}.png"
            file_path = os.path.join(ASSETS_DIR, file_name_png)
        
        # Final fallback to default icon
        if not os.path.exists(file_path):
            file_path = os.path.join(ASSETS_DIR, default_file)
            if not os.path.exists(file_path):
                return ""

    try:
        with open(file_path, "rb") as f:
            data = f.read()
            encoded = base64.b64encode(data).decode()
        # Dynamically set the MIME type based on the file extension
        mime_type = "image/gif" if file_path.endswith(".gif") else "image/png"
        return f"data:{mime_type};base64,{encoded}"
    except Exception:
        return ""
    
def display_hero_card_global(player_data, rank):
    colors = {1: "#FFD700", 2: "#C0C0C0", 3: "#CD7F32"}
    border_color = colors.get(rank, "#444")
    role = player_data.get('Assigned_Role', 'Batsman')
    global_rank = int(player_data.get('Global_Rank', 0))
    
    with st.container(border=True):
        # st.markdown(f"<h1 style='text-align: center; color: {border_color}; margin-bottom: 0;'>#{rank}</h1>", unsafe_allow_html=True)
        st.markdown(f"""<div style="display: flex; flex-direction: column;align-items: center;justify-content: center;text-align: center;
            ">
                <h1 style="color: {border_color}; margin: 0;line-height: 1;">{rank}</h1>
            </div>
            """, 
            unsafe_allow_html=True
        )
        # Request the .gif extension (falls back to png in get_base64_asset)
        icon_base64 = get_base64_asset(player_data['Player_Name'], type="player", extension="gif")
        
        if icon_base64:
            st.image(icon_base64, use_container_width=True)
        
        st.markdown(f"<h3 style='text-align: center; margin-top: 0;'>{player_data['Player_Name']}</h3>", unsafe_allow_html=True)
        
        # Dynamic Metric Layout based on Role
        c1, c2, c3 = st.columns(3)
        
        if role == "Batsman":
            with c1:
                st.metric("Runs", player_data['Total_Runs'])
            with c2:
                st.metric("Avg", f"{player_data['Bat_Avg']:.1f}")
            with c3:
                st.metric("50+", player_data['Count_50s']) 
            st.caption(f"SR: {player_data['Bat_SR']:.1f} | MoM: {int(player_data['Total_MoMs'])}")
            
        elif role == "Bowler":
            with c1:
                st.metric("Wickets", int(player_data['Total_Wickets']))
            with c2:
                st.metric("Econ", f"{player_data['Bowl_Econ']:.2f}")
            with c3:
                st.metric("3W+", player_data['Count_3W']) 
            st.caption(f"Avg: {player_data['Bowl_Avg']:.1f} | MoM: {int(player_data['Total_MoMs'])}")
            
        elif role == "All-Rounder":
            with c1:
                st.metric("Runs", player_data['Total_Runs'])
            with c2:
                st.metric("Wkts", int(player_data['Total_Wickets']))
            with c3:
                st.metric("MoM", player_data['Total_MoMs'])
            st.caption(f"Bat Avg: {player_data['Bat_Avg']:.1f} | Bowl Avg: {player_data['Bowl_Avg']:.1f}")



# --- PAGE CONFIG ---
st.set_page_config(page_title="Global Rankings", page_icon="🌍", layout="wide")

st.title("🌍 Global Player Rankings")
st.markdown("### Performance Leaderboards")

DB_FILE = get_db_path()

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
    "Player_Icon": st.column_config.ImageColumn("", width="small"),
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
    (tab1, 'Batsman', 'Pts_Batting', ['Player_Icon', 'Rank', 'Player_Link', 'Team_Name', 'Matches_Played', 'Total_Runs', 'Bat_Avg', 'Bat_SR', 'Count_30s', 'Count_50s', 'Total_MoMs', 'Points_Value', 'Points_Visual']),
    (tab2, 'Bowler', 'Pts_Bowling', ['Player_Icon', 'Rank', 'Player_Link', 'Team_Name', 'Matches_Played', 'Total_Wickets', 'Bowl_Avg', 'Bowl_SR', 'Bowl_Econ', 'Count_3W', 'Count_4W', 'Count_5W', 'Total_MoMs', 'Points_Value', 'Points_Visual']),
    (tab3, 'All-Rounder', 'Pts_AllRounder', ['Player_Icon', 'Rank', 'Player_Link', 'Team_Name', 'Matches_Played', 'Total_Runs', 'Total_Wickets', 'Bat_Avg', 'Bowl_Avg', 'Bowl_Econ', 'Total_MoMs', 'Points_Value', 'Points_Visual'])
]:

    with t:
        if not df_career.empty:
            # 1. Filter, Sort, and Create Base Columns
            df_role = df_career[df_career['Role'] == role].sort_values(pts_col, ascending=False).head(50).copy()
            df_role['Rank'] = range(1, len(df_role) + 1)
            df_role['Global_Rank'] = df_role['Rank']
            df_role['Assigned_Role'] = role
            df_role['Player_Link'] = df_role['Player_Name'].apply(make_trajectory_link)
            
            # 2. Generate the Icons (Assigned directly to df_role)
            # This ensures the 'Player_Icon' column exists before we call the dataframe
            df_role['Player_Icon'] = df_role['Player_Name'].apply(
                lambda x: get_base64_asset(x, type="player")
            )

            # 3. Calculate Performance Metrics
            top_score = df_role[pts_col].max()
            df_role['Points_Visual'] = (df_role[pts_col] / top_score * 100) if top_score > 0 else 0
            df_role['Points_Value'] = df_role[pts_col]
            
            # Bowling specific safety checks
            df_role['Bowl_SR'] = np.where(df_role['Total_Wickets'] > 0, 
                                        df_role['Total_Runs_Conceded'] / df_role['Total_Wickets'], 0.0)
            df_role['Bowl_Avg'] = np.where(df_role['Total_Wickets'] > 0, 
                                          df_role['Total_Runs_Conceded'] / df_role['Total_Wickets'], 0.0)

            # 4. HERO SECTION
            st.markdown('### 🏆 Top Global Performers')
            top_3 = df_role.head(3)
            p_cols = st.columns([1, 1, 1, 1, 1])

            for i in range(len(top_3)):
                with p_cols[i + 1]:
                    display_hero_card_global(top_3.iloc[i].to_dict(), i + 1)
                    
            st.divider()

            # 5. TABLE SECTION (Cleaned up)
            st.markdown("#### 📊 Leaderboard")
            st.dataframe(
                df_role[cols], 
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