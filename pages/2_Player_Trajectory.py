import streamlit as st
import sqlite3
import pandas as pd
import os
import altair as alt

# --- CONFIGURATION ---
st.set_page_config(page_title="Player Trajectory", page_icon="📈", layout="wide")

# --- 1. ROBUST PATH FINDER ---
def get_db_path():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(script_dir)
    db_path = os.path.join(root_dir, "cricket_data.db")
    return db_path

DB_FILE = get_db_path()

# --- 2. CONSTANTS & MAPPINGS ---
CURRENT_TOUR_PREFIX = "026"  
BAT_DROP_TOLERANCE = 0.30
BOWL_AVG_TOLERANCE = 0.40

TEAM_ALIASES = {
    "IND": "India", "PAK": "Pakistan", "NZ": "New Zealand", "AUS": "Australia",
    "ENG": "England", "SA": "South Africa", "WI": "West Indies", "SL": "Sri Lanka",
    "BAN": "Bangladesh", "AFG": "Afghanistan", "NED": "Netherlands"
}

# --- 3. DATABASE ENGINE ---
def run_query(query, params=None):
    if not os.path.exists(DB_FILE):
        st.error(f"❌ Database not found at: {DB_FILE}")
        return pd.DataFrame()
    conn = sqlite3.connect(DB_FILE)
    try:
        return pd.read_sql_query(query, conn, params=params)
    except Exception as e:
        st.error(f"SQL Error: {e}")
        return pd.DataFrame()
    finally:
        conn.close()

@st.cache_data
def get_teams():
    df = run_query("SELECT DISTINCT Team_Name FROM player_stats ORDER BY Team_Name")
    return df['Team_Name'].tolist() if not df.empty else []

def get_squad(team):
    t_code = next((k for k, v in TEAM_ALIASES.items() if v == team), team)
    q = f"""
    SELECT DISTINCT Player_Name FROM player_stats 
    WHERE Team_Name = '{team}' OR Team_Name = '{t_code}'
    ORDER BY Player_Name
    """
    df = run_query(q)
    return df['Player_Name'].tolist()

def get_opponents(player):
    q = f"SELECT DISTINCT Opposition FROM player_stats WHERE Player_Name = '{player}' ORDER BY Opposition"
    df = run_query(q)
    return df['Opposition'].tolist()

# --- 4. ADVANCED LOGIC: ROLE CLASSIFIER ---
def determine_role(df):
    if df.empty: return "Newcomer"
    
    runs = df['Runs_Scored'].sum()
    outs = df['Innings_Out'].sum()
    wkts = df['Wickets_Taken'].sum()
    balls_faced = df['Balls_Faced'].sum()
    moms = df['Is_MoM'].sum()
    
    balls_bowled = df['Total_Balls_Bowled'].sum()
    overs = balls_bowled / 6.0
    runs_conceded = df['Runs_Conceded'].sum()
    bowl_econ = runs_conceded / overs if overs > 0 else 0
    
    ducks = len(df[(df['Runs_Scored'] == 0) & (df['Innings_Out'] == 1)])
    wkt_hauls = len(df[df['Wickets_Taken'] >= 3])
    zero_wkts = len(df[(df['Wickets_Taken'] == 0) & (df['Total_Balls_Bowled'] > 0)])
    
    bat_avg = runs / outs if outs > 0 else runs
    bat_pts = (runs * 0.5) + (bat_avg * 1.0) + (moms * 10) - (ducks * 5)
    
    bowl_pts = (wkts * 20) + (moms * 10) + (wkt_hauls * 15) - (zero_wkts * 5)
    if bowl_econ > 0 and bowl_econ < 6.0: bowl_pts += 20 
    
    ar_threshold = 200 
    
    if bat_pts > ar_threshold and bowl_pts > ar_threshold:
        return "All-Rounder"
    elif bowl_pts > bat_pts:
        return "Bowler"
    else:
        return "Batsman"

# --- 5. FORM AUDIT ENGINE ---
def audit_form(hist, curr, role):
    # Batting
    h_outs = hist['Innings_Out'].sum()
    h_avg = hist['Runs_Scored'].sum() / h_outs if h_outs > 0 else 0
    
    c_outs = curr['Innings_Out'].sum()
    c_avg = curr['Runs_Scored'].sum() / c_outs if c_outs > 0 else 0
    
    bat_drop = (h_avg - c_avg) / h_avg if h_avg > 0 else 0
    bat_bad = bat_drop > BAT_DROP_TOLERANCE
    
    # Bowling
    h_wkts = hist['Wickets_Taken'].sum()
    h_bowl_avg = hist['Runs_Conceded'].sum() / h_wkts if h_wkts > 0 else 50
    
    c_wkts = curr['Wickets_Taken'].sum()
    c_runs = curr['Runs_Conceded'].sum()
    c_bowl_avg = (c_runs / c_wkts) if c_wkts > 0 else (99 if curr['Total_Balls_Bowled'].sum() > 0 else h_bowl_avg)
    
    bowl_spike = (c_bowl_avg - h_bowl_avg) / h_bowl_avg if h_bowl_avg > 0 else 0
    bowl_bad = bowl_spike > BOWL_AVG_TOLERANCE

    if role == "Batsman":
        if bat_bad: return "🔴 OUT OF FORM", f"Avg dropped {h_avg:.1f} -> {c_avg:.1f}"
        return "🟢 IN FORM", "Batting consistent"
        
    elif role == "Bowler":
        if bowl_bad: return "🔴 OUT OF FORM", f"Bowl Avg worsened {h_bowl_avg:.1f} -> {c_bowl_avg:.1f}"
        return "🟢 IN FORM", "Bowling consistent"
        
    elif role == "All-Rounder":
        if bat_bad and bowl_bad: return "🔴 CRITICAL", "Both disciplines failing"
        if bat_bad: return "⚠️ DIP", "Batting slump, saved by bowling"
        if bowl_bad: return "⚠️ DIP", "Bowling expensive, saved by batting"
        return "⭐ PEAK FORM", "Firing on all cylinders"
        
    return "⚪ NEUTRAL", "Not enough recent games"

# --- 6. MAIN UI ---
def app():
    st.title("📈 Player Career Trajectory")
    
    col1, col2, col3 = st.columns(3)
    team = col1.selectbox("Select Team", get_teams())
    
    if team:
        players = get_squad(team)
        player = col2.selectbox("Select Player", players)
        
        if player:
            opps = ["All Teams"] + get_opponents(player)
            opp_select = col3.selectbox("Filter Opponent", opps)
            
            opp_clause = f"AND Opposition = '{opp_select}'" if opp_select != "All Teams" else ""
            
            query = f"""
            SELECT Match_ID, Team_Name, Opposition, Runs_Scored, Balls_Faced, Innings_Out, Is_MoM,
                   Wickets_Taken, Runs_Conceded, Total_Balls_Bowled, Dismissal_Type
            FROM player_stats
            WHERE Player_Name = '{player}' {opp_clause}
            ORDER BY Match_ID ASC
            """
            df = run_query(query)
            
            if df.empty:
                st.warning("No data found.")
                st.stop()
            
            current_df = df[df['Match_ID'].astype(str).str.startswith(CURRENT_TOUR_PREFIX)]
            hist_df = df[~df['Match_ID'].astype(str).str.startswith(CURRENT_TOUR_PREFIX)]
            
            role = determine_role(df)
            form_status, form_reason = audit_form(hist_df, current_df, role)
            
            with st.container():
                c1, c2 = st.columns([1, 3])
                with c1:
                    st.subheader(player)
                    st.caption(f"Team: {team}")
                    st.markdown(f"**Role:** `{role}`")
                
                with c2:
                    m1, m2, m3, m4 = st.columns(4)
                    total_runs = df['Runs_Scored'].sum()
                    total_wkts = df['Wickets_Taken'].sum()
                    matches = len(df)
                    moms = df['Is_MoM'].sum()
                    
                    m1.metric("Matches", matches)
                    m2.metric("Total Runs", total_runs)
                    m3.metric("Total Wickets", total_wkts)
                    m4.metric("MoM Awards", moms)

                if "🔴" in form_status or "⚠️" in form_status:
                    st.error(f"**{form_status}**: {form_reason}")
                else:
                    st.success(f"**{form_status}**: {form_reason}")

            st.divider()

            # --- 8. TRAJECTORY GRAPHS ---
            tab_bat, tab_bowl = st.tabs(["🏏 Batting Arc", "⚾ Bowling Impact"])
            
            with tab_bat:
                df['Bat_MA_5'] = df['Runs_Scored'].rolling(window=5).mean()
                
                base = alt.Chart(df.reset_index()).encode(x=alt.X('index', title='Match Timeline'))
                bars = base.mark_bar(color='#4CAF50', opacity=0.6).encode(
                    y=alt.Y('Runs_Scored', title='Runs'),
                    tooltip=['Match_ID', 'Opposition', 'Runs_Scored', 'Dismissal_Type']
                )
                line = base.mark_line(color='red', size=2).encode(
                    y=alt.Y('Bat_MA_5', title='5-Match Avg Trend')
                )
                
                chart = (bars + line).properties(height=350).interactive()
                st.altair_chart(chart, use_container_width=True)
                
                # UPDATED: Key Stats Grid (30s and 50s)
                kb1, kb2, kb3, kb4 = st.columns(4)
                hs = df['Runs_Scored'].max()
                outs = df['Innings_Out'].sum()
                avg = total_runs / outs if outs > 0 else total_runs
                
                # 50s: Score >= 50 (Includes 100s)
                count_50s = len(df[df['Runs_Scored'] >= 50])
                # 30s: Score between 30 and 49 (inclusive)
                count_30s = len(df[(df['Runs_Scored'] >= 30) & (df['Runs_Scored'] < 50)])
                
                kb1.metric("Highest Score", hs)
                kb2.metric("Career Average", f"{avg:.2f}")
                kb3.metric("50s", count_50s)
                kb4.metric("30s", count_30s)

            with tab_bowl:
                bowl_df = df[df['Total_Balls_Bowled'] > 0].reset_index()
                
                if not bowl_df.empty:
                    chart_bowl = alt.Chart(bowl_df).mark_bar(color='#2196F3').encode(
                        x=alt.X('index', title='Matches Bowled'),
                        y=alt.Y('Wickets_Taken', title='Wickets'),
                        tooltip=['Match_ID', 'Opposition', 'Wickets_Taken', 'Runs_Conceded']
                    ).properties(height=350).interactive()
                    st.altair_chart(chart_bowl, use_container_width=True)
                    
                    kw1, kw2, kw3, kw4 = st.columns(4)
                    best_fig_row = bowl_df.sort_values(by=['Wickets_Taken', 'Runs_Conceded'], ascending=[False, True]).iloc[0]
                    best_fig = f"{best_fig_row['Wickets_Taken']}/{best_fig_row['Runs_Conceded']}"
                    
                    tot_overs = bowl_df['Total_Balls_Bowled'].sum() / 6.0
                    econ = bowl_df['Runs_Conceded'].sum() / tot_overs if tot_overs > 0 else 0
                    
                    kw1.metric("Best Bowling", best_fig)
                    kw2.metric("Economy Rate", f"{econ:.2f}")
                    kw3.metric("5-Wicket Hauls", len(bowl_df[bowl_df['Wickets_Taken'] >= 5]))
                    kw4.metric("3-Wicket Hauls", len(bowl_df[bowl_df['Wickets_Taken'] >= 3]))
                else:
                    st.info("This player has never bowled in the selected matches.")

if __name__ == "__main__":
    app()