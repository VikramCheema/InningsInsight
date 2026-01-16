import streamlit as st
import sqlite3
import pandas as pd

# 1. PAGE CONFIGURATION
st.set_page_config(page_title="Cricket Rivalry Analytics", layout="wide")
st.title("🏏 Brother vs. Brother: Match Analytics")

# 2. CONNECT TO DATABASE
@st.cache_data
def load_data(query):
    # Ensure 'cricket_data.db' is in the same folder
    conn = sqlite3.connect('cricket_data.db')
    try:
        df = pd.read_sql_query(query, conn)
    except Exception as e:
        st.error(f"SQL Error: {e}")
        df = pd.DataFrame()
    conn.close()
    return df

# 3. SIDEBAR CONTROLS
st.sidebar.header("Match Filters")

# Fetch unique tournaments
tournaments = load_data("SELECT DISTINCT Tournament_ID FROM player_stats")

if not tournaments.empty:
    selected_tournament = st.sidebar.selectbox("Select Tournament", tournaments['Tournament_ID'])
    
    # --- MAIN DASHBOARD ---
    
    # Get list of players for the dropdown based on tournament
    players = load_data(f"SELECT DISTINCT Player_Name FROM player_stats WHERE Tournament_ID = '{selected_tournament}'")
    
    if not players.empty:
        selected_player = st.selectbox("Select a Player to Analyze", players['Player_Name'])

        # Updated Query using UNDERSCORES based on your snippet
        # I am assuming 'Strike_Rate' and 'Balls_Faced' also follow this pattern based on 'Runs_Scored'
        player_sql = f"""
            SELECT 
                Match_ID, 
                Opposition, 
                Runs_Scored, 
                Balls_Faced,
                Strike_Rate, 
                Dismissal_Type,
                Dismissed_By,
                Is_MoM
            FROM player_stats
            WHERE Player_Name = '{selected_player}' 
            AND Tournament_ID = '{selected_tournament}'
            ORDER BY Match_ID ASC
        """
        player_df = load_data(player_sql)

        if not player_df.empty:
            st.subheader(f"Stats for {selected_player}")
            
            # --- METRICS SECTION ---
            # Calculating metrics using Pandas (easier than complex SQL for simple sums)
            total_runs = player_df["Runs_Scored"].sum()
            # Handle cases where Strike_Rate might be missing or 0 to avoid errors
            avg_sr = player_df["Strike_Rate"].mean() if "Strike_Rate" in player_df.columns else 0
            highest = player_df["Runs_Scored"].max()
            mom_count = player_df["Is_MoM"].sum()
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Runs", int(total_runs))
            col2.metric("Avg Strike Rate", round(avg_sr, 2))
            col3.metric("Highest Score", int(highest))
            col4.metric("Man of Match", int(mom_count))
            
            # --- CHARTS SECTION ---
            
            # 1. Runs over time (Line Chart)
            st.subheader("Run Scoring Trajectory")
            st.line_chart(player_df.set_index("Match_ID")["Runs_Scored"])

            # 2. Performance vs Bowlers (Bar Chart)
            # This helps you see who gets this player out the most or gives up most runs
            if "Dismissed_By" in player_df.columns:
                st.subheader("Performance vs. Bowlers")
                # Group data to sum runs against specific bowlers
                vs_bowler = player_df.groupby("Dismissed_By")[["Runs_Scored", "Balls_Faced"]].sum().reset_index()
                # Filter out empty bowler names (e.g. if Not Out)
                vs_bowler = vs_bowler[vs_bowler["Dismissed_By"] != ""]
                
                st.bar_chart(vs_bowler.set_index("Dismissed_By")["Runs_Scored"])

            # --- RAW DATA ---
            with st.expander("See detailed match logs"):
                st.dataframe(player_df)
        else:
            st.warning("No stats found for this player in this tournament.")
    else:
        st.error("No players found in this tournament.")
else:
    st.error("Could not load tournaments. Check database connection.")