import streamlit as st
import sqlite3
import pandas as pd
import os
import numpy as np

# --- CONFIGURATION ---
st.set_page_config(page_title="Match Day Center", page_icon="⚔️", layout="wide")

DB_FILE = "cricket_data.db"

# --- 1. NORMALIZATION LOGIC ---
TEAM_ALIASES = {
    "IND": "India", "PAK": "Pakistan", "NZ": "New Zealand", "AUS": "Australia",
    "ENG": "England", "SA": "South Africa", "WI": "West Indies", "SL": "Sri Lanka",
    "BAN": "Bangladesh", "AFG": "Afghanistan", "NED": "Netherlands", "ZIM": "Zimbabwe",
    "IRE": "Ireland", "SCO": "Scotland", "USA": "United States", "CAN": "Canada",
    "NEP": "Nepal", "OMN": "Oman", "PNG": "Papua New Guinea", "NAM": "Namibia", "UGA": "Uganda"
}

def normalize(name):
    """Standardizes names to lowercase for comparison."""
    return str(name).strip().lower()

def is_same_team(name1, name2):
    """
    Checks if two team names are the same, handling aliases.
    e.g. 'IND' == 'India' returns True.
    """
    if not name1 or not name2: return False
    n1, n2 = normalize(name1), normalize(name2)
    
    # Direct match
    if n1 == n2: return True
    
    # Check aliases (Forward and Reverse)
    # Check if n1 is an alias for n2
    alias_n1 = normalize(TEAM_ALIASES.get(name1.upper(), ""))
    if alias_n1 == n2: return True
    
    # Check if n2 is an alias for n1
    alias_n2 = normalize(TEAM_ALIASES.get(name2.upper(), ""))
    if alias_n2 == n1: return True
    
    return False

def determine_winner_abbr(t1_abbr, t2_abbr, winner_str):
    """Returns the ABBR of the winner based on match inputs."""
    if not winner_str: return None
    if is_same_team(t1_abbr, winner_str): return t1_abbr
    if is_same_team(t2_abbr, winner_str): return t2_abbr
    return None

# --- 2. DATA LOADING ---
def run_query(query):
    conn = sqlite3.connect(DB_FILE)
    try:
        return pd.read_sql_query(query, conn)
    except Exception as e:
        st.error(f"SQL Error: {e}")
        return pd.DataFrame()
    finally:
        conn.close()

@st.cache_data
def get_teams():
    df = run_query("SELECT DISTINCT Team_Name FROM player_stats ORDER BY Team_Name")
    return df['Team_Name'].tolist() if not df.empty else []

@st.cache_data
def get_venues():
    df = run_query("SELECT DISTINCT Venue FROM innings_summary ORDER BY Venue")
    return df['Venue'].tolist() if not df.empty else []

# --- 3. ANALYTICS ---

def get_venue_stats(venue):
    """
    Calculates Wins based on Winner matching Innings Team.
    Does NOT require 'Win_Margin_Type'.
    """
    query = f"""
    SELECT Match_ID, Team_Name, Winner, Innings_No, Total_Runs
    FROM innings_summary 
    WHERE Venue = '{venue}'
    """
    df = run_query(query)
    
    if df.empty: return None
    
    total_matches = df['Match_ID'].nunique()
    avg_score = df['Total_Runs'].mean()
    
    bat1_wins = 0
    bat2_wins = 0
    
    # We iterate through rows to check who won
    for _, row in df.iterrows():
        winner = row['Winner']
        team = row['Team_Name']
        innings = row['Innings_No']
        
        # If this row's team IS the winner
        if is_same_team(team, winner):
            if innings == 1:
                bat1_wins += 1
            elif innings == 2:
                bat2_wins += 1

    return {
        "matches": total_matches,
        "avg_score": int(avg_score) if not np.isnan(avg_score) else 0,
        "bat1_win_pct": (bat1_wins / total_matches * 100) if total_matches else 0,
        "bat2_win_pct": (bat2_wins / total_matches * 100) if total_matches else 0
    }

def get_h2h_rivalry(team_a, team_b):
    # Query fetches pairs of teams for every match
    query = f"""
    SELECT t1.Match_ID, t1.Venue, t1.Team_Name as T1, t2.Team_Name as T2, t1.Winner
    FROM innings_summary t1
    JOIN innings_summary t2 ON t1.Match_ID = t2.Match_ID
    WHERE (t1.Team_Name = '{team_a}' OR t1.Team_Name = '{TEAM_ALIASES.get(team_a, "XXX")}')
      AND (t2.Team_Name = '{team_a}' OR t2.Team_Name = '{TEAM_ALIASES.get(team_a, "XXX")}')
      AND t1.Team_Name != t2.Team_Name 
    """
    # Note: The logic above simplifies searching. 
    # We essentially want matches where A matches T1/T2 and B matches T2/T1.
    
    # Let's do it purely in Python for 100% safety against SQL mix-ups
    query_all = "SELECT Match_ID, Venue, Team_Name, Winner FROM innings_summary"
    df_all = run_query(query_all)
    
    if df_all.empty: return None

    # Filter in Python
    stats = {
        'total': 0, 'a_wins': 0, 'b_wins': 0, 'venues': {}
    }
    
    # Group by Match ID
    matches = df_all.groupby('Match_ID')
    
    for mid, group in matches:
        teams = group['Team_Name'].tolist()
        if len(teams) < 2: continue
        
        # Check if this match involves BOTH team A and team B
        has_a = any(is_same_team(t, team_a) for t in teams)
        has_b = any(is_same_team(t, team_b) for t in teams)
        
        if has_a and has_b:
            stats['total'] += 1
            
            # Venue
            v = group.iloc[0]['Venue']
            stats['venues'][v] = stats['venues'].get(v, 0) + 1
            
            # Winner
            w = group.iloc[0]['Winner']
            if is_same_team(w, team_a): stats['a_wins'] += 1
            elif is_same_team(w, team_b): stats['b_wins'] += 1

    return stats

def get_star_performers(team, opp):
    # Safe aliases for SQL
    t_alias = TEAM_ALIASES.get(team, team)
    o_alias = TEAM_ALIASES.get(opp, opp)
    
    bat_query = f"""
    SELECT Player_Name, SUM(Runs_Scored) as Runs, ROUND(AVG(Runs_Scored), 1) as Avg, SUM(Is_MoM) as MoM
    FROM player_stats 
    WHERE (Team_Name = '{team}' OR Team_Name = '{t_alias}')
      AND (Opposition = '{opp}' OR Opposition = '{o_alias}')
    GROUP BY Player_Name ORDER BY Runs DESC LIMIT 3
    """
    
    bowl_query = f"""
    SELECT Player_Name, SUM(Wickets_Taken) as Wkts, ROUND(AVG(Runs_Conceded), 1) as Avg_Cost,
           ROUND(CAST(SUM(Total_Balls_Bowled) AS FLOAT) / NULLIF(SUM(Wickets_Taken), 0), 1) as SR
    FROM player_stats 
    WHERE (Team_Name = '{team}' OR Team_Name = '{t_alias}')
      AND (Opposition = '{opp}' OR Opposition = '{o_alias}')
      AND Overs_Balled > 0
    GROUP BY Player_Name ORDER BY Wkts DESC LIMIT 3
    """
    return run_query(bat_query), run_query(bowl_query)

def get_bunny_alert(bat_team, bowl_team):
    bat_alias = TEAM_ALIASES.get(bat_team, bat_team)
    bowl_alias = TEAM_ALIASES.get(bowl_team, bowl_team)
    
    query = f"""
    SELECT Player_Name as Batter, Dismissed_By as Bowler, COUNT(*) as Count
    FROM player_stats
    WHERE (Team_Name = '{bat_team}' OR Team_Name = '{bat_alias}')
      AND (Opposition = '{bowl_team}' OR Opposition = '{bowl_alias}')
      AND Dismissal_Type != 'Run Out' 
      AND Dismissed_By IS NOT NULL AND Dismissed_By != ''
    GROUP BY Player_Name, Dismissed_By
    HAVING Count >= 1
    ORDER BY Count DESC LIMIT 5
    """
    return run_query(query)

# --- UI ---
def app():
    st.title("⚔️ Match Preview Center")
    
    teams, venues = get_teams(), get_venues()
    c1, c2, c3 = st.columns(3)
    team_a = c1.selectbox("Team A", teams, index=0)
    team_b = c2.selectbox("Team B", teams, index=1 if len(teams)>1 else 0)
    venue = c3.selectbox("Venue", venues)

    if st.button("Generate Report", type="primary"):
        if team_a == team_b:
            st.error("Select different teams.")
            return

        # 1. VENUE STATS
        v_data = get_venue_stats(venue)
        if v_data and v_data['matches'] > 0:
            st.subheader(f"🏟️ Venue: {venue}")
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Matches", v_data['matches'])
            k2.metric("Avg Score", v_data['avg_score'])
            k3.metric("Bat 1st Win %", f"{v_data['bat1_win_pct']:.0f}%")
            k4.metric("Bat 2nd Win %", f"{v_data['bat2_win_pct']:.0f}%")
        else:
            st.info(f"No match data found for venue: {venue}")
            
        st.divider()

        # 2. H2H RIVALRY
        st.subheader("⚔️ Rivalry Facts")
        h_data = get_h2h_rivalry(team_a, team_b)
        
        if h_data and h_data['total'] > 0:
            st.write(f"**Total Matches:** {h_data['total']}")
            
            c1, c2 = st.columns(2)
            # Percentages
            a_pct = (h_data['a_wins'] / h_data['total']) * 100
            b_pct = (h_data['b_wins'] / h_data['total']) * 100
            
            c1.info(f"**{team_a} Wins: {h_data['a_wins']}** ({a_pct:.0f}%)")
            c2.success(f"**{team_b} Wins: {h_data['b_wins']}** ({b_pct:.0f}%)")
            
            sorted_venues = sorted(h_data['venues'].items(), key=lambda x: x[1], reverse=True)
            venue_str = ", ".join([f"{v} ({c})" for v, c in sorted_venues[:5]])
            st.write(f"**Played At:** {venue_str}")
        else:
            st.warning("No Head-to-Head matches found.")

        st.divider()

        # 3. STARS & BUNNIES
        t1, t2 = st.tabs(["🌟 Star Performers", "🐇 Bunny Alert"])
        
        with t1:
            c1, c2 = st.columns(2)
            
            ba, bo = get_star_performers(team_a, team_b)
            c1.markdown(f"**🦁 {team_a} vs {team_b}**")
            if not ba.empty: c1.dataframe(ba, hide_index=True)
            else: c1.caption("No batting stats.")
            if not bo.empty: c1.dataframe(bo, hide_index=True)
            else: c1.caption("No bowling stats.")
            
            bb, bbo = get_star_performers(team_b, team_a)
            c2.markdown(f"**🦅 {team_b} vs {team_a}**")
            if not bb.empty: c2.dataframe(bb, hide_index=True)
            else: c2.caption("No batting stats.")
            if not bbo.empty: c2.dataframe(bbo, hide_index=True)
            else: c2.caption("No bowling stats.")
            
        with t2:
            c1, c2 = st.columns(2)
            
            bun_a = get_bunny_alert(team_b, team_a)
            c1.markdown(f"**{team_a} Bowlers vs {team_b} Batters**")
            if not bun_a.empty: c1.table(bun_a)
            else: c1.caption("No rivalries found.")
            
            bun_b = get_bunny_alert(team_a, team_b)
            c2.markdown(f"**{team_b} Bowlers vs {team_a} Batters**")
            if not bun_b.empty: c2.table(bun_b)
            else: c2.caption("No rivalries found.")

if __name__ == "__main__":
    app()