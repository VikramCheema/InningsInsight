import streamlit as st
import sqlite3
import pandas as pd
import os
import numpy as np

# --- CONFIGURATION ---
st.set_page_config(page_title="Match Center", page_icon="⚔️", layout="wide")

# --- 1. ROBUST PATH FINDER ---
def get_db_path():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(script_dir)
    db_path = os.path.join(root_dir, "cricket_data.db")
    return db_path

DB_FILE = get_db_path()

# --- 2. NORMALIZATION LOGIC ---
TEAM_ALIASES = {
    "IND": "India", "PAK": "Pakistan", "NZ": "New Zealand", "AUS": "Australia",
    "ENG": "England", "SA": "South Africa", "WI": "West Indies", "SL": "Sri Lanka",
    "BAN": "Bangladesh", "AFG": "Afghanistan", "NED": "Netherlands", "ZIM": "Zimbabwe",
    "IRE": "Ireland", "SCO": "Scotland", "USA": "United States", "CAN": "Canada",
    "NEP": "Nepal", "OMN": "Oman", "PNG": "Papua New Guinea", "NAM": "Namibia", "UGA": "Uganda"
}

def normalize(name):
    return str(name).strip().lower()

def is_same_team(name1, name2):
    if not name1 or not name2: return False
    n1, n2 = normalize(name1), normalize(name2)
    if n1 == n2: return True
    alias_n1 = normalize(TEAM_ALIASES.get(name1.upper(), ""))
    if alias_n1 == n2: return True
    alias_n2 = normalize(TEAM_ALIASES.get(name2.upper(), ""))
    if alias_n2 == n1: return True
    return False

script_dir = os.path.dirname(os.path.abspath(__file__))

# Go UP one level to the parent directory (the main app folder)
root_dir = os.path.dirname(script_dir)

# Now join with 'assets'
ASSETS_DIR = os.path.join(root_dir, "assets")
import base64
def get_base64_image(team_name):
    """
    Finds local image, converts to Base64.
    Includes Debugging and Whitespace cleaning.
    """
    # 1. Clean the input (Remove invisible spaces)
    clean_name = str(team_name).strip()
    
    # 2. Get the full name from Alias map
    # Note: We upper() the clean name to match keys like 'IND'
    full_name = TEAM_ALIASES.get(clean_name.upper(), clean_name)
    
    # 3. Construct the path
    file_name = f"{full_name}.png"
    file_path = os.path.join(ASSETS_DIR, file_name)

    # 4. Check existence
    if not os.path.exists(file_path):
        # If we fail, print a warning to the UI so you know exactly which file failed
        st.toast(f"⚠️ Missing flag: {file_name}", icon="❌")
        return "https://cdn-icons-png.flaticon.com/512/1165/1165249.png"
    try:
        with open(file_path, "rb") as f:
            data = f.read()
            encoded = base64.b64encode(data).decode()
        return f"data:image/png;base64,{encoded}"
    except Exception as e:
        st.error(f"Error loading {full_name}: {e}")
        return ""
    
# Update the helper to use the local function
def get_flag_url(team_name):
    return get_base64_image(team_name)

# --- 3. DATABASE HELPERS ---
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

@st.cache_data
def get_venues():
    df = run_query("SELECT DISTINCT Venue FROM innings_summary ORDER BY Venue")
    return df['Venue'].tolist() if not df.empty else []

# --- 4. ADVANCED ANALYTICS ---

def get_venue_records(venue):
    query_match = f"""
    SELECT Match_ID, Team_Name, Winner, Innings_No, Total_Runs
    FROM innings_summary 
    WHERE Venue = '{venue}'
    """
    df = run_query(query_match)
    
    stats = {}
    
    if not df.empty:
        stats['matches'] = df['Match_ID'].nunique()
        stats['avg_score_1st'] = int(df[df['Innings_No'] == 1]['Total_Runs'].mean())
        stats['highest_total'] = int(df['Total_Runs'].max())
        stats['lowest_total'] = int(df['Total_Runs'].min())
        
        bat1_wins = 0
        bat2_wins = 0
        highest_chase = 0
        lowest_defend = 9999
        
        for mid in df['Match_ID'].unique():
            m = df[df['Match_ID'] == mid]
            if m.empty: continue
            
            winner = m.iloc[0]['Winner']
            inn1 = m[m['Innings_No'] == 1]
            inn2 = m[m['Innings_No'] == 2]
            
            if inn1.empty or inn2.empty: continue
            
            t1 = inn1.iloc[0]['Team_Name']
            score1 = inn1.iloc[0]['Total_Runs']
            score2 = inn2.iloc[0]['Total_Runs']
            
            if is_same_team(winner, t1):
                bat1_wins += 1
                if score1 < lowest_defend: lowest_defend = score1
            else:
                bat2_wins += 1
                if score2 > highest_chase: highest_chase = score2

        stats['bat1_win_pct'] = (bat1_wins / stats['matches'] * 100)
        stats['bat2_win_pct'] = (bat2_wins / stats['matches'] * 100)
        stats['highest_chase'] = highest_chase if highest_chase > 0 else "N/A"
        stats['lowest_defend'] = lowest_defend if lowest_defend != 9999 else "N/A"

    q_bat = f"""
    SELECT p.Player_Name, p.Runs_Scored, p.Team_Name
    FROM player_stats p
    JOIN innings_summary i ON p.Match_ID = i.Match_ID
    WHERE i.Venue = '{venue}'
    ORDER BY p.Runs_Scored DESC LIMIT 1
    """
    best_bat = run_query(q_bat)
    if not best_bat.empty:
        stats['best_bat'] = f"{best_bat.iloc[0]['Player_Name']} ({best_bat.iloc[0]['Runs_Scored']})"
    else:
        stats['best_bat'] = "N/A"

    q_bowl = f"""
    SELECT p.Player_Name, p.Wickets_Taken, p.Runs_Conceded, p.Team_Name
    FROM player_stats p
    JOIN innings_summary i ON p.Match_ID = i.Match_ID
    WHERE i.Venue = '{venue}'
    ORDER BY p.Wickets_Taken DESC, p.Runs_Conceded ASC LIMIT 1
    """
    best_bowl = run_query(q_bowl)
    if not best_bowl.empty:
        stats['best_bowl'] = f"{best_bowl.iloc[0]['Player_Name']} ({best_bowl.iloc[0]['Wickets_Taken']}/{best_bowl.iloc[0]['Runs_Conceded']})"
    else:
        stats['best_bowl'] = "N/A"

    return stats

def get_h2h_rivalry(team_a, team_b):
    query_all = "SELECT Match_ID, Venue, Team_Name, Winner FROM innings_summary"
    df_all = run_query(query_all)
    
    if df_all.empty: return None

    stats = {'total': 0, 'a_wins': 0, 'b_wins': 0, 'venues': {}}
    matches = df_all.groupby('Match_ID')
    
    for mid, group in matches:
        teams = group['Team_Name'].tolist()
        if len(teams) < 2: continue
        
        has_a = any(is_same_team(t, team_a) for t in teams)
        has_b = any(is_same_team(t, team_b) for t in teams)
        
        if has_a and has_b:
            stats['total'] += 1
            v = group.iloc[0]['Venue']
            stats['venues'][v] = stats['venues'].get(v, 0) + 1
            w = group.iloc[0]['Winner']
            if is_same_team(w, team_a): stats['a_wins'] += 1
            elif is_same_team(w, team_b): stats['b_wins'] += 1

    return stats

def get_recent_h2h_matches(team_a, team_b, limit=5):
    """
    Fetches recent matches and constructs the result string using 
    Win_Type and Win_Margin columns.
    """
    # 1. Fetch Win_Type and Margin columns as per your schema
    query = f"""
    SELECT Match_ID, Venue, Winner, Team_Name, 
           Win_Type, Win_Margin_Runs, Win_Margin_Wickets
    FROM innings_summary
    ORDER BY Match_ID DESC
    """
    df_raw = run_query(query)
    
    if df_raw.empty: return []

    matches = []
    
    # Group by Match_ID 
    for mid, group in df_raw.groupby("Match_ID", sort=False):
        teams = group['Team_Name'].tolist()
        
        # Ensure both rival teams are in this match
        has_a = any(is_same_team(t, team_a) for t in teams)
        has_b = any(is_same_team(t, team_b) for t in teams)
        
        if has_a and has_b:
            row = group.iloc[0]
            winner = row['Winner']
            
            # --- CONSTRUCT RESULT STRING ---
            win_type = str(row['Win_Type']).strip()  # e.g., 'Runs' or 'Wickets'
            result_text = "Won Match" # Default fallback
            
            try:
                if win_type == 'Runs':
                    margin = int(row['Win_Margin_Runs'])
                    result_text = f"Won by {margin} Runs"
                    
                elif win_type == 'Wickets':
                    margin = int(row['Win_Margin_Wickets'])
                    result_text = f"Won by {margin} Wickets"
                    
                elif win_type == 'Super Over':
                    result_text = "Won via Super Over"
                    
            except Exception:
                pass

            matches.append({
                'Match_ID': mid,
                'Venue': row['Venue'],
                'Winner': winner,
                'Result': result_text 
            })
            
            if len(matches) >= limit:
                break
                
    return matches
def get_star_performers(team, opp):
    t_alias = TEAM_ALIASES.get(team, team)
    o_alias = TEAM_ALIASES.get(opp, opp)
    
    bat_query = f"""
    SELECT Player_Name, SUM(Runs_Scored) as Runs, MAX(Runs_Scored) as HS, ROUND(AVG(Runs_Scored), 1) as Avg, SUM(Is_MoM) as MoM
    FROM player_stats 
    WHERE (Team_Name = '{team}' OR Team_Name = '{t_alias}')
      AND (Opposition = '{opp}' OR Opposition = '{o_alias}')
    GROUP BY Player_Name ORDER BY Runs DESC LIMIT 3
    """
    
    # UPDATED: Added SUM(Is_MoM)
    bowl_query = f"""
    SELECT Player_Name, SUM(Wickets_Taken) as Wkts, ROUND(AVG(Runs_Conceded), 1) as BA,
           ROUND(CAST(SUM(Total_Balls_Bowled) AS FLOAT) / NULLIF(SUM(Wickets_Taken), 0), 1) as SR,
           SUM(Is_MoM) as MoM
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

def render_timeline(matches, team_a, team_b):
    if not matches:
        return

    st.markdown("""
    <style>
        .timeline-container {
            display: flex;
            flex-direction: row;
            align-items: flex-start;
            justify-content: flex-start;
            overflow-x: auto;
            padding: 10px 5px;
            gap: 15px; /* Increased gap slightly */
            font-family: sans-serif;
            scrollbar-width: thin;
        }
        
        .match-card {
            /* USE THEME BACKGROUND */
            background: var(--secondary-background-color); 
            border-radius: 12px;
            padding: 12px 10px;
            min-width: 140px;
            max-width: 140px;
            display: flex;
            flex-direction: column;
            align-items: center;
            position: relative;
            /* Thin border that matches text color but very faint */
            border: 1px solid rgba(128, 128, 128, 0.2);
            transition: transform 0.2s;
        }
        .match-card:hover {
            transform: translateY(-3px);
            border-color: var(--primary-color);
        }
        
        .flag-img {
            width: 70px;
            height: 46px;
            object-fit: cover;
            border-radius: 6px;
            margin-bottom: 8px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.2); 
        }
        
        /* --- DYNAMIC TEXT COLORS --- */
        .result-text {
            font-size: 0.75rem;
            font-weight: 700;
            color: var(--text-color); /* ADAPTS TO THEME */
            margin-bottom: 4px;
            text-align: center;
            line-height: 1.2;
        }
        
        .venue-text {
            font-size: 0.65rem;
            color: var(--text-color); /* ADAPTS TO THEME */
            opacity: 0.7; /* Make it slightly dimmer than main text */
            text-align: center;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
            width: 100%;
        }
        
        .match-label {
            font-size: 0.6rem;
            text-transform: uppercase;
            letter-spacing: 1px;
            color: var(--text-color); /* ADAPTS TO THEME */
            opacity: 0.5;
            margin-bottom: 8px;
            text-align: center;
            font-weight: 600;
        }
        
        .arrow {
            font-size: 1.5rem;
            color: var(--text-color); /* ADAPTS TO THEME */
            opacity: 0.3;
            font-weight: 600;
            margin: 0 5px;
            margin-top: 35px;
            user-select: none;
        }
        
        /* Colored bottom borders still okay as they are distinct colors */
        .win-a .flag-img { border-bottom: 4px solid #2ecc71; }
        .win-b .flag-img { border-bottom: 4px solid #e67e22; }

    </style>
    """, unsafe_allow_html=True)

    html_content = '<div class="timeline-container">'
    
    for i, m in enumerate(matches):
        winner = m['Winner']
        venue = m['Venue']
        result = m.get('Result', 'Won')
        
        flag_url = get_flag_url(winner)
        
        if is_same_team(winner, team_a):
            win_class = "win-a"
        elif is_same_team(winner, team_b):
            win_class = "win-b"
        else:
            win_class = ""

        label = "LATEST" if i == 0 else f"{i} MATCHES AGO"

        card_html = f"""
        <div class="match-card {win_class}">
            <div class="match-label">{label}</div>
            <img src="{flag_url}" class="flag-img" onerror="this.style.display='none'">
            <div class="result-text">{result}</div>
            <div class="venue-text" title="{venue}">{venue}</div>
        </div>
        """
        
        html_content += card_html
        
        if i < len(matches) - 1:
            html_content += '<div class="arrow">←</div>'

    html_content += '</div>'
    
    st.markdown(html_content, unsafe_allow_html=True)

# --- UI LAYOUT ---
def app():
    st.title("⚔️ Match Center")
    
    if not get_teams():
        st.warning("⚠️ Database appears empty or inaccessible. Please check 'cricket_data.db'.")
        st.stop()
    
    teams, venues = get_teams(), get_venues()
    c1, c2, c3 = st.columns(3)
    team_a = c1.selectbox("Team A", teams, index=0)
    team_b = c2.selectbox("Team B", teams, index=1 if len(teams)>1 else 0)
    venue = c3.selectbox("Venue", venues)

    if st.button("Generate Report", type="primary"):
        if team_a == team_b:
            st.error("Select different teams.")
            return

        # VENUE INTELLIGENCE
        st.subheader(f"🏟️ Venue Intelligence: {venue}")
        v = get_venue_records(venue)
        
        if v and v.get('matches', 0) > 0:
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Matches Played", v['matches'])
            k2.metric("Avg 1st Inn Score", v['avg_score_1st'])
            k3.metric("Bat 1st Win %", f"{v['bat1_win_pct']:.0f}%")
            k4.metric("Bat 2nd Win %", f"{v['bat2_win_pct']:.0f}%")
            
            st.markdown("---")
            r1, r2, r3, r4 = st.columns(4)
            r1.metric("Highest Total", v['highest_total'])
            r2.metric("Lowest Total", v['lowest_total'])
            r3.metric("Highest Chased", v['highest_chase'])
            r4.metric("Lowest Defended", v['lowest_defend'])
            
            st.markdown("---")
            p1, p2 = st.columns(2)
            p1.success(f"🏏 **Best Batting:** {v['best_bat']}")
            p2.info(f"⚾ **Best Bowling:** {v['best_bowl']}")
        else:
            st.info(f"No match data found for venue: {venue}")
            
        st.divider()

        # H2H RIVALRY
        st.subheader("⚔️ Head-to-Head")
        
        # 1. Fetch Data
        h = get_h2h_rivalry(team_a, team_b)
        # --- MISSING LINE ADDED BELOW ---
        recent_matches = get_recent_h2h_matches(team_a, team_b) 
        
        if h and h['total'] > 0:
            # --- NEW: RENDER TIMELINE ---
            st.markdown("**Recent Form:**")
            render_timeline(recent_matches, team_a, team_b)
            st.write("") # Spacer
            # -----------------------------

            st.write(f"**Total Matches:** {h['total']}")
            c1, c2 = st.columns(2)
            a_pct = (h['a_wins'] / h['total']) * 100
            b_pct = (h['b_wins'] / h['total']) * 100
            
            # Using progress bars for better visuals
            c1.metric(f"{team_a} Wins", f"{h['a_wins']} ({a_pct:.0f}%)")
            c1.progress(min(a_pct / 100, 1.0))
            
            c2.metric(f"{team_b} Wins", f"{h['b_wins']} ({b_pct:.0f}%)")
            c2.progress(min(b_pct / 100, 1.0))
            
            sorted_venues = sorted(h['venues'].items(), key=lambda x: x[1], reverse=True)
            venue_str = ", ".join([f"{v} ({c})" for v, c in sorted_venues[:5]])
            st.caption(f"**Top Battlegrounds:** {venue_str}")
        else:
            st.warning("No Head-to-Head history found.")

        st.divider()

        # PLAYER BATTLES
        t1, t2 = st.tabs(["🌟 Key Performers", "🐇 Matchups"])
        
        with t1:
            c1, c2 = st.columns(2)
            
            # Team A Stars
            ba, bo = get_star_performers(team_a, team_b)
            c1.markdown(f"**{team_a} Key Players**")
            
            if not ba.empty: c1.dataframe(ba[['Player_Name', 'Runs', 'HS', 'Avg', 'MoM']].rename(columns = {'Player_Name':'Player'}), hide_index=True)
            if not bo.empty: c1.dataframe(bo[['Player_Name', 'Wkts', 'BA', 'SR', 'MoM']].rename(columns = {'Player_Name':'Player'}), hide_index=True)
            
            # Team B Stars
            bb, bbo = get_star_performers(team_b, team_a)
            c2.markdown(f"**{team_b} Key Players**")
            
            if not bb.empty: c2.dataframe(bb[['Player_Name', 'Runs', 'HS', 'Avg', 'MoM']].rename(columns = {'Player_Name':'Player'}), hide_index=True)
            if not bbo.empty: c2.dataframe(bbo[['Player_Name', 'Wkts', 'BA', 'SR', 'MoM']].rename(columns = {'Player_Name':'Player'}), hide_index=True)
            
        with t2:
            c1, c2 = st.columns(2)
            bun_a = get_bunny_alert(team_b, team_a)
            c1.markdown(f"**{team_a} Bowlers vs {team_b}**")
            if not bun_a.empty: c1.table(bun_a)
            else: c1.caption("No notable matchups.")
            
            bun_b = get_bunny_alert(team_a, team_b)
            c2.markdown(f"**{team_b} Bowlers vs {team_a}**")
            if not bun_b.empty: c2.table(bun_b)
            else: c2.caption("No notable matchups.")

if __name__ == "__main__":
    app()