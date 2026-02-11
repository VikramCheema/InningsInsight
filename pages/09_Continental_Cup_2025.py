import streamlit as st
import pandas as pd
import sqlite3
import plotly.express as px
import os
import numpy as np
import base64

import streamlit.components.v1 as components

def retro_metric(label, value, prefix="", suffix="", color="#FFD700"):
    """
    Robust version using Streamlit Components to guarantee JS execution.
    """
    html_code = f"""
    <!DOCTYPE html>
    <html>
    <head>
    <style>
        body {{
            margin: 0;
            padding: 0;
            background-color: transparent;
            font-family: 'Courier New', Courier, monospace;
        }}
        .scoreboard-card {{
            background: linear-gradient(to bottom, #222, #333);
            border: 2px solid #555;
            border-radius: 8px;
            padding: 10px;
            text-align: center;
            box-shadow: 0 4px 6px rgba(0,0,0,0.5);
            color: white;
            height: 90px;
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
        }}
        .scoreboard-label {{
            font-size: 14px;
            color: #aaa;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 5px;
        }}
        .scoreboard-value {{
            font-size: 28px;
            font-weight: bold;
            color: {color};
            text-shadow: 0 0 10px {color}55;
        }}
    </style>
    </head>
    <body>
        <div class="scoreboard-card">
            <div class="scoreboard-label">{label}</div>
            <div class="scoreboard-value">
                {prefix}<span id="counter">0</span>{suffix}
            </div>
        </div>

        <script>
            const duration = 2000;
            const endValue = {value};
            const obj = document.getElementById("counter");
            
            let startTimestamp = null;
            const step = (timestamp) => {{
                if (!startTimestamp) startTimestamp = timestamp;
                const progress = Math.min((timestamp - startTimestamp) / duration, 1);
                const current = Math.floor(progress * endValue);
                obj.innerHTML = current.toLocaleString();
                
                if (progress < 1) {{
                    window.requestAnimationFrame(step);
                }} else {{
                    obj.innerHTML = endValue.toLocaleString();
                }}
            }};
            window.requestAnimationFrame(step);
        </script>
    </body>
    </html>
    """
    
    components.html(html_code, height=120, scrolling=False)


# --- CONFIGURATION ---
st.set_page_config(page_title="Continental Cup 2025 Headquarters", page_icon="🏆", layout="wide")
MAX_OVERS = 10 
TARGET_TOUR_PATTERN = "010"  # Continental Cup 2025

# --- ALL TEAMS (Single Group) ---
ALL_TEAMS = [ "West Indies", "New Zealand", "England",
              "South Africa", "Pakistan", ]

FULL_NAMES = {
    "WI": "West Indies","NZ": "New Zealand", "ENG": "England",
     "SA": "South Africa", "PAK": "Pakistan"
}
CODE_MAP = {v: k for k, v in FULL_NAMES.items()}

# --- 1. DB LOADER ---
def get_db_path():
    parent_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../cricket_data.db")
    if os.path.exists(parent_path): return parent_path
    if os.path.exists("cricket_data.db"): return "cricket_data.db"
    return None

DB_FILE = get_db_path()

# --- 2. FACTS ENGINE ---
def get_tournament_facts():
    """Fetches expanded headline stats specifically for this tournament pattern"""
    if not DB_FILE: return {}
    
    conn = sqlite3.connect(DB_FILE)
    facts = {}
    
    try:
        match_filter = f"Match_ID LIKE '{TARGET_TOUR_PATTERN}%'"
        
        facts['matches'] = pd.read_sql(f"SELECT count(DISTINCT Match_ID) FROM innings_summary WHERE {match_filter}", conn).iloc[0,0]
        facts['venues'] = pd.read_sql(f"SELECT count(DISTINCT Venue) FROM innings_summary WHERE {match_filter}", conn).iloc[0,0]
        
        facts['fifties'] = pd.read_sql(f"SELECT count(*) FROM player_stats WHERE {match_filter} AND Runs_Scored >= 50", conn).iloc[0,0]
        facts['ducks'] = pd.read_sql(f"SELECT count(*) FROM player_stats WHERE {match_filter} AND Runs_Scored = 0 AND Innings_Out = 1", conn).iloc[0,0]
        
        facts['3w_hauls'] = pd.read_sql(f"SELECT count(*) FROM player_stats WHERE {match_filter} AND Wickets_Taken = 3", conn).iloc[0,0]
        facts['4w_plus'] = pd.read_sql(f"SELECT count(*) FROM player_stats WHERE {match_filter} AND Wickets_Taken >= 4", conn).iloc[0,0]
        
        facts['sub_100'] = pd.read_sql(f"SELECT count(*) FROM innings_summary WHERE {match_filter} AND Total_Runs < 100", conn).iloc[0,0]
        facts['all_outs'] = pd.read_sql(f"SELECT count(*) FROM innings_summary WHERE {match_filter} AND Total_Wickets_Lost = 10", conn).iloc[0,0]

        try:
            facts['extras'] = pd.read_sql(f"SELECT COALESCE(SUM(Extras_Given), 0) FROM player_stats WHERE {match_filter}", conn).iloc[0,0]
        except:
            facts['extras'] = 0

        facts['bowled'] = pd.read_sql(f"SELECT count(*) FROM player_stats WHERE {match_filter} AND Dismissal_Type = 'Bowled'", conn).iloc[0,0]
        facts['lbw'] = pd.read_sql(f"SELECT count(*) FROM player_stats WHERE {match_filter} AND Dismissal_Type = 'LBW'", conn).iloc[0,0]
        facts['run_outs'] = pd.read_sql(f"SELECT count(*) FROM player_stats WHERE {match_filter} AND Dismissal_Type = 'Run Out'", conn).iloc[0,0]

    except Exception as e:
        print(f"Error fetching facts: {e}")
        return {}
    finally:
        conn.close()
        
    return facts

# --- 3. CALCULATION ENGINES ---

def determine_primary_role(row):
    """Classifies player based on highest point category"""
    bat = row['Pts_Batting']
    bowl = row['Pts_Bowling']
    ar = row['Pts_AllRounder']
    
    if bat >= bowl and bat >= ar:
        return "Batsman"
    elif bowl >= bat and bowl >= ar:
        return "Bowler"
    else:
        return "All-Rounder"

@st.cache_data
def get_mvp_data():
    if not DB_FILE: return pd.DataFrame()
    conn = sqlite3.connect(DB_FILE)
    
    query = f"""
    SELECT Player_Name, Team_Name, Runs_Scored, Innings_Out, Balls_Faced, 
           Wickets_Taken, Runs_Conceded, Overs_Balled, Is_MoM
    FROM player_stats 
    WHERE Match_ID LIKE '{TARGET_TOUR_PATTERN}%'
    """
    df_raw = pd.read_sql_query(query, conn)
    conn.close()
    
    if df_raw.empty: return pd.DataFrame()

    def calc_balls(overs):
        return sum(int(o)*6 + int(round((o%1)*10)) for o in overs)

    stats = df_raw.groupby(['Player_Name', 'Team_Name']).agg(
        Total_Runs=('Runs_Scored', 'sum'),
        Total_Outs=('Innings_Out', 'sum'),
        Total_Balls_Faced=('Balls_Faced', 'sum'),
        HS=('Runs_Scored', 'max'),
        Count_30s=('Runs_Scored', lambda x: ((x >= 30) & (x < 50)).sum()),
        Count_50s=('Runs_Scored', lambda x: ((x >= 50) & (x < 100)).sum()),
        Count_100s=('Runs_Scored', lambda x: (x >= 100).sum()),
        Count_Ducks=('Runs_Scored', lambda x: ((x==0) & (df_raw.loc[x.index, 'Innings_Out']==1)).sum()),
        
        Total_Wickets=('Wickets_Taken', 'sum'),
        Total_Runs_Conceded=('Runs_Conceded', 'sum'),
        Total_Balls_Bowled=('Overs_Balled', calc_balls),
        Count_Zero_Wkt=('Wickets_Taken', lambda x: (x == 0).sum()),
        Count_3W=('Wickets_Taken', lambda x: (x == 3).sum()),
        Count_4W=('Wickets_Taken', lambda x: (x == 4).sum()),
        Count_5W=('Wickets_Taken', lambda x: (x >= 5).sum()),
        Total_MoMs=('Is_MoM', 'sum')
    ).reset_index()

    best_figs = df_raw.sort_values(['Wickets_Taken', 'Runs_Conceded'], ascending=[False, True]).drop_duplicates('Player_Name')
    best_figs['Best_Fig'] = best_figs['Wickets_Taken'].astype(str) + '/' + best_figs['Runs_Conceded'].astype(str)
    stats = stats.merge(best_figs[['Player_Name', 'Best_Fig']], on='Player_Name', how='left')

    stats['Bat_Avg'] = np.where(stats['Total_Outs'] > 0, stats['Total_Runs'] / stats['Total_Outs'], stats['Total_Runs'])
    stats['Bat_SR'] = np.where(stats['Total_Balls_Faced'] > 0, (stats['Total_Runs'] / stats['Total_Balls_Faced']) * 100, 0.0)
    
    stats['Total_Overs_Precise'] = stats['Total_Balls_Bowled'] / 6.0
    stats['Bowl_Avg'] = np.where(stats['Total_Wickets'] > 0, stats['Total_Runs_Conceded'] / stats['Total_Wickets'], 0.0)
    stats['Bowl_SR'] = np.where(stats['Total_Wickets'] > 0, stats['Total_Balls_Bowled'] / stats['Total_Wickets'], 0.0)
    stats['Bowl_Econ'] = np.where(stats['Total_Overs_Precise'] > 0, stats['Total_Runs_Conceded'] / stats['Total_Overs_Precise'], 0.0)

    b_sr_pts = np.where(stats['Bat_SR'] > 100, (stats['Bat_SR'] - 100)/5, 0)
    
    stats['Pts_Batting'] = (
        (stats['Total_Runs'] * 0.5) + 
        (stats['Bat_Avg'] * 0.5) +
        (b_sr_pts) +
        (stats['Count_100s']*50 + stats['Count_50s']*20 + stats['Count_30s']*10) +
        (stats['Total_MoMs']*10) - 
        (stats['Count_Ducks']*10)
    )
    
    w_econ_pts = np.where(stats['Bowl_Econ'] < 12.0, (12.0 - stats['Bowl_Econ']) * 2, 0.0)
    
    stats['Pts_Bowling'] = (
        (stats['Total_Wickets'] * 15) + 
        (stats['Total_Overs_Precise'] * 1.0) +
        (w_econ_pts) +
        (stats['Count_5W']*30 + stats['Count_4W']*20 + stats['Count_3W']*10) +
        (stats['Total_MoMs']*10) - 
        (stats['Count_Zero_Wkt']*5)
    )

    stats['Pts_AllRounder'] = (
        (stats['Total_Runs'] * 1) + 
        (stats['Total_Wickets'] * 10.0) +
        (stats['Total_MoMs'] * 10.0) - 
        (stats['Count_Ducks'] * 1.1) - 
        (stats['Count_Zero_Wkt'] * 1.1)
    )
    
    stats.loc[stats['Total_Wickets'] < 1, 'Pts_AllRounder'] = 0

    stats['Role'] = stats.apply(determine_primary_role, axis=1)
    
    return stats

@st.cache_data
def get_records_data():
    conn = sqlite3.connect(DB_FILE)
    
    q_bat = f"SELECT Player_Name, Runs_Scored, Balls_Faced, Team_Name, Opposition as Vs FROM player_stats WHERE Match_ID LIKE '{TARGET_TOUR_PATTERN}%' ORDER BY Runs_Scored DESC LIMIT 10"
    df_bat = pd.read_sql_query(q_bat, conn)
    
    q_bowl = f"SELECT Player_Name, Wickets_Taken, Runs_Conceded, Overs_Balled, Team_Name, Opposition as Vs FROM player_stats WHERE Match_ID LIKE '{TARGET_TOUR_PATTERN}%' AND Overs_Balled > 0 ORDER BY Wickets_Taken DESC, Runs_Conceded ASC LIMIT 10"
    df_bowl = pd.read_sql_query(q_bowl, conn)
    
    q_matches = f"SELECT Match_ID, Team_Name, Total_Runs, Winner, Venue FROM innings_summary WHERE Match_ID LIKE '{TARGET_TOUR_PATTERN}%' ORDER BY Match_ID"
    df_matches = pd.read_sql_query(q_matches, conn)
    
    conn.close()
    return df_bat, df_bowl, df_matches

# --- 4. PROGRESSION ENGINE (MODIFIED FOR SINGLE GROUP) ---
@st.cache_data
def generate_race_data():
    """Modified to handle single group with all teams"""
    if not DB_FILE: return pd.DataFrame()
    conn = sqlite3.connect(DB_FILE)
    try:
        # Only get group stage matches (first 20)
        q_raw = f"""
        SELECT Match_ID, Team_Name, Winner, Total_Runs, Total_Wickets_Lost 
        FROM innings_summary 
        WHERE Match_ID LIKE '{TARGET_TOUR_PATTERN}%' 
        AND CAST(SUBSTR(Match_ID, INSTR(Match_ID, '2025-') + 5, 2) AS INTEGER) <= 20
        ORDER BY Match_ID
        """
        df_raw = pd.read_sql_query(q_raw, conn)
        
        q_balls = f"""
        SELECT Match_ID, Team_Name, SUM(Total_Balls_Bowled) as Balls 
        FROM player_stats 
        WHERE Match_ID LIKE '{TARGET_TOUR_PATTERN}%' 
        AND CAST(SUBSTR(Match_ID, INSTR(Match_ID, '2025-') + 5, 2) AS INTEGER) <= 20
        GROUP BY Match_ID, Team_Name
        """
        df_balls = pd.read_sql_query(q_balls, conn)
    finally:
        conn.close()

    if df_raw.empty: return pd.DataFrame()

    standings = {t: {'Pts': 0, 'Runs_For': 0, 'Balls_Faced': 0, 'Runs_Agst': 0, 'Balls_Bowled': 0, 'Played': 0, 'Won': 0, 'Lost': 0} for t in ALL_TEAMS}
    snapshots = []
    
    for t in ALL_TEAMS:
        snapshots.append({
            'Match_Label': "Start", 'Match_Order': 0, 'Team': t, 
            'Points': 0, 'NRR': 0.0, 'Rank': len(ALL_TEAMS),
            'Played': 0, 'Won': 0, 'Lost': 0
        })

    match_cnt = 0
    grouped = df_raw.groupby('Match_ID')

    for mid, mdata in grouped:
        if len(mdata) != 2: continue
        t1, t2 = mdata.iloc[0]['Team_Name'], mdata.iloc[1]['Team_Name']
        n1, n2 = FULL_NAMES.get(t1, t1), FULL_NAMES.get(t2, t2)
        
        if n1 not in ALL_TEAMS or n2 not in ALL_TEAMS: continue
        match_cnt += 1
        
        standings[n1]['Played'] += 1
        standings[n2]['Played'] += 1
        winner = str(mdata.iloc[0]['Winner']).strip()
        
        if t1 in winner or n1 in winner:
            standings[n1]['Pts'] += 2
            standings[n1]['Won'] += 1
            standings[n2]['Lost'] += 1
        elif t2 in winner or n2 in winner:
            standings[n2]['Pts'] += 2
            standings[n2]['Won'] += 1
            standings[n1]['Lost'] += 1
        else:
            standings[n1]['Pts'] += 1
            standings[n2]['Pts'] += 1

        def get_b(b_team):
            r = df_balls[(df_balls['Match_ID']==mid) & (df_balls['Team_Name']==b_team)]
            return int(r['Balls'].iloc[0]) if not r.empty else MAX_OVERS*6

        r1 = mdata.iloc[0]['Total_Runs']
        bf1 = (MAX_OVERS*6) if mdata.iloc[0]['Total_Wickets_Lost'] == 10 else get_b(t2)
        standings[n1]['Runs_For'] += r1
        standings[n1]['Balls_Faced'] += bf1
        standings[n2]['Runs_Agst'] += r1
        standings[n2]['Balls_Bowled'] += bf1
        
        r2 = mdata.iloc[1]['Total_Runs']
        bf2 = (MAX_OVERS*6) if mdata.iloc[1]['Total_Wickets_Lost'] == 10 else get_b(t1)
        standings[n2]['Runs_For'] += r2
        standings[n2]['Balls_Faced'] += bf2
        standings[n1]['Runs_Agst'] += r2
        standings[n1]['Balls_Bowled'] += bf2

        curr = []
        for t in ALL_TEAMS:
            s = standings[t]
            nrr = (s['Runs_For']/(s['Balls_Faced']/6)) - (s['Runs_Agst']/(s['Balls_Bowled']/6)) if s['Played']>0 else 0
            curr.append({
                'Team': t, 'Points': s['Pts'], 'NRR': nrr,
                'Played': s['Played'], 'Won': s['Won'], 'Lost': s['Lost']
            })
        
        df_curr = pd.DataFrame(curr).sort_values(by=['Points', 'NRR'], ascending=[False, False]).reset_index(drop=True)
        df_curr['Rank'] = df_curr.index + 1
        
        label = f"Match {match_cnt}"
        for _, row in df_curr.iterrows():
            snapshots.append({
                'Match_Label': label, 'Match_Order': match_cnt, 
                'Team': row['Team'], 'Points': row['Points'], 'NRR': row['NRR'], 
                'Rank': row['Rank'], 'Played': row['Played'], 
                'Won': row['Won'], 'Lost': row['Lost']
            })

    return pd.DataFrame(snapshots)


@st.cache_data
def get_final_data():
    """Fetches the final match data (Match 21)"""
    if not DB_FILE: return {}
    conn = sqlite3.connect(DB_FILE)
    package = {}
    
    try:
        summary_query = f"""
        SELECT Match_ID, Team_Name, Total_Runs, Total_Wickets_Lost, Winner, Venue 
        FROM innings_summary 
        WHERE Match_ID LIKE '{TARGET_TOUR_PATTERN}%' 
        AND CAST(SUBSTR(Match_ID, INSTR(Match_ID, '2025-') + 5, 2) AS INTEGER) = 21
        ORDER BY Match_ID ASC
        """
        df_summary = pd.read_sql_query(summary_query, conn)
        
        if df_summary.empty:
            return {}

        match_ids = df_summary['Match_ID'].unique().tolist()
        id_filter = "('" + "','".join(match_ids) + "')"
        
        player_query = f"""
        SELECT Match_ID, Team_Name, Player_Name, Runs_Scored, Balls_Faced, 
               Wickets_Taken, Runs_Conceded, Is_MoM
        FROM player_stats 
        WHERE Match_ID IN {id_filter}
        """
        df_players = pd.read_sql_query(player_query, conn)
        
        package['summary'] = df_summary
        package['players'] = df_players
        
        return package
    except Exception as e:
        st.error(f"Error fetching final data: {e}")
        return {}
    finally:
        conn.close()

# --- PATH & NORMALIZATION ---
TEAM_ALIASES = {
    "PAK": "Pakistan", "NZ": "New Zealand",
    "ENG": "England", "SA": "South Africa", "WI": "West Indies"
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
root_dir = os.path.dirname(script_dir)
ASSETS_DIR = os.path.join(root_dir, "assets")

def get_base64_image(team_name):
    """Finds local image, converts to Base64"""
    clean_name = str(team_name).strip()
    full_name = TEAM_ALIASES.get(clean_name.upper(), clean_name)
    
    file_name = f"{full_name}.png"
    file_path = os.path.join(ASSETS_DIR, file_name)

    if not os.path.exists(file_path):
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
    
def get_flag_url(team_name):
    return get_base64_image(team_name)

# --- DATABASE HELPERS ---
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


# --- 5. MAIN APP ---
def app():
    st.title("🏆 Continental Cup 2025: Headquarters")
    st.markdown("""
    <style>
    [data-testid="stImage"] img {
        height: 80px !important;
        width: 120px !important;
        object-fit: contain !important;
        background-color: transparent;
    }
    </style>
    """, unsafe_allow_html=True)
    
    if not DB_FILE: 
        st.error("Database missing.")
        st.stop()

    # --- TOURNAMENT FACTS ---
    facts = get_tournament_facts()
    
    if facts:
        gold = "#FFD700"
        blue = "#00BFFF"
        red  = "#FF4500"
        
        c1, c2, c3, c4 = st.columns(4)
        with c1: retro_metric("Matches Played", facts.get('matches', 0), color=blue)
        with c2: retro_metric("Venues Used", facts.get('venues', 0), color=blue)
        with c3: retro_metric("Total Extras", int(facts.get('extras', 0)), color=blue)
        with c4: retro_metric("Teams All Out", facts.get('all_outs', 0), color=red)
        
        c5, c6, c7, c8 = st.columns(4)
        with c5: retro_metric("50+ Scores", facts.get('fifties', 0), color=gold)
        with c6: retro_metric("Ducks 🦆", facts.get('ducks', 0), color=gold)
        with c7: retro_metric("3-Wicket Hauls", facts.get('3w_hauls', 0), color=gold)
        with c8: retro_metric("4+ Wicket Hauls", facts.get('4w_plus', 0), color=gold)

        c9, c10, c11, c12 = st.columns(4)
        with c9: retro_metric("Scores < 100", facts.get('sub_100', 0), color=red)
        with c10: retro_metric("Bowled 🎯", facts.get('bowled', 0), color=red)
        with c11: retro_metric("LBW 🦵", facts.get('lbw', 0), color=red)
        with c12: retro_metric("Run Outs 🏃", facts.get('run_outs', 0), color=red)
        
    else:
        st.warning("Could not load tournament facts.")
        
    st.divider()

    # --- TABS ---
    tabs = st.tabs(["🏆 The Final", "🎢 Points Table Progression", "🌟 MVP Leaderboard", "🔥 Tournament Records", "🏟️ Venue Stats"])

    # --- TAB 0: THE FINAL ---
    with tabs[0]:
        st.header("🏁 The Grand Final")
        
        data_package = get_final_data()
        
        if data_package and 'summary' in data_package:
            df_summary = data_package['summary']
            df_players = data_package['players']
            
            if not df_summary.empty and len(df_summary) >= 2:
                t1_row = df_summary.iloc[0]
                t2_row = df_summary.iloc[1]
                
                # Get team names
                final_t1 = t1_row['Team_Name']
                final_t2 = t2_row['Team_Name']
                venue_f = t1_row['Venue']
                
                # --- FINAL PREVIEW ---
                st.markdown("### 🏆 Match Preview")
                
                with st.container(border=True):
                    col_t1, col_vs_f, col_t2 = st.columns([2, 1, 2])
                    h2h_stats = get_h2h_rivalry(final_t1, final_t2)
                    
                    with col_t1:
                        st.image(get_flag_url(final_t1), width=100)
                        st.markdown(f"<h3 style='text-align: center;'>{h2h_stats['a_wins'] if h2h_stats else 0} Wins</h3>", unsafe_allow_html=True)
                    with col_vs_f:
                        st.markdown("<h1 style='text-align: center;'>H2H</h1>", unsafe_allow_html=True)
                        st.caption(f"<p style='text-align: center;'>{h2h_stats['total'] if h2h_stats else 0} Total Played</p>", unsafe_allow_html=True)
                    with col_t2:
                        st.image(get_flag_url(final_t2), width=100)
                        st.markdown(f"<h3 style='text-align: center;'>{h2h_stats['b_wins'] if h2h_stats else 0} Wins</h3>", unsafe_allow_html=True)

                    st.divider()

                    # Venue Stats
                    st.markdown(f"#### 🏟️ Venue: {venue_f}")
                    v_stats = get_venue_records(venue_f)
                    
                    vc1, vc2, vc3, vc4 = st.columns(4)
                    vc1.metric("Highest Total", v_stats.get('highest_total', 'N/A'))
                    vc2.metric("Bat 1st Win %", f"{v_stats.get('bat1_win_pct', 0):.1f}%")
                    vc3.metric("Highest Chase", v_stats.get('highest_chase', 'N/A'))
                    vc4.metric("Lowest Defended", v_stats.get('lowest_defend', 'N/A'))

                    st.divider()

                    # Star Performers
                    st.markdown("#### 🔥 Key Players in This Rivalry")
                    m_bat, m_bowl = st.columns(2)
                    
                    bat1, bowl1 = get_star_performers(final_t1, final_t2)
                    bat2, bowl2 = get_star_performers(final_t2, final_t1)

                    with m_bat:
                        st.write("**🏏 Best Batsmen**")
                        if not bat1.empty: 
                            st.success(f"**{final_t1}**: {bat1.iloc[0]['Player_Name']} ({int(bat1.iloc[0]['Runs'])} Runs)")
                        if not bat2.empty: 
                            st.success(f"**{final_t2}**: {bat2.iloc[0]['Player_Name']} ({int(bat2.iloc[0]['Runs'])} Runs)")

                    with m_bowl:
                        st.write("**⚾ Best Bowlers**")
                        if not bowl1.empty: 
                            st.warning(f"**{final_t1}**: {bowl1.iloc[0]['Player_Name']} ({int(bowl1.iloc[0]['Wkts'])} Wkts)")
                        if not bowl2.empty: 
                            st.warning(f"**{final_t2}**: {bowl2.iloc[0]['Player_Name']} ({int(bowl2.iloc[0]['Wkts'])} Wkts)")

                    bunny = get_bunny_alert(final_t1, final_t2)
                    if not bunny.empty:
                        st.error(f"⚠️ **Bunny Alert:** {bunny.iloc[0]['Bowler']} has dismissed {bunny.iloc[0]['Batter']} {bunny.iloc[0]['Count']} times!")

                st.divider()

                # --- MATCH RESULT ---
                st.markdown("### 🏟️ Match Result")
                
                with st.container(border=True):
                    col1, col_vs, col2 = st.columns([2, 1, 2])
                    
                    with col1:
                        st.image(get_flag_url(final_t1), width=80)
                        st.metric(final_t1, f"{int(t1_row['Total_Runs'])}/{int(t1_row['Total_Wickets_Lost'])}")
                    with col_vs:
                        st.markdown("<h1 style='text-align: center; padding-top: 15px;'>VS</h1>", unsafe_allow_html=True)
                    with col2:
                        st.image(get_flag_url(final_t2), width=80)
                        st.metric(final_t2, f"{int(t2_row['Total_Runs'])}/{int(t2_row['Total_Wickets_Lost'])}")

                    st.info(f"🏆 Winner: {t1_row['Winner']} | Venue: {venue_f}")

                    with st.expander("📊 Player Highlights"):
                        c_bat, c_bowl = st.columns(2)
                        
                        with c_bat:
                            st.write("**🏏 Top Scorers**")
                            for team in [final_t1, final_t2]:
                                p_bat = df_players[df_players['Team_Name'] == team].sort_values('Runs_Scored', ascending=False).head(1)
                                if not p_bat.empty:
                                    p = p_bat.iloc[0]
                                    st.markdown(f"**{p['Player_Name']}** ({team})\n{int(p['Runs_Scored'])} ({int(p['Balls_Faced'])})")

                        with c_bowl:
                            st.write("**⚾ Top Bowlers**")
                            for team in [final_t1, final_t2]:
                                p_bowl = df_players[df_players['Team_Name'] == team].sort_values(['Wickets_Taken', 'Runs_Conceded'], ascending=[False, True]).head(1)
                                if not p_bowl.empty:
                                    p = p_bowl.iloc[0]
                                    st.markdown(f"**{p['Player_Name']}** ({team})\n{int(p['Wickets_Taken'])}/{int(p['Runs_Conceded'])}")
                        
                        mom = df_players[df_players['Is_MoM'] == 1]
                        if not mom.empty:
                            st.divider()
                            st.success(f"🌟 **Man of the Match:** {mom.iloc[0]['Player_Name']} ({mom.iloc[0]['Team_Name']})")
            else:
                st.info("The Final match hasn't been played yet.")
        else:
            st.info("The Final match data is not available yet.")

    # TAB 1: PROGRESSION
    with tabs[1]:
        st.header("📊 Points Table Evolution (Group Stage)")
        df_race = generate_race_data()
        
        if not df_race.empty:
            max_m = df_race['Match_Order'].max()
            final = df_race[df_race['Match_Order'] == max_m].sort_values('Rank').reset_index(drop=True)
            
            st.markdown("#### 📊 Final Group Standings")
            
            def highlight_top_two(row):
                return ['background-color: #87efd8' if row.name < 2 else '' for _ in row]
            
            cols_show = ['Rank', 'Team', 'Played', 'Won', 'Lost', 'Points', 'NRR']
            
            st.dataframe(
                final[cols_show].style
                .format({'NRR': "{:+.3f}"})
                .apply(highlight_top_two, axis=1), 
                use_container_width=True, 
                hide_index=True
            )
            
            st.divider()
            
            st.subheader("Race Chart: Points Progression")
            df_race['Inv_Rank'] = len(ALL_TEAMS) + 1 - df_race['Rank']
            fig = px.scatter(df_race, x="Points", y="Inv_Rank", 
                             animation_frame="Match_Label", 
                             animation_group="Team", 
                             size="Points", color="Team", text="Team", 
                             range_x=[-1, df_race['Points'].max()+2], 
                             range_y=[0.5, len(ALL_TEAMS)+0.5], height=550)
            
            fig.update_traces(textposition='middle right', marker=dict(size=30, line=dict(width=2)))
            fig.update_layout(yaxis=dict(tickvals=list(range(1, len(ALL_TEAMS)+1)), 
                             ticktext=[f'{i}th' if i > 3 else ['1st','2nd','3rd'][i-1] for i in range(len(ALL_TEAMS), 0, -1)], 
                             title="Position"), 
                             xaxis_title="Points", showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
            
            st.subheader("NRR Trajectory")
            fig_nrr = px.line(df_race[df_race['Match_Order'] > 0], x="Match_Order", y="NRR", 
                              color="Team", markers=True, height=400,
                              labels={"Match_Order": "Match Number", "NRR": "NRR"})
            st.plotly_chart(fig_nrr, use_container_width=True)
        else:
            st.warning("No progression data available.")

    # TAB 2: MVP
    with tabs[2]:
        df_mvp = get_mvp_data()
        
        if not df_mvp.empty:
            st.markdown("### 🏏 Top Batsmen")
            st.caption("Ranking based on Impact Points")
            
            best_bat = df_mvp[df_mvp['Role'] == 'Batsman'].sort_values('Pts_Batting', ascending=False).head(10)
            
            cols_bat = {
                'Player_Name': 'Player', 'Team_Name': 'Team', 
                'Total_Runs': 'Runs', 'Bat_Avg': 'Avg', 'HS': 'High Score',
                'Bat_SR': 'SR', 'Count_50s': '50s', 'Pts_Batting': 'Pts'
            }
            show_bat = best_bat[cols_bat.keys()].rename(columns=cols_bat)
            
            st.dataframe(
                show_bat.style.format({'Avg': "{:.2f}", 'SR': "{:.1f}", 'Pts': "{:.0f}"})
                .background_gradient(subset=['Pts'], cmap='Greens'),
                use_container_width=True, hide_index=True
            )
            
            st.divider()

            st.markdown("### ⚾ Top Bowlers")
            st.caption("Ranking based on Impact Points")
            
            best_bowl = df_mvp[df_mvp['Role'] == 'Bowler'].sort_values('Pts_Bowling', ascending=False).head(10)
            
            cols_bowl = {
                'Player_Name': 'Player', 'Team_Name': 'Team', 
                'Total_Wickets': 'Wickets', 'Best_Fig': 'Best', 
                'Bowl_Econ': 'Econ', 'Bowl_Avg': 'Avg', 'Bowl_SR': 'SR', 
                'Count_3W': '3W+', 'Pts_Bowling': 'Pts'
            }
            show_bowl = best_bowl[cols_bowl.keys()].rename(columns=cols_bowl)
            
            st.dataframe(
                show_bowl.style.format({'Econ': "{:.2f}", 'Avg': "{:.2f}", 'SR': "{:.1f}", 'Pts': "{:.0f}"})
                .background_gradient(subset=['Pts'], cmap='Blues'),
                use_container_width=True, hide_index=True
            )
            
            st.divider()

            st.markdown("### ⭐ Top All-Rounders")
            st.caption("Ranking based on Combined Impact")
            
            best_ar = df_mvp[df_mvp['Role'] == 'All-Rounder'].sort_values('Pts_AllRounder', ascending=False).head(10)
            
            cols_ar = {
                'Player_Name': 'Player', 'Team_Name': 'Team', 
                'Total_Runs': 'Runs', 'Total_Wickets': 'Wkts',
                'Bat_Avg': 'Bat Avg', 'Bowl_Avg': 'Bowl Avg',
                'Count_50s': '50s', 'Count_3W': '3W+', 'Pts_AllRounder': 'Pts'
            }
            show_ar = best_ar[cols_ar.keys()].rename(columns=cols_ar)
            
            st.dataframe(
                show_ar.style.format({'Bat Avg': "{:.1f}", 'Bowl Avg': "{:.1f}", 'Pts': "{:.0f}"})
                .background_gradient(subset=['Pts'], cmap='Oranges'),
                use_container_width=True, hide_index=True
            )
        else:
            st.info("No stats available yet.")

    # TAB 3: RECORDS
    with tabs[3]:
        df_bat, df_bowl, df_matches = get_records_data()
        c1, c2 = st.columns(2)
        c1.markdown("##### 🏏 Highest Scores")
        c1.dataframe(df_bat[['Player_Name', 'Runs_Scored', 'Balls_Faced', 'Vs']].head(10), hide_index=True)
        
        c2.markdown("##### ⚾ Best Bowling")
        df_bowl['Figures'] = df_bowl['Wickets_Taken'].astype(str) + "/" + df_bowl['Runs_Conceded'].astype(str)
        c2.dataframe(df_bowl[['Player_Name', 'Figures', 'Overs_Balled', 'Vs']].head(10), hide_index=True)
        
        st.divider()
        matches = df_matches.groupby('Match_ID')
        chases, defends = [], []
        
        for mid, mdata in matches:
            if len(mdata) != 2: continue
            inn1, inn2 = mdata.iloc[0], mdata.iloc[1]
            winner = str(inn1['Winner']).strip()
            if not winner or winner == 'nan': continue
            
            t1_name = inn1['Team_Name']
            t1_full = FULL_NAMES.get(t1_name, "")
            
            if t1_name in winner or (t1_full and t1_full in winner):
                run_margin = int(inn1['Total_Runs']) - int(inn2['Total_Runs'])
                defends.append({'Match': f"{t1_name} vs {inn2['Team_Name']}",'Winner': t1_name,'Score': int(inn1['Total_Runs']),'Margin': f"Won by {run_margin} runs"})
            else:
                chases.append({'Match': f"{t1_name} vs {inn2['Team_Name']}", 'Winner': inn2['Team_Name'],'Target': int(inn1['Total_Runs']), 'Chased': int(inn2['Total_Runs'])})
        
        c3, c4 = st.columns(2)
        if chases:
            df_chase = pd.DataFrame(chases).sort_values('Chased', ascending=False).head(5)
            c3.markdown("##### 🚀 Highest Successful Chases")
            c3.table(df_chase)
        if defends:
            df_defend = pd.DataFrame(defends).sort_values('Score', ascending=True).head(5)
            c4.markdown("##### 🛡️ Lowest Totals Defended")
            c4.table(df_defend)

    # TAB 4: VENUES
    with tabs[4]:
        if not df_matches.empty:
            venue_stats = []
            for venue, data in df_matches.groupby('Venue'):
                matches = data['Match_ID'].nunique()
                bat_1_wins, bat_2_wins = 0, 0
                for _, m in data.groupby('Match_ID'):
                    if len(m) != 2: continue
                    winner = str(m.iloc[0]['Winner']).strip()
                    t1 = m.iloc[0]['Team_Name']
                    t2 = m.iloc[1]['Team_Name']
                    t1_full = FULL_NAMES.get(t1, "")
                    t2_full = FULL_NAMES.get(t2, "")
                    if t1 in winner or (t1_full and t1_full in winner): bat_1_wins += 1
                    elif t2 in winner or (t2_full and t2_full in winner): bat_2_wins += 1
                
                venue_stats.append({'Venue': venue, 'Matches': matches, 'Bat 1st Won': bat_1_wins, 'Bat 2nd Won': bat_2_wins, 'Win % Bat 1st': f"{(bat_1_wins/matches)*100:.0f}%" if matches>0 else "0%"})
            
            df_ven = pd.DataFrame(venue_stats).sort_values('Matches', ascending=False)
            st.markdown("#### 🏟️ Venue Bias Analysis")
            st.dataframe(df_ven, hide_index=True, use_container_width=True)
            fig_v = px.bar(df_ven, x='Venue', y=['Bat 1st Won', 'Bat 2nd Won'], title="Wins by Innings", barmode='group')
            st.plotly_chart(fig_v, use_container_width=True)

if __name__ == "__main__":
    app()
