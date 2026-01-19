import streamlit as st
import pandas as pd
import sqlite3
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Circle
from matplotlib.lines import Line2D
import matplotlib.ticker as ticker
import seaborn as sns
import os

# --- CONFIGURATION ---
st.set_page_config(page_title="Team Ranking & Form", page_icon="🏆", layout="wide")
MAX_OVERS = 10 

# --- MAPPING ---
TEAM_ALIASES = {
    "IND": "India", "PAK": "Pakistan", "NZ": "New Zealand", "AUS": "Australia",
    "ENG": "England", "SA": "South Africa", "WI": "West Indies", "SL": "Sri Lanka",
    "BAN": "Bangladesh", "AFG": "Afghanistan", "NED": "Netherlands", "ZIM": "Zimbabwe",
    "IRE": "Ireland", "SCO": "Scotland", "USA": "United States", "CAN": "Canada",
    "NEP": "Nepal", "OMN": "Oman", "PNG": "Papua New Guinea", "NAM": "Namibia", 
    "UGA": "Uganda"
}

# --- 1. LOCAL DB LOADER ---
def get_db_path():
    # Check parent directory first (common in some deployments)
    parent_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../cricket_data.db")
    if os.path.exists(parent_path):
        return parent_path
    # Check current directory
    if os.path.exists("cricket_data.db"):
        return "cricket_data.db"
    return None

DB_FILE = get_db_path()

# --- 2. HELPERS ---
def normalize(name):
    return str(name).strip().lower()

def determine_winner(t1_abbr, t2_abbr, winner_str):
    raw_winner = normalize(winner_str)
    t1_full = normalize(TEAM_ALIASES.get(t1_abbr, ""))
    t2_full = normalize(TEAM_ALIASES.get(t2_abbr, ""))
    t1_norm, t2_norm = normalize(t1_abbr), normalize(t2_abbr)
    
    if raw_winner == t1_norm or raw_winner == t1_full: return t1_abbr
    if t1_full and t1_full in raw_winner: return t1_abbr
    if raw_winner == t2_norm or raw_winner == t2_full: return t2_abbr
    if t2_full and t2_full in raw_winner: return t2_abbr
    return None 

def calculate_streaks(results_list):
    if not results_list: return 0, 0
    max_win, max_loss, cur_win, cur_loss = 0, 0, 0, 0
    for res in results_list:
        if res == 'W':
            cur_win += 1; cur_loss = 0; max_win = max(max_win, cur_win)
        elif res == 'L':
            cur_loss += 1; cur_win = 0; max_loss = max(max_loss, cur_loss)
        else: cur_win = 0; cur_loss = 0
    return max_win, max_loss

# --- 3. DATA PROCESSING ---
@st.cache_data
def get_matches_data():
    if not DB_FILE: return pd.DataFrame(), pd.DataFrame()
    
    conn = sqlite3.connect(DB_FILE)
    try:
        # VALIDATION: Check table existence
        check = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table' AND name='innings_summary'", conn)
        if check.empty:
            return pd.DataFrame(), pd.DataFrame()

        df_inn = pd.read_sql_query("SELECT Match_ID, Team_Name, Total_Runs, Total_Wickets_Lost, Winner FROM innings_summary ORDER BY Match_ID", conn)
        df_bowl = pd.read_sql_query("SELECT Match_ID, Team_Name, SUM(Total_Balls_Bowled) as Balls_Delivered FROM player_stats GROUP BY Match_ID, Team_Name", conn)
        return df_inn, df_bowl
    except Exception:
        return pd.DataFrame(), pd.DataFrame()
    finally:
        conn.close()

def calculate_standings():
    df_inn, df_bowl = get_matches_data()
    if df_inn.empty: return pd.DataFrame()

    teams = {}
    matches = df_inn.groupby('Match_ID')

    for match_id, match_data in matches:
        if len(match_data) != 2: continue
        t1_row, t2_row = match_data.iloc[0], match_data.iloc[1]
        t1, t2 = t1_row['Team_Name'], t2_row['Team_Name']
        
        for t in [t1, t2]:
            if t not in teams: teams[t] = {'Results':[], 'Played':0, 'Won':0, 'Lost':0, 'Points':0, 'Runs_For':0, 'Balls_Faced':0, 'Runs_Agst':0, 'Balls_Bowled':0}
        
        teams[t1]['Played'] += 1; teams[t2]['Played'] += 1
        winner = determine_winner(t1, t2, t1_row['Winner'])
        
        if winner == t1:
            teams[t1]['Results'].append('W'); teams[t1]['Won'] += 1; teams[t1]['Points'] += 2
            teams[t2]['Results'].append('L'); teams[t2]['Lost'] += 1; teams[t2]['Points'] -= 1
        elif winner == t2:
            teams[t2]['Results'].append('W'); teams[t2]['Won'] += 1; teams[t2]['Points'] += 2
            teams[t1]['Results'].append('L'); teams[t1]['Lost'] += 1; teams[t1]['Points'] -= 1
        else:
            teams[t1]['Results'].append('T'); teams[t2]['Results'].append('T')

        def get_balls(bowling_team):
            r = df_bowl[(df_bowl['Match_ID'] == match_id) & (df_bowl['Team_Name'] == bowling_team)]
            return int(r['Balls_Delivered'].iloc[0]) if not r.empty else 0

        t1_balls = (MAX_OVERS*6) if t1_row['Total_Wickets_Lost'] == 10 else get_balls(t2)
        teams[t1]['Runs_For'] += t1_row['Total_Runs']; teams[t1]['Balls_Faced'] += t1_balls
        teams[t2]['Runs_Agst'] += t1_row['Total_Runs']; teams[t2]['Balls_Bowled'] += t1_balls
        
        t2_balls = (MAX_OVERS*6) if t2_row['Total_Wickets_Lost'] == 10 else get_balls(t1)
        teams[t2]['Runs_For'] += t2_row['Total_Runs']; teams[t2]['Balls_Faced'] += t2_balls
        teams[t1]['Runs_Agst'] += t2_row['Total_Runs']; teams[t1]['Balls_Bowled'] += t2_balls

    ranking_data = []
    for t, s in teams.items():
        off = s['Runs_For'] / (s['Balls_Faced']/6) if s['Balls_Faced']>0 else 0
        defn = s['Runs_Agst'] / (s['Balls_Bowled']/6) if s['Balls_Bowled']>0 else 0
        w_strk, l_strk = calculate_streaks(s['Results'])
        ranking_data.append({
            'Team': t, 'Mat': s['Played'], 'Won': s['Won'], 'Lost': s['Lost'], 'Pts': s['Points'],
            'NRR': off - defn, 'Results': s['Results'], 'Win_Streak': w_strk, 'Loss_Streak': l_strk
        })
        
    df_rank = pd.DataFrame(ranking_data).sort_values(by=['Pts', 'NRR'], ascending=[False, False]).reset_index(drop=True)
    df_rank['Rank'] = df_rank.index + 1
    return df_rank

# --- 4. HISTORY REPLAY ---
def get_team_history(target_team):
    df_inn, df_bowl = get_matches_data()
    if df_inn.empty: return pd.DataFrame()

    teams_stats = {}
    for t in df_inn['Team_Name'].unique():
        teams_stats[t] = {'Points': 0, 'Runs_For': 0, 'Balls_Faced': 0, 'Runs_Agst': 0, 'Balls_Bowled': 0, 'Played': 0}

    history = []
    match_counter = 0
    matches = df_inn.groupby('Match_ID')

    for match_id, match_data in matches:
        if len(match_data) != 2: continue
        match_counter += 1
        t1, t2 = match_data.iloc[0]['Team_Name'], match_data.iloc[1]['Team_Name']
        winner = determine_winner(t1, t2, match_data.iloc[0]['Winner'])
        
        teams_stats[t1]['Played'] += 1; teams_stats[t2]['Played'] += 1
        if winner == t1: teams_stats[t1]['Points'] += 2; teams_stats[t2]['Points'] -= 1
        elif winner == t2: teams_stats[t2]['Points'] += 2; teams_stats[t1]['Points'] -= 1
        
        def get_b(bowling):
            r = df_bowl[(df_bowl['Match_ID']==match_id) & (df_bowl['Team_Name']==bowling)]
            return int(r['Balls_Delivered'].iloc[0]) if not r.empty else 0

        for team, row, opp in [(t1, match_data.iloc[0], t2), (t2, match_data.iloc[1], t1)]:
            bf = (MAX_OVERS*6) if row['Total_Wickets_Lost']==10 else get_b(opp)
            teams_stats[team]['Runs_For'] += row['Total_Runs']; teams_stats[team]['Balls_Faced'] += bf
            teams_stats[opp]['Runs_Agst'] += row['Total_Runs']; teams_stats[opp]['Balls_Bowled'] += bf

        # Snapshot
        standings = []
        for t, s in teams_stats.items():
            if s['Played'] == 0: continue
            off = s['Runs_For']/(s['Balls_Faced']/6) if s['Balls_Faced']>0 else 0
            defn = s['Runs_Agst']/(s['Balls_Bowled']/6) if s['Balls_Bowled']>0 else 0
            standings.append({'Team': t, 'Pts': s['Points'], 'NRR': off - defn})
        
        if not standings: continue
        df_snap = pd.DataFrame(standings).sort_values(by=['Pts', 'NRR'], ascending=[False, False]).reset_index(drop=True)
        df_snap['Rank'] = df_snap.index + 1
        
        target_row = df_snap[df_snap['Team'] == target_team]
        if not target_row.empty:
            history.append({
                'Match_Num': match_counter, 'Rank': target_row['Rank'].values[0],
                'NRR': target_row['NRR'].values[0], 'Played': (target_team == t1 or target_team == t2)
            })
    return pd.DataFrame(history)

# --- 5. VISUALS ---
def plot_form_guide(df):
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(16, len(df)*0.6 + 1))
    ax.set_facecolor('#1e1e1e'); fig.patch.set_facecolor('#1e1e1e')
    df['Recent'] = df['Results'].apply(lambda x: x[-15:])
    max_len = df['Recent'].apply(len).max()
    ax.set_ylim(-0.5, len(df)-0.5); ax.set_xlim(0.5, max_len+1.5); ax.invert_yaxis()
    ax.set_yticks(range(len(df))); ax.set_yticklabels(df['Team'], color='white', fontsize=12, fontweight='bold')
    ax.get_xaxis().set_visible(False)
    for s in ax.spines.values(): s.set_visible(False)
    for i, row in df.iterrows():
        for j, res in enumerate(row['Recent'], start=1):
            color = '#2ecc71' if res == 'W' else '#e74c3c' if res == 'L' else '#95a5a6'
            ax.add_patch(plt.Circle((j, i), 0.35, color=color))
            ax.text(j, i, res, color='white', ha='center', va='center', fontweight='bold', fontsize=10)
    st.pyplot(fig)

def plot_history(df, team):
    if df.empty: return
    sns.set_theme(style="darkgrid", rc={"axes.facecolor": "#1c1c1c", 'figure.facecolor': '#1c1c1c', 'text.color': 'white', 'axes.labelcolor': 'white', 'xtick.color': 'white', 'ytick.color': 'white'})
    fig, ax1 = plt.subplots(figsize=(12, 6))
    for _, row in df.iterrows():
        if row['Played']: ax1.axvspan(row['Match_Num']-0.5, row['Match_Num']+0.5, color='#334444', alpha=0.5, lw=0)
    
    color_rank = '#00ffcc'
    sns.lineplot(data=df, x='Match_Num', y='Rank', color=color_rank, lw=3, marker='o', ax=ax1)
    ax1.set_ylabel("Global Rank", color=color_rank, fontweight='bold'); ax1.tick_params(axis='y', colors=color_rank)
    ax1.invert_yaxis(); ax1.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax1.set_ylim([df['Rank'].max()+1, 0.5])

    color_nrr = '#ff9933'
    ax2 = ax1.twinx()
    sns.lineplot(data=df, x='Match_Num', y='NRR', color=color_nrr, linestyle='--', lw=2, marker='d', ax=ax2, alpha=0.8)
    ax2.set_ylabel("NRR", color=color_nrr, fontweight='bold'); ax2.tick_params(axis='y', colors=color_nrr)
    ax2.axhline(0, color='white', alpha=0.3, lw=1)
    
    ax1.set_title(f"Tournament Trajectory: {team}", fontsize=16, color='white', fontweight='bold')
    st.pyplot(fig)

# --- 6. APP ---
def app():
    st.title("🏆 Tournament Headquarters")
    if not DB_FILE: 
        st.error("No Database File Found.")
        st.stop()
    
    df_rank = calculate_standings()
    if df_rank.empty: 
        st.warning("Found database but unable to calculate rankings (tables might be empty).")
        st.stop()

    tab1, tab2 = st.tabs(["📊 Points Table & Form", "📈 Team Trajectory"])
    with tab1:
        st.dataframe(df_rank[['Rank', 'Team', 'Mat', 'Won', 'Lost', 'Pts', 'NRR', 'Win_Streak', 'Loss_Streak']].style.format({'NRR': "{:+.3f}"}).highlight_max(subset=['Pts', 'NRR'], color='#1f4e3d'), use_container_width=True, hide_index=True)
        st.divider()
        st.subheader("🔥 Form Guide (Last 15 Matches)")
        plot_form_guide(df_rank)
    with tab2:
        teams = sorted(df_rank['Team'].unique())
        sel_team = st.selectbox("Select Team to Trace", teams)
        if sel_team:
            df_hist = get_team_history(sel_team)
            if not df_hist.empty:
                curr = df_hist.iloc[-1]
                best_rank = int(df_hist['Rank'].min())
                worst_rank = int(df_hist['Rank'].max())
                
                # UPDATED METRICS ROW
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Current Rank", int(curr['Rank']))
                c2.metric("Best Rank", best_rank)
                c3.metric("Worst Rank", worst_rank)
                c4.metric("Current NRR", f"{curr['NRR']:.3f}")
                
                plot_history(df_hist, sel_team)
            else: st.warning("No history.")

if __name__ == "__main__":
    app()