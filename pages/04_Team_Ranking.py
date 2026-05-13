import streamlit as st
import pandas as pd
import sqlite3
import plotly.express as px
import plotly.graph_objects as go
import os
import numpy as np

import base64
import os


from cricket_utils import highlight_podium, determine_primary_role
# --- CONFIGURATION ---
st.set_page_config(page_title="Team Ranking & Form", page_icon="🏆", layout="wide")
MAX_OVERS = 10 


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

# --- MAPPING ---
TEAM_ALIASES = {
    "IND": "India", "PAK": "Pakistan", "NZ": "New Zealand", "AUS": "Australia",
    "ENG": "England", "SA": "South Africa", "WI": "West Indies", "SL": "Sri Lanka",
    "BAN": "Bangladesh", "AFG": "Afghanistan", "NED": "Netherlands", "ZIM": "Zimbabwe",
    "IRE": "Ireland", "SCO": "Scotland", "USA": "United States", "CAN": "Canada",
    "NEP": "Nepal", "OMN": "Oman", "PNG": "Papua New Guinea", "NAM": "Namibia", 
    "UGA": "Uganda"
}

    
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
    for entry in results_list:
        res = entry['res']
        if res == 'W':
            cur_win += 1; cur_loss = 0; max_win = max(max_win, cur_win)
        elif res == 'L':
            cur_loss += 1; cur_win = 0; max_loss = max(max_loss, cur_loss)
        else: cur_win = 0; cur_loss = 0
    return max_win, max_loss

def get_db_path():
    parent_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../cricket_data.db")
    if os.path.exists(parent_path): return parent_path
    if os.path.exists("cricket_data.db"): return "cricket_data.db"
    return None

DB_FILE = get_db_path()

@st.cache_data
def get_matches_data():
    if not DB_FILE: return pd.DataFrame(), pd.DataFrame()
    conn = sqlite3.connect(DB_FILE)
    try:
        df_inn = pd.read_sql_query("""
            SELECT Match_ID, Tournament_ID, Team_Name, Innings_No, Total_Runs, 
                   Total_Wickets_Lost, Winner, Win_Type 
            FROM innings_summary ORDER BY Match_ID
        """, conn)
        df_bowl = pd.read_sql_query("SELECT Match_ID, Team_Name, SUM(Total_Balls_Bowled) as Balls_Delivered FROM player_stats GROUP BY Match_ID, Team_Name", conn)
        return df_inn, df_bowl
    finally: conn.close()

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
            teams[t1]['Results'].append({'res': 'W', 'opp': t2}); teams[t1]['Won'] += 1; teams[t1]['Points'] += 2
            teams[t2]['Results'].append({'res': 'L', 'opp': t1}); teams[t2]['Lost'] += 1; teams[t2]['Points'] -= 1
        elif winner == t2:
            teams[t2]['Results'].append({'res': 'W', 'opp': t1}); teams[t2]['Won'] += 1; teams[t2]['Points'] += 2
            teams[t1]['Results'].append({'res': 'L', 'opp': t2}); teams[t1]['Lost'] += 1; teams[t1]['Points'] -= 1
        else:
            teams[t1]['Results'].append({'res': 'T', 'opp': t2}); teams[t2]['Results'].append({'res': 'T', 'opp': t1})
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
        ranking_data.append({'Team': t, 'Mat': s['Played'], 'Won': s['Won'], 'Lost': s['Lost'], 'Pts': s['Points'], 'NRR': off - defn, 'Results': s['Results'], 'Win_Streak': w_strk, 'Loss_Streak': l_strk})
    df_rank = pd.DataFrame(ranking_data).sort_values(by=['Pts', 'NRR'], ascending=[False, False]).reset_index(drop=True)
    df_rank['Rank'] = df_rank.index + 1
    return df_rank

def get_team_history(target_team):
    df_inn, df_bowl = get_matches_data()
    if df_inn.empty: return pd.DataFrame()
    teams_stats = {}
    for t in df_inn['Team_Name'].unique(): teams_stats[t] = {'Points': 0, 'Runs_For': 0, 'Balls_Faced': 0, 'Runs_Agst': 0, 'Balls_Bowled': 0, 'Played': 0}
    history = []
    match_counter = 0
    matches = df_inn.groupby('Match_ID')
    for match_id, match_data in matches:
        if len(match_data) != 2: continue
        match_counter += 1
        t1_row = match_data.iloc[0]
        t1, t2 = t1_row['Team_Name'], match_data.iloc[1]['Team_Name']
        winner = determine_winner(t1, t2, t1_row['Winner'])
        is_playing = (target_team == t1 or target_team == t2)
        match_opp, match_res = "N/A", "DNP"
        tourn_id = t1_row.get('Tournament_ID', 'N/A')
        if is_playing:
            match_opp = t2 if target_team == t1 else t1
            match_res = "W" if winner == target_team else "L" if winner else "T/NR"
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
        standings = []
        for t, s in teams_stats.items():
            if s['Played'] == 0: continue
            off = s['Runs_For']/(s['Balls_Faced']/6) if s['Balls_Faced']>0 else 0
            defn = s['Runs_Agst']/(s['Balls_Bowled']/6) if s['Balls_Bowled']>0 else 0
            standings.append({'Team': t, 'Pts': s['Points'], 'NRR': off - defn})
        df_snap = pd.DataFrame(standings).sort_values(by=['Pts', 'NRR'], ascending=[False, False]).reset_index(drop=True)
        df_snap['Rank'] = df_snap.index + 1
        target_row = df_snap[df_snap['Team'] == target_team]
        if not target_row.empty:
            history.append({'Match_Num': match_counter, 'Rank': target_row['Rank'].values[0], 'NRR': target_row['NRR'].values[0], 'Played': is_playing, 'Opponent': match_opp, 'Result': match_res, 'Tournament': tourn_id})
    return pd.DataFrame(history)

def calculate_h2h_metrics(target_team):
    df_inn, _ = get_matches_data()
    if df_inn.empty: return pd.DataFrame()
    target_match_ids = df_inn[df_inn['Team_Name'] == target_team]['Match_ID'].unique()
    df_relevant = df_inn[df_inn['Match_ID'].isin(target_match_ids)]
    h2h_records = []
    for m_id, group in df_relevant.groupby('Match_ID'):
        if len(group) < 2: continue
        t_row = group[group['Team_Name'] == target_team].iloc[0]
        o_row = group[group['Team_Name'] != target_team].iloc[0]
        opp_abbr = o_row['Team_Name']
        match_winner = determine_winner(target_team, opp_abbr, t_row['Winner'])
        h2h_records.append({'Opponent': opp_abbr, 'Winner': match_winner, 'Innings_No': t_row['Innings_No']})
    if not h2h_records: return pd.DataFrame()
    df_results = pd.DataFrame(h2h_records)
    summary = []
    for opp, group in df_results.groupby('Opponent'):
        total = len(group)
        wins = len(group[group['Winner'] == target_team])
        losses = len(group[group['Winner'] == opp])
        defend_group = group[group['Innings_No'] == 1]
        chase_group = group[group['Innings_No'] == 2]
        def_win_pct = (len(defend_group[defend_group['Winner'] == target_team]) / len(defend_group) * 100) if len(defend_group) > 0 else 0
        cha_win_pct = (len(chase_group[chase_group['Winner'] == target_team]) / len(chase_group) * 100) if len(chase_group) > 0 else 0
        summary.append({'Opponent': opp, 'Played': total, 'Win %': (wins / total) * 100, 'Loss %': (losses / total) * 100, 'Chase Win %': cha_win_pct, 'Defend Win %': def_win_pct})
    return pd.DataFrame(summary)

@st.cache_data
def get_rankings_data():
    if not DB_FILE: return pd.DataFrame()
    conn = sqlite3.connect(DB_FILE)
    try:
        query = """
        SELECT 
            Player_Name, Team_Name, COUNT(DISTINCT Match_ID) as Mat,
            SUM(Runs_Scored) as Total_Runs, SUM(Balls_Faced) as Total_Balls_Faced,
            SUM(Innings_Out) as Total_Outs, SUM(Wickets_Taken) as Total_Wickets,
            SUM(Runs_Conceded) as Total_Runs_Conceded, SUM(CASE WHEN Is_MoM = 1 THEN 1 ELSE 0 END) as Total_MoMs,
            SUM(CASE WHEN Runs_Scored = 0 AND Innings_Out = 1 THEN 1 ELSE 0 END) as Count_Ducks,
            SUM(CASE WHEN Runs_Scored >= 30 AND Runs_Scored < 50 THEN 1 ELSE 0 END) as Count_30s,
            SUM(CASE WHEN Runs_Scored >= 50 AND Runs_Scored < 100 THEN 1 ELSE 0 END) as Count_50s,
            SUM(CASE WHEN Runs_Scored >= 100 THEN 1 ELSE 0 END) as Count_100s,
            SUM(CASE WHEN Wickets_Taken = 0 THEN 1 ELSE 0 END) as Matches_Zero_Wickets,
            SUM(CASE WHEN Wickets_Taken = 3 THEN 1 ELSE 0 END) as Count_3W,
            SUM(CASE WHEN Wickets_Taken = 4 THEN 1 ELSE 0 END) as Count_4W,
            SUM(CASE WHEN Wickets_Taken >= 5 THEN 1 ELSE 0 END) as Count_5W,
            SUM(CAST(Overs_Balled AS INT) * 6 + CAST(ROUND((Overs_Balled - CAST(Overs_Balled AS INT)) * 10) AS INT)) as Total_Balls_Bowled
        FROM player_stats
        GROUP BY Player_Name, Team_Name
        """
        df = pd.read_sql_query(query, conn)
        if df.empty: return pd.DataFrame()
        df['Bat_Avg'] = np.where(df['Total_Outs'] > 0, df['Total_Runs'] / df['Total_Outs'], df['Total_Runs'])
        df['Bat_SR'] = np.where(df['Total_Balls_Faced'] > 0, (df['Total_Runs'] / df['Total_Balls_Faced']) * 100, 0.0)
        df['Total_Overs'] = df['Total_Balls_Bowled'] / 6.0
        df['Bowl_Econ'] = np.where(df['Total_Overs'] > 0, df['Total_Runs_Conceded'] / df['Total_Overs'], 0.0)
        df['Bowl_Avg'] = np.where(df['Total_Wickets'] > 0, df['Total_Runs_Conceded'] / df['Total_Wickets'], 0.0)

        sr_bonus = np.where(df['Bat_SR'] > 100, (df['Bat_SR'] - 100) / 5.0, 0.0)
        milestone_pts = (df['Count_100s'] * 50) + (df['Count_50s'] * 20) + (df['Count_30s'] * 10)
        duck_penalty = df['Count_Ducks'] * 10
        df['Pts_Batting'] = ((df['Total_Runs'] * 0.5) + (df['Bat_Avg'] * 0.5) + sr_bonus + milestone_pts + (df['Total_MoMs'] * 10) - duck_penalty)

        econ_bonus = np.where(df['Bowl_Econ'] < 12.0, (12.0 - df['Bowl_Econ']) * 2, 0.0)
        wicket_haul_pts = (df['Count_3W'] * 10) + (df['Count_4W'] * 20) + (df['Count_5W'] * 30)
        zero_w_penalty = df['Matches_Zero_Wickets'] * 5
        df['Pts_Bowling'] = ((df['Total_Wickets'] * 15) + (df['Total_Overs'] * 1.0) + econ_bonus + wicket_haul_pts + (df['Total_MoMs'] * 10) - zero_w_penalty)

        ar_duck_penalty = df['Count_Ducks'] * 1.1
        ar_zero_w_penalty = df['Matches_Zero_Wickets'] * 1.1
        df['Pts_AllRounder'] = ((df['Total_Runs'] * 1.0) + (df['Total_Wickets'] * 10) + (df['Total_MoMs'] * 10) - ar_duck_penalty - ar_zero_w_penalty)
        mask_not_ar = (df['Total_Wickets'] <= 5) | (df['Total_Runs'] < 50)
        df.loc[mask_not_ar, 'Pts_AllRounder'] = -1.0
        return df
    finally: conn.close()

# def determine_primary_role(row):
#     bat, bowl, ar = row['Pts_Batting'], row['Pts_Bowling'], row['Pts_AllRounder']
#     if bat >= bowl and bat >= ar: return "Batsman"
#     elif bowl >= bat and bowl >= ar: return "Bowler"
#     return "All-Rounder"

TEAM_LEADERBOARD_CONFIG = {
    "Player_Icon": st.column_config.ImageColumn("", width="small"),
    "Team_Rank": st.column_config.NumberColumn("Team Rank", width="small", format="%d"),
    "Global_Rank": st.column_config.NumberColumn("Global Rank", width="small", format="%d"),
    "Player_Link": st.column_config.LinkColumn("Player Name", display_text=r"player=(.*)$", width="large"),
    "Mat": st.column_config.NumberColumn("Matches", width="small"),
    "Total_Runs": st.column_config.NumberColumn("Runs", width="small"),
    "Total_Wickets": st.column_config.NumberColumn("Wkts", width="small"),
    "Bat_Avg": st.column_config.NumberColumn("Avg", width="small", format="%.1f"),
    "Bat_SR": st.column_config.NumberColumn("Bat SR", width="small", format="%.1f"),
    "Bowl_Avg": st.column_config.NumberColumn("Bowl Avg", width="small", format="%.1f", help="Bowling Average (Runs/Wicket)"),
    "Bowl_Econ": st.column_config.NumberColumn("Econ", width="small", format="%.1f"),
    "Total_MoMs": st.column_config.NumberColumn("MoM", width="small", help="Man of the Match Awards"),
    "Count_30s": st.column_config.NumberColumn("30s", width="small"),
    "Count_50s": st.column_config.NumberColumn("50s", width="small"),
    "Count_3W": st.column_config.NumberColumn("3W", width="small"),
    "Count_4W": st.column_config.NumberColumn("4W", width="small"),
    "Count_5W": st.column_config.NumberColumn("5W", width="small"),
    "Points_Visual": st.column_config.ProgressColumn(
        "Rel. Performance", 
        help="Performance relative to the Rank 1 player (100%)",
        format="%.1f%%", # Displays the percentage on the bar
        min_value=0,
        max_value=100,
    ),
    "Points_Value": st.column_config.NumberColumn("Points", width="small", format="%.1f"),
}


def display_hero_card(player_data, rank):
    colors = {1: "#FFD700", 2: "#C0C0C0", 3: "#CD7F32"}
    border_color = colors.get(rank, "#444")
    role = player_data.get('Assigned_Role', 'Batsman')
    global_rank = int(player_data.get('Global_Rank', 0))
    
    with st.container(border=True):
        # st.markdown(f"<h1 style='text-align: center; color: {border_color}; margin-bottom: 0;'>#{rank}</h1>", unsafe_allow_html=True)
        st.markdown(f"""<div style="display: flex; flex-direction: column;align-items: center;justify-content: center;text-align: center;
            ">
                <h1 style="color: {border_color}; margin: 0;line-height: 1;">{rank}</h1>
                <p style="font-size: 0.9em;color: gray; margin: 0;padding-top: 2px;
                ">Global Rank: {global_rank}</p>
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

def make_trajectory_link(player_name):
    """Generates a query parameter link for the Deep Dive page."""
    return f"/Player_Trajectory?player={player_name.replace(' ', '+')}"

def plot_form_guide_plotly(df):
    plot_data = []
    for _, row in df.iterrows():
        recent = row['Results'][-15:] 
        for idx, entry in enumerate(recent):
            plot_data.append({'Team': row['Team'], 'Match_Order': idx + 1, 'Result': entry['res'], 'Opponent': entry['opp'], 'Status': 'Win' if entry['res'] == 'W' else 'Loss' if entry['res'] == 'L' else 'Tie/NR'})
    pdf = pd.DataFrame(plot_data)
    fig = px.scatter(pdf, x="Match_Order", y="Team", color="Status", text="Result", custom_data=["Opponent"], color_discrete_map={'Win': '#2ecc71', 'Loss': '#e74c3c', 'Tie/NR': '#95a5a6'}, category_orders={"Team": df['Team'].tolist()})
    fig.update_traces(marker=dict(size=28), textfont=dict(color='white', size=9, family="Arial Black"), hovertemplate="<b>Vs: %{customdata[0]}</b><br><extra></extra>")
    fig.update_layout(xaxis=dict(title=dict(text="⬅ Older matches . . . . . . Recent matches ➡"), tickvals=[1, 15], ticktext=['15 Matches Ago', 'Latest'], range=[0.5, 15.5]), yaxis=dict(title="", autorange="reversed"), template="plotly_dark", showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

def plot_history_plotly(df, team):
    if df.empty: return
    fig = go.Figure()
    for i in range(len(df)):
        if df.iloc[i]['Played']: fig.add_vrect(x0=df.iloc[i]['Match_Num'] - 0.45, x1=df.iloc[i]['Match_Num'] + 0.45, fillcolor="gray", opacity=0.15, layer="below", line_width=0)
    fig.add_trace(go.Scatter(x=df['Match_Num'], y=df['Rank'], name="Global Rank", mode='lines+markers', line=dict(color='#008080', width=3), customdata=df[['Opponent', 'Result', 'Tournament', 'NRR']], hovertemplate="<b>Vs:</b> %{customdata[0]}<br><b>Result:</b> %{customdata[1]}<br><b>Rank:</b> %{y}<extra></extra>"))
    fig.add_trace(go.Scatter(x=df['Match_Num'], y=df['NRR'], name="NRR", mode='lines+markers', line=dict(color='#D35400', width=2, dash='dot'), yaxis="y2"))
    fig.update_layout(template="plotly_dark", yaxis=dict(title="Rank", autorange="reversed", dtick=1), yaxis2=dict(title="NRR", anchor="x", overlaying="y", side="right", showgrid=False), legend=dict(orientation="h", y=1.1, x=0.5, xanchor="center"))
    st.plotly_chart(fig, use_container_width=True)


# --- MAIN APP ---
def app():
    st.title("🏆 Tournament Headquarters")
    df_rank = calculate_standings()
    
    tab1, tab2 = st.tabs(["📊 Points Table & Form", "📈 Team Trajectory"])
    
    with tab1:
        display_cols = ['Rank', 'Team', 'Mat', 'Won', 'Lost', 'Pts', 'NRR', 'Win_Streak', 'Loss_Streak']
        styled_df = df_rank[display_cols].style.format({'NRR': "{:+.3f}"}).apply(highlight_podium, axis=1)
        st.dataframe(styled_df, use_container_width=True, hide_index=True)
        st.divider(); st.subheader("📈 Team Form"); plot_form_guide_plotly(df_rank)

    with tab2:
        teams = sorted(df_rank['Team'].unique())
        sel_team = st.selectbox("Select Team to Trace", teams)
        
        if sel_team:
            df_hist = get_team_history(sel_team)
            df_inn, _ = get_matches_data()
            df_player_rank = get_rankings_data() 
            
            if not df_hist.empty:
                # 1. KEY METRICS
                st.subheader(f"📈 {sel_team} Performance Overview")
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Best Rank", int(df_hist['Rank'].min()))
                m2.metric("Worst Rank", int(df_hist['Rank'].max()))
                m3.metric("Highest NRR", f"{df_hist['NRR'].max():+.3f}")
                m4.metric("Matches Played", len(df_hist[df_hist['Played'] == True]))
                plot_history_plotly(df_hist, sel_team)
                st.divider()

                # 2. TOURNAMENTS & TEAM LEADERBOARDS
                st.subheader("🌟 Team Leaderboards")
                df_player_rank['Assigned_Role'] = df_player_rank.apply(determine_primary_role, axis=1)
                
                # --- PRE-CALCULATE GLOBAL RANKS ---
                for role_name, pts_col in [('Batsman', 'Pts_Batting'), ('Bowler', 'Pts_Bowling'), ('All-Rounder', 'Pts_AllRounder')]:
                    role_mask = df_player_rank['Assigned_Role'] == role_name
                    df_player_rank.loc[role_mask, 'Global_Rank'] = df_player_rank[role_mask][pts_col].rank(ascending=False, method='min').astype(int)

                team_players = df_player_rank[df_player_rank['Team_Name'] == sel_team].copy()
                
                lb_t1, lb_t2, lb_t3 = st.tabs(["🏏 Top Batsmen", "⚾ Top Bowlers", "⭐ Top All-Rounders"])
                

                with lb_t1:
                    df_b = team_players[team_players['Assigned_Role'] == 'Batsman'].sort_values('Pts_Batting', ascending=False).head(10)
                    if not df_b.empty:
                        st.markdown('### 🏆 Top Performers')
                        top_3 = df_b.head(3).copy()
                        p_cols = st.columns([1, 1, 1, 1, 1])
                        for i in range(len(top_3)):
                            with p_cols[i + 1]:
                                display_hero_card(top_3.iloc[i].to_dict(), i + 1)
                        
                        st.divider()

                        st.markdown("#### 📊 Other Perfomers")

                        top_score = df_b['Pts_Batting'].max()
                        df_b['Points_Visual'] = (df_b['Pts_Batting'] / top_score * 100) if top_score > 0 else 0
                        df_b['Points_Value'] = df_b['Pts_Batting']

                        df_remaining = df_b.iloc[:].copy()

                        if not df_remaining.empty:
                            df_remaining['Team_Rank'] = range(1, len(df_remaining) + 1)
                            df_remaining['Player_Link'] = df_remaining['Player_Name'].apply(make_trajectory_link)
                            df_remaining['Player_Icon'] = df_remaining['Player_Name'].apply(lambda x: get_base64_asset(x, type="player"))
                            
                            cols = ['Player_Icon', 'Team_Rank', 'Global_Rank', 'Player_Link', 'Mat', 'Total_Runs', 'Bat_Avg', 'Bat_SR','Count_30s','Count_50s', 'Total_MoMs', 'Points_Value', 'Points_Visual']
                            
                            st.dataframe(
                                df_remaining[cols], 
                                column_config=TEAM_LEADERBOARD_CONFIG, 
                                hide_index=True, 
                                use_container_width=True
                            )
                    else:
                        st.caption("No specialized batsmen found.")

                with lb_t2:
                    df_bo = team_players[team_players['Assigned_Role'] == 'Bowler'].sort_values('Pts_Bowling', ascending=False).head(10)
                    if not df_bo.empty:
                        st.markdown('### 🏆 Top Performers')
                        top_3 = df_bo.head(3).copy()
                        p_cols = st.columns([1, 1, 1, 1, 1])
                        for i in range(len(top_3)):
                            with p_cols[i + 1]:
                                display_hero_card(top_3.iloc[i].to_dict(), i + 1)
                        
                        st.divider()

                        st.markdown("#### 📊 Other Perfomers")

                        top_score = df_bo['Pts_Bowling'].max()
                        df_bo['Points_Visual'] = (df_bo['Pts_Bowling'] / top_score * 100) if top_score > 0 else 0
                        df_bo['Points_Value'] = df_bo['Pts_Bowling']

                        df_remaining = df_bo.iloc[:].copy()

                        if not df_remaining.empty:
                            df_remaining['Team_Rank'] = range(1, len(df_remaining) + 1)
                            df_remaining['Player_Link'] = df_remaining['Player_Name'].apply(make_trajectory_link)
                            df_remaining['Player_Icon'] = df_remaining['Player_Name'].apply(lambda x: get_base64_asset(x, type="player"))
                            
                            cols = ['Player_Icon','Team_Rank', 'Global_Rank', 'Player_Link', 'Mat', 'Total_Wickets', 'Bowl_Avg', 'Bowl_Econ', 'Count_3W', 'Count_4W', 'Count_5W', 'Total_MoMs', 'Points_Value', 'Points_Visual']
                        
                            st.dataframe(
                                df_remaining[cols], 
                                column_config=TEAM_LEADERBOARD_CONFIG, 
                                hide_index=True, 
                                use_container_width=True
                            )
                    else:
                        st.caption("No specialized batsmen found.")

                with lb_t3:
                    df_ar = team_players[team_players['Assigned_Role'] == 'All-Rounder'].sort_values('Pts_AllRounder', ascending=False).head(10)
                    if not df_ar.empty:
                        st.markdown('### 🏆 Top Performers')
                        top_3 = df_ar.head(3).copy()
                        p_cols = st.columns([1, 1, 1, 1, 1])

                        for i in range(len(top_3)):
                            with p_cols[i + 1]:
                                display_hero_card(top_3.iloc[i].to_dict(), i + 1)
                        
                        st.divider()

                        st.markdown("#### 📊 Other Perfomers")

                        top_score = df_ar['Pts_AllRounder'].max()
                        df_ar['Points_Visual'] = (df_ar['Pts_AllRounder'] / top_score * 100) if top_score > 0 else 0
                        df_ar['Points_Value'] = df_ar['Pts_AllRounder']

                        df_remaining = df_ar.iloc[:].copy()

                        if not df_remaining.empty:
                            df_remaining['Team_Rank'] = range(1, len(df_remaining) + 1)
                            df_remaining['Player_Link'] = df_remaining['Player_Name'].apply(make_trajectory_link)
                            df_remaining['Player_Icon'] = df_remaining['Player_Name'].apply(lambda x: get_base64_asset(x, type="player"))
                            
                            cols = ['Player_Icon','Team_Rank', 'Global_Rank', 'Player_Link', 'Mat', 'Total_Runs', 'Total_Wickets', 'Bat_Avg', 'Bowl_Avg', 'Total_MoMs', 'Points_Value', 'Points_Visual']
                        
                            st.dataframe(
                                df_remaining[cols], 
                                column_config=TEAM_LEADERBOARD_CONFIG, 
                                hide_index=True, 
                                use_container_width=True
                            )
                    else:
                        st.caption("No specialized batsmen found.")
                        

                st.divider()
                st.subheader("⚔ Head-to-Head Performance")
                h2h_df = calculate_h2h_metrics(sel_team)
                if not h2h_df.empty:
                    st.dataframe(h2h_df.style.format({c: "{:.1f}%" for c in h2h_df.columns if '%' in c}).background_gradient(subset=['Win %'], cmap='Greens'), use_container_width=True, hide_index=True)

if __name__ == "__main__":
    app()