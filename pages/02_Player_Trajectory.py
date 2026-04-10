import streamlit as st
import sqlite3
import pandas as pd
import os
import altair as alt
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import re

# --- CONFIGURATION ---
st.set_page_config(page_title="Player Deep Dive", page_icon="📈", layout="wide")

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

HOME_COUNTRIES = {
    "IND": "India", "PAK": "Pakistan", "AUS": "Australia", "ENG": "England",
    "NZ": "New Zealand", "SA": "South Africa", "WI": "West Indies", 
    "SL": "Sri Lanka", "BAN": "Bangladesh", "AFG": "Afghanistan"
}

# --- 3. DATABASE ENGINE ---
def run_query(query, params=None):
    if not os.path.exists(DB_FILE):
        return pd.DataFrame()
    conn = sqlite3.connect(DB_FILE)
    try:
        return pd.read_sql_query(query, conn, params=params)
    except Exception:
        return pd.DataFrame()
    finally:
        conn.close()

# --- 4. HELPERS ---
def normalize(name):
    return str(name).strip().upper()

def is_same_team(name1, name2):
    if not name1 or not name2: return False
    n1, n2 = normalize(name1), normalize(name2)
    if n1 == n2: return True
    t1_full = normalize(TEAM_ALIASES.get(n1, ""))
    if t1_full == n2: return True
    t2_full = normalize(TEAM_ALIASES.get(n2, ""))
    if t2_full == n1: return True
    return False

@st.cache_data
def get_teams():
    df = run_query("SELECT DISTINCT Team_Name FROM player_stats ORDER BY Team_Name")
    return df['Team_Name'].tolist() if not df.empty else []

def get_squad(team):
    t_code = next((k for k, v in TEAM_ALIASES.items() if v == team), team)
    q = f"SELECT DISTINCT Player_Name FROM player_stats WHERE Team_Name='{team}' OR Team_Name='{t_code}' ORDER BY Player_Name"
    df = run_query(q)
    return df['Player_Name'].tolist() if not df.empty else []

def get_opponents(player):
    q = f"SELECT DISTINCT Opposition FROM player_stats WHERE Player_Name = '{player}' ORDER BY Opposition"
    df = run_query(q)
    return df['Opposition'].tolist()

def extract_tournament_info(match_id):
    m = re.match(r"^(\d+)_(.+?)-\d+", str(match_id))
    if m: return m.group(1), m.group(2).replace('_', ' ')
    return "999", "Unknown"

def get_latest_tour_prefix(team):
    t_code = next((k for k, v in TEAM_ALIASES.items() if v == team), team)
    query = f"SELECT Match_ID FROM player_stats WHERE Team_Name = '{team}' OR Team_Name = '{t_code}' ORDER BY Match_ID DESC LIMIT 1"
    df = run_query(query)
    if not df.empty:
        parts = str(df.iloc[0]['Match_ID']).split('_')
        if len(parts) > 0: return parts[0], parts[1].replace('-', ' ') if len(parts) > 1 else "Unknown"
    return None, None

# --- 5. POINTS CALCULATION ---
def calculate_points(df):
    # Batting
    df['Bat_Avg'] = np.where(df['Total_Outs'] > 0, df['Total_Runs'] / df['Total_Outs'], df['Total_Runs'])
    df['Bat_SR'] = np.where(df['Total_Balls_Faced'] > 0, (df['Total_Runs'] / df['Total_Balls_Faced']) * 100, 0.0)
    for col in ['Count_100s', 'Count_50s', 'Count_30s', 'Total_MoMs', 'Count_Ducks', 'Count_5W', 'Count_4W', 'Count_3W', 'Count_Zero_Wkt']:
        if col not in df.columns: df[col] = 0

    df['Pts_Batting'] = (
        (df['Total_Runs'] * 0.5) + (df['Bat_Avg'] * 0.5) +
        (np.where(df['Bat_SR'] > 100, (df['Bat_SR'] - 100)/5, 0)) +
        (df['Count_100s']*50 + df['Count_50s']*20 + df['Count_30s']*10) +
        (df['Total_MoMs']*10) - (df['Count_Ducks']*10)
    )

    # Bowling
    df['Total_Overs_Precise'] = df['Total_Balls_Bowled'] / 6.0
    df['Bowl_Econ'] = np.where(df['Total_Overs_Precise'] > 0, df['Total_Runs_Conceded'] / df['Total_Overs_Precise'], 0.0)
    df['Pts_Bowling'] = (
        (df['Total_Wickets'] * 15) + (df['Total_Overs_Precise'] * 1.0) +
        (np.where(df['Bowl_Econ'] < 12.0, (12.0 - df['Bowl_Econ']) * 2, 0.0)) +
        (df['Count_5W']*30 + df['Count_4W']*20 + df['Count_3W']*10) +
        (df['Total_MoMs']*10) - (df['Count_Zero_Wkt']*5)
    )

    # All-Rounder
    df['Pts_AllRounder'] = (
        (df['Total_Runs'] * 1) + (df['Total_Wickets'] * 10.0) +
        (df['Total_MoMs'] * 10.0) - (df['Count_Ducks'] * 1.1) - (df['Count_Zero_Wkt'] * 1.1)
    )
    return df

# --- 6. RANKING STORYLINE ENGINE ---
@st.cache_data(show_spinner="Replaying Match History...")
def get_ranking_history_data(target_player):
    conn = sqlite3.connect(DB_FILE)
    
    global_query = """
    SELECT Player_Name, SUM(Runs_Scored) as Total_Runs, SUM(Innings_Out) as Total_Outs, SUM(Balls_Faced) as Total_Balls_Faced,
    SUM(Wickets_Taken) as Total_Wickets, SUM(Runs_Conceded) as Total_Runs_Conceded, SUM(CASE WHEN Is_MoM=1 THEN 1 ELSE 0 END) as Total_MoMs,
    SUM(CAST(Overs_Balled AS INT)*6 + CAST(ROUND((Overs_Balled-CAST(Overs_Balled AS INT))*10) AS INT)) as Total_Balls_Bowled,
    SUM(CASE WHEN Runs_Scored>=100 THEN 1 ELSE 0 END) as Count_100s, SUM(CASE WHEN Runs_Scored>=50 AND Runs_Scored<100 THEN 1 ELSE 0 END) as Count_50s,
    SUM(CASE WHEN Runs_Scored>=30 AND Runs_Scored<50 THEN 1 ELSE 0 END) as Count_30s, SUM(CASE WHEN Runs_Scored=0 AND Innings_Out=1 THEN 1 ELSE 0 END) as Count_Ducks,
    SUM(CASE WHEN Wickets_Taken>=5 THEN 1 ELSE 0 END) as Count_5W, SUM(CASE WHEN Wickets_Taken=4 THEN 1 ELSE 0 END) as Count_4W,
    SUM(CASE WHEN Wickets_Taken=3 THEN 1 ELSE 0 END) as Count_3W, SUM(CASE WHEN Wickets_Taken=0 AND Overs_Balled>0 THEN 1 ELSE 0 END) as Count_Zero_Wkt
    FROM player_stats GROUP BY Player_Name
    """
    df_final = pd.read_sql_query(global_query, conn)
    
    if target_player not in df_final['Player_Name'].values:
        conn.close(); return None, None, None

    df_final = calculate_points(df_final)
    
    # --- MATCH GLOBAL RANKINGS DISQUALIFICATION MASK ---
    mask_not_ar = (df_final['Total_Wickets'] == 0) | (df_final['Total_Runs'] < 50)
    df_final.loc[mask_not_ar, 'Pts_AllRounder'] = -1.0 # Use -1.0 for consistency

    target_row = df_final[df_final['Player_Name'] == target_player].iloc[0]
    
    # Determine role by highest points (the logic used in Global Rankings)
    bat = target_row['Pts_Batting']
    bowl = target_row['Pts_Bowling']
    ar = target_row['Pts_AllRounder']
    
    if bat >= bowl and bat >= ar:
        best_role_col = 'Pts_Batting'
        conclusive_role = "Batsman"
    elif bowl >= bat and bowl >= ar:
        best_role_col = 'Pts_Bowling'
        conclusive_role = "Bowler"
    else:
        best_role_col = 'Pts_AllRounder'
        conclusive_role = "All-Rounder"
    
    history_query = """
    SELECT Match_ID, Player_Name, Runs_Scored, Balls_Faced, Innings_Out, Wickets_Taken, Runs_Conceded, Is_MoM, Overs_Balled
    FROM player_stats ORDER BY Match_ID ASC
    """
    df_history = pd.read_sql_query(history_query, conn)
    conn.close()

    match_ids = df_history['Match_ID'].unique()
    stats = {}
    history_data = [] 
    
    for p in df_history['Player_Name'].unique():
        stats[p] = {
            'Total_Runs':0, 'Total_Balls_Faced':0, 'Total_Outs':0, 'Total_Wickets':0, 'Total_Runs_Conceded':0, 'Total_MoMs':0, 'Total_Balls_Bowled':0,
            'Count_100s':0, 'Count_50s':0, 'Count_30s':0, 'Count_Ducks':0, 'Count_5W':0, 'Count_4W':0, 'Count_3W':0, 'Count_Zero_Wkt':0
        }

    for i, match_id in enumerate(match_ids):
        match_perfs = df_history[df_history['Match_ID'] == match_id]
        
        for _, row in match_perfs.iterrows():
            p = row['Player_Name']
            stats[p]['Total_Runs'] += row['Runs_Scored']
            stats[p]['Total_Balls_Faced'] += row['Balls_Faced']
            stats[p]['Total_Outs'] += row['Innings_Out']
            stats[p]['Total_Wickets'] += row['Wickets_Taken']
            stats[p]['Total_Runs_Conceded'] += row['Runs_Conceded']
            if row['Is_MoM']: stats[p]['Total_MoMs'] += 1
            overs = row['Overs_Balled']
            balls = int(overs) * 6 + int(round((overs - int(overs)) * 10))
            stats[p]['Total_Balls_Bowled'] += balls
            r = row['Runs_Scored']
            if r >= 100: stats[p]['Count_100s'] += 1
            elif r >= 50: stats[p]['Count_50s'] += 1
            elif r >= 30: stats[p]['Count_30s'] += 1
            if r == 0 and row['Innings_Out'] == 1: stats[p]['Count_Ducks'] += 1
            w = row['Wickets_Taken']
            if w >= 5: stats[p]['Count_5W'] += 1
            elif w == 4: stats[p]['Count_4W'] += 1
            elif w == 3: stats[p]['Count_3W'] += 1
            if w == 0 and overs > 0: stats[p]['Count_Zero_Wkt'] += 1

        if i < 3: continue 

        current_df = pd.DataFrame.from_dict(stats, orient='index')
        current_df.reset_index(inplace=True); current_df.rename(columns={'index': 'Player_Name'}, inplace=True)
        current_df = calculate_points(current_df)
        # Apply same global threshold mask during the history replay
        m = (current_df['Total_Wickets'] == 0) | (current_df['Total_Runs'] < 50)
        current_df.loc[m, 'Pts_AllRounder'] = -1.0
        
        ranked_df = current_df.sort_values(by=best_role_col, ascending=False).reset_index(drop=True)
        ranked_df['Rank'] = ranked_df.index + 1
        
        player_rank_row = ranked_df[ranked_df['Player_Name'] == target_player]
        if not player_rank_row.empty:
            rank = player_rank_row['Rank'].values[0]
            played_this = target_player in match_perfs['Player_Name'].values
            if ranked_df.loc[0, 'Total_Runs'] > 0 or ranked_df.loc[0, 'Total_Wickets'] > 0:
                history_data.append({'Match_Num': i+1, 'Rank': rank, 'Played': played_this})

    return pd.DataFrame(history_data), conclusive_role, best_role_col

def plot_ranking_curve_altair(hist_df, player, role):
    if hist_df.empty:
        return None

    # 1. Prepare data for the participation shading (spans)
    # Altair's rect needs a start and end point to center on the match number
    hist_df['Match_Start'] = hist_df['Match_Num'] - 0.5
    hist_df['Match_End'] = hist_df['Match_Num'] + 0.5

    # 2. Background Shading (Only where Played == True)
    participation_shading = alt.Chart(hist_df).transform_filter(
        alt.datum.Played == True
    ).mark_rect(
        color='lightgray',
        opacity=0.4
    ).encode(
        x='Match_Start:Q',
        x2='Match_End:Q',
        tooltip=alt.value(None)
    )

    # 3. The Line and Points
    # sort='descending' on Y ensures Rank 1 is at the top
    base = alt.Chart(hist_df).encode(
        x=alt.X('Match_Num:Q', title='Match Timeline', axis=alt.Axis(tickMinStep=1)),
        y=alt.Y('Rank:Q', title='Rank Position', sort='descending', axis=alt.Axis(tickMinStep=1)),
        tooltip=[alt.Tooltip('Rank:Q', title='Rank')]
    )

    line = base.mark_line(point=True, strokeWidth=2).encode(
        color=alt.value('#0072B2') # Standard blue
    )

    # 4. Layering and Title
    # We use subtitle to display the Best/Worst/Current facts clearly
    best, worst, curr = hist_df['Rank'].min(), hist_df['Rank'].max(), hist_df['Rank'].iloc[-1]
    
    chart = alt.layer(participation_shading, line).properties(
        width=800,
        height=400,
        title={
            "text": f"Ranking History: {player} ({role})",
            "subtitle": [f"Best: #{best} | Worst: #{worst} | Current: #{curr}"]
        }
    ).interactive() # Allows zooming/panning

    return chart

# --- 7. FORM AUDIT ---
def audit_form(hist, curr, role):
    if hist.empty: return "⚪ NEW", "No history"
    if curr.empty: return "⚪ INACTIVE", "Didn't play last tour"
    h_outs = hist['Innings_Out'].sum(); h_avg = hist['Runs_Scored'].sum()/h_outs if h_outs>0 else 0
    c_outs = curr['Innings_Out'].sum(); c_avg = curr['Runs_Scored'].sum()/c_outs if c_outs>0 else 0
    bat_bad = ((h_avg-c_avg)/h_avg > BAT_DROP_TOLERANCE) if h_avg>0 else False
    h_wkts = hist['Wickets_Taken'].sum(); h_bowl_avg = hist['Runs_Conceded'].sum()/h_wkts if h_wkts>0 else 50
    c_wkts = curr['Wickets_Taken'].sum(); c_runs = curr['Runs_Conceded'].sum()
    if c_wkts > 0: c_bowl_avg = c_runs/c_wkts
    elif curr['Total_Balls_Bowled'].sum() > 0: c_bowl_avg = 99.0 
    else: c_bowl_avg = h_bowl_avg 
    bowl_bad = ((c_bowl_avg-h_bowl_avg)/h_bowl_avg > BOWL_AVG_TOLERANCE) if h_bowl_avg>0 else False
    if role == "Batsman": return ("🔴 OUT OF FORM", f"Avg dropped {h_avg:.1f} -> {c_avg:.1f}") if bat_bad else ("🟢 IN FORM", "Batting consistent")
    elif role == "Bowler": return ("🔴 OUT OF FORM", f"Bowl Avg worsened {h_bowl_avg:.1f} -> {c_bowl_avg:.1f}") if bowl_bad else ("🟢 IN FORM", "Bowling consistent")
    elif role == "All-Rounder":
        if bat_bad and bowl_bad: return "🔴 CRITICAL", "Both failing"
        if bat_bad: return "⚠️ DIP", "Batting slump"
        if bowl_bad: return "⚠️ DIP", "Bowling expensive"
        return "⭐ PEAK FORM", "Firing on all cylinders"
    return "⚪ NEUTRAL", "Insufficient data"

# --- 8. PLOT FUNCTIONS ---
def plot_batting_tournaments_altair(df):
    df = df.copy()
    def extract_info(match_id):
        m = re.match(r"^(\d+)_(.+?)-\d+", str(match_id))
        if m: return m.group(1), m.group(2).replace('_', ' ')
        return "999", "Unknown"
    
    df[['Tour_Code_Raw', 'Tour_Name']] = df['Match_ID'].apply(lambda x: pd.Series(extract_info(x)))
    df['Sort_Key'] = pd.to_numeric(df['Tour_Code_Raw'], errors='coerce').fillna(999).astype(int)
    df['Tour_Code'] = df['Tour_Code_Raw']

    grouped = df.groupby(['Sort_Key', 'Tour_Code', 'Tour_Name']).agg({
        'Runs_Scored': 'sum', 'Innings_Out': 'sum', 'Match_ID': 'count'
    }).reset_index()
    
    grouped['Avg'] = grouped.apply(lambda x: x['Runs_Scored']/x['Innings_Out'] if x['Innings_Out']>0 else x['Runs_Scored'], axis=1)

    x_axis = alt.X('Sort_Key:O', title='Tournament', axis=alt.Axis(labelExpr="'T' + datum.value", labelAngle=0))

    # Fix 1: Explicitly define the Y-axis for bars
    bars = alt.Chart(grouped).mark_bar(color='#00E5FF', opacity=0.5).encode(
        x=x_axis,
        y=alt.Y('Runs_Scored:Q', title='Total Runs'), # This defaults to left
        tooltip=['Tour_Name', 'Runs_Scored']
    )

    # Fix 2: Ensure text labels share the SAME Y-axis scale as bars
    bar_text = bars.mark_text(dy=-10, fontWeight='bold').encode(
        text='Runs_Scored:Q'
    ).transform_filter(alt.datum.Runs_Scored > 0)

    # Fix 3: Ensure the line is the only one using a colored/named right axis
    line = alt.Chart(grouped).mark_line(color='#FF007F', point=True).encode(
        x=x_axis,
        y=alt.Y('Avg:Q', title='Batting Average', axis=alt.Axis(titleColor='#FF007F')),
        tooltip=['Tour_Name', alt.Tooltip('Avg:Q', title='Batting Avg', format='.1f')]
    )

    # Combine with independent scales
    chart = alt.layer(bars, bar_text, line).resolve_scale(
        y='independent'
    ).properties(width=800, height=400, title="Tournament Batting: Runs & Average")

    return chart, grouped

def plot_bowling_tournaments_altair(df):
    df_bowl = df[df['Total_Balls_Bowled'] > 0].copy()
    if df_bowl.empty: return None, pd.DataFrame()
    df_bowl[['Tour_Code_Raw', 'Tour_Name']] = df_bowl['Match_ID'].apply(lambda x: pd.Series(extract_tournament_info(x)))
    df_bowl['Sort_Key'] = pd.to_numeric(df_bowl['Tour_Code_Raw'], errors='coerce').fillna(999).astype(int)
    grouped = df_bowl.groupby(['Sort_Key', 'Tour_Code_Raw', 'Tour_Name']).agg({'Wickets_Taken':'sum','Runs_Conceded':'sum'}).reset_index()
    grouped['Bowl_Avg'] = grouped.apply(lambda x: x['Runs_Conceded']/x['Wickets_Taken'] if x['Wickets_Taken']>0 else 0, axis=1)
    
    x_axis = alt.X('Sort_Key:O', title='Tournament', axis=alt.Axis(labelExpr="'T' + datum.value", labelAngle=0))
    bars = alt.Chart(grouped).mark_bar(color='#00E5FF', opacity=0.4).encode(x=x_axis, y=alt.Y('Wickets_Taken:Q', title='Wickets'))
    text = bars.mark_text(dy=-10, fontWeight='bold').encode(text='Wickets_Taken:Q').transform_filter(alt.datum.Wickets_Taken > 0)
    line = alt.Chart(grouped).mark_line(color='#FF007F', point=True).encode(x=x_axis, y=alt.Y('Bowl_Avg:Q', title='Bowling Average', axis=alt.Axis(titleColor='#FF007F')))
    
    chart = alt.layer(alt.layer(bars, text), line).resolve_scale(y='independent').properties(width=850, height=400, title="Tournament Bowling")
    return chart, grouped

def plot_impact_altair(stats_df, title_prefix, is_batting=True):
    # Prepare data
    plot_data = stats_df.reindex(['Won', 'Lost']).reset_index()
    color_scale = alt.Scale(domain=['Won', 'Lost'], range=['#00b894', '#d63031'])

    # --- Chart 1: Avg ---
    y_metric = 'Bat_Avg' if is_batting else 'Average'
    base1 = alt.Chart(plot_data).encode(
        x=alt.X('Result:N', title=None),
        tooltip=[alt.Tooltip('Result:N'), alt.Tooltip(f'{y_metric}:Q', title='Avg', format='.1f')]
    )
    bars1 = base1.mark_bar(size=40, color='#0984e3').encode(y=alt.Y(f'{y_metric}:Q', title="Avg"))
    # Filter: Only show text if Avg > 0
    text1 = base1.mark_text(dy=-5, fontWeight='bold').transform_filter(
        alt.datum[y_metric] > 0
    ).encode(y=f'{y_metric}:Q', text=alt.Text(f'{y_metric}:Q', format='.1f'))
    chart1 = (bars1 + text1).properties(width=160, height=200, title="Avg by Result")

    # --- Chart 2: Total Volume ---
    main_val = 'Runs_Scored' if is_batting else 'Wickets_Taken'
    main_label = "Runs" if is_batting else "Wickets"
    base2 = alt.Chart(plot_data).encode(
        x=alt.X('Result:N', title=None),
        tooltip=[alt.Tooltip('Result:N'), alt.Tooltip(f'{main_val}:Q', title=f'Total {main_label}')]
    )
    bars2 = base2.mark_bar().encode(y=alt.Y(f'{main_val}:Q', title=main_label), color=alt.Color('Result:N', scale=color_scale, legend=None))
    # Filter: Only show text if Runs/Wickets > 0
    text2 = base2.mark_text(dy=-5, fontWeight='bold').transform_filter(
        alt.datum[main_val] > 0
    ).encode(y=f'{main_val}:Q', text=f'{main_val}:Q')
    chart2 = (bars2 + text2).properties(width=160, height=200, title=f"Total {main_label}")

    # --- Chart 3: Milestones & Awards ---
    label_map = {'Count_30s': '30+', 'Count_50s': '50+', 'Is_MoM': 'MoM', '3W+': '3W+', '5W+': '5W+', 'MoM': 'MoM'}
    milestone_cols = ['Count_30s', 'Count_50s', 'Is_MoM'] if is_batting else ['3W+', '5W+', 'MoM']
    
    base3 = alt.Chart(plot_data).transform_fold(
        milestone_cols, as_=['Original', 'Count']
    ).transform_calculate(
        Category=f"datum.Original == '{milestone_cols[0]}' ? '{label_map[milestone_cols[0]]}' : datum.Original == '{milestone_cols[1]}' ? '{label_map[milestone_cols[1]]}' : 'MoM'"
    ).encode(
        x=alt.X('Category:N', title=None, sort=['30+', '50+', '3W+', '5W+', 'MoM']),
        xOffset='Result:N',
        tooltip=[alt.Tooltip('Result:N'), alt.Tooltip('Category:N', title='Milestone'), alt.Tooltip('Count:Q', title='Count')]
    )
    bars3 = base3.mark_bar().encode(
        y=alt.Y('Count:Q', axis=alt.Axis(tickMinStep=1), title="Count"),
        color=alt.Color('Result:N', scale=color_scale, legend=alt.Legend(orient='bottom', title=None))
    )
    # Filter: Only show text if Count > 0
    text3 = base3.mark_text(dy=-5, fontWeight='bold', fontSize=10).transform_filter(
        alt.datum.Count > 0
    ).encode(y='Count:Q', text='Count:Q')
    chart3 = (bars3 + text3).properties(width=220, height=200, title="Milestones & Awards")

    # --- Chart 4: Split ---
    base4 = alt.Chart(plot_data).encode(
        theta=alt.Theta(f'{main_val}:Q', stack=True),
        color=alt.Color('Result:N', scale=color_scale, legend=None),
        tooltip=[alt.Tooltip('Result:N'), alt.Tooltip(f'{main_val}:Q', title=main_label)]
    )
    
    arcs4 = base4.mark_arc(innerRadius=45, outerRadius=80)
    
    # Instead of arc-positioning, we use standard X/Y to put text in the dead center
    # We use Result:N to separate Won and Lost vertically
    text4 = alt.Chart(plot_data).mark_text(fontWeight='bold').encode(
        x=alt.value(80), # Half of the 160 width
        y=alt.Y('Result:N', axis=None).scale(padding=40), # Spreads them vertically in the hole
        text=alt.Text(f'{main_val}:Q'),
        color=alt.Color('Result:N', scale=color_scale, legend=None)
    ).transform_filter(
        alt.datum[main_val] > 0
    ).properties(width=160, height=200)
    
    chart4 = alt.layer(arcs4, text4).properties(width=160, height=200, title=f"{main_label} Split")

    return alt.hconcat(chart1, chart2, chart3, chart4).resolve_scale(color='shared').configure_view(strokeOpacity=0)

# --- HELPER: ADVANCED CHART DATA PROCESSING ---

def process_batting_data(df):
    if df.empty: return df
    df = df.sort_values("Match_ID").copy()
    
    # 1. Match Numbering
    df['Match_Number'] = range(1, len(df) + 1)
    
    # 2. HYBRID "Did Not Bat" (NB) Check
    # We check BOTH the DB column AND the stats to be 100% sure.
    # Condition A: Your DB says they didn't play the inning
    cond_db = (pd.to_numeric(df['Innings_Played'], errors='coerce').fillna(1) == 0)
    
    # Condition B: Heuristic (0 Runs, 0 Balls, No Dismissal Info)
    # This catches cases where Innings_Played might be 1 but they never batted.
    cond_stats = ((df['Runs_Scored'] == 0) & 
                  (df['Balls_Faced'] == 0) & 
                  (df['Dismissal_Type'].isna() | (df['Dismissal_Type'].astype(str).str.strip() == "")))
    
    df['Is_NB'] = (cond_db | cond_stats)

    # 3. Cumulative Stats
    df['Total Runs Scored'] = df['Runs_Scored'].cumsum()
    
    # Divisor: Only count innings where they were OUT
    # Standard Batting Avg = Total Runs / Times Out
    # We use 'Not_Out_Innings' (1=Not Out, 0=Out)
    # If Is_NB is True, we treat it as Not Out (doesn't increment divisor)
    
    # Calculate if this specific inning counts as a dismissal
    df['Is_Dismissal'] = np.where(df['Is_NB'], 0, (1 - df['Not_Out_Innings']))
    df['CumOuts'] = df['Is_Dismissal'].cumsum()
    
    # Batting Average
    df['Batting Average'] = df.apply(
        lambda x: x['Total Runs Scored'] / x['CumOuts'] if x['CumOuts'] > 0 else x['Total Runs Scored'], axis=1
    )
    
    # 4. Rolling Average (Skip NB)
    df['Runs_For_Rolling'] = np.where(df['Is_NB'], np.nan, df['Runs_Scored'])
    df['Rolling_Avg'] = df['Runs_For_Rolling'].rolling(window=10, min_periods=1).mean()
    
    # 5. Visualization Helpers
    def get_bat_props(row):
        # Case A: Did Not Bat
        if row['Is_NB']: 
            return pd.Series(["NB", "Gray"])
        
        val = int(row['Runs_Scored'])
        is_not_out = (row['Not_Out_Innings'] == 1)
        
        # Label: Add * if Not Out
        label = f"{val}*" if is_not_out else str(val)
        
        # Color Logic
        if val >= 100: 
            color = 'Century' # Pink/Red
        elif val >= 50: 
            color = 'Fifty'   # Yellow
        elif val == 0 and not is_not_out: 
            color = 'Duck'    # RED (Only if actually OUT)
        elif val == 0 and is_not_out:
            color = 'Normal'  # Teal (0* is not a duck)
        else: 
            color = 'Normal'  # Teal
            
        return pd.Series([label, color])

    df[['Bar_Label', 'Form_Color']] = df.apply(get_bat_props, axis=1)
    df['MoM_Marker'] = df['Is_MoM'].apply(lambda x: "★" if x == 1 else "")
    
    return df

def process_bowling_data(df):
    if df.empty: return df
    df = df.sort_values("Match_ID").copy()
    
    df['Match_Number'] = range(1, len(df) + 1)
    df['Is_NB'] = (df['Total_Balls_Bowled'] == 0)

    if 'Runs_Conceded' not in df.columns: df['Runs_Conceded'] = 0
    
    df['Total Wickets Taken'] = df['Wickets_Taken'].cumsum()
    df['CumRunsConceded'] = df['Runs_Conceded'].cumsum()
    
    df['Bowling Average'] = df.apply(
        lambda x: x['CumRunsConceded'] / x['Total Wickets Taken'] if x['Total Wickets Taken'] > 0 else None, axis=1
    )
    
    df['Wkts_For_Rolling'] = np.where(df['Is_NB'], np.nan, df['Wickets_Taken'])
    df['Rolling_Wkts'] = df['Wkts_For_Rolling'].rolling(window=10, min_periods=1).mean()
    
    def get_bowl_props(row):
        if row['Is_NB']: return pd.Series(["NB", "DidNotBowl"])
        
        w = int(row['Wickets_Taken'])
        label = str(w)
        
        if w >= 5: color = '5-Fer'
        elif w >= 3: color = '3-Fer'
        elif w == 0: color = 'Wicketless'
        else: color = 'Normal'
        
        return pd.Series([label, color])
    
    df[['Bar_Label', 'Form_Color']] = df.apply(get_bowl_props, axis=1)
    df['MoM_Marker'] = df['Is_MoM'].apply(lambda x: "★" if x == 1 else "")
    
    return df

# --- 9. DATA FETCHERS ---
def fetch_comprehensive_stats(player, opp_filter=None):
    opp_clause = f"AND p.Opposition = '{opp_filter}'" if opp_filter and opp_filter != "All Teams" else ""
    query = f"""
    SELECT 
        p.Tournament_ID, p.Match_ID, p.Team_Name, p.Opposition, p.Runs_Scored, 
        p.Balls_Faced, p.Innings_Out, p.Is_MoM, p.Wickets_Taken, p.Runs_Conceded, 
        p.Total_Balls_Bowled, p.Dismissal_Type, p.Innings_Played, p.Not_Out_Innings,
        p.Fours_Hit, p.Sixes_Hit, p.Dot_Balls_Bowled, -- From Schema v2.2
        i.Winner, i.Venue, i.Country
    FROM player_stats p
    JOIN (SELECT DISTINCT Match_ID, Winner, Venue, Country FROM innings_summary) i ON p.Match_ID = i.Match_ID
    WHERE p.Player_Name = '{player}' {opp_clause}
    ORDER BY p.Match_ID ASC
    """
    return run_query(query)

# --- 10. MAIN UI ---
def app():
    # 1. READ URL PARAMETERS
    params = st.query_params
    url_player = params.get("player") # Catch the 'player' key from the URL

    st.title("📈 Player Deep Dive")
    
    # 2. SETUP DEFAULTS
    all_teams = get_teams()
    default_team_idx = 0
    default_player_idx = 0

    # 3. IF ACCESSED VIA LINK: FIND THE PLAYER'S TEAM
    if url_player:
        # Check database for this player's team to sync the first dropdown
        search_team = run_query("SELECT Team_Name FROM player_stats WHERE Player_Name = ? LIMIT 1", (url_player,))
        if not search_team.empty:
            p_team = search_team.iloc[0]['Team_Name']
            if p_team in all_teams:
                default_team_idx = all_teams.index(p_team)

    c1, c2, c3 = st.columns(3)
    team = c1.selectbox("Select Team", all_teams, index=default_team_idx)
    
    if team:
        last_tour_prefix, last_tour_name = get_latest_tour_prefix(team)
        players = get_squad(team)
        
        # 4. SYNC PLAYER DROPDOWN
        if url_player in players:
            default_player_idx = players.index(url_player)
            
        player = c2.selectbox("Select Player", players, index=default_player_idx)
        
        if player:
            opps = ["All Teams"] + get_opponents(player)
            opp_select = c3.selectbox("Filter Opponent", opps)
            
            # 1. FETCH MAIN DATA (Global Stats)
            df = fetch_comprehensive_stats(player, opp_select)
            if df.empty: st.warning("No data found."); st.stop()
            
            df['Result'] = df.apply(lambda x: "Won" if is_same_team(x['Team_Name'], x['Winner']) else "Lost", axis=1)
            
            # 2. DETERMINE ROLE & FORM
            # Use the already calculated conclusive_role from the ranking engine
            hist_data, conclusive_role, metric_col = get_ranking_history_data(player)
            quick_role = conclusive_role # Rename for compatibility with the rest of your UI
            
            if last_tour_prefix:
                curr = df[df['Match_ID'].astype(str).str.startswith(last_tour_prefix)]
                hist = df[~df['Match_ID'].astype(str).str.startswith(last_tour_prefix)]
                status, reason = audit_form(hist, curr, quick_role)
                form_label = f"vs History (Last Tour: {last_tour_name})"
            else:
                status, reason = "⚪ NEUTRAL", "No recent data"
                form_label = "Form Check"

            # 3. HEADER METRICS
            with st.container():
                cols = st.columns([2, 5])
                with cols[0]:
                    st.subheader(player)
                    st.caption(f"**Est Role:** {quick_role} | **{form_label}:** {status}")
                    st.caption(f"*{reason}*")
                with cols[1]:
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Matches", len(df))
                    m2.metric("Runs", df['Runs_Scored'].sum())
                    m3.metric("Wickets", df['Wickets_Taken'].sum())
                    m4.metric("MoMs", df['Is_MoM'].sum())
            st.divider()

            # 4. TABS
            tabs = st.tabs(["📊 Career Arc", "⚖️ Win vs Loss", "🏠 Home vs Away", "🆚 Opponents", "🏆 Tournaments", "⭐ Ranking Story", "🔥 Power Metrics"])
            show_bat_first = True if quick_role != "Bowler" else False

            # --- TAB 1: ADVANCED CAREER ARC (PRO DASHBOARD) ---
            with tabs[0]:
                
                # Use Main DF Directly
                bat_data = process_batting_data(df.copy())
                bowl_data = process_bowling_data(df.copy())
                
                # --- BATTING VIZ ---
                bat_chart_final = None
                if not bat_data.empty:
                    
                    bat_scale = alt.Scale(
                        domain=['Century', 'Fifty', 'Duck', 'Normal', 'Gray'], 
                        range=['#ff6b6b', '#ffe66d', '#e63946', '#4ecdc4', '#2c3e50']
                    )
                    
                    # LAYER A: Trajectory (Runs) - Left Axis
                    base_traj = alt.Chart(bat_data).encode(x=alt.X('Match_Number:Q', axis=alt.Axis(labels=False, title=None)))
                    
                    traj_area = base_traj.mark_area(color='#1a535c', opacity=0.4).encode(
                        y=alt.Y('Total Runs Scored:Q', axis=alt.Axis(title='Total Runs', titleColor='#4ecdc4')),
                        tooltip=['Match_ID', 'Venue', 'Total Runs Scored']
                    )
                    
                    # LAYER B: Average - Right Axis
                    traj_line = base_traj.mark_line(color='#ff9f1c', strokeDash=[5,5]).encode(
                        y=alt.Y('Batting Average:Q', axis=alt.Axis(title='Batting Avg', titleColor='#ff9f1c', orient='right')),
                        tooltip=['Batting Average']
                    )
                    
                    # Combined Top Panel
                    # NOTE: width=850 is safe for 'wide' layout without clipping the right axis
                    chart_A = alt.layer(traj_area, traj_line).resolve_scale(y='independent').properties(
                        height=200, width=850, title=f"BATTING TRAJECTORY: {player}"
                    )
                    
                    # FORM CHART
                    base_form = alt.Chart(bat_data).encode(x=alt.X('Match_Number:Q', title='Match Timeline'))
                    
                    form_bars = base_form.mark_bar().encode(
                        y=alt.Y('Runs_Scored:Q', title='Runs'),
                        color=alt.Color('Form_Color:N', scale=bat_scale, legend=None),
                        tooltip=['Match_ID', 'Venue', 'Runs_Scored', 'Bar_Label']
                    )
                    
                    form_text = base_form.mark_text(dy=-10, color='white', size=10).encode(
                        y=alt.Y('Runs_Scored:Q'), text='Bar_Label'
                    )
                    
                    form_roll = alt.Chart(bat_data.dropna(subset=['Rolling_Avg'])).mark_line(color='#ff9f1c', strokeDash=[5,5], opacity=0.9).encode(
                        x='Match_Number:Q', y='Rolling_Avg:Q'
                    )
                    
                    form_mom = base_form.mark_text(dy=-20, color='gold', size=14).encode(
                        y='Runs_Scored:Q', text='MoM_Marker'
                    )
                    
                    chart_B = alt.layer(form_bars, form_text, form_roll, form_mom).properties(
                        height=250, width=850, title="FORM GUIDE"
                    )
                    
                    # Final Assembly (Removed configure_layout to fix crash)
                    bat_chart_final = alt.vconcat(chart_A, chart_B).resolve_scale(x='shared')

                # --- BOWLING VIZ ---
                bowl_chart_final = None
                if not bowl_data.empty:
                    
                    bowl_scale = alt.Scale(
                        domain=['5-Fer', '3-Fer', 'Wicketless', 'Normal', 'DidNotBowl'], 
                        range=['#f72585', '#4361ee', '#7209b7', '#4cc9f0', 'transparent']
                    )
                    
                    base_b_traj = alt.Chart(bowl_data).encode(x=alt.X('Match_Number:Q', axis=alt.Axis(labels=False, title=None)))
                    
                    # LAYER A: Wickets - Left Axis
                    b_traj_area = base_b_traj.mark_area(color='#3a0ca3', opacity=0.4).encode(
                        y=alt.Y('Total Wickets Taken:Q', axis=alt.Axis(title='Total Wickets', titleColor='#4cc9f0')),
                        tooltip=['Match_ID', 'Venue', 'Total Wickets Taken']
                    )
                    
                    # LAYER B: Average - Right Axis
                    b_traj_line = base_b_traj.mark_line(color='#f77f00', strokeDash=[5,5]).encode(
                        y=alt.Y('Bowling Average:Q', axis=alt.Axis(title='Bowl Avg', titleColor='#f77f00', orient='right')),
                        tooltip=['Bowling Average']
                    )
                    
                    chart_C = alt.layer(b_traj_area, b_traj_line).resolve_scale(y='independent').properties(
                        height=200, width=850, title=f"BOWLING TRAJECTORY: {player}"
                    )
                    
                    # FORM CHART
                    base_b_form = alt.Chart(bowl_data).encode(x=alt.X('Match_Number:Q', title='Match Timeline'))
                    
                    b_form_bars = base_b_form.mark_bar().encode(
                        y=alt.Y('Wickets_Taken:Q', title='Wickets', axis=alt.Axis(tickMinStep=1, format='d')),
                        color=alt.Color('Form_Color:N', scale=bowl_scale, legend=None),
                        tooltip=['Match_ID', 'Venue', 'Wickets_Taken', 'Bar_Label']
                    )
                    
                    b_form_text = base_b_form.mark_text(dy=-10, color='white', size=10).encode(
                        y=alt.Y('Wickets_Taken:Q'), text='Bar_Label'
                    )
                    
                    b_form_roll = alt.Chart(bowl_data.dropna(subset=['Rolling_Wkts'])).mark_line(color='#ff9f1c', strokeDash=[5,5], opacity=0.9).encode(
                        x='Match_Number:Q', y='Rolling_Wkts:Q'
                    )
                    
                    b_form_mom = base_b_form.mark_text(dy=-20, color='gold', size=14).encode(
                        y='Wickets_Taken:Q', text='MoM_Marker'
                    )
                    
                    chart_D = alt.layer(b_form_bars, b_form_text, b_form_roll, b_form_mom).properties(
                        height=250, width=850, title="FORM GUIDE"
                    )
                    
                    bowl_chart_final = alt.vconcat(chart_C, chart_D).resolve_scale(x='shared')

                # --- RENDER ---
                # IMPORTANT: use_container_width=False is required to respect the fixed 850px width
                # This prevents Streamlit from squashing the right-side axis.
                if show_bat_first:
                    if bat_chart_final: st.altair_chart(bat_chart_final, use_container_width=False)
                    if bowl_chart_final: st.divider(); st.altair_chart(bowl_chart_final, use_container_width=False)
                else:
                    if bowl_chart_final: st.altair_chart(bowl_chart_final, use_container_width=False)
                    if bat_chart_final: st.divider(); st.altair_chart(bat_chart_final, use_container_width=False)

            # TAB 2: WIN vs LOSS (Keeping original logic)
            with tabs[1]:
                wl_grp = df.groupby('Result').agg({
                    'Match_ID': 'count', 'Runs_Scored': 'sum', 'Balls_Faced': 'sum', 'Innings_Out': 'sum',
                    'Is_MoM': 'sum', 'Wickets_Taken': 'sum', 'Runs_Conceded': 'sum', 'Total_Balls_Bowled': 'sum'
                }).reset_index().set_index('Result').reindex(['Won', 'Lost']).fillna(0)

                wl_grp['Bat_Avg'] = np.where(wl_grp['Innings_Out']>0, wl_grp['Runs_Scored']/wl_grp['Innings_Out'], wl_grp['Runs_Scored'])
                wl_grp['Strike_Rate'] = np.where(wl_grp['Balls_Faced']>0, (wl_grp['Runs_Scored']/wl_grp['Balls_Faced'])*100, 0)
                for res in ['Won', 'Lost']:
                    sub = df[df['Result'] == res]
                    wl_grp.loc[res, 'Count_30s'] = len(sub[(sub['Runs_Scored']>=30) & (sub['Runs_Scored']<50)])
                    wl_grp.loc[res, 'Count_50s'] = len(sub[sub['Runs_Scored']>=50])

                wl_grp['Average'] = np.where(wl_grp['Wickets_Taken']>0, wl_grp['Runs_Conceded']/wl_grp['Wickets_Taken'], 0)
                wl_grp['Economy'] = np.where(wl_grp['Total_Balls_Bowled']>0, (wl_grp['Runs_Conceded']/wl_grp['Total_Balls_Bowled'])*6, 0)
                for res in ['Won', 'Lost']:
                    sub = df[df['Result'] == res]
                    wl_grp.loc[res, '3W+'] = len(sub[sub['Wickets_Taken']>=3])
                    wl_grp.loc[res, '5W+'] = len(sub[sub['Wickets_Taken']>=5])
                    wl_grp.loc[res, 'MoM'] = sub['Is_MoM'].sum()

                if show_bat_first:
                    st.altair_chart(plot_impact_altair(wl_grp, "Batting", is_batting=True), use_container_width=True)
                    st.divider()
                    st.altair_chart(plot_impact_altair(wl_grp, "Bowling", is_batting=False), use_container_width=True)
                else:
                    st.altair_chart(plot_impact_altair(wl_grp, "Bowling", is_batting=False), use_container_width=True)
                    st.divider()
                    st.altair_chart(plot_impact_altair(wl_grp, "Batting", is_batting=True), use_container_width=True)

            # TAB 3: HOME vs AWAY
            with tabs[2]:
                def check_venue(row):
                    if 'Country' not in row or not row['Country']: return "Neutral"
                    t = normalize(row['Team_Name'])
                    home = HOME_COUNTRIES.get(t, TEAM_ALIASES.get(t, "Unknown"))
                    return "Home" if normalize(home) == normalize(row['Country']) else "Away"
                df['Condition'] = df.apply(check_venue, axis=1)
                
                ha_grp = df.groupby('Condition').agg({'Match_ID':'count', 'Runs_Scored':'sum', 'Wickets_Taken':'sum', 'Innings_Out':'sum', 'Runs_Conceded':'sum'}).reset_index()
                ha_grp['Bat_Avg'] = np.where(ha_grp['Innings_Out']>0, ha_grp['Runs_Scored']/ha_grp['Innings_Out'], ha_grp['Runs_Scored'])
                ha_grp['Bowl_Avg'] = np.where(ha_grp['Wickets_Taken']>0, ha_grp['Runs_Conceded']/ha_grp['Wickets_Taken'], 0)
                
                c1, c2 = st.columns(2)
                base = alt.Chart(ha_grp).encode(x='Condition')
                chart_bat = base.mark_bar(color='teal').encode(y='Bat_Avg', tooltip=['Bat_Avg']).properties(title="Batting Avg")
                chart_bowl = base.mark_bar(color='orange').encode(y='Bowl_Avg', tooltip=['Bowl_Avg']).properties(title="Bowling Avg")
                
                if show_bat_first: c1.altair_chart(chart_bat, use_container_width=True); c2.altair_chart(chart_bowl, use_container_width=True)
                else: c1.altair_chart(chart_bowl, use_container_width=True); c2.altair_chart(chart_bat, use_container_width=True)

            # TAB 4: OPPONENTS
            with tabs[3]:
                # --- PREPARE DATA ---
                
                # 1. Batting Stats
                bat_opp = df.groupby('Opposition').agg({'Runs_Scored': 'sum','Balls_Faced': 'sum','Innings_Out': 'sum','Match_ID': 'count'}).reset_index()

                # Calculate Batting Metrics
                bat_opp['Bat_Avg'] = bat_opp.apply(lambda x: x['Runs_Scored'] / x['Innings_Out'] if x['Innings_Out'] > 0 else x['Runs_Scored'], axis=1)
                bat_opp['Bat_SR'] = bat_opp.apply(lambda x: (x['Runs_Scored'] / x['Balls_Faced'] * 100) if x['Balls_Faced'] > 0 else 0, axis=1)
                
                # Create Display Label (e.g., "450 | Avg: 55.2 | SR: 140")
                bat_opp['Label'] = bat_opp.apply(lambda x: f"{int(x['Runs_Scored'])}  |  Avg: {x['Bat_Avg']:.1f}  |  SR: {x['Bat_SR']:.0f}", axis=1)

                # 2. Bowling Stats
                bowl_opp = df.groupby('Opposition').agg({'Wickets_Taken': 'sum','Runs_Conceded': 'sum','Total_Balls_Bowled': 'sum','Match_ID': 'count'}).reset_index()

                # Calculate Bowling Metrics
                bowl_opp['Bowl_Avg'] = bowl_opp.apply(lambda x: x['Runs_Conceded'] / x['Wickets_Taken'] if x['Wickets_Taken'] > 0 else 0, axis=1)
                bowl_opp['Bowl_SR'] = bowl_opp.apply(lambda x: x['Total_Balls_Bowled'] / x['Wickets_Taken'] if x['Wickets_Taken'] > 0 else 0, axis=1)
                
                # Create Display Label
                bowl_opp['Label'] = bowl_opp.apply(lambda x: f"{int(x['Wickets_Taken'])}  |  Avg: {x['Bowl_Avg']:.1f}  |  SR: {x['Bowl_SR']:.1f}", axis=1)

                # --- VISUALIZATION (Horizontal Bars for Readability) ---
                
                # BATTING CHART
                b_base = alt.Chart(bat_opp).encode(y=alt.Y('Opposition', sort='-x', title=None),x=alt.X('Runs_Scored', title='Runs Scored'));b_bars = b_base.mark_bar(color='#4CAF50').encode(tooltip=['Opposition', 'Runs_Scored', alt.Tooltip('Bat_Avg', format='.1f'), alt.Tooltip('Bat_SR', format='.1f')]);b_text = b_base.mark_text(align='left', dx=5, color='white').encode(text='Label');chart_bat_opp = (b_bars + b_text).properties(title="BATTING Performance vs Opponents")

                # BOWLING CHART
                w_base = alt.Chart(bowl_opp).encode(y=alt.Y('Opposition', sort='-x', title=None),x=alt.X('Wickets_Taken', title='Wickets Taken'));w_bars = w_base.mark_bar(color='#2196F3').encode(tooltip=['Opposition', 'Wickets_Taken', alt.Tooltip('Bowl_Avg', format='.1f'), alt.Tooltip('Bowl_SR', format='.1f')]);w_text = w_base.mark_text(align='left', dx=5, color='white').encode(text='Label');chart_bowl_opp = (w_bars + w_text).properties(title="BOWLING Performance vs Opponents")

                # --- RENDER ---
                # We use container width but add padding for the text labels
                if show_bat_first:
                    st.altair_chart(chart_bat_opp, use_container_width=True)
                    st.divider()
                    st.altair_chart(chart_bowl_opp, use_container_width=True)
                else:
                    st.altair_chart(chart_bowl_opp, use_container_width=True)
                    st.divider()
                    st.altair_chart(chart_bat_opp, use_container_width=True)
            # TAB 5: TOURNAMENTS
            with tabs[4]:
                st.markdown("#### 🏏 Batting Tournament History")
                chart_bat, df_bat_grouped = plot_batting_tournaments_altair(df)
                st.altair_chart(chart_bat, use_container_width=True)
                with st.expander("View Batting Table"):
                    st.dataframe(df_bat_grouped, hide_index=True)
                    
                st.divider()
                
                st.markdown("#### ⚾ Bowling Tournament History")
                chart_bowl, df_bowl_grouped = plot_bowling_tournaments_altair(df)
                if chart_bowl:
                    st.altair_chart(chart_bowl, use_container_width=True)
                    with st.expander("View Bowling Table"):
                        st.dataframe(df_bowl_grouped, hide_index=True)
                else:
                    st.info("Player has not bowled in any tournaments.")
           # TAB 6: RANKING
            with tabs[5]:
                st.markdown("#### ⭐ Player Ranking Journey")
                st.caption(f"Determining {player}'s best role relative to global stats and tracking rank over time.")
                
                # This function is already in your code fetching the data
                hist_data, conclusive_role, metric_col = get_ranking_history_data(player)
                
                if hist_data is not None and not hist_data.empty:
                    # Call the new Altair function
                    chart_rank = plot_ranking_curve_altair(hist_data, player, conclusive_role)
                    st.altair_chart(chart_rank, use_container_width=True)
                    st.success(f"Tracked based on **{conclusive_role}** metrics.")
                else: 
                    st.warning("Insufficient data.")

            # --- TAB 7: POWER & PRESSURE (FILTERED FROM T028) ---
            with tabs[6]:
                st.header("🔥 Power & Pressure Analysis")
                
                # Define the threshold for modern metrics
                START_TOUR = "028"
                
                # SYNCED DATASET: All calculations below use this filtered DF
                modern_df = df[df['Tournament_ID'].astype(str) >= START_TOUR].copy()
                
                if modern_df.empty:
                    st.warning(f"No data available for this player since Tournament {START_TOUR}.")
                else:
                    # 1. ERA AGGREGATES (Top Section)
                    m_4s = int(modern_df['Fours_Hit'].sum())
                    m_6s = int(modern_df['Sixes_Hit'].sum())
                    m_dots = int(modern_df['Dot_Balls_Bowled'].sum())
                    
                    st.markdown(f"### Power Metrics since end of World Cup 2025")
                    # st.caption("Synchronized totals used for factor analysis below.")
                    t1, t2, t3 = st.columns(3)
                    t1.metric("Total Fours", m_4s)
                    t2.metric("Total Sixes", m_6s)
                    t3.metric("Total Dot Balls", m_dots)
                    
                    st.divider()

                    # 2. FACTOR ANALYSIS (Bottom Section)
                    m_col1, m_col2 = st.columns(2)
                    
                    # --- BATTING: BOUNDARY FACTOR ---
                    with m_col1:
                        m_runs = modern_df['Runs_Scored'].sum()
                        m_bound_runs = (m_4s * 4) + (m_6s * 6)
                        
                        st.subheader("🏏 Boundary Factor")
                        if m_runs > 0:
                            bound_perc = (m_bound_runs / m_runs) * 100
                            st.write(f"**{bound_perc:.1f}%** of runs come from boundaries.")
                            
                            dist_data = pd.DataFrame({
                                'Type': ['Boundaries', 'Singles/Doubles'],
                                'Runs': [m_bound_runs, m_runs - m_bound_runs]
                            })
                            
                            chart_b = alt.Chart(dist_data).mark_arc(innerRadius=60).encode(
                                theta="Runs:Q",
                                color=alt.Color("Type:N", scale=alt.Scale(range=['#f72585', '#4cc9f0'])),
                                tooltip=['Type', 'Runs']
                            ).properties(height=300, title="Run Scoring Distribution")
                            st.altair_chart(chart_b, use_container_width=True)
                        else:
                            st.info("No runs scored in this era.")

                    # --- BOWLING: PRESSURE FACTOR ---
                    with m_col2:
                        m_balls = modern_df['Total_Balls_Bowled'].sum()
                        
                        st.subheader("🛑 Pressure Factor")
                        if m_balls > 0:
                            dot_perc = (m_dots / m_balls) * 100
                            st.write(f"**{dot_perc:.1f}%** of deliveries are dot balls.")
                            
                            dot_dist_data = pd.DataFrame({
                                'Type': ['Dot Balls', 'Scoring Balls'],
                                'Count': [m_dots, m_balls - m_dots]
                            })
                            
                            chart_d = alt.Chart(dot_dist_data).mark_arc(innerRadius=60).encode(
                                theta="Count:Q",
                                color=alt.Color("Type:N", scale=alt.Scale(range=['#4ecdc4', '#2c3e50'])),
                                tooltip=['Type', 'Count']
                            ).properties(height=300, title="Delivery Pressure Distribution")
                            st.altair_chart(chart_d, use_container_width=True)
                        else:
                            st.info("No bowling records in this era.")

if __name__ == "__main__":
    app()