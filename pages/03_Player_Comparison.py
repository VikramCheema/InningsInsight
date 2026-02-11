import streamlit as st
import sqlite3
import pandas as pd
import os
import numpy as np
import altair as alt
import matplotlib.pyplot as plt

# --- CONFIGURATION ---
st.set_page_config(page_title="Player Comparison", page_icon="🆚", layout="wide")

# --- 1. ROBUST PATH FINDER ---
def get_db_path():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(script_dir) # Go up one level to root
    
    # Check parent dir first (standard structure)
    db_path = os.path.join(root_dir, "cricket_data.db")
    if os.path.exists(db_path): return db_path
    
    # Fallback to current dir
    return "cricket_data.db"

DB_FILE = get_db_path()

# --- 2. DATABASE ENGINE ---
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

@st.cache_data
def get_all_players():
    df = run_query("SELECT DISTINCT Player_Name FROM player_stats ORDER BY Player_Name")
    return df['Player_Name'].tolist() if not df.empty else []

# --- 3. POINTS CALCULATION (Strict Logic) ---
def determine_primary_role(row):
    """Assigns ONE primary role based on highest points"""
    bat = row['Pts_Batting']
    bowl = row['Pts_Bowling']
    ar = row['Pts_AllRounder']
    
    if bat >= bowl and bat >= ar:
        return "Batsman"
    elif bowl >= bat and bowl >= ar:
        return "Bowler"
    else:
        return "All-Rounder"

def calculate_points_and_roles(df):
    """Applies standard points logic and determines specific Role"""
    
    # --- PRE-CALCULATIONS ---
    df['Bat_Avg'] = np.where(df['Total_Outs'] > 0, df['Total_Runs'] / df['Total_Outs'], df['Total_Runs'])
    df['Bat_SR'] = np.where(df['Total_Balls_Faced'] > 0, (df['Total_Runs'] / df['Total_Balls_Faced']) * 100, 0.0)
    
    df['Total_Overs_Precise'] = df['Total_Balls_Bowled'] / 6.0
    df['Bowl_Econ'] = np.where(df['Total_Overs_Precise'] > 0, df['Total_Runs_Conceded'] / df['Total_Overs_Precise'], 0.0)
    
    # Fill missing columns with 0
    cols_needed = ['Count_100s', 'Count_50s', 'Count_30s', 'Total_MoMs', 'Count_Ducks', 'Count_5W', 'Count_4W', 'Count_3W', 'Count_Zero_Wkt']
    for col in cols_needed:
        if col not in df.columns: df[col] = 0

    # --- 1. BATTING POINTS ---
    b_sr_pts = np.where(df['Bat_SR'] > 100, (df['Bat_SR'] - 100)/5, 0)
    df['Pts_Batting'] = (
        (df['Total_Runs'] * 0.5) + (df['Bat_Avg'] * 0.5) + b_sr_pts +
        (df['Count_100s']*50 + df['Count_50s']*20 + df['Count_30s']*10) +
        (df['Total_MoMs']*10) - (df['Count_Ducks']*10)
    )

    # --- 2. BOWLING POINTS ---
    w_econ_pts = np.where(df['Bowl_Econ'] < 12.0, (12.0 - df['Bowl_Econ']) * 2, 0.0)
    df['Pts_Bowling'] = (
        (df['Total_Wickets'] * 15) + (df['Total_Overs_Precise'] * 1.0) + w_econ_pts +
        (df['Count_5W']*30 + df['Count_4W']*20 + df['Count_3W']*10) +
        (df['Total_MoMs']*10) - (df['Count_Zero_Wkt']*5)
    )

    # --- 3. ALL-ROUNDER POINTS ---
    df['Pts_AllRounder'] = (
        (df['Total_Runs'] * 1) + (df['Total_Wickets'] * 10.0) +
        (df['Total_MoMs'] * 10.0) - (df['Count_Ducks'] * 1.1) - (df['Count_Zero_Wkt'] * 1.1)
    )
    
    # --- 4. AR DISQUALIFICATION ---
    # Global rule: Must have >0 Wickets AND >50 Runs to be an All-Rounder
    # Otherwise, points are voided so they fall back to Bat/Bowl
    mask_not_ar = (df['Total_Wickets'] == 0) | (df['Total_Runs'] < 50)
    df.loc[mask_not_ar, 'Pts_AllRounder'] = -1.0

    # --- 5. ASSIGN ROLE ---
    df['Role'] = df.apply(determine_primary_role, axis=1)
    
    return df

# --- 4. RANKING ENGINE (Role-Specific) ---
@st.cache_data
def get_player_ranks(selected_players):
    """
    Calculates global ranks but ONLY ranks players against others with the SAME Role.
    """
    conn = sqlite3.connect(DB_FILE)
    # Fetch ALL data to build global leaderboards
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
    df_global = pd.read_sql_query(global_query, conn)
    conn.close()
    
    if df_global.empty: return {}

    # 1. Calculate Points & Assign Roles for EVERYONE
    df_calc = calculate_points_and_roles(df_global)

    # 2. Filter Lists by Role (Exclusive Lists)
    # A player only exists in one of these three dataframes
    bat_board = df_calc[df_calc['Role'] == 'Batsman'].sort_values('Pts_Batting', ascending=False).reset_index(drop=True)
    bat_board['Rank'] = bat_board.index + 1
    
    bowl_board = df_calc[df_calc['Role'] == 'Bowler'].sort_values('Pts_Bowling', ascending=False).reset_index(drop=True)
    bowl_board['Rank'] = bowl_board.index + 1
    
    ar_board = df_calc[df_calc['Role'] == 'All-Rounder'].sort_values('Pts_AllRounder', ascending=False).reset_index(drop=True)
    ar_board['Rank'] = ar_board.index + 1

    # 3. Lookup Selected Players
    results = {}
    
    for p in selected_players:
        # Find the player's row in the master calculated df
        p_row = df_calc[df_calc['Player_Name'] == p]
        
        if p_row.empty:
            results[p] = {"Rank": "-", "Role": "N/A", "Points": 0}
            continue
            
        role = p_row['Role'].values[0]
        
        # Find their rank in their SPECIFIC role list
        if role == 'Batsman':
            rank_info = bat_board[bat_board['Player_Name'] == p]
            points = p_row['Pts_Batting'].values[0]
        elif role == 'Bowler':
            rank_info = bowl_board[bowl_board['Player_Name'] == p]
            points = p_row['Pts_Bowling'].values[0]
        else:
            rank_info = ar_board[ar_board['Player_Name'] == p]
            points = p_row['Pts_AllRounder'].values[0]
            
        if not rank_info.empty:
            rank_val = rank_info['Rank'].values[0]
            results[p] = {"Rank": f"#{rank_val}", "Role": role, "Points": int(points)}
        else:
            results[p] = {"Rank": "?", "Role": role, "Points": int(points)}
            
    return results

# --- 5. TRAJECTORY ENGINE ---
def get_trajectories(players):
    placeholders = ','.join([f"'{p}'" for p in players])
    
    q_bat = f"""
    WITH Innings AS (
        SELECT Player_Name, Runs_Scored, 
        ROW_NUMBER() OVER(PARTITION BY Player_Name ORDER BY Match_ID) as N,
        SUM(Runs_Scored) OVER(PARTITION BY Player_Name ORDER BY Match_ID) as CumRuns,
        SUM(Innings_Out) OVER(PARTITION BY Player_Name ORDER BY Match_ID) as CumOut
        FROM player_stats WHERE Player_Name IN ({placeholders})
    )
    SELECT Player_Name, N, 
    CASE WHEN CumOut=0 THEN CumRuns ELSE ROUND(CAST(CumRuns AS FLOAT)/CumOut, 2) END as Value,
    'Batting Average' as Type
    FROM Innings ORDER BY Player_Name, N
    """
    
    q_bowl = f"""
    WITH Innings AS (
        SELECT Player_Name, Wickets_Taken,
        ROW_NUMBER() OVER(PARTITION BY Player_Name ORDER BY Match_ID) as N,
        SUM(Wickets_Taken) OVER(PARTITION BY Player_Name ORDER BY Match_ID) as CumWickets,
        SUM(Runs_Conceded) OVER(PARTITION BY Player_Name ORDER BY Match_ID) as CumRuns
        FROM player_stats WHERE Player_Name IN ({placeholders}) AND Overs_Balled > 0
    )
    SELECT Player_Name, N, 
    CASE WHEN CumWickets=0 THEN 0 ELSE ROUND(CAST(CumRuns AS FLOAT)/CumWickets, 2) END as Value,
    'Bowling Average' as Type
    FROM Innings ORDER BY Player_Name, N
    """
    return run_query(q_bat), run_query(q_bowl)

# --- 6. STATS FETCHING ---
def fetch_comparison_data(players):
    placeholders = ','.join([f"'{p}'" for p in players])
    
    q_bat = f"""
    SELECT Player_Name, COUNT(Match_ID) as Matches, SUM(Runs_Scored) as Runs, MAX(Runs_Scored) as HS,
    SUM(Is_MoM) as MoM,
    CASE WHEN SUM(Innings_Out)=0 THEN SUM(Runs_Scored) ELSE ROUND(CAST(SUM(Runs_Scored) AS FLOAT)/SUM(Innings_Out), 2) END as Bat_Avg,
    CASE WHEN SUM(Balls_Faced)=0 THEN 0 ELSE ROUND(CAST(SUM(Runs_Scored) AS FLOAT)/SUM(Balls_Faced)*100, 2) END as SR
    FROM player_stats WHERE Player_Name IN ({placeholders}) GROUP BY Player_Name
    """
    df_bat = run_query(q_bat)

    q_bowl = f"""
    WITH BestFigures AS (
        SELECT Player_Name, Wickets_Taken as BW, Runs_Conceded as BR,
        ROW_NUMBER() OVER(PARTITION BY Player_Name ORDER BY Wickets_Taken DESC, Runs_Conceded ASC) as Rank
        FROM player_stats WHERE Overs_Balled > 0 AND Player_Name IN ({placeholders})
    )
    SELECT p.Player_Name, SUM(p.Wickets_Taken) as Wickets, bf.BW || '/' || bf.BR as Best_Bowl,
    CASE WHEN SUM(p.Wickets_Taken)=0 THEN 0 ELSE ROUND(CAST(SUM(p.Runs_Conceded) AS FLOAT)/SUM(p.Wickets_Taken), 2) END as Bowl_Avg,
    CASE WHEN SUM(p.Total_Balls_Bowled)=0 THEN 0 ELSE ROUND(CAST(SUM(p.Runs_Conceded) AS FLOAT)/(SUM(p.Total_Balls_Bowled)/6.0), 2) END as Econ
    FROM player_stats p LEFT JOIN BestFigures bf ON p.Player_Name = bf.Player_Name AND bf.Rank = 1
    WHERE p.Player_Name IN ({placeholders}) AND p.Overs_Balled > 0 GROUP BY p.Player_Name
    """
    df_bowl = run_query(q_bowl)

    if not df_bat.empty:
        # Merge Batting and Bowling Data
        final_df = df_bat.merge(df_bowl, on='Player_Name', how='outer').fillna(0)
        # Fix string formatting for missing bowl stats
        final_df['Best_Bowl'] = final_df['Best_Bowl'].replace(0, "-")
        return final_df
    return pd.DataFrame()

# --- 7. VISUALIZATIONS ---

def plot_radar_chart(df):
    """Matplotlib Radar Chart with Normalized Stats"""
    # Metric Calculations
    df['WPM'] = df.apply(lambda x: x['Wickets'] / x['Matches'] if x['Matches'] > 0 else 0, axis=1)

    categories = ['Bat Avg', 'Strike Rate', 'Runs', 'Wickets', 'Wkts/Match']
    metric_cols = ['Bat_Avg', 'SR', 'Runs', 'Wickets', 'WPM']
    
    # Normalize
    plot_df = pd.DataFrame()
    plot_df['Player'] = df['Player_Name']
    
    for col, metric in zip(metric_cols, categories):
        max_val = df[col].max()
        if max_val == 0: plot_df[metric] = 0
        else: plot_df[metric] = df[col] / max_val

    # Radar Setup
    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    colors = ['#00E5FF', '#FF007F', '#FFD700', '#00FF00'] 
    
    for i, row in plot_df.iterrows():
        values = row[categories].values.flatten().tolist()
        values += values[:1]
        color = colors[i % len(colors)]
        ax.plot(angles, values, linewidth=2, linestyle='solid', label=row['Player'], color=color)
        ax.fill(angles, values, color=color, alpha=0.25)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=10, fontweight='bold', color='white')
    ax.set_yticklabels([])
    ax.spines['polar'].set_visible(False)
    ax.grid(color='#333333', linestyle='--')
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), facecolor='#1c1c1c', edgecolor='white')
    st.pyplot(fig)

def plot_batting_dual_axis(df):
    base = alt.Chart(df).encode(x=alt.X('Player_Name', axis=None))
    bar = base.mark_bar(width=40, color='#4CAF50').encode(y=alt.Y('Bat_Avg', title='Avg', axis=alt.Axis(titleColor='#4CAF50')), tooltip=['Player_Name', 'Bat_Avg'])
    point = base.mark_circle(size=150, color='#FF5722', opacity=1).encode(y=alt.Y('SR', title='SR', axis=alt.Axis(titleColor='#FF5722')), tooltip=['Player_Name', 'SR'])
    st.altair_chart(alt.layer(bar, point).resolve_scale(y='independent').properties(title="Batting Quality: Avg vs SR"), use_container_width=True)

def plot_trajectory_lines(df, title, y_label, invert=False):
    if df.empty: st.info(f"No {title} data."); return
    scale = alt.Scale(reverse=True) if invert else alt.Scale()
    chart = alt.Chart(df).mark_line(point=True).encode(
        x=alt.X('N', title='Innings'),
        y=alt.Y('Value', title=y_label, scale=scale),
        color='Player_Name', tooltip=['Player_Name', 'N', alt.Tooltip('Value', format='.2f')]
    ).properties(title=title, height=350).interactive()
    st.altair_chart(chart, use_container_width=True)

# --- 8. MAIN APP ---
def app():
    st.title("🆚 Player Comparison")
    
    all_players = get_all_players()
    if not all_players: st.stop()

    # Default Selection Logic
    default_sel = all_players[:2] if len(all_players)>=2 else all_players[:1]
    selected = st.multiselect("Select 2-4 Players", all_players, default=default_sel, max_selections=4)

    if len(selected) < 2: 
        st.info("Select at least 2 players to start comparison.")
        st.stop()

    if st.button("Compare", type="primary"):
        # 1. Fetch Ranks
        results = get_player_ranks(selected)
        
        # Display Leaderboard
        st.subheader("🏆 Role-Based Rankings")
        cols = st.columns(len(selected))
        for idx, p in enumerate(selected):
            data = results.get(p, {})
            # Rank (e.g., #1), Label (e.g., Batsman), Value (e.g., 1500 pts)
            cols[idx].metric(
                label=f"{p} ({data.get('Role', 'N/A')})", 
                value=data.get('Rank', '-'), 
                delta=f"{data.get('Points', 0)} pts"
            )
        st.divider()

        # 2. Fetch Stats
        df = fetch_comparison_data(selected)
        if df.empty: st.error("No data."); st.stop()

        # Stats Table
        st.subheader("📋 Head-to-Head Stats")
        cols_show = ['Player_Name', 'Matches', 'Runs', 'HS', 'Bat_Avg', 'SR', 'MoM', 'Wickets', 'Best_Bowl', 'Bowl_Avg', 'Econ']
        
        # Safe column selection
        available_cols = [c for c in cols_show if c in df.columns]
        df_display = df[available_cols].rename(columns={'Bat_Avg': 'BA', 'Best_Bowl': 'Best Bowl', 'Bowl_Avg': 'Bowl Avg'})
        
        format_dict = {'BA':"{:.1f}", 'SR':"{:.1f}", 'Bowl Avg':"{:.1f}", 'Econ':"{:.1f}"}
        
        st.dataframe(
            df_display.style
            .format(format_dict)
            .highlight_max(axis=0, color='#1f4e3d', subset=[c for c in ['Runs','BA','SR','Wickets'] if c in df_display.columns]), 
            use_container_width=True, 
            hide_index=True
        )
        st.divider()

        # 3. Visuals
        st.subheader("📊 Visual Battle")
        c1, c2 = st.columns(2)
        with c1: plot_radar_chart(df)
        with c2:
            st.markdown("#### 🏏 Batting Quality")
            plot_batting_dual_axis(df)

        st.divider()

        # 4. Trajectories
        st.subheader("📈 Career Trajectories")
        df_traj_bat, df_traj_bowl = get_trajectories(selected)
        
        t1, t2 = st.tabs(["🏏 Batting Progression", "⚾ Bowling Progression"])
        with t1: plot_trajectory_lines(df_traj_bat, "Cumulative Batting Average", "Batting Avg")
        with t2: plot_trajectory_lines(df_traj_bowl, "Cumulative Bowling Average", "Bowling Avg", invert=True)

if __name__ == "__main__":
    app()