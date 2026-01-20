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
    root_dir = os.path.dirname(script_dir)
    db_path = os.path.join(root_dir, "cricket_data.db")
    return db_path

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

# --- 3. POINTS CALCULATION (Exact Logic) ---
def calculate_points(df):
    """Applies standard points logic for Role determination and Ranking"""
    # Batting
    df['Bat_Avg'] = np.where(df['Total_Outs'] > 0, df['Total_Runs'] / df['Total_Outs'], df['Total_Runs'])
    df['Bat_SR'] = np.where(df['Total_Balls_Faced'] > 0, (df['Total_Runs'] / df['Total_Balls_Faced']) * 100, 0.0)
    
    # Fill missing columns
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

# --- 4. RANKING ENGINE (Role-Specific) ---
@st.cache_data
def get_player_ranks(selected_players):
    """
    1. Calculates points for EVERY player in the DB.
    2. Builds 3 separate leaderboards (Batting, Bowling, AR).
    3. For selected players, finds their best role and returns Rank + Points for THAT role.
    """
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
    df_global = pd.read_sql_query(global_query, conn)
    conn.close()
    
    if df_global.empty: return {}

    # Calculate Points
    df_calc = calculate_points(df_global)
    df_calc.loc[df_calc['Total_Wickets'] < 1, 'Pts_AllRounder'] = -9999.0

    # Create 3 Independent Leaderboards
    bat_board = df_calc.sort_values('Pts_Batting', ascending=False).reset_index(drop=True)
    bat_board['Rank'] = bat_board.index + 1
    
    bowl_board = df_calc.sort_values('Pts_Bowling', ascending=False).reset_index(drop=True)
    bowl_board['Rank'] = bowl_board.index + 1
    
    ar_board = df_calc.sort_values('Pts_AllRounder', ascending=False).reset_index(drop=True)
    ar_board['Rank'] = ar_board.index + 1

    results = {}
    
    for p in selected_players:
        # Get player stats
        p_row = df_calc[df_calc['Player_Name'] == p]
        if p_row.empty:
            results[p] = {"Rank": "-", "Role": "N/A", "Points": 0}
            continue
            
        # Determine Primary Role based on Max Points
        scores = {
            'Batsman': p_row['Pts_Batting'].values[0],
            'Bowler': p_row['Pts_Bowling'].values[0],
            'All-Rounder': p_row['Pts_AllRounder'].values[0]
        }
        best_role = max(scores, key=scores.get)
        best_points = scores[best_role]
        
        # Look up rank in the SPECIFIC leaderboard for that role
        if best_role == 'Batsman':
            rank = bat_board[bat_board['Player_Name'] == p]['Rank'].values[0]
        elif best_role == 'Bowler':
            rank = bowl_board[bowl_board['Player_Name'] == p]['Rank'].values[0]
        else:
            rank = ar_board[ar_board['Player_Name'] == p]['Rank'].values[0]
            
        results[p] = {"Rank": f"#{rank}", "Role": best_role, "Points": int(best_points)}
            
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
        final_df = df_bat.merge(df_bowl, on='Player_Name', how='outer').fillna(0)
        final_df['Best_Bowl'] = final_df['Best_Bowl'].replace(0, "-")
        return final_df
    return pd.DataFrame()

# --- 7. VISUALIZATIONS ---

def plot_radar_chart(df):
    """
    Matplotlib Radar Chart with Wickets Per Match (Positive Slope)
    """
    # 1. Create Derived Metric: Wickets Per Match
    # Avoid division by zero
    df['WPM'] = df.apply(lambda x: x['Wickets'] / x['Matches'] if x['Matches'] > 0 else 0, axis=1)

    categories = ['Bat Avg', 'Strike Rate', 'Runs', 'Wickets', 'Wkts/Match']
    metric_cols = ['Bat_Avg', 'SR', 'Runs', 'Wickets', 'WPM']
    
    # 2. Normalize Data (0-1 Scale)
    plot_df = pd.DataFrame()
    plot_df['Player'] = df['Player_Name']
    
    for col, metric in zip(metric_cols, categories):
        max_val = df[col].max()
        # If max is 0 (e.g. no wickets taken by anyone), set scale to 1 to avoid div/0
        if max_val == 0: 
            plot_df[metric] = 0
        else:
            plot_df[metric] = df[col] / max_val

    # 3. Setup Radar
    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1] # Close the loop
    
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    
    # Custom colors
    colors = ['#00E5FF', '#FF007F', '#FFD700', '#00FF00'] 
    
    for i, row in plot_df.iterrows():
        values = row[categories].values.flatten().tolist()
        values += values[:1] # Close loop
        
        color = colors[i % len(colors)]
        ax.plot(angles, values, linewidth=2, linestyle='solid', label=row['Player'], color=color)
        ax.fill(angles, values, color=color, alpha=0.25)

    # 4. Styling
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=10, fontweight='bold', color='white')
    ax.set_yticklabels([]) # Hide radial grid numbers
    ax.spines['polar'].set_visible(False)
    ax.grid(color='#333333', linestyle='--')
    
    # Legend
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

    selected = st.multiselect("Select 2-4 Players", all_players, default=all_players[:2] if len(all_players)>=2 else all_players[:1], max_selections=4)

    if len(selected) < 2: st.info("Select at least 2 players."); st.stop()

    if st.button("Compare", type="primary"):
        # 1. Fetch Ranks & Points (Role Specific)
        results = get_player_ranks(selected)
        
        # Display Leaderboard
        st.subheader("🏆 Role-Based Rankings")
        cols = st.columns(len(selected))
        for idx, p in enumerate(selected):
            data = results.get(p, {})
            # Display: Rank (Big), Role (Label), Points (Delta)
            cols[idx].metric(
                label=f"{p} ({data['Role']})", 
                value=data['Rank'], 
                delta=f"{data['Points']} pts"
            )
        st.divider()

        # 2. Fetch Stats
        df = fetch_comparison_data(selected)
        if df.empty: st.error("No data."); st.stop()

        # Stats Table
        st.subheader("📋 Head-to-Head Stats")
        cols_show = ['Player_Name', 'Matches', 'Runs', 'HS', 'Bat_Avg', 'SR', 'MoM', 'Wickets', 'Best_Bowl', 'Bowl_Avg', 'Econ'];df_display = df[cols_show].rename(columns={'Bat_Avg': 'BA','Best_Bowl': 'Best Bowl','Bowl_Avg': 'Bowl Avg'})
        format_dict = {'BA':"{:.1f}",'SR':"{:.1f}",'Bowl Avg':"{:.1f}",'Econ':"{:.1f}" }
        st.dataframe(df_display.style.format(format_dict).highlight_max(axis=0, color='#1f4e3d', subset=['Runs','BA','SR','Wickets']), use_container_width=True, hide_index=True)
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
        with t2: plot_trajectory_lines(df_traj_bowl, "Cumulative Bowling Average (Lower is Better)", "Bowling Avg", invert=True)

if __name__ == "__main__":
    app()