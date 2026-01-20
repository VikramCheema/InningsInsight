import streamlit as st
import pandas as pd
import sqlite3
import plotly.express as px
import os
import numpy as np

# --- CONFIGURATION ---
st.set_page_config(page_title="World Cup 2026 Headquarters", page_icon="🏆", layout="wide")
MAX_OVERS = 10 
TARGET_TOUR_PATTERN = "026" # Locks to World Cup 2026

# --- GROUPS ---
GROUPS = {
    "Group A": ["India", "West Indies", "Bangladesh", "New Zealand", "England"],
    "Group B": ["Australia", "South Africa", "Pakistan", "Sri Lanka", "Afghanistan"]
}

FULL_NAMES = {
    "IND": "India", "WI": "West Indies", "BAN": "Bangladesh", "NZ": "New Zealand", "ENG": "England",
    "AUS": "Australia", "SA": "South Africa", "PAK": "Pakistan", "SL": "Sri Lanka", "AFG": "Afghanistan"
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
    """Fetches headline stats specifically for this tournament pattern"""
    if not DB_FILE: return 0, 0, 0, 0
    conn = sqlite3.connect(DB_FILE)
    try:
        # Filter everything by TARGET_TOUR_PATTERN
        m = pd.read_sql(f"SELECT count(DISTINCT Match_ID) as val FROM innings_summary WHERE Match_ID LIKE '{TARGET_TOUR_PATTERN}%'", conn).iloc[0]['val']
        v = pd.read_sql(f"SELECT count(DISTINCT Venue) as val FROM innings_summary WHERE Match_ID LIKE '{TARGET_TOUR_PATTERN}%'", conn).iloc[0]['val']
        f = pd.read_sql(f"SELECT count(*) as val FROM player_stats WHERE Match_ID LIKE '{TARGET_TOUR_PATTERN}%' AND Runs_Scored >= 50", conn).iloc[0]['val']
        w = pd.read_sql(f"SELECT count(*) as val FROM player_stats WHERE Match_ID LIKE '{TARGET_TOUR_PATTERN}%' AND Wickets_Taken >= 3", conn).iloc[0]['val']
        return m, v, f, w
    except:
        return 0, 0, 0, 0
    finally:
        conn.close()

# --- 3. CALCULATION ENGINES ---

def determine_primary_role(row):
    """Classifies player based on highest point category"""
    bat = row['Pts_Batting']
    bowl = row['Pts_Bowling']
    ar = row['Pts_AllRounder']
    
    # Simple Max Check
    if bat >= bowl and bat >= ar:
        return "Batsman"
    elif bowl >= bat and bowl >= ar:
        return "Bowler"
    else:
        return "All-Rounder"

def calculate_impact_points(df):
    # Batting
    df['Bat_Avg'] = np.where(df['Total_Outs'] > 0, df['Total_Runs'] / df['Total_Outs'], df['Total_Runs'])
    df['Bat_SR'] = np.where(df['Total_Balls_Faced'] > 0, (df['Total_Runs'] / df['Total_Balls_Faced']) * 100, 0.0)
    
    for col in ['Total_MoMs', 'Count_Ducks', 'Count_5W', 'Count_4W', 'Count_3W', 'Count_Zero_Wkt', 'Count_100s', 'Count_50s', 'Count_30s']:
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
    
    # Penalize AR if no contribution in one department
    df.loc[(df['Total_Wickets'] == 0) | (df['Total_Runs'] < 10), 'Pts_AllRounder'] = -999
    
    # Determine Role
    df['Role'] = df.apply(determine_primary_role, axis=1)
    return df

@st.cache_data
def get_mvp_data():
    if not DB_FILE: return pd.DataFrame()
    conn = sqlite3.connect(DB_FILE)
    
    # 1. Fetch Raw Data
    query = f"""
    SELECT Player_Name, Team_Name, Runs_Scored, Innings_Out, Balls_Faced, 
           Wickets_Taken, Runs_Conceded, Overs_Balled, Is_MoM
    FROM player_stats 
    WHERE Match_ID LIKE '{TARGET_TOUR_PATTERN}%'
    """
    df_raw = pd.read_sql_query(query, conn)
    conn.close()
    
    if df_raw.empty: return pd.DataFrame()

    # 2. Aggregations (UPDATED to include 30s, 4Ws, and Zero Wickets)
    def calc_balls(overs):
        return sum(int(o)*6 + int(round((o%1)*10)) for o in overs)

    stats = df_raw.groupby(['Player_Name', 'Team_Name']).agg(
        Total_Runs=('Runs_Scored', 'sum'),
        Total_Outs=('Innings_Out', 'sum'),
        Total_Balls_Faced=('Balls_Faced', 'sum'),
        HS=('Runs_Scored', 'max'),
        # Batting Milestones
        Count_30s=('Runs_Scored', lambda x: ((x >= 30) & (x < 50)).sum()), # Exclusive 30-49
        Count_50s=('Runs_Scored', lambda x: ((x >= 50) & (x < 100)).sum()), # Exclusive 50-99
        Count_100s=('Runs_Scored', lambda x: (x >= 100).sum()),
        Count_Ducks=('Runs_Scored', lambda x: ((x==0) & (df_raw.loc[x.index, 'Innings_Out']==1)).sum()),
        
        # Bowling Stats
        Total_Wickets=('Wickets_Taken', 'sum'),
        Total_Runs_Conceded=('Runs_Conceded', 'sum'),
        Total_Balls_Bowled=('Overs_Balled', calc_balls),
        # Bowling Milestones
        Count_Zero_Wkt=('Wickets_Taken', lambda x: (x == 0).sum()),
        Count_3W=('Wickets_Taken', lambda x: (x == 3).sum()),
        Count_4W=('Wickets_Taken', lambda x: (x == 4).sum()),
        Count_5W=('Wickets_Taken', lambda x: (x >= 5).sum()),
        Total_MoMs=('Is_MoM', 'sum')
    ).reset_index()

    # 3. Best Bowling Figures
    best_figs = df_raw.sort_values(['Wickets_Taken', 'Runs_Conceded'], ascending=[False, True]).drop_duplicates('Player_Name')
    best_figs['Best_Fig'] = best_figs['Wickets_Taken'].astype(str) + '/' + best_figs['Runs_Conceded'].astype(str)
    stats = stats.merge(best_figs[['Player_Name', 'Best_Fig']], on='Player_Name', how='left')

    # 4. Derived Metrics
    stats['Bat_Avg'] = np.where(stats['Total_Outs'] > 0, stats['Total_Runs'] / stats['Total_Outs'], stats['Total_Runs'])
    stats['Bat_SR'] = np.where(stats['Total_Balls_Faced'] > 0, (stats['Total_Runs'] / stats['Total_Balls_Faced']) * 100, 0.0)
    
    stats['Total_Overs_Precise'] = stats['Total_Balls_Bowled'] / 6.0
    stats['Bowl_Avg'] = np.where(stats['Total_Wickets'] > 0, stats['Total_Runs_Conceded'] / stats['Total_Wickets'], 0.0)
    stats['Bowl_SR'] = np.where(stats['Total_Wickets'] > 0, stats['Total_Balls_Bowled'] / stats['Total_Wickets'], 0.0)
    stats['Bowl_Econ'] = np.where(stats['Total_Overs_Precise'] > 0, stats['Total_Runs_Conceded'] / stats['Total_Overs_Precise'], 0.0)

    # 5. Impact Points Calculation (FIXED)
    
    # --- Batting Points ---
    b_sr_pts = np.where(stats['Bat_SR'] > 100, (stats['Bat_SR'] - 100)/5, 0)
    
    stats['Pts_Batting'] = (
        (stats['Total_Runs'] * 0.5) + 
        (stats['Bat_Avg'] * 0.5) +
        (b_sr_pts) +
        (stats['Count_100s']*50 + stats['Count_50s']*20 + stats['Count_30s']*10) +
        (stats['Total_MoMs']*10) - 
        (stats['Count_Ducks']*10)
    )
    
    # --- Bowling Points ---
    w_econ_pts = np.where(stats['Bowl_Econ'] < 12.0, (12.0 - stats['Bowl_Econ']) * 2, 0.0)
    
    stats['Pts_Bowling'] = (
        (stats['Total_Wickets'] * 15) + 
        (stats['Total_Overs_Precise'] * 1.0) +
        (w_econ_pts) +
        (stats['Count_5W']*30 + stats['Count_4W']*20 + stats['Count_3W']*10) +
        (stats['Total_MoMs']*10) - 
        (stats['Count_Zero_Wkt']*5)
    )

    # --- All-Rounder Points (FIXED LOGIC) ---
    stats['Pts_AllRounder'] = (
        (stats['Total_Runs'] * 1) + 
        (stats['Total_Wickets'] * 10.0) +
        (stats['Total_MoMs'] * 10.0) - 
        (stats['Count_Ducks'] * 1.1) - 
        (stats['Count_Zero_Wkt'] * 1.1)
    )
    
    # Qualification Check: Must have at least 1 wicket to be an "All-Rounder" here
    stats.loc[stats['Total_Wickets'] < 1, 'Pts_AllRounder'] = 0

    # 6. Role Classification (Re-apply helper logic)
    stats['Role'] = stats.apply(determine_primary_role, axis=1)
    
    return stats

@st.cache_data
def get_records_data():
    conn = sqlite3.connect(DB_FILE)
    
    # 1. High Scores (Added 'Vs' column)
    q_bat = f"SELECT Player_Name, Runs_Scored, Balls_Faced, Team_Name, Opposition as Vs FROM player_stats WHERE Match_ID LIKE '{TARGET_TOUR_PATTERN}%' ORDER BY Runs_Scored DESC LIMIT 10"
    df_bat = pd.read_sql_query(q_bat, conn)
    
    # 2. Best Bowling (Added 'Vs' column)
    q_bowl = f"SELECT Player_Name, Wickets_Taken, Runs_Conceded, Overs_Balled, Team_Name, Opposition as Vs FROM player_stats WHERE Match_ID LIKE '{TARGET_TOUR_PATTERN}%' AND Overs_Balled > 0 ORDER BY Wickets_Taken DESC, Runs_Conceded ASC LIMIT 10"
    df_bowl = pd.read_sql_query(q_bowl, conn)
    
    # 3. Match Results
    q_matches = f"SELECT Match_ID, Team_Name, Total_Runs, Winner, Venue FROM innings_summary WHERE Match_ID LIKE '{TARGET_TOUR_PATTERN}%' ORDER BY Match_ID"
    df_matches = pd.read_sql_query(q_matches, conn)
    
    conn.close()
    return df_bat, df_bowl, df_matches

# --- 4. PROGRESSION ENGINE ---
@st.cache_data
def generate_race_data(group_name, team_list):
    if not DB_FILE: return pd.DataFrame()
    conn = sqlite3.connect(DB_FILE)
    try:
        q_raw = f"SELECT Match_ID, Team_Name, Winner, Total_Runs, Total_Wickets_Lost FROM innings_summary WHERE Match_ID LIKE '{TARGET_TOUR_PATTERN}%' ORDER BY Match_ID"
        df_raw = pd.read_sql_query(q_raw, conn)
        q_balls = f"SELECT Match_ID, Team_Name, SUM(Total_Balls_Bowled) as Balls FROM player_stats WHERE Match_ID LIKE '{TARGET_TOUR_PATTERN}%' GROUP BY Match_ID, Team_Name"
        df_balls = pd.read_sql_query(q_balls, conn)
    finally:
        conn.close()

    if df_raw.empty: return pd.DataFrame()

    standings = {t: {'Pts': 0, 'Runs_For': 0, 'Balls_Faced': 0, 'Runs_Agst': 0, 'Balls_Bowled': 0, 'Played': 0, 'Won': 0, 'Lost': 0} for t in team_list}
    snapshots = []
    
    for t in team_list:
        snapshots.append({
            'Match_Label': "Start", 'Match_Order': 0, 'Team': t, 
            'Points': 0, 'NRR': 0.0, 'Rank': len(team_list),
            'Played': 0, 'Won': 0, 'Lost': 0
        })

    match_cnt = 0
    grouped = df_raw.groupby('Match_ID')

    for mid, mdata in grouped:
        if len(mdata) != 2: continue
        t1, t2 = mdata.iloc[0]['Team_Name'], mdata.iloc[1]['Team_Name']
        n1, n2 = FULL_NAMES.get(t1, t1), FULL_NAMES.get(t2, t2)
        
        if n1 not in team_list or n2 not in team_list: continue
        match_cnt += 1
        
        standings[n1]['Played'] += 1; standings[n2]['Played'] += 1
        winner = str(mdata.iloc[0]['Winner']).strip()
        
        if t1 in winner or n1 in winner:
            standings[n1]['Pts'] += 2; standings[n1]['Won'] += 1; standings[n2]['Lost'] += 1
        elif t2 in winner or n2 in winner:
            standings[n2]['Pts'] += 2; standings[n2]['Won'] += 1; standings[n1]['Lost'] += 1
        else:
            standings[n1]['Pts'] += 1; standings[n2]['Pts'] += 1

        def get_b(b_team):
            r = df_balls[(df_balls['Match_ID']==mid) & (df_balls['Team_Name']==b_team)]
            return int(r['Balls'].iloc[0]) if not r.empty else MAX_OVERS*6

        r1 = mdata.iloc[0]['Total_Runs']
        bf1 = (MAX_OVERS*6) if mdata.iloc[0]['Total_Wickets_Lost'] == 10 else get_b(t2)
        standings[n1]['Runs_For'] += r1; standings[n1]['Balls_Faced'] += bf1
        standings[n2]['Runs_Agst'] += r1; standings[n2]['Balls_Bowled'] += bf1
        
        r2 = mdata.iloc[1]['Total_Runs']
        bf2 = (MAX_OVERS*6) if mdata.iloc[1]['Total_Wickets_Lost'] == 10 else get_b(t1)
        standings[n2]['Runs_For'] += r2; standings[n2]['Balls_Faced'] += bf2
        standings[n1]['Runs_Agst'] += r2; standings[n1]['Balls_Bowled'] += bf2

        curr = []
        for t in team_list:
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

# --- 5. MAIN APP ---
def app():
    st.title("🏆 World Cup 2026: Headquarters")
    if not DB_FILE: st.error("Database missing."); st.stop()

    # --- TOURNAMENT FACTS (NEW) ---
    m_count, v_count, f_count, w_count = get_tournament_facts()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Matches Played", m_count)
    c2.metric("Venues Used", v_count)
    c3.metric("50+ Scores", f_count)
    c4.metric("3-Wicket Hauls", w_count)
    st.divider()

    # --- TABS ---
    tabs = st.tabs(["🎢 Group Progression", "🌟 MVP Leaderboard", "🔥 Tournament Records", "🏟️ Venue Stats"])

    # TAB 1: PROGRESSION
    with tabs[0]:
        grp = st.selectbox("Select Group", list(GROUPS.keys()))
        df_race = generate_race_data(grp, GROUPS[grp])
        
        if not df_race.empty:
            max_m = df_race['Match_Order'].max()
            final = df_race[df_race['Match_Order'] == max_m].sort_values('Rank')
            st.markdown(f"#### 📊 {grp} Final Standings")
            
            # SHOW Won/Lost
            cols_show = ['Rank', 'Team', 'Played', 'Won', 'Lost', 'Points', 'NRR']
            st.dataframe(final[cols_show].style.format({'NRR': "{:+.3f}"}).highlight_max(subset=['Points'], color='#1f4e3d'), use_container_width=True, hide_index=True)
            
            st.divider()
            
            # ANIMATION
            st.subheader("Race Chart")
            df_race['Inv_Rank'] = 6 - df_race['Rank']
            fig = px.scatter(df_race, x="Points", y="Inv_Rank", animation_frame="Match_Label", animation_group="Team", size="Points", color="Team", text="Team", range_x=[-1, df_race['Points'].max()+2], range_y=[0.5, 5.5], height=550)
            fig.update_traces(textposition='middle right', marker=dict(size=30, line=dict(width=2)))
            fig.update_layout(yaxis=dict(tickvals=[1,2,3,4,5], ticktext=['5th','4th','3rd','2nd','1st'], title="Position"), xaxis_title="Points", showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
            
            # RESTORED NRR CHART
            st.subheader("NRR Trajectory")
            fig_nrr = px.line(df_race[df_race['Match_Order'] > 0], x="Match_Order", y="NRR", color="Team", markers=True, height=400,labels={"Match_Order": "Match Number", "NRR": "NRR"})
            st.plotly_chart(fig_nrr, use_container_width=True)
        else:
            st.warning("No data for this group.")

    # TAB 2: MVP
    with tabs[1]:
        df_mvp = get_mvp_data()
        
        if not df_mvp.empty:
            # --- SECTION 1: BATSMEN ---
            st.markdown("### 🏏 Top Batsmen")
            st.caption("Ranking based on Impact Points (Runs + Avg + SR + Milestones)")
            
            best_bat = df_mvp[df_mvp['Role'] == 'Batsman'].sort_values('Pts_Batting', ascending=False).head(10)
            
            # Select & Rename Columns
            cols_bat = {
                'Player_Name': 'Player', 'Team_Name': 'Team', 
                'Total_Runs': 'Runs', 'Bat_Avg': 'Avg', 'HS': 'High Score',
                'Bat_SR': 'SR', 'Count_50s': '50s', 'Pts_Batting': 'Pts'
            }
            show_bat = best_bat[cols_bat.keys()].rename(columns=cols_bat)
            
            # Format Formatting
            st.dataframe(
                show_bat.style.format({'Avg': "{:.2f}", 'SR': "{:.1f}", 'Pts': "{:.0f}"})
                .background_gradient(subset=['Pts'], cmap='Greens'),
                use_container_width=True, hide_index=True
            )
            
            st.divider()

            # --- SECTION 2: BOWLERS ---
            st.markdown("### ⚾ Top Bowlers")
            st.caption("Ranking based on Impact Points (Wickets + Econ + SR + Milestones)")
            
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

            # --- SECTION 3: ALL-ROUNDERS ---
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
            st.info("No stats available for this tournament yet.")

    # TAB 3: RECORDS
    with tabs[2]:
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
            
            # Check if Team 1 (Batting First) won
            if t1_name in winner or (t1_full and t1_full in winner):
                run_margin = int(inn1['Total_Runs']) - int(inn2['Total_Runs'])
                defends.append({'Match': f"{t1_name} vs {inn2['Team_Name']}",'Winner': t1_name,'Score': int(inn1['Total_Runs']),'Margin': f"Won by {run_margin} runs"})
            else:
                # Chasing team won
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
    with tabs[3]:
        if not df_matches.empty:
            venue_stats = []
            for venue, data in df_matches.groupby('Venue'):
                matches = data['Match_ID'].nunique()
                bat_1_wins, bat_2_wins = 0, 0
                for _, m in data.groupby('Match_ID'):
                    if len(m) != 2: continue
                    winner = str(m.iloc[0]['Winner']).strip()
                    t1 = m.iloc[0]['Team_Name']; t2 = m.iloc[1]['Team_Name']
                    t1_full = FULL_NAMES.get(t1, ""); t2_full = FULL_NAMES.get(t2, "")
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