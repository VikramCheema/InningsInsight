import streamlit as st
import pandas as pd
import sqlite3
import plotly.express as px
import plotly.graph_objects as go
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
    for entry in results_list:
        res = entry['res'] # Extract the 'W' or 'L' from the dictionary
        if res == 'W':
            cur_win += 1; cur_loss = 0; max_win = max(max_win, cur_win)
        elif res == 'L':
            cur_loss += 1; cur_win = 0; max_loss = max(max_loss, cur_loss)
        else: 
            cur_win = 0; cur_loss = 0
    return max_win, max_loss

# --- 3. DATA PROCESSING ---
@st.cache_data
def get_matches_data():
    if not DB_FILE: return pd.DataFrame(), pd.DataFrame()
    
    conn = sqlite3.connect(DB_FILE)
    try:
        check = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table' AND name='innings_summary'", conn)
        if check.empty:
            return pd.DataFrame(), pd.DataFrame()

        # Added Tournament_ID to the SELECT statement
        df_inn = pd.read_sql_query("""
            SELECT Match_ID, Tournament_ID, Team_Name, Total_Runs, Total_Wickets_Lost, Winner 
            FROM innings_summary 
            ORDER BY Match_ID
        """, conn)
        
        df_bowl = pd.read_sql_query("""
            SELECT Match_ID, Team_Name, SUM(Total_Balls_Bowled) as Balls_Delivered 
            FROM player_stats 
            GROUP BY Match_ID, Team_Name
        """, conn)
        
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
        
        # Initialize team data if not exists
        for t in [t1, t2]:
            if t not in teams: 
                teams[t] = {
                    'Results':[], 'Played':0, 'Won':0, 'Lost':0, 
                    'Points':0, 'Runs_For':0, 'Balls_Faced':0, 
                    'Runs_Agst':0, 'Balls_Bowled':0
                }
        
        teams[t1]['Played'] += 1; teams[t2]['Played'] += 1
        winner = determine_winner(t1, t2, t1_row['Winner'])
        
        # Update Results with Dictionaries for Interactive Tooltips
        if winner == t1:
            teams[t1]['Results'].append({'res': 'W', 'opp': t2})
            teams[t1]['Won'] += 1; teams[t1]['Points'] += 2
            
            teams[t2]['Results'].append({'res': 'L', 'opp': t1})
            teams[t2]['Lost'] += 1; teams[t2]['Points'] -= 1
        elif winner == t2:
            teams[t2]['Results'].append({'res': 'W', 'opp': t1})
            teams[t2]['Won'] += 1; teams[t2]['Points'] += 2
            
            teams[t1]['Results'].append({'res': 'L', 'opp': t2})
            teams[t1]['Lost'] += 1; teams[t1]['Points'] -= 1
        else:
            # Handle Ties or No Results
            teams[t1]['Results'].append({'res': 'T', 'opp': t2})
            teams[t2]['Results'].append({'res': 'T', 'opp': t1})

        # --- NRR Calculation Logic ---
        def get_balls(bowling_team):
            r = df_bowl[(df_bowl['Match_ID'] == match_id) & (df_bowl['Team_Name'] == bowling_team)]
            return int(r['Balls_Delivered'].iloc[0]) if not r.empty else 0

        # Team 1 Batting Stats
        t1_balls = (MAX_OVERS*6) if t1_row['Total_Wickets_Lost'] == 10 else get_balls(t2)
        teams[t1]['Runs_For'] += t1_row['Total_Runs']; teams[t1]['Balls_Faced'] += t1_balls
        teams[t2]['Runs_Agst'] += t1_row['Total_Runs']; teams[t2]['Balls_Bowled'] += t1_balls
        
        # Team 2 Batting Stats
        t2_balls = (MAX_OVERS*6) if t2_row['Total_Wickets_Lost'] == 10 else get_balls(t1)
        teams[t2]['Runs_For'] += t2_row['Total_Runs']; teams[t2]['Balls_Faced'] += t2_balls
        teams[t1]['Runs_Agst'] += t2_row['Total_Runs']; teams[t1]['Balls_Bowled'] += t2_balls

    # Compile Final Ranking Data
    ranking_data = []
    for t, s in teams.items():
        # Calculate NRR components
        off = s['Runs_For'] / (s['Balls_Faced']/6) if s['Balls_Faced']>0 else 0
        defn = s['Runs_Agst'] / (s['Balls_Bowled']/6) if s['Balls_Bowled']>0 else 0
        
        # Calculate Streaks (Make sure to update calculate_streaks helper to read entry['res'])
        w_strk, l_strk = calculate_streaks(s['Results'])
        
        ranking_data.append({
            'Team': t, 
            'Mat': s['Played'], 
            'Won': s['Won'], 
            'Lost': s['Lost'], 
            'Pts': s['Points'],
            'NRR': off - defn, 
            'Results': s['Results'], 
            'Win_Streak': w_strk, 
            'Loss_Streak': l_strk
        })
        
    # Sort by Points, then NRR
    df_rank = pd.DataFrame(ranking_data).sort_values(
        by=['Pts', 'NRR'], 
        ascending=[False, False]
    ).reset_index(drop=True)
    
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
        
        t1_row = match_data.iloc[0]
        t1, t2 = t1_row['Team_Name'], match_data.iloc[1]['Team_Name']
        winner = determine_winner(t1, t2, t1_row['Winner'])
        
        # --- Metadata Extraction ---
        is_playing = (target_team == t1 or target_team == t2)
        match_opp = "N/A"
        match_res = "DNP"
        tourn_id = t1_row.get('Tournament_ID', 'N/A') # Get the Tournament ID

        if is_playing:
            match_opp = t2 if target_team == t1 else t1
            match_res = "W" if winner == target_team else "L" if winner else "T/NR"
        # -------------------------------------------

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
            history.append({
                'Match_Num': match_counter, 
                'Rank': target_row['Rank'].values[0],
                'NRR': target_row['NRR'].values[0], 
                'Played': is_playing,
                'Opponent': match_opp,
                'Result': match_res,
                'Tournament': tourn_id # New key added here
            })
            
    return pd.DataFrame(history)

# --- 5. VISUALS ---
def plot_form_guide_plotly(df):
    plot_data = []
    for _, row in df.iterrows():
        # results are now [{'res': 'W', 'opp': 'PAK'}, ...]
        recent = row['Results'][-15:] 
        for idx, entry in enumerate(recent):
            plot_data.append({
                'Team': row['Team'],
                'Match_Order': idx + 1,
                'Result': entry['res'],
                'Opponent': entry['opp'],
                'Status': 'Win' if entry['res'] == 'W' else 'Loss' if entry['res'] == 'L' else 'Tie/NR'
            })
    
    pdf = pd.DataFrame(plot_data)
    
    fig = px.scatter(
        pdf, x="Match_Order", y="Team", color="Status", text="Result",
        # Pass Opponent into the hover data
        custom_data=["Opponent"], 
        color_discrete_map={'Win': '#2ecc71', 'Loss': '#e74c3c', 'Tie/NR': '#95a5a6'},
        category_orders={"Team": df['Team'].tolist()}
    )

    fig.update_traces(
        marker=dict(size=28),
        textfont=dict(color='white', size=9, family="Arial Black"),
        # Update the template to show the Opponent
        hovertemplate="<b>Vs: %{customdata[0]}</b><br><extra></extra>"
    )

    fig.update_layout(
        annotations=[
            dict(
                x=1, y=-0.12, # Position at the bottom right
                xref="paper", yref="paper",
                text="Most Recent Match",
                showarrow=True,
                arrowhead=2,
                ax=-100, ay=0, # Points the arrow to the right
                font=dict(size=12, color="gray"),
                align="right"
            )
        ],
        xaxis=dict(
            title=dict(
                text="⬅ Older matches . . . . . . Recent matches ➡",
                font=dict(size=14, family="Arial Black", color="gray")
            ),
            showgrid=False,
            showticklabels=True,
            tickmode='array',
            tickvals=[1, 15],
            ticktext=['15 Matches Ago', 'Latest'], # Clear labels on the ends
            range=[0.5, 15.5],
            fixedrange=True
        ),
        template="plotly_white" if st.get_option("theme.base") == "light" else "plotly_dark",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        # xaxis=dict(showgrid=False, showticklabels=False, title=""),
        yaxis=dict(title="", autorange="reversed"),
        showlegend=False
    )
    
    st.plotly_chart(fig, use_container_width=True, height = 600)

def plot_history_plotly(df, team):
    if df.empty: return
    fig = go.Figure()

    # 1. NEUTRAL SHADING
    # Using a mid-gray with low alpha works on both light and dark backgrounds
    for i in range(len(df)):
        if df.iloc[i]['Played']:
            fig.add_vrect(
                x0=df.iloc[i]['Match_Num'] - 0.45, 
                x1=df.iloc[i]['Match_Num'] + 0.45,
                fillcolor="gray", 
                opacity=0.15, # Neutral opacity
                layer="below", 
                line_width=0,
            )

    # 2. UNIVERSAL ACCENT COLORS
    # Primary (Rank): Teal/Cyan
    fig.add_trace(go.Scatter(
        x=df['Match_Num'], y=df['Rank'],
        name="Global Rank",
        mode='lines+markers',
        line=dict(color='#008080', width=3),
        marker=dict(size=8),
        customdata=df[['Opponent', 'Result', 'Tournament', 'NRR']], # Added NRR to customdata
        hovertemplate=(
            "<i>%{customdata[2]}</i><br><br>" +
            "<b>Vs:</b> %{customdata[0]}<br>" +
            "<b>Result:</b> %{customdata[1]}<br>" +
            "<b>Rank:</b> %{y}<br>" +
            "<b>NRR:</b> %{customdata[3]:.3f}" + # Show NRR here too
            "<extra></extra>"
        )
    ))

    # Secondary (NRR): Burnt Orange
    fig.add_trace(go.Scatter(
        x=df['Match_Num'], y=df['NRR'],
        name="NRR",
        mode='lines+markers',
        line=dict(color='#D35400', width=2, dash='dot'), # Universal Burnt Orange
        yaxis="y2",
        hoverinfo = 'skip'
    ))
    

    fig.update_layout(
        # template=None allows Streamlit to inject its own theme colors (Light/Dark)
        template="plotly_white" if st.get_option("theme.base") == "light" else "plotly_dark",
        hovermode="closest",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(title=dict(text="Match Number"), dtick=1, range=[0.5, df['Match_Num'].max() + 0.5]),
        yaxis=dict(
            title=dict(text="Rank", font=dict(color='#008080')),
            tickfont=dict(color='#008080'),
            autorange="reversed",
            dtick=1
        ),
        yaxis2=dict(
            title=dict(text="Net Run Rate", font=dict(color='#D35400')),
            tickfont=dict(color='#D35400'),
            anchor="x", overlaying="y", side="right", showgrid = False, zeroline = False
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="center", x=0.5)
    )
    
    # Zero line for NRR - semi-transparent gray
    fig.add_hline(y=0, line_dash="solid", line_color="gray", opacity=0.3, yref="y2")

    st.plotly_chart(fig, use_container_width=True, height = 500)


def highlight_podium(row):
    """Applies Olympic colors to the first three ranks."""
    # Define colors with low opacity for readability
    gold = 'background-color: rgba(255, 215, 0, 0.3)'   # Gold
    silver = 'background-color: rgba(192, 192, 192, 0.3)' # Silver
    bronze = 'background-color: rgba(205, 127, 50, 0.3)'  # Bronze
    default = ''

    if row['Rank'] == 1:
        return [gold] * len(row)
    elif row['Rank'] == 2:
        return [silver] * len(row)
    elif row['Rank'] == 3:
        return [bronze] * len(row)
    else:
        return [default] * len(row)
# --- 6. APP ---
def app():
    st.title("🏆 Tournament Headquarters")
    # ... (Keep logic for DB checking and calculate_standings)

    df_rank = calculate_standings()
    
    tab1, tab2 = st.tabs(["📊 Points Table & Form", "📈 Team Trajectory"])
    with tab1:
        # Define the columns we want to show
        display_cols = ['Rank', 'Team', 'Mat', 'Won', 'Lost', 'Pts', 'NRR', 'Win_Streak', 'Loss_Streak']
        
        # Apply the formatting and custom podium highlighting
        styled_df = df_rank[display_cols].style \
            .format({'NRR': "{:+.3f}"}) \
            .apply(highlight_podium, axis=1) # axis=1 applies it row-wise
            
        st.dataframe(styled_df, use_container_width=True, hide_index=True)
        st.divider()
        st.subheader("📈 Team Form")
        plot_form_guide_plotly(df_rank)

    with tab2:
        teams = sorted(df_rank['Team'].unique())
        sel_team = st.selectbox("Select Team to Trace", teams)
        if sel_team:
            df_hist = get_team_history(sel_team)
            if not df_hist.empty:
                # (Keep metrics row code here)
                plot_history_plotly(df_hist, sel_team)

if __name__ == "__main__":
    app()