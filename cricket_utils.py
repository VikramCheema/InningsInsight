import pandas as pd
import numpy as np
import os


########################### GET DB PATH ####################################
# def get_db_path():
#     """Locates the database file in the root directory relative to the script."""
#     script_dir = os.path.dirname(os.path.abspath(__file__))
#     root_dir = os.path.dirname(script_dir)
#     # The schema specifies 'cricket data.db' but code uses 'cricket_data.db'
#     # Checking for the version that works with your Deep Dive app
#     db_path = os.path.join(root_dir, "cricket_data.db")
#     if os.path.exists(db_path):
#         return db_path
#     return "cricket_data.db" 

def get_db_path():
    """Locates the database file in the root directory relative to the script."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(script_dir)
 
    db_path = os.path.join(root_dir, "cricket_data.db")
    if os.path.exists(db_path):
        return db_path
   
    grandparent_dir = os.path.dirname(root_dir)
    db_path_upper = os.path.join(grandparent_dir, "cricket_data.db")
    if os.path.exists(db_path_upper):
        return db_path_upper

    db_path_local = os.path.join(script_dir, "cricket_data.db")
    if os.path.exists(db_path_local):
        return db_path_local

    return db_path
########################### LINK PLAYER NAME BETWEEN PAGES ####################################
def make_trajectory_link(player_name):
    """Generates a query parameter link for the Deep Dive page."""
    return f"/Player_Trajectory?player={player_name.replace(' ', '+')}"

########################### GIVE OLYMPIC STYLE COLOR TO TOP THREE ####################################
def highlight_podium(row):
    gold, silver, bronze = 'background-color: rgba(255, 215, 0, 0.3)', 'background-color: rgba(192, 192, 192, 0.3)', 'background-color: rgba(205, 127, 50, 0.3)'
    val = row.get('Rank') if row.get('Rank') is not None else row.get('Team_Rank')
    
    if val == 1: return [gold] * len(row)
    if val == 2: return [silver] * len(row)
    if val == 3: return [bronze] * len(row)
    return [''] * len(row)

########################### PICK PRIMARY ROLE FOR PLAYER BASED ON POINTS ####################################
def determine_primary_role(row):
    """Assigns the primary role based on the highest impact points."""
    bat, bowl, ar = row['Pts_Batting'], row['Pts_Bowling'], row['Pts_AllRounder']
    if bat >= bowl and bat >= ar: return "Batsman"
    elif bowl >= bat and bowl >= ar: return "Bowler"
    return "All-Rounder"


########################### GET PLAYER ROLE AND RANK ####################################
def calculate_player_metrics(df):
    if df.empty: return df

    # --- Pre-calculations ---
    df['Bat_Avg'] = np.where(df['Total_Outs'] > 0, df['Total_Runs'] / df['Total_Outs'], df['Total_Runs'])
    df['Bat_SR'] = np.where(df['Total_Balls_Faced'] > 0, (df['Total_Runs'] / df['Total_Balls_Faced']) * 100, 0.0)
    df['Total_Overs'] = df['Total_Balls_Bowled'] / 6.0
    df['Bowl_Econ'] = np.where(df['Total_Overs'] > 0, df['Total_Runs_Conceded'] / df['Total_Overs'], 0.0)

    # --- 1. Official Batsman Points ---
    sr_bonus = np.where(df['Bat_SR'] > 100, (df['Bat_SR'] - 100) / 5, 0)
    milestones = (df.get('Count_30s', 0) * 10) + (df.get('Count_50s', 0) * 20) + (df.get('Count_100s', 0) * 50)
    
    df['Pts_Batting'] = (
        (df['Total_Runs'] * 0.5) + 
        (df['Bat_Avg'] * 0.5) + 
        sr_bonus + milestones + 
        (df.get('Total_MoMs', 0) * 10) - 
        (df.get('Count_Ducks', 0) * 10)
    )

    # --- 2. Official Bowler Points ---
    econ_bonus = np.where(df['Bowl_Econ'] < 12, (12 - df['Bowl_Econ']) * 2, 0)
    hauls = (df.get('Count_3W', 0) * 10) + (df.get('Count_4W', 0) * 20) + (df.get('Count_5W', 0) * 30)
    
    df['Pts_Bowling'] = (
        (df['Total_Wickets'] * 15) + 
        df['Total_Overs'] + 
        econ_bonus + hauls + 
        (df.get('Total_MoMs', 0) * 10) - 
        (df.get('Matches_Zero_Wickets', 0) * 5)
    )

    # --- 3. Official All-Rounder Points ---
    df['Pts_AllRounder'] = (
        df['Total_Runs'] + 
        (df['Total_Wickets'] * 10) + 
        (df.get('Total_MoMs', 0) * 10) - 
        (df.get('Count_Ducks', 0) * 1.1) - 
        (df.get('Matches_Zero_Wickets', 0) * 1.1)
    )

    # APPLY YOUR MASK: Disqualify specialists from AR rankings
    mask_not_ar = (df['Total_Wickets'] == 0) | (df['Total_Runs'] < 50)
    df.loc[mask_not_ar, 'Pts_AllRounder'] = -1.0

    return df

########################### HERO CARDS ################################
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
########################### HERO CARDS ################################
import base64
import os
import streamlit as st
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
            st.image(icon_base64, use_container_width=150)
        
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
