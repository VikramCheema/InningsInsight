import streamlit as st
import pandas as pd
import sqlite3
import os
import folium
from streamlit_folium import st_folium

st.set_page_config(page_title="Venue Atlas", page_icon="🌍", layout="wide")

# --- 1. SETUP & HELPERS ---
def get_db_path():
    return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cricket_data.db")

DB_FILE = get_db_path()

def run_query(query, params=None):
    if not os.path.exists(DB_FILE): return pd.DataFrame()
    conn = sqlite3.connect(DB_FILE)
    try:
        return pd.read_sql_query(query, conn, params=params)
    except Exception as e:
        st.error(f"DB Error: {e}")
        return pd.DataFrame()
    finally:
        conn.close()

# --- 2. CONFIGURATION DATA ---

# A. NORMALIZATION MAP (Your provided list)
VENUE_MAP = {
    # --- AUSTRALIA ---
    "adelaide": "Adelaide Oval", "adelaide oval": "Adelaide Oval", "allen border fields": "Allen Border Fields",
    "bellerive oval": "Bellerive Oval", "coffs harbour": "Coffs Harbour", "docklands stadium": "Docklands Stadium",
    "gabba": "The Gabba", "the gabba": "The Gabba", "junction oval": "Junction Oval",
    "karen rolton oval": "Karen Rolton Oval", "manuka oval": "Manuka Oval", "melbourne": "Melbourne Cricket Ground",
    "melbourne cricket ground": "Melbourne Cricket Ground", "perth stadium": "Perth Stadium", "sydney cricket ground": "Sydney Cricket Ground","Sydney Cricket Ground": "Sydney Cricket Ground", "SCG":"Sydney Cricket Ground",
    # --- ENGLAND / WALES ---
    "ageas bowl": "Utilita Bowl", "utilita bowl": "Utilita Bowl", "headingley": "Headingley",
    "kia oval": "Kia Oval", "lords": "Lords", "lord's": "Lords", "lord’s": "Lords",
    "sophia gardens": "Sophia Gardens", "trent bridge": "Trent Bridge",
    # --- INDIA ---
    "ahmedabad": "Narendra Modi Stadium", "narendra modi stadium": "Narendra Modi Stadium",
    "arun jaitley cricket stadium": "Arun Jaitley Cricket Stadium", "delhi": "Arun Jaitley Cricket Stadium",
    "chennai": "Chennai", "dharamshala": "Dharamshala", "eden gardens": "Kolkata Eden Gardens",
    "kolkata": "Kolkata Eden Gardens", "kolkata park": "Kolkata Eden Gardens", "guwahati": "Guwahati",
    "hydrabad": "Rajiv Gandhi International", "rajiv gandhi international": "Rajiv Gandhi International",
    "lucknow": "Lucknow", "mullanpur": "Mullanpur Stadium", "mullanpur stadium": "Mullanpur Stadium",
    "new chandigarh": "Mullanpur Stadium", "mumbai": "Wankhede Stadium", "wankhede stadium": "Wankhede Stadium",
    # --- NEW ZEALAND ---
    "auckland": "Auckland Stadium", "auckland stadium": "Auckland Stadium", "aukland oval": "Auckland Stadium",
    "bay oval": "Bay Oval", "eden park": "Eden Park", "hagley oval": "Hagley Oval",
    "seddon park": "Seddon Park", "wellington": "Wellington Park", "wellington park": "Wellington Park",
    # --- PAKISTAN ---
    "gaddafi stadium": "Lahore Gaddafi Stadium", "lahore": "Lahore Gaddafi Stadium",
    "old lahore cricket ground": "Lahore Gaddafi Stadium", "karachi": "Karachi", "multan": "Multan Cricket Stadium",
    "multan cricket stadium": "Multan Cricket Stadium", "rawalpindi": "Rawalpindi Cricket Stadium",
    "rawalpindi cricket stadium": "Rawalpindi Cricket Stadium",
    # --- SOUTH AFRICA ---
    "cape town": "Cape Town", "cape town\u200b": "Cape Town", "durban": "Durban Fields",
    "durban fields": "Durban Fields", "mangaung oval": "Mangaung Oval",
    # --- WEST INDIES ---
    "brian lara stadium": "Brian Lara Stadium", "daren sammy stadium": "Daren Sammy Stadium",
    "derral sammy stadium": "Daren Sammy Stadium", "kensington oval": "Kensington Oval",
    "queen's park oval": "Queen's Park Oval", "sabina garden": "Sabina Park", "sabina park": "Sabina Park",
    # --- SRI LANKA / BANGLADESH ---
    "colombo stadium": "Colombo Stadium", "dhaka stadium": "Dhaka Stadium",
    # --- Zimbabwe ---
    "harare sports club": "Harare Sports Club","harare": "Harare Sports Club","salisbury sports club": "Harare Sports Club"
}

# B. COORDINATES
VENUE_COORDS = {
    "Adelaide Oval": [-34.9155, 138.5961], "Allen Border Fields": [-27.4435, 153.0416],
    "Bellerive Oval": [-42.8772, 147.3736], "Coffs Harbour": [-30.3167, 153.1167],
    "Sydney Cricket Ground": [-33.8917,151.2248],
    "Docklands Stadium": [-37.8160, 144.9478], "The Gabba": [-27.4859, 153.0380],
    "Junction Oval": [-37.8540, 144.9816], "Karen Rolton Oval": [-34.9238, 138.5830],
    "Manuka Oval": [-35.3181, 149.1354], "Melbourne Cricket Ground": [-37.8199, 144.9834],
    "Perth Stadium": [-31.9511, 115.8890], "Utilita Bowl": [50.9248, -1.3223],
    "Headingley": [53.8176, -1.5822], "Kia Oval": [51.4837, -0.1150],
    "Lords": [51.5292, -0.1722], "Sophia Gardens": [51.4893, -3.1906],
    "Trent Bridge": [52.9366, -1.1325], "Narendra Modi Stadium": [23.0904, 72.5975],
    "Arun Jaitley Cricket Stadium": [28.6369, 77.2410], "Chennai": [13.0628, 80.2793],
    "Dharamshala": [32.1976, 76.3259], "Kolkata Eden Gardens": [22.5646, 88.3433],
    "Guwahati": [26.1436, 91.7371], "Rajiv Gandhi International": [17.4065, 78.5505],
    "Lucknow": [26.8113, 81.0181], "Mullanpur Stadium": [30.8037, 76.7119],
    "Wankhede Stadium": [18.9389, 72.8258], "Auckland Stadium": [-36.8718, 174.7456],
    "Bay Oval": [-37.6565, 176.2081], "Eden Park": [-36.8718, 174.7456],
    "Hagley Oval": [-43.5309, 172.6203], "Seddon Park": [-37.7872, 175.2731],
    "Wellington Park": [-41.2730, 174.7850], "Lahore Gaddafi Stadium": [31.5126, 74.3364],
    "Karachi": [24.8931, 67.0736], "Multan Cricket Stadium": [30.1601, 71.5222],
    "Rawalpindi Cricket Stadium": [33.6421, 73.0766], "Cape Town": [-33.9702, 18.4682],
    "Durban Fields": [-29.8504, 31.0298], "Mangaung Oval": [-29.1130, 26.2057],
    "Brian Lara Stadium": [10.2920, -61.4285], "Daren Sammy Stadium": [14.0714, -60.9575],
    "Kensington Oval": [13.1042, -59.6212], "Queen's Park Oval": [10.6698, -61.5235],
    "Sabina Park": [17.9790, -76.7828], "Colombo Stadium": [6.9397, 79.8687],
    "Dhaka Stadium": [23.8069, 90.3636],"Harare Sports Club": [-17.8141, 31.0506],
}

COUNTRY_CENTERS = {
    "India": [20.5937, 78.9629], "Australia": [-25.2744, 133.7751],
    "England": [52.3555, -1.1743], "South Africa": [-30.5595, 22.9375],
    "New Zealand": [-40.9006, 174.8860], "Pakistan": [30.3753, 69.3451],
    "West Indies": [18.1096, -77.2975], "Sri Lanka": [7.8731, 80.7718],
    "Bangladesh": [23.6850, 90.3563],"Zimbabwe": [-19.0154, 29.1549],
}

# --- 3. TEAM HELPERS ---
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

# --- 4. CORE ANALYTICS LOGIC ---
def get_venue_stats(venue_original_name):
    """
    Calculates detailed venue stats using provided logic.
    """
    stats = {}
    
    # A. MATCH RECORDS (Win %, Chases, Defends)
    query_match = f"""
    SELECT Match_ID, Team_Name, Winner, Innings_No, Total_Runs
    FROM innings_summary 
    WHERE Venue = '{venue_original_name}'
    """
    df = run_query(query_match)
    
    if not df.empty:
        stats['matches'] = df['Match_ID'].nunique()
        # Avg 1st Inn Score
        avg_1st = df[df['Innings_No'] == 1]['Total_Runs'].mean()
        stats['avg_score_1st'] = int(avg_1st) if not pd.isna(avg_1st) else 0
        
        # Totals
        stats['highest_total'] = int(df['Total_Runs'].max())
        stats['lowest_total'] = int(df['Total_Runs'].min())
        
        bat1_wins = 0
        bat2_wins = 0
        highest_chase = 0
        lowest_defend = 9999
        
        for mid in df['Match_ID'].unique():
            m = df[df['Match_ID'] == mid]
            if m.empty: continue
            
            # Safe access to Winner
            if 'Winner' not in m.columns or pd.isna(m.iloc[0]['Winner']): continue
            winner = m.iloc[0]['Winner']
            
            inn1 = m[m['Innings_No'] == 1]
            inn2 = m[m['Innings_No'] == 2]
            
            if inn1.empty or inn2.empty: continue
            
            t1 = inn1.iloc[0]['Team_Name']
            score1 = int(inn1.iloc[0]['Total_Runs'])
            score2 = int(inn2.iloc[0]['Total_Runs'])
            
            if is_same_team(winner, t1):
                bat1_wins += 1
                if score1 < lowest_defend: lowest_defend = score1
            else:
                bat2_wins += 1
                if score2 > highest_chase: highest_chase = score2

        # Final Calculations
        total_valid = bat1_wins + bat2_wins
        if total_valid > 0:
            stats['bat1_win_pct'] = (bat1_wins / total_valid * 100)
            stats['bat2_win_pct'] = (bat2_wins / total_valid * 100)
        else:
            stats['bat1_win_pct'] = 0
            stats['bat2_win_pct'] = 0
            
        stats['highest_chase'] = highest_chase if highest_chase > 0 else "N/A"
        stats['lowest_defend'] = lowest_defend if lowest_defend != 9999 else "N/A"
    else:
        return None # No data for venue

    # B. BEST BATTER (Your Query)
    q_bat = f"""
    SELECT p.Player_Name, p.Runs_Scored, p.Team_Name
    FROM player_stats p
    JOIN innings_summary i ON p.Match_ID = i.Match_ID
    WHERE i.Venue = '{venue_original_name}'
    ORDER BY p.Runs_Scored DESC LIMIT 1
    """
    best_bat = run_query(q_bat)
    if not best_bat.empty:
        stats['best_bat'] = f"{best_bat.iloc[0]['Player_Name']} ({best_bat.iloc[0]['Runs_Scored']})"
    else:
        stats['best_bat'] = "N/A"

    # C. BEST BOWLER (Your Query)
    q_bowl = f"""
    SELECT p.Player_Name, p.Wickets_Taken, p.Runs_Conceded, p.Team_Name
    FROM player_stats p
    JOIN innings_summary i ON p.Match_ID = i.Match_ID
    WHERE i.Venue = '{venue_original_name}'
    ORDER BY p.Wickets_Taken DESC, p.Runs_Conceded ASC LIMIT 1
    """
    best_bowl = run_query(q_bowl)
    if not best_bowl.empty:
        stats['best_bowl'] = f"{best_bowl.iloc[0]['Player_Name']} ({best_bowl.iloc[0]['Wickets_Taken']}/{best_bowl.iloc[0]['Runs_Conceded']})"
    else:
        stats['best_bowl'] = "N/A"
        
    # D. TEAMS HOSTED (New Query)
    q_teams = f"""
    SELECT DISTINCT Team_Name FROM innings_summary 
    WHERE Venue = '{venue_original_name}'
    ORDER BY Team_Name
    """
    teams_df = run_query(q_teams)
    if not teams_df.empty:
        stats['teams_hosted'] = teams_df['Team_Name'].tolist()
    else:
        stats['teams_hosted'] = []

    return stats

# --- 5. UI LAYOUT ---
st.title("🌍 Global Venue Intelligence")

countries_df = run_query("SELECT DISTINCT Country FROM innings_summary WHERE Country IS NOT NULL ORDER BY Country")
available_countries = countries_df['Country'].tolist() if not countries_df.empty else ["India", "England", "Australia"]

selected_country = st.sidebar.selectbox("Select Host Country", available_countries)

# --- MAP LOGIC ---
q_venues = f"SELECT DISTINCT Venue FROM innings_summary WHERE Country = '{selected_country}'"
venues_df = run_query(q_venues)

map_data = []
for v in venues_df['Venue'].unique():
    clean_v = str(v).strip().lower()
    standard_name = VENUE_MAP.get(clean_v, v)
    
    if standard_name in VENUE_COORDS:
        map_data.append({
            "name": standard_name,
            "original_name": v, # IMPORTANT: Used for DB queries
            "coords": VENUE_COORDS[standard_name]
        })

# RENDER MAP
st.subheader(f"📍 Venues in {selected_country}")

start_coords = COUNTRY_CENTERS.get(selected_country, [20, 0])
m = folium.Map(location=start_coords, zoom_start=5, tiles="CartoDB positron")

for venue in map_data:
    folium.Marker(
        location=venue["coords"],
        tooltip=venue["name"],
        popup=venue["name"],
        icon=folium.Icon(color="green", icon="info-sign"),
    ).add_to(m)

map_output = st_folium(m, width="100%", height=500)

# --- 6. DISPLAY STATS ---
if map_output["last_object_clicked_tooltip"]:
    clicked_std_name = map_output["last_object_clicked_tooltip"]
    
    # Reverse lookup to find the Original DB Name associated with this pin
    # (Matches what we stored in map_data)
    original_db_name = next((x['original_name'] for x in map_data if x['name'] == clicked_std_name), clicked_std_name)
    
    st.divider()
    st.markdown(f"## 🏟️ Analytics: {clicked_std_name}")
    
    # Fetch Data
    stats = get_venue_stats(original_db_name)
    
    if stats:
        # 1. High Level Metrics
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Matches Played", stats['matches'])
        c2.metric("Avg 1st Inn Score", stats['avg_score_1st'])
        c3.metric("Highest Total", stats['highest_total'])
        c4.metric("Lowest Total", stats['lowest_total'])
        
        st.write("")
        
        # 2. Win Bias & Chase Records
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Win Bat 1st", f"{stats['bat1_win_pct']:.0f}%")
        k2.metric("Win Bat 2nd", f"{stats['bat2_win_pct']:.0f}%")
        k3.metric("Highest Chased", stats['highest_chase'])
        k4.metric("Lowest Defended", stats['lowest_defend'])
        
        # Ease of Chase Badge
        bias = "⚖️ Balanced"
        if stats['bat2_win_pct'] > 55: bias = "🟢 Chasing Paradise"
        elif stats['bat1_win_pct'] > 55: bias = "🔴 Defend Fortress"
        st.caption(f"**Venue Bias:** {bias}")
        
        st.divider()
        
        # 3. Star Performers
        s1, s2 = st.columns(2)
        s1.success(f"🏏 **Best Batting Performance:**\n\n {stats['best_bat']}")
        s2.info(f"⚾ **Best Bowling Figures:**\n\n {stats['best_bowl']}")
        
        # 4. Teams Hosted
        if stats['teams_hosted']:
            with st.expander("Show All Teams Hosted"):
                st.write(", ".join(stats['teams_hosted']))
        
    else:
        st.warning("No match data available for this venue.")
    
elif not map_data:
    st.warning(f"No coordinates found for venues in {selected_country}.")
else:
    st.info("👆 Click on a green pin above to reveal venue intelligence.")