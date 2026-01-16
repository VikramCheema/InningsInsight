import streamlit as st
import sqlite3
import pandas as pd

st.set_page_config(page_title="Venue Analysis", layout="wide")
st.title("🏟️ Venue Analysis")

# Reuse your load_data function logic
@st.cache_data
def load_data(query):
    conn = sqlite3.connect('cricket_data.db')
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

# Filter by Tournament
tournaments = load_data("SELECT DISTINCT Tournament_ID FROM innings_summary")
selected_tournament = st.sidebar.selectbox("Select Tournament", tournaments['Tournament_ID'])

# Get Venues
venues = load_data(f"SELECT DISTINCT Venue FROM innings_summary WHERE Tournament_ID = '{selected_tournament}'")
selected_venue = st.selectbox("Select Venue", venues['Venue'])

# Get Match Data for this Venue
venue_sql = f"""
    SELECT Match_Date, Team_Name, Innings_No, Total_Runs, Total_Wickets_Lost, Winner
    FROM innings_summary 
    WHERE Venue = '{selected_venue}' AND Tournament_ID = '{selected_tournament}'
"""
venue_df = load_data(venue_sql)

st.subheader(f"Matches at {selected_venue}")
col1, col2 = st.columns(2)
col1.metric("Avg 1st Inn Score", int(venue_df[venue_df['Innings_No']==1]['Total_Runs'].mean()))
col2.metric("Highest Chased", venue_df[venue_df['Innings_No']==2]['Total_Runs'].max())

st.dataframe(venue_df)