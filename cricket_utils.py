import pandas as pd
import numpy as np

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

def determine_primary_role(row):
    """
    Role logic: Specialists are determined by their primary impact points.
    We check for All-Rounder status ONLY if they pass your mask criteria.
    """
    bat, bowl, ar = row['Pts_Batting'], row['Pts_Bowling'], row['Pts_AllRounder']
    
    # 1. If disqualified by mask, they cannot be an All-Rounder
    if ar == -1.0:
        return "Batsman" if bat >= bowl else "Bowler"
    
    # 2. To be an AR, their AR points must significantly outweigh their specialist rank,
    # or they must show balanced elite skill. 
    # Because AR points (Runs * 1.0) scale higher than Batsman points (Runs * 0.5),
    # we default to specialist roles unless the bowling impact is high.
    if bowl > 20 and bat > 20:
        return "All-Rounder"
        
    return "Batsman" if bat >= bowl else "Bowler"