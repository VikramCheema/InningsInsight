import streamlit as st
import pandas as pd
import altair as alt
import os
from sqlalchemy import create_engine, inspect
from langchain_community.utilities import SQLDatabase
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# --- PAGE CONFIG ---
st.set_page_config(page_title="Cricket SQL Analyst", page_icon="🏏", layout="wide")

# --- DATABASE SETUP ---
def get_db_path():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(script_dir)
    db_path = os.path.join(root_dir, "cricket_data.db")
    if not os.path.exists(db_path):
        db_path = os.path.join(script_dir, "cricket_data.db")
    return db_path

DB_FILE = get_db_path()
TABLES = ["player_stats", "innings_summary", "partnership_stats"]

@st.cache_resource
def get_db_engine():
    if not os.path.exists(DB_FILE):
        st.error(f"Database not found at {DB_FILE}")
        st.stop()
    engine = create_engine(f"sqlite:///{DB_FILE}")
    return SQLDatabase(engine, include_tables=TABLES), engine

# --- LLM SETUP ---
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY")

@st.cache_resource
def get_llm():
    return ChatGroq(
        model="llama-3.1-8b-instant",
        api_key=GROQ_API_KEY,
        temperature=0.0,
        stop_sequences=[";"]
    )

# --- MASTER PROMPT ---
def build_smart_sql_chain(db, llm):
    base_prompt = """
    You are an expert Cricket Analyst and SQLite developer.
    ### DATA TRANSLATION LAYER (CRITICAL)
    The database uses 3-letter CODES in 'player_stats' (ps) and FULL NAMES in 'innings_summary' (inv).
    When joining or comparing these tables, you MUST translate the code to the full name.
    ### TEAM NAME MAPPING KNOWLEDGE:
    Use this mapping to bridge 'player_stats' (3-letter codes) and 'innings_summary' (Full Names):
    - 'IND' <-> 'India' / 'INDIA'
    - 'PAK' <-> 'Pakistan' / 'PAKISTAN'
    - 'ENG' <-> 'England' / 'ENGLAND'
    - 'AUS' <-> 'Australia' / 'AUSTRALIA'
    - 'NZ'  <-> 'New Zealand' / 'NEW ZEALAND'
    - 'BAN' <-> 'Bangladesh' / 'BANGLADESH'
    - 'SL'  <-> 'Sri Lanka' / 'SRI LANKA'
    - 'AFG' <-> 'Afghanistan' / 'AFGHANISTAN'
    - 'SA'  <-> 'South Africa' / 'SOUTH AFRICA'
    - 'WI'  <-> 'West Indies' / 'WEST INDIES'
    - 'IRE' <-> 'Ireland' / 'IRELAND'
    - 'ZIM' <-> 'Zimbabwe' / 'ZIMBABWE'

    ### SEMANTIC KEYWORD MAPPING:
    - "Personal Best/Highest Score" -> MAX("Runs_Scored")
    - "Economy" -> "Economy_Rate"
    - "Winning Margin" -> "Win_Margin_Runs" or "Win_Margin_Wickets"
    - "Big Stand" -> "Partnership_Runs"
    - "Egg Match" -> A match where "Overs Balled" > 0 AND "Wickets Taken" = 0.
    - "Clutch Finish" -> A match where "Not_Out_Innings" = 1, "Runs_Scored" >= 20, and the player's team won.

    ### DATABASE SCHEMA & CRITICAL RULES:
    1. Table 'player_stats' (ps): 
       - "Team Name" uses 3-letter codes.
       - For Bowling: ALWAYS check "Overs Balled" > 0.
       - For Batting: ALWAYS check "Innings_Played" > 0.
       
    2. Table 'innings_summary' (inv): 
       - "Winner" and "Team Name" use Full Names.
       
    3. Table 'partnership_stats' (part): Use for batting pair stands.

    ### BRIDGE LOGIC FOR JOINS:
    - When comparing `ps.Team_Name` (code) to `inv.Winner` (full name), you MUST translate the code into the full name within the SQL.
    - Example: `WHERE inv.Winner = (CASE ps.Team_Name WHEN 'IND' THEN 'India' WHEN 'AUS' THEN 'Australia' ... END)`

    ### SQL CONSTRUCTION RULES:
    - Output ONLY raw SQL. No markdown.
    - ALIASES: NEVER use "is" as an alias. Use 'ps', 'inv', or 'part'.
    - PERCENTAGES: Ensure the denominator is relevant (e.g., Total Bowling Matches where Overs Balled > 0).
    - JOINS: Join on "Match_ID". Use the Bridge Logic above to ensure codes and full names match during filters.
    
    Current Schema:
    {table_info}
    
    User Question: {input}
    
    SQL Query:
    """
    
    prompt = PromptTemplate.from_template(base_prompt)
    return (
        RunnablePassthrough.assign(
            table_info=lambda _: db.get_table_info(table_names=TABLES),
            input=lambda x: x["question"],
        )
        | prompt
        | llm
        | StrOutputParser()
    )

# --- INITIALIZATION ---
try:
    db, engine = get_db_engine()
    llm = get_llm()
    chain = build_smart_sql_chain(db, llm)
except Exception as e:
    st.error(f"Init Error: {e}")
    st.stop()

# --- MAIN UI ---
st.title("🏏 Cricket Natural Language Analyst")
st.markdown("Ask complex questions about players, teams, or match trends.")

# Manual Entry
question = st.text_area(
    "Enter your analysis request:", 
    height=150,
    placeholder="e.g., List players with at least 10 matches who have the highest percentage of 'egg matches' (bowled but took 0 wickets)..."
)

col1, col2 = st.columns([1, 5])
with col1:
    run_button = st.button("Run Analysis", type="primary")

# if run_button:
#     if not question:
#         st.warning("Please enter a prompt first.")
#     else:
#         with st.spinner("Querying database..."):
#             try:
#                 # 1. SQL Generation
#                 raw_sql = chain.invoke({"question": question})
#                 cleaned_sql = raw_sql.strip().replace("```sql", "").replace("```", "")
#                 if not cleaned_sql.endswith(";"): cleaned_sql += ";"

#                 # 2. Results Preview
#                 with st.expander("View Logic (SQL)"):
#                     st.code(cleaned_sql, language="sql")

#                 # 3. Data Execution
#                 with engine.connect() as conn:
#                     df = pd.read_sql(cleaned_sql, conn)

#                 if df.empty:
#                     st.info("No records match your criteria.")
#                 else:
#                     st.subheader("Results")
#                     st.dataframe(df, use_container_width=True)
                    
#                     # 4. Auto-Chart
#                     if len(df.columns) >= 2:
#                         num_cols = df.select_dtypes(include=['number']).columns
#                         cat_cols = df.select_dtypes(include=['object']).columns
#                         if not num_cols.empty and not cat_cols.empty:
#                             st.bar_chart(df.set_index(cat_cols[0])[num_cols[0]])

#             except Exception as e:
#                 st.error(f"Analysis failed: {e}")

# --- INITIALIZE SESSION STATE ---
if "analysis_df" not in st.session_state:
    st.session_state.analysis_df = None
if "last_sql" not in st.session_state:
    st.session_state.last_sql = ""
# --- EXECUTION SECTION ---
# --- EXECUTION ---
if run_button:
    if not question:
        st.warning("Please enter a prompt.")
    else:
        with st.spinner("Analyzing..."):
            try:
                # Generate SQL
                raw_sql = chain.invoke({"question": question})
                st.session_state.last_sql = raw_sql.strip().replace("```sql", "").replace("```", "")
                if not st.session_state.last_sql.endswith(";"): 
                    st.session_state.last_sql += ";"

                # Execute and store in session state
                with engine.connect() as conn:
                    st.session_state.analysis_df = pd.read_sql(st.session_state.last_sql, conn)
            except Exception as e:
                st.error(f"Error: {e}")

# --- DISPLAY PERSISTENT RESULTS ---
if st.session_state.analysis_df is not None:
    df = st.session_state.analysis_df
    
    with st.expander("View Logic (SQL)"):
        st.code(st.session_state.last_sql, language="sql")

    if df.empty:
        st.info("No records found.")
    else:
        st.subheader("Results")
        st.dataframe(df, use_container_width=True)
        
        # Interactive Plotting (Now survives reruns!)
        # --- INTERACTIVE PLOTTING WITH SORTING ---
        all_cols = df.columns.tolist()
        num_cols = df.select_dtypes(include=['number']).columns.tolist()

        if num_cols:
            st.divider()
            st.subheader("📊 Interactive Performance Chart")
            
            # 1. UI Controls
            row1_c1, row1_c2 = st.columns(2)
            with row1_c1:
                selected_metric = st.selectbox("Y-Axis (Metric):", num_cols)
            with row1_c2:
                selected_group = st.selectbox("X-Axis (Category):", all_cols, 
                                            index=all_cols.index("Match_Order") if "Match_Order" in all_cols else 0)
            
            row2_c1, row2_c2 = st.columns(2)
            with row2_c1:
                # NEW: Choose which axis to sort by
                sort_target = st.radio(
                    "Sort By:", 
                    options=["X-Axis", "Y-Axis"], 
                    horizontal=True,
                    help="Choose to sort by labels (X) or values (Y)"
                )
            with row2_c2:
                sort_order = st.radio("Direction:", ["Ascending", "Descending"], horizontal=True)

            # 2. Sorting Logic
            # Determine the column to sort on based on the radio selection
            sort_col = selected_group if sort_target == "X-Axis" else selected_metric
            ascending_bool = (sort_order == "Ascending")
            
            # Apply sorting to the DataFrame
            df_sorted = df.sort_values(by=sort_col, ascending=ascending_bool)

            # 3. Create Altair Chart
            # :O (Ordinal) treats numbers like Match_Order as discrete labels
            chart = alt.Chart(df_sorted).mark_bar().encode(
                x=alt.X(f"{selected_group}:O", 
                        sort=None,  # CRITICAL: Respects the DataFrame order
                        title=selected_group.replace("_", " ")),
                y=alt.Y(f"{selected_metric}:Q", 
                        title=selected_metric.replace("_", " ")),
                color=alt.condition(
                    alt.datum[selected_metric] > 0,
                    alt.value("#0068c9"), # Standard Blue
                    alt.value("#ff4b4b")  # Red for zero/negative if applicable
                ),
                tooltip=all_cols
            ).properties(height=450).interactive()

        st.altair_chart(chart, use_container_width=True)