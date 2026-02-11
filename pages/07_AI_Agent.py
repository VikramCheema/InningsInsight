import streamlit as st
import pandas as pd
from sqlalchemy import create_engine
from langchain_community.utilities import SQLDatabase
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import os

# --- PAGE CONFIG ---
st.set_page_config(page_title="Cricket SQL Agent", page_icon="🏏", layout="wide")

# --- CONFIG ---
def get_db_path():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(script_dir)
    db_path = os.path.join(root_dir, "cricket_data.db")
    return db_path

DB_FILE = get_db_path()
# We explicitly list the tables defined in your PDF [cite: 3, 7, 14]
TABLES = ["player_stats", "innings_summary", "partnership_stats"]

# --- SETUP CREDENTIALS ---
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY")

@st.cache_resource
def get_db_engine():
    """Cache the DB connection to avoid reconnecting on every interaction."""
    engine = create_engine(f"sqlite:///{DB_FILE}")
    return SQLDatabase(engine, include_tables=TABLES), engine

@st.cache_resource
def get_llm():
    """
    Initialize the Groq client.
    """
    if not GROQ_API_KEY:
        st.error("❌ Groq API Key is missing. Please set it in Streamlit Secrets.")
        st.stop()
        
    return ChatGroq(
        # model="llama-3.3-70b-versatile",
        model="llama-3.1-8b-instant",
        api_key=GROQ_API_KEY,
        temperature=0.0, # Zero temperature for strict SQL generation
        stop_sequences=[";"]
    )

def build_sql_chain(db, llm):
    """
    Creates the chain that converts natural language -> SQL.
    Refined with rules from Schema v2.1 
    """
    base_prompt = """
    You are an expert Cricket Analyst and SQLite developer.
    
    Database Schema & Rules:
    1. Table 'player_stats': [cite: 3]
       - Contains granular performance per player per innings.
       - IMPORTANT: For Bowling stats (Economy, Wickets), YOU MUST FILTER WHERE "Overs_Balled" > 0. 
       - IMPORTANT: For Batting stats (Runs, Strike Rate), YOU MUST FILTER WHERE "Innings_Played" > 0. 
       - "Opposition" column is crucial for "vs Team" queries. 
       
    2. Table 'innings_summary': [cite: 7]
       - Contains match results, venues, and total scores.
       - Use this for queries about "Venues", "Winners", "Total Runs", or "Margins". 
       
    3. Table 'partnership_stats': [cite: 14]
       - Contains partnership details between two batters.
       - Use "Partnership_Runs" and "Wicket" columns here. [cite: 16]

    General Instructions:
    - Output ONLY the raw SQL query. No markdown, no explanations.
    - Handle columns with spaces (e.g., "Runs Scored") by wrapping them in double quotes like "Runs Scored".
    - Use aliases for clarity (e.g., SELECT p.Player_Name...).
    - Current Schema:
    {table_info}
    
    User Question: {input}
    
    SQL Query:
    """
    
    prompt = PromptTemplate.from_template(base_prompt)
    
    def get_table_info(_):
        # This pulls the EXACT column names from your actual .db file
        return db.get_table_info(table_names=TABLES)

    return (
        RunnablePassthrough.assign(
            table_info=get_table_info,
            input=lambda x: x["question"],
        )
        | prompt
        | llm
        | StrOutputParser()
    )

# --- UI LAYOUT ---
st.title("🏏 Cricket Data Analyst")
st.markdown("""
Ask questions about player stats, match summaries, or partnerships.
* **Try:** "Who has the best economy rate in matches played in England?"
* **Try:** "List the top 5 partnerships by runs."
""")

# Initialize Backend
try:
    db, engine = get_db_engine()
    llm = get_llm()
    chain = build_sql_chain(db, llm)
except Exception as e:
    st.error(f"System Error: {e}")
    st.stop()

# User Input
question = st.text_input(
    "Enter your question:", 
    placeholder="e.g., Who scored the most runs against Australia?"
)

if st.button("Run Analysis", type="primary"):
    if not question:
        st.warning("Please enter a question first.")
    else:
        with st.spinner("Analyzing cricket stats..."):
            try:
                # 1. Generate SQL
                raw_response = chain.invoke({"question": question})
                
                # Clean up SQL
                cleaned_sql = raw_response.replace("```sql", "").replace("```", "").strip()
                if ";" in cleaned_sql:
                    cleaned_sql = cleaned_sql.split(";")[0] + ";"
                else:
                    cleaned_sql += ";"
                
                # Show SQL for debugging/trust
                with st.expander("View Generated SQL Query"):
                    st.code(cleaned_sql, language="sql")

                # 2. Execute SQL
                with engine.connect() as conn:
                    df = pd.read_sql(cleaned_sql, conn)
                
                # 3. Show Results
                if df.empty:
                    st.info("The query ran successfully but returned no data.")
                else:
                    st.subheader("Results")
                    st.dataframe(df, use_container_width=True)

                    # 4. Auto-Plotting Logic
                    if len(df.columns) == 2:
                        num_cols = df.select_dtypes(include=['number']).columns
                        cat_cols = df.select_dtypes(include=['object']).columns
                        
                        if len(num_cols) == 1 and len(cat_cols) == 1:
                            st.subheader("Visualization")
                            st.bar_chart(df.set_index(cat_cols[0]))
                            
            except Exception as e:
                error_msg = str(e)
                if "429" in error_msg and "Rate limit" in error_msg:
                    st.warning("⚠️ Daily Usage Limit Reached. Please wait a few minutes or switch models.")
                else:
                    st.error(f"Analysis failed. Error details:\n{e}")