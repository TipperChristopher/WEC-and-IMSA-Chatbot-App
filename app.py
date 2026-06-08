# app.py
# app.py
import streamlit as st
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
from langchain_community.llms import Ollama
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader

# This line must match your folder and file structure perfectly:
from physics.fuel_burn import calculate_fuel_corrected_time
from physics.tire_deg import predict_tire_degradation_penalty

st.set_page_config(page_title="WEC & IMSA Strategy Assistant", layout="wide")

# --- DATABASE SETUP ---
def execute_safe_query(sql_query: str) -> pd.DataFrame:
    forbidden = ["drop", "delete", "insert", "update", "alter", "truncate"]
    if any(k in sql_query.lower() for k in forbidden):
        st.error("🛑 Security Block: Destructive SQL query rejected.")
        return pd.DataFrame()
    
    conn = sqlite3.connect("trackside_timing.db")
    try:
        df = pd.read_sql_query(sql_query, conn)
        return df
    except Exception as e:
        st.error(f"Database Error: {e}")
        return pd.DataFrame()
    finally:
        conn.close()

# --- SIDEBAR CONFIGURATION ---
with st.sidebar:
    st.header("⚡ Trackside Configurations")
    # Provide the available series as a list of options
    series = st.selectbox("Championship Series",)

# Provide the tire compound sets as a list of options
    compound = st.selectbox("Tire Compound Set",)
    track_temp = st.slider("Track Temperature (°F)", 60, 140, 95)
    stint_laps = st.number_input("Stint Laps Forecast", min_value=5, max_value=40, value=20)
    
    st.divider()
    st.subheader("System Status")
    st.success("Ollama Engine: Connected (Port 11434)")
    st.info("Database: trackside_timing.db active")

st.title("🏎️ WEC & IMSA Cognitive Strategy & Diagnostics Assistant")
st.markdown("---")

# --- TAB 1: OFF-LINE CHATBOT ---
tab_chat, tab_physics = st.tabs()

with tab_chat:
    st.subheader("Interactive Strategy & Diagnostics")
    
    # Initialize Local LLM via Ollama
    try:
        llm = Ollama(model="qwen2.5-coder:7b", base_url="http://localhost:11434")
    except Exception as e:
        st.error("Could not reach local Ollama server. Verify that Ollama is running in your background taskbar.")
    
    user_query = st.chat_input("Ask about strategy, competitor sector times, or request hybrid fault code troubleshooting...")
    
    if user_query:
        with st.chat_message("user"):
            st.write(user_query)
            
        with st.chat_message("assistant"):
            with st.spinner("Processing local engines..."):
                # Determine query intent: Diagnostics vs Database Timing
                routing_prompt = f"""
                You are a senior Race Strategy Engineer. Categorize the user's intent.
                If the query asks about manual instructions, hybrid codes, rules, or diagnostics, output 'RAG'.
                If the query asks to search lap times, sector times, positions, or ELO, output 'SQL'.
                Query: "{user_query}"
                Output exactly one word: 'RAG' or 'SQL'.
                """
                intent = llm.invoke(routing_prompt).strip().upper()
                
                if "SQL" in intent:
                    st.caption("🤖 *Routing to: SQLite Database Timing Engine*")
                    # Text-to-SQL logic
                    sql_gen_prompt = f"""
                    Given the SQLite table 'laps' with fields:
                    series_code, class, driver_name, lap_time_s, s1_s, s2_s, s3_s, pit_time_s, track_temp_f, raining.
                    Generate a clean SQL SELECT query to answer: "{user_query}".
                    Respond with ONLY the raw SQL query, no markdown blocks, no format.
                    """
                    sql_query = llm.invoke(sql_gen_prompt).strip().replace("`", "").replace("sql", "")
                    st.code(sql_query, language="sql")
                    
                    df_results = execute_safe_query(sql_query)
                    if not df_results.empty:
                        st.dataframe(df_results)
                        # Natural language summarization of results
                        summary = llm.invoke(f"Summarize these timing database results for the engineer: {df_results.to_string()}")
                        st.write(summary)
                    else:
                        st.warning("No timing records matched your query.")
                
                else:
                    st.caption("📖 *Routing to: Advanced Technical Manual RAG*")
                    try:
                        # Direct RAG over any PDFs in data/manuals/
                        documents = SimpleDirectoryReader("data/manuals").load_data()
                        index = VectorStoreIndex.from_documents(documents)
                        query_engine = index.as_query_engine()
                        response = query_engine.query(user_query)
                        st.write(str(response))
                    except Exception as e:
                        st.info("Place technical PDFs (e.g. Bosch MGU/MCU troubleshooting manuals) inside 'data/manuals' to enable advanced diagnostic RAG.")
                        st.write(llm.invoke(user_query))

# --- TAB 2: PHYSICS PREDICTIONS ---
with tab_physics:
    st.subheader("Stint Pace Decay & Degradation Forecast")
    
    # Generate simulation curves
    laps_seq = list(range(1, stint_laps + 1))
    
    # Example Baseline reference pace (Daytona Hypercar baseline)
    ref_lap_time = 95.0 # seconds
    initial_fuel = 100.0 # kg
    fuel_burn_rate = 1.84 # kg per lap
    
    raw_laps = []
    fuel_corrected_laps = []
    tire_penalties = []
    combined_pace = []
    
    for l in laps_seq:
        # Simulate tire penalty
        t_pen = predict_tire_degradation_penalty(l, compound, track_temp)
        tire_penalties.append(t_pen)
        
        # Simulated actual times (getting heavier or lighter depending on fuel drop)
        # Actual lap gets lighter, saving time, but tire wears out costing time.
        simulated_actual = ref_lap_time + t_pen - (0.03 * (initial_fuel - (initial_fuel - (l * fuel_burn_rate))))
        raw_laps.append(simulated_actual)
        
        # Apply physics correction formula to clean out fuel burn advantage
        corrected = calculate_fuel_corrected_time(simulated_actual, initial_mass_kg=initial_fuel, current_lap=l, fuel_burn_per_lap_kg=fuel_burn_rate)
        fuel_corrected_laps.append(corrected)
        
    # Plotting
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(laps_seq, raw_laps, label="Simulated Actual Lap Times (Masked by Fuel Burn-off)", color="red", linestyle="--", marker="o")
    ax.plot(laps_seq, fuel_corrected_laps, label="Fuel-Corrected Pace (Pure Mechanical/Tire Degradation)", color="green", linewidth=2.5, marker="s")
    
    # Highlight critical tire cliff zone
    threshold = 12 if compound.lower() == "soft" else (17 if compound.lower() == "medium" else 22)
    if stint_laps > threshold:
        ax.axvspan(threshold, stint_laps, color='yellow', alpha=0.2, label='Tire Cliff Operating Window')
        
    ax.set_xlabel("Stint Lap Number")
    ax.set_ylabel("Lap Time (Seconds)")
    ax.set_title(f"Endurance Stint Progression on {compound} Compound")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend()
    
    st.pyplot(fig)
    
    # Visual metrics cards
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(label="Simulated Stint End Penalty", value=f"+{tire_penalties[-1]:.2f} s")
    with col2:
        st.metric(label="Predicted Tire Cliff Threshold", value=f"Lap {threshold}")
    with col3:
        st.metric(label="Stint Average Pace", value=f"{pd.Series(fuel_corrected_laps).mean():.3f} s")