import os
import sys
import sqlite3
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Ensure root directory is in path for local package imports
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from backend_utils import (
    execute_safe_query,
    get_mode_prompt,
    route_query_source,
)
from llm_provider import get_llm
from physics.fuel_burn import calculate_fuel_corrected_time
from physics.tire_deg import predict_tire_degradation_penalty

# 1. Page Configuration & Setup
st.set_page_config(page_title="IMSA Strategy Hub", layout="wide")

# Initialize isolated chat history for IMSA
if "imsa_chat_history" not in st.session_state:
    st.session_state.imsa_chat_history = []
if "query_mode" not in st.session_state:
    st.session_state.query_mode = "Standard"

# 2. Authentic IMSA Championship Branding
col_logo, col_title = st.columns([1, 4])
with col_logo:
    # Official IMSA Logo URL
    st.image("https://upload.wikimedia.org/wikipedia/commons/e/ea/IMSA_Logo.svg", width=160)
with col_title:
    st.title("IMSA WeatherTech SportsCar Championship")
    st.caption("🇺🇸 GTP, GTD Pro, & GTD Pit Wall Strategy Dashboard")

st.markdown("---")

# 3. Sidebar Configurations (Isolated to IMSA Parameters & Physics)
with st.sidebar:
    st.header("⚡ IMSA Trackside Parameters")
    
    # Engine Mode selector matching the app logic
    st.session_state.query_mode = st.radio(
        "Select Engine Mode:",
        ["Simple", "Standard", "Advanced"],
        index=1
    )
    
    st.subheader("🏎️ Stint Simulation Inputs")
    compound = st.selectbox("Michelin Tyre Allocation", ["Soft", "Medium", "Hard"])
    
    # IMSA teams traditionally communicate in Fahrenheit for track temps
    track_temp_f = st.slider("Track Temp (°F)", 60, 140, 95)
    stint_laps = st.number_input("Stint Length Target", min_value=5, max_value=45, value=25)
    
    st.divider()
    
    # Quick Simulation Trigger
    run_sim = st.button("Run Live Stint Predictive Model")
    
    st.subheader("Telemetry Feeds")
    st.success("IMSA Al Kamel Stream: Connected")

# 4. Live Physics Engine Panel
if run_sim:
    st.subheader("📊 IMSA Stint Simulation Output")
    
    # Simulation Constants (Daytona Prototype Parameters)
    baseline_pace = 95.0
    initial_fuel = 100.0
    fuel_burn_rate = 1.84
    
    raw_laps = []
    fuel_corrected_laps = []
    laps_seq = list(range(1, stint_laps + 1))
    
    for l in laps_seq:
        tire_penalty = predict_tire_degradation_penalty(l, compound, track_temp_f)
        fuel_weight_delta = (l - 1) * fuel_burn_rate * 0.03 # weight-drop advantage
        simulated_actual = baseline_pace + tire_penalty - fuel_weight_delta
        raw_laps.append(simulated_actual)
        
        corrected = calculate_fuel_corrected_time(
            simulated_actual, 
            initial_mass_kg=initial_fuel, 
            current_lap=l, 
            fuel_burn_per_lap_kg=fuel_burn_rate
        )
        fuel_corrected_laps.append(corrected)
        
    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.plot(laps_seq, raw_laps, label="Simulated Raw Laps (Masked by Fuel Drop)", color="red", linestyle="--", marker="o")
    ax.plot(laps_seq, fuel_corrected_laps, label="Fuel-Corrected Pace (Pure Rubber Deg)", color="green", linewidth=2.5, marker="s")
    
    # Highlight tire cliff window
    threshold = 12 if compound.lower() == "soft" else (17 if compound.lower() == "medium" else 22)
    if stint_laps > threshold:
        ax.axvspan(threshold, stint_laps, color='yellow', alpha=0.2, label='Tire Cliff Window')
        
    ax.set_xlabel("Lap Number")
    ax.set_ylabel("Lap Time (Seconds)")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend()
    st.pyplot(fig)

# 5. Cognitive Strategy Assistant (Chatbot Interface)
st.subheader("IMSA Pit Wall Assistant")

# Display historical messages in the screen container
for message in st.session_state.imsa_chat_history:
    with st.chat_message(message["role"]):
        st.write(message["content"])

user_query = st.chat_input("Query IMSA rules, drive-time constraints, or timing records...")

if user_query:
    st.session_state.imsa_chat_history.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.write(user_query)

    with st.chat_message("assistant"):
        with st.spinner("Processing Strategy Core..."):
            # Prompt modifier selection
            prompt_modifier = get_mode_prompt(st.session_state.query_mode)
            
            # Explicitly append contextual boundaries to keep query localized to IMSA
            augmented_query = f"Focusing strictly on IMSA data: {user_query}"
            intent = route_query_source(user_query)
            llm = get_llm()
            
            if intent == "SQL":
                st.caption("🤖 *Routing to: SQLite Database Timing Engine (IMSA Sector)*")
                
                sql_gen_prompt = f"""
                Given the SQLite table 'laps' with fields:
                series_code, class, driver_name, lap_time_s, s1_s, s2_s, s3_s, pit_time_s, track_temp_f, raining.
                Generate a clean SQL SELECT query to answer: "{augmented_query}".
                Make sure to explicitly filter WHERE series_code = 'imsa' or series_code = 'IMSA'.
                Respond with ONLY the raw SQL query, no markdown blocks, no format.
                """
                # Invoke model and sanitize string formatting anomalies
                sql_query = llm.invoke(sql_gen_prompt).strip().replace("`", "").replace("sql", "")
                st.code(sql_query, language="sql")
                
                df_results = execute_safe_query(sql_query)
                if not df_results.empty:
                    st.dataframe(df_results)
                    summary_prompt = f"{prompt_modifier} Summarize these IMSA timing database results for the strategy engineer: {df_results.to_string()}"
                    assistant_text = llm.invoke(summary_prompt)
                else:
                    st.warning("No IMSA timing records matched your parameters.")
                    assistant_text = "No matching metrics found in the IMSA dataset allocation."
            else:
                st.caption("📖 *Routing to: IMSA Sporting Regulations & Rules Manuals RAG*")
                # Direct query fallback to baseline model with contextual reinforcement 
                rag_prompt = f"{prompt_modifier} Context: IMSA Regulations Group. Question: {user_query}"
                assistant_text = llm.invoke(rag_prompt)
            
            st.write(assistant_text)
            st.session_state.imsa_chat_history.append({"role": "assistant", "content": assistant_text})