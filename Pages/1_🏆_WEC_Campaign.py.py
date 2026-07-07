import os
import sys
import streamlit as st

# Force python to recognize the root workspace folder
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, ".."))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

# Now these imports will execute without throwing an error
from backend_utils import execute_safe_query, get_mode_prompt, route_query_source
from llm_provider import get_llm
from physics.fuel_burn import calculate_fuel_corrected_time
from physics.tire_deg import predict_tire_degradation_penalty

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
st.set_page_config(page_title="WEC Strategy Hub", layout="wide")

# Initialize isolated chat history for WEC
if "wec_chat_history" not in st.session_state:
    st.session_state.wec_chat_history = []
if "query_mode" not in st.session_state:
    st.session_state.query_mode = "Standard"

# 2. Authentic WEC Championship Branding
col_logo, col_title = st.columns([1, 4])
with col_logo:
    # Official FIA WEC Logo URL
    st.image("https://upload.wikimedia.org/wikipedia/en/thumb/4/43/FIA_WEC_logo.svg/320px-FIA_WEC_logo.svg.png", width=140)
with col_title:
    st.title("FIA World Endurance Championship")
    st.caption("🤖 Hypercar & LMGT3 Pit Wall Strategy Engine")

st.markdown("---")

# 3. Sidebar Configurations (Isolated to WEC Parameters & Physics)
with st.sidebar:
    st.header("⚡ WEC Trackside Parameters")
    
    # Engine Mode selector matching original app rules
    st.session_state.query_mode = st.radio(
        "Select Engine Mode:",
        ["Simple", "Standard", "Advanced"],
        index=1
    )
    
    st.subheader("🏎️ Stint Simulation Inputs")
    compound = st.selectbox("Michelin Compound Specs", ["Soft Hot Weather", "Soft Cold Weather", "Medium", "Hard"])
    
    # WEC relies strictly on Celsius for track temps
    track_temp_c = st.slider("Track Temp (°C)", 15, 55, 32)
    stint_laps = st.number_input("Stint Length Forecast", min_value=5, max_value=45, value=28)
    
    st.divider()
    
    # Quick Simulation Trigger
    run_sim = st.button("Run Live Stint Predictive Model")
    
    st.subheader("Telemetry Feeds")
    st.success("WEC Al Kamel Stream: Connected")

# 4. Live Physics Engine Panel
if run_sim:
    st.subheader("📊 WEC Stint Simulation Output")
    
    # Convert Celsius to Fahrenheit behind the scenes for the core package requirements
    track_temp_f = (track_temp_c * 9/5) + 32
    
    # Simulation Constants (Hypercar baseline references)
    baseline_pace = 101.5
    initial_fuel = 90.0
    fuel_burn_rate = 2.45
    
    raw_laps = []
    fuel_corrected_laps = []
    laps_seq = list(range(1, stint_laps + 1))
    
    for l in laps_seq:
        # Pass converted temps seamlessly to avoid breaking tire cliff mapping math
        comp_clean = "soft" if "soft" in compound.lower() else ("medium" if "medium" in compound.lower() else "hard")
        tire_penalty = predict_tire_degradation_penalty(l, comp_clean, track_temp_f)
        fuel_weight_delta = (l - 1) * fuel_burn_rate * 0.03
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
    ax.plot(laps_seq, raw_laps, label="Simulated Raw Laps (Masked by Fuel Drop)", color="crimson", linestyle="--", marker="o")
    ax.plot(laps_seq, fuel_corrected_laps, label="Fuel-Corrected Pace (Pure Rubber Deg)", color="blue", linewidth=2.5, marker="s")
    
    # Highlight tire cliff window dynamically
    threshold = 12 if "soft" in compound.lower() else (17 if "medium" in compound.lower() else 22)
    if stint_laps > threshold:
        ax.axvspan(threshold, stint_laps, color='orange', alpha=0.15, label='Thermal Degradation Cliff')
        
    ax.set_xlabel("Lap Number")
    ax.set_ylabel("Lap Time (Seconds)")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend()
    st.pyplot(fig)

# 5. Cognitive Strategy Assistant (Chatbot Interface)
st.subheader("WEC Cognitive Strategy Assistant")

# Render historical messages
for message in st.session_state.wec_chat_history:
    with st.chat_message(message["role"]):
        st.write(message["content"])

user_query = st.chat_input("Query WEC sporting regulations, hybrid allocation thresholds, or timing details...")

if user_query:
    st.session_state.wec_chat_history.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.write(user_query)

    with st.chat_message("assistant"):
        with st.spinner("Analyzing Pit Wall Feed..."):
            prompt_modifier = get_mode_prompt(st.session_state.query_mode)
            
            # Context injection ensuring query is anchored entirely within WEC bounds
            augmented_query = f"Focusing strictly on WEC data: {user_query}"
            intent = route_query_source(user_query)
            llm = get_llm()
            
            if intent == "SQL":
                st.caption("🤖 *Routing to: SQLite Database Timing Engine (WEC Sector)*")
                
                sql_gen_prompt = f"""
                Given the SQLite table 'laps' with fields:
                series_code, class, driver_name, lap_time_s, s1_s, s2_s, s3_s, pit_time_s, track_temp_f, raining.
                Generate a clean SQL SELECT query to answer: "{augmented_query}".
                Make sure to explicitly filter WHERE series_code = 'wec' or series_code = 'WEC'.
                Respond with ONLY the raw SQL query, no markdown blocks, no format.
                """
                sql_query = llm.invoke(sql_gen_prompt).strip().replace("`", "").replace("sql", "")
                st.code(sql_query, language="sql")
                
                df_results = execute_safe_query(sql_query)
                if not df_results.empty:
                    st.dataframe(df_results)
                    summary_prompt = f"{prompt_modifier} Summarize these WEC timing database results for the strategy engineer: {df_results.to_string()}"
                    assistant_text = llm.invoke(summary_prompt)
                else:
                    st.warning("No WEC timing records matched your parameters.")
                    assistant_text = "No matching metrics found in the WEC dataset allocation."
            else:
                st.caption("📖 *Routing to: WEC Sporting Regulations & Manuals RAG*")
                rag_prompt = f"{prompt_modifier} Context: FIA WEC Regulations Handbook. Question: {user_query}"
                assistant_text = llm.invoke(rag_prompt)
            
            # Safely output the text (fixed from original streaming None bug)
            st.write(assistant_text)
            st.session_state.wec_chat_history.append({"role": "assistant", "content": assistant_text})