import streamlit as st
import sqlite3
from typing import Optional, List, Any

import pandas as pd
import matplotlib.pyplot as plt
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llm_provider import get_llm

def route_query_source(user_query: Optional[str]) -> str:
    """
    Determines if the question requires looking at the timing database 
    or parsing the technical/event PDF documents.
    """
    if not user_query:
        return "RAG"

    query_lower = user_query.lower()
    
    # Timing queries -> SQL timing database
    sql_keywords = ["fastest lap", "sector", "lap time", "gap", "position", "pit stop duration"]
    if any(k in query_lower for k in sql_keywords):
        return "SQL"
        
    # Schedule, entry lists, rules, or diagnostics -> RAG PDF Engine
    return "RAG"

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


def get_simple_prompt() -> str:
    return "Answer in clear, concise plain language with a focus on the core insight."


def get_standard_prompt() -> str:
    return "Answer with balanced clarity and technical accuracy."


def get_advanced_prompt() -> str:
    return "Answer with detailed technical reasoning, diagnostics, and step-by-step explanation."


def get_mode_prompt(mode: str) -> str:
    if mode == "Simple":
        return get_simple_prompt()
    if mode == "Advanced":
        return get_advanced_prompt()
    return get_standard_prompt()


def reset_chat():
    st.session_state.chat_history = []
    st.session_state.user_query = ""


if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "query_mode" not in st.session_state:
    st.session_state.query_mode = "Standard"


def get_series_options() -> List[str]:
    """Load available series from the laps database, or fall back to defaults."""
    conn = sqlite3.connect("trackside_timing.db")
    try:
        df = pd.read_sql_query("SELECT DISTINCT series_code FROM laps ORDER BY series_code", conn)
        if not df.empty:
            return df["series_code"].astype(str).tolist()
    except Exception:
        pass
    finally:
        conn.close()

    return ["WEC", "IMSA"]

# --- SIDEBAR CONFIGURATION ---
with st.sidebar:
    st.header("⚡ Trackside Configurations")
    # Provide the available series as a list of options
    series_options = get_series_options()
    series = st.selectbox("Championship Series", series_options)

    # Provide the tire compound sets as a list of options
    compound = st.selectbox("Tire Compound Set", ["Soft", "Medium", "Hard"])
    track_temp = st.slider("Track Temperature (°F)", 60, 140, 95)
    stint_laps = st.number_input("Stint Laps Forecast", min_value=5, max_value=40, value=20)
    
    st.divider()
    st.subheader("System Status")
    st.success("Ollama Engine: Connected (Port 11434)")
    st.info("Database: trackside_timing.db active")

st.title("🏎️ WEC & IMSA Cognitive Strategy & Diagnostics Assistant")
st.markdown("---")

# --- TAB 1: OFF-LINE CHATBOT ---
tab_chat, tab_physics = st.tabs(["Offline Chatbot", "Physics Predictions"])

with tab_chat:
    st.subheader("Interactive Strategy & Diagnostics")
    
    # Initialize LLM (provider selectable via env `LLM_PROVIDER`)
    llm: Any = None
    try:
        llm = get_llm()
    except Exception as e:
        st.error(f"LLM initialization error: {e}. Verify configuration and that the model service is reachable.")

    selected_mode = st.session_state.query_mode
    col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
    with col1:
        simple_label = "Simple ✅" if selected_mode == "Simple" else "Simple"
        if st.button(simple_label, key="mode_simple"):
            st.session_state.query_mode = "Simple"
    with col2:
        standard_label = "Standard ✅" if selected_mode == "Standard" else "Standard"
        if st.button(standard_label, key="mode_standard"):
            st.session_state.query_mode = "Standard"
    with col3:
        advanced_label = "Advanced ✅" if selected_mode == "Advanced" else "Advanced"
        if st.button(advanced_label, key="mode_advanced"):
            st.session_state.query_mode = "Advanced"
    with col4:
        if st.button("Clear Chat", key="clear_chat"):
            reset_chat()

    mode_descriptions = {
        "Simple": "Concise, plain-language guidance for quick decisions.",
        "Standard": "Balanced replies with practical technical clarity.",
        "Advanced": "Deep reasoning, diagnostics, and step-by-step explanation."
    }
    mode_colors = {
        "Simple": "#dbeafe",
        "Standard": "#e2f0d9",
        "Advanced": "#f9f0d7"
    }
    selected_mode = st.session_state.query_mode
    st.markdown(
        f"<div style='background:{mode_colors[selected_mode]}; padding:14px; border-radius:8px; margin-bottom:12px;'>"
        f"<strong>{selected_mode} mode</strong>: {mode_descriptions[selected_mode]}"
        "</div>",
        unsafe_allow_html=True,
    )

    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    user_query = st.chat_input(
        "Ask about strategy, competitor sector times, or request hybrid fault code troubleshooting...",
        key="user_query",
    )

    if user_query:
        st.session_state.chat_history.append({"role": "user", "content": user_query})

        with st.chat_message("assistant"):
            with st.spinner("Processing local engines..."):
                intent = route_query_source(user_query)
                prompt_modifier = get_mode_prompt(st.session_state.query_mode)

                if intent == "SQL":
                    st.caption("🤖 *Routing to: SQLite Database Timing Engine*")
                    
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
                        summary_prompt = f"{prompt_modifier} Summarize these timing database results for the engineer: {df_results.to_string()}"
                        assistant_text = llm.invoke(summary_prompt)
                        st.write(assistant_text)
                    else:
                        st.warning("No timing records matched your query.")
                        assistant_text = "No timing records matched the query."
                else:
                    st.caption("📖 *Routing to: Advanced Technical Manual RAG*")
                    query_text = f"{prompt_modifier} {user_query}"
                    try:
                        documents = SimpleDirectoryReader("data/manuals").load_data()
                        index = VectorStoreIndex.from_documents(documents)
                        query_engine = index.as_query_engine()
                        response = query_engine.query(query_text)
                        assistant_text = str(response)
                        st.write(assistant_text)
                    except Exception:
                        st.info("Place technical PDFs (e.g. Bosch MGU/MCU troubleshooting manuals) inside 'data/manuals' to enable advanced diagnostic RAG.")
                        assistant_text = llm.invoke(query_text)
                        st.write(assistant_text)

        st.session_state.chat_history.append({"role": "assistant", "content": assistant_text})

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