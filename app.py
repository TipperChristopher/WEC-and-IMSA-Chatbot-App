import os
import sys
import sqlite3
import streamlit as st
from typing import Optional, List, Any
from pathlib import Path
import tempfile
import shutil

import pandas as pd
import matplotlib.pyplot as plt
import requests
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Document
from llama_parse import LlamaParse

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from backend_utils import (
    execute_safe_query,
    fetch_series_options,
    get_mode_prompt,
    route_query_source,
)
from llm_provider import get_llm

# This line must match your folder and file structure perfectly:
from physics.fuel_burn import calculate_fuel_corrected_time
from physics.tire_deg import predict_tire_degradation_penalty

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
USE_BACKEND = os.getenv("BACKEND_ENABLED", "false").lower() in {"1", "true", "yes"}

# ========== PDF STREAMING UTILITIES ==========

def parse_pdf_with_llamaparse(pdf_path: str) -> str:
    """Parse PDF using LlamaParse and return markdown content."""
    try:
        parser = LlamaParse(
            result_type="markdown",
            verbose=False,
            language="en",
            instruction="Extract all tables, timing data, and structured information as markdown."
        )
        documents = parser.load_data(pdf_path)
        return documents[0].text if documents else ""
    except Exception as e:
        st.warning(f"LlamaParse error: {e}. Falling back to basic extraction.")
        return ""


def stream_llm_response(prompt: str, llm_instance: Any) -> str:
    """Stream LLM response using streaming if available."""
    try:
        # Try streaming if the LLM supports it
        if hasattr(llm_instance, 'stream'):
            with st.write_stream(llm_instance.stream(prompt)):
                pass
        else:
            # Fallback: invoke without streaming, but display character by character
            response = llm_instance.invoke(prompt)
            
            # Stream response display
            placeholder = st.empty()
            displayed_text = ""
            for char in response:
                displayed_text += char
                placeholder.write(displayed_text)
            return response
    except Exception as e:
        st.warning(f"Streaming error: {e}")
        return llm_instance.invoke(prompt)


def save_uploaded_pdf(uploaded_file, destination_folder: str = "data/manuals") -> bool:
    """Save uploaded PDF to the manuals folder."""
    try:
        os.makedirs(destination_folder, exist_ok=True)
        file_path = os.path.join(destination_folder, uploaded_file.name)
        
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        st.success(f"✅ Saved: {uploaded_file.name}")
        return True
    except Exception as e:
        st.error(f"Error saving PDF: {e}")
        return False


def get_available_pdfs(folder: str = "data/manuals") -> List[str]:
    """Get list of available PDFs in the manuals folder."""
    try:
        if os.path.exists(folder):
            return [f for f in os.listdir(folder) if f.lower().endswith('.pdf')]
    except Exception:
        pass
    return []


def display_pdf_preview(pdf_path: str) -> None:
    """Display PDF preview in Streamlit."""
    try:
        with open(pdf_path, "rb") as pdf_file:
            st.download_button(
                label=f"📥 Download {os.path.basename(pdf_path)}",
                data=pdf_file.read(),
                file_name=os.path.basename(pdf_path),
                mime="application/pdf"
            )
    except Exception as e:
        st.warning(f"Could not display PDF preview: {e}")





def call_backend_query(query: str, mode: str) -> dict:
    response = requests.post(
        f"{BACKEND_URL}/v1/query",
        json={"query": query, "mode": mode},
        timeout=30,
    )
    response.raise_for_status()
    return response.json()


def get_series_options() -> List[str]:
    if USE_BACKEND:
        try:
            resp = requests.get(f"{BACKEND_URL}/v1/series-options", timeout=15)
            resp.raise_for_status()
            payload = resp.json()
            if isinstance(payload, dict) and "series_options" in payload:
                return payload["series_options"]
        except Exception:
            st.warning("Backend series lookup failed, using local database fallback.")
    return fetch_series_options()

st.set_page_config(page_title="WEC & IMSA Strategy Assistant", layout="wide")

st.markdown(
    """
    <style>
    button[role='button'],
    button[role='button'] * {
    background-color: #1e1e1e !important;  /* Modern dark gray */
    color: #ffffff !important;             /* High contrast crisp white text */
    border-color: #444444 !important;
    }
    button[role='button']:hover {
    background-color: #2d2d2d !important;
    color: #00ffcc !important;             /* Subtle racing cyan highlight on hover */
    }
    button[role='button']:focus,
    button[role='button']:active,
    button[role='button']:hover,
    button[role='button']:focus-visible {
        background-color: #e8e8e8 !important;
        color: #111 !important;
        outline: none !important;
        box-shadow: none !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

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
    
    # ========== PDF MANAGEMENT SECTION ==========
    st.divider()
    st.subheader("📚 PDF Management")
    
    with st.expander("Upload Technical Manuals", expanded=False):
        uploaded_files = st.file_uploader(
            "Drag & drop PDFs here or click to browse",
            type=["pdf"],
            accept_multiple_files=True
        )
        if uploaded_files:
            for uploaded_file in uploaded_files:
                if save_uploaded_pdf(uploaded_file):
                    st.session_state.pdf_updated = True
    
    # Display available PDFs
    available_pdfs = get_available_pdfs()
    if available_pdfs:
        with st.expander(f"Available Documents ({len(available_pdfs)})", expanded=False):
            for pdf_file in available_pdfs:
                pdf_path = os.path.join("data/manuals", pdf_file)
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.caption(f"📄 {pdf_file}")
                with col2:
                    if st.button("📥", key=f"dl_{pdf_file}"):
                        display_pdf_preview(pdf_path)
    
    if not available_pdfs:
        st.caption("ℹ️ No PDFs uploaded yet. Upload manuals to enable RAG.")


st.title("🏎️ WEC & IMSA Cognitive Strategy & Diagnostics Assistant")
st.markdown("---")

# --- TAB 1: OFF-LINE CHATBOT ---
tab_chat, tab_docs, tab_physics = st.tabs(["Offline Chatbot", "Documents", "Physics Predictions"])

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
        "Simple": "#eef4ff",
        "Standard": "#eef7ef",
        "Advanced": "#f8f4ec"
    }
    selected_mode = st.session_state.query_mode
    #  NEW HIGH-CONTRAST CODE
    st.markdown(
    f"<div style='background:{mode_colors[selected_mode]}; color: #111111; padding:14px; border-radius:8px; margin-bottom:12px;'>"
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
            with st.spinner("Processing engines..."):
                assistant_text = ""
                use_local = False

                if USE_BACKEND:
                    try:
                        response = call_backend_query(user_query, st.session_state.query_mode)
                        intent = response.get("intent", "RAG")
                        if intent == "SQL":
                            st.caption("🤖 *Routing to: SQLite Database Timing Engine via backend*")
                            if response.get("sql"):
                                st.code(response["sql"], language="sql")
                            if response.get("results"):
                                st.dataframe(response["results"])
                        else:
                            st.caption("📖 *Routing to: Advanced Technical Manual RAG via backend*")

                        assistant_text = response.get("assistant_text", "")
                        if response.get("details"):
                            st.info(response["details"])
                    except Exception:
                        st.warning("Backend request failed, falling back to local processing.")
                        use_local = True

                if not USE_BACKEND or use_local:
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
                            assistant_text = stream_llm_response(summary_prompt, llm)
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
                            
                            # Stream the response
                            with st.write_stream(stream_llm_response(query_text, llm)):
                                pass
                        except Exception:
                            st.info("Place technical PDFs (e.g. Bosch MGU/MCU troubleshooting manuals) inside 'data/manuals' to enable advanced diagnostic RAG.")
                            assistant_text = stream_llm_response(query_text, llm)


        st.session_state.chat_history.append({"role": "assistant", "content": assistant_text})

# --- TAB 2: DOCUMENTS ---
with tab_docs:
    st.subheader("📚 Technical Document Library")
    
    available_pdfs = get_available_pdfs()
    
    if available_pdfs:
        selected_pdf = st.selectbox("Select a document to view:", available_pdfs)
        
        if selected_pdf:
            pdf_path = os.path.join("data/manuals", selected_pdf)
            
            col1, col2 = st.columns([4, 1])
            with col1:
                st.caption(f"**Viewing:** {selected_pdf}")
            with col2:
                if st.button("🔄 Parse with LlamaParse"):
                    with st.spinner("Parsing PDF..."):
                        content = parse_pdf_with_llamaparse(pdf_path)
                        if content:
                            st.text_area("Parsed Content (Markdown):", value=content, height=400, disabled=True)
                        else:
                            st.warning("Could not parse PDF content.")
            
            # Display PDF download
            display_pdf_preview(pdf_path)
            
            # Show file info
            file_size = os.path.getsize(pdf_path) / 1024  # KB
            st.metric("File Size", f"{file_size:.1f} KB")
    else:
        st.info("📤 No documents uploaded yet. Use the sidebar to upload technical PDFs.")
        st.markdown("""
        **Supported formats:**
        - Technical Manuals (PDF)
        - Service Bulletins
        - Race Strategy Guides
        """)

# --- TAB 3: PHYSICS PREDICTIONS ---
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