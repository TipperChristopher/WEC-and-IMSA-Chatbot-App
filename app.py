import streamlit as st

st.set_page_config(page_title="Race Strategy Core Control", layout="centered")

st.title("🏎️ Trackside Cognitive Strategy Terminal")
st.markdown("### WEC and IMSA Edge-Deployment Interface")
st.markdown("---")

st.info("👈 Use the left sidebar navigation menu to choose your active race campaign workspace.")

col1, col2 = st.columns(2)
with col1:
    st.metric("Active Database Records", "trackside_timing.db")
with col2:
    st.metric("Edge Engine Status", "Ollama: Connected")