# diagnostics/orchestrator.py
from typing import Literal, Dict, Any
from llm_provider import get_llm
from langchain_core.tools import tool

llm: Any = get_llm()

@tool
def route_engineer_query(query: str) -> Literal["sql_agent", "rag_diagnostics"]:
    """
    Routes query based on engineer intent.
    If they ask about lap times, positions, sectors, or entry lists -> SQL Agent.
    If they ask about trouble-shooting, rulebooks, fault codes -> RAG Diagnostics.
    """
    routing_prompt = f"""
    Analyze this motorsport engineer query: "{query}".
    Classify the routing path. Respond with exactly "sql_agent" or "rag_diagnostics".
    """
    decision = llm.invoke(routing_prompt).strip().lower()
    return "sql_agent" if "sql" in decision else "rag_diagnostics"