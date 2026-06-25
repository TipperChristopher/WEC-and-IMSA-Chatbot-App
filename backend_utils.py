import sqlite3
from typing import Any, List, Optional

import pandas as pd
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex


def route_query_source(user_query: Optional[str]) -> str:
    """
    Determines if the question requires the timing database or RAG document engine.
    """
    if not user_query:
        return "RAG"

    query_lower = user_query.lower()

    sql_keywords = ["fastest lap", "sector", "lap time", "gap", "position", "pit stop duration"]
    if any(k in query_lower for k in sql_keywords):
        return "SQL"

    return "RAG"


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


def execute_safe_query(sql_query: str) -> pd.DataFrame:
    forbidden = ["drop", "delete", "insert", "update", "alter", "truncate"]
    if any(k in sql_query.lower() for k in forbidden):
        return pd.DataFrame()

    conn = sqlite3.connect("trackside_timing.db")
    try:
        df = pd.read_sql_query(sql_query, conn)
        return df
    except Exception:
        return pd.DataFrame()
    finally:
        conn.close()


def get_series_options() -> List[str]:
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


fetch_series_options = get_series_options
