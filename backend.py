import os
import sys
from typing import Any, Dict, List, Optional

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from backend_utils import (
    execute_safe_query,
    get_mode_prompt,
    get_series_options,
    route_query_source,
)
from llm_provider import get_llm
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex


class QueryRequest(BaseModel):
    query: str
    mode: Optional[str] = "Standard"


class QueryResponse(BaseModel):
    intent: str
    sql: Optional[str] = None
    results: Optional[List[Dict[str, Any]]] = None
    assistant_text: str
    details: Optional[str] = None


app = FastAPI(
    title="WEC & IMSA Chatbot Backend",
    description="Dedicated backend API for the WEC & IMSA strategy assistant.",
    version="1.0.0",
)

_llm: Any = None


def get_llm_instance() -> Any:
    global _llm
    if _llm is None:
        _llm = get_llm()
    return _llm


@app.get("/v1/health")
def health_check() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/v1/series-options")
def series_options() -> Dict[str, List[str]]:
    return {"series_options": get_series_options()}


@app.post("/v1/query", response_model=QueryResponse)
def query_backend(request: QueryRequest) -> QueryResponse:
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query text must not be empty.")

    llm = get_llm_instance()
    intent = route_query_source(request.query)
    assistant_text = ""
    sql_text = None
    results = None
    details = None

    if intent == "SQL":
        sql_text = (
            f"""
            Given the SQLite table 'laps' with fields:
            series_code, class, driver_name, lap_time_s, s1_s, s2_s, s3_s, pit_time_s, track_temp_f, raining.
            Generate a clean SQL SELECT query to answer: \"{request.query}\".
            Respond with ONLY the raw SQL query, no markdown blocks, no format.
            """
        )
        sql_text = llm.invoke(sql_text).strip().replace("`", "").replace("sql", "")
        df_results = execute_safe_query(sql_text)
        if not df_results.empty:
            results = df_results.to_dict(orient="records")
            summary_prompt = f"{get_mode_prompt(request.mode)} Summarize these timing database results for the engineer: {df_results.to_string()}"
            assistant_text = llm.invoke(summary_prompt)
        else:
            assistant_text = "No timing records matched the query."
            details = "The SQL query returned no rows."
    else:
        query_text = f"{get_mode_prompt(request.mode)} {request.query}"
        try:
            documents = SimpleDirectoryReader("data/manuals").load_data()
            index = VectorStoreIndex.from_documents(documents)
            query_engine = index.as_query_engine()
            response = query_engine.query(query_text)
            assistant_text = str(response)
        except Exception:
            assistant_text = llm.invoke(query_text)
            details = "No RAG documents were available, response came from the base LLM."

    return QueryResponse(
        intent=intent,
        sql=sql_text,
        results=results,
        assistant_text=assistant_text,
        details=details,
    )
