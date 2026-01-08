"""
Agent definitions for the multi‑agent RAG system.

This module defines the shared state type and four agents: planner, retriever,
tool executor and synthesiser.  These agents operate on ``WorkflowState``
dictionaries and mutate state to reflect decisions, retrieved context, tool
outputs and final answers.  The module also exposes functions to set the
global vector store and LLM used by the retrieval and synthesis agents.
"""

from __future__ import annotations

from typing import List, Dict, Any, TypedDict
import re
import json

from .tools import weather_tool_call, calculator_tool_call, python_custom_tool_call
from .rag import retrieve_rag_chunks, generate_answer_from_context

# Globals to store the vector store and LLM; these are set by orchestrator
_vector_store = None
_llm = None


def set_vector_store(vector_store) -> None:
    """Set the global FAISS vector store for retrieval."""
    global _vector_store
    _vector_store = vector_store


def set_llm(llm) -> None:
    """Set the global HuggingFace pipeline for synthesis."""
    global _llm
    _llm = llm


class WorkflowState(TypedDict):
    """Typed dictionary defining the state shared across agents."""
    user_query: str
    operation: str            # "rag" | "tool"
    plan: str
    react_steps: List[Dict[str, Any]]
    retrieved_context: str
    citations: List[Dict[str, Any]]
    tool_name: str
    tool_input: Dict[str, Any]
    tool_result: Dict[str, Any]
    final_answer: str
    error: str


def planning_agent(state: WorkflowState) -> WorkflowState:
    """Determine whether to call RAG or a tool based on the query.

    Uses simple heuristics: if the query contains keywords related to weather,
    call the weather tool; if it contains arithmetic terms or the word
    "calculate", call the calculator; if it contains certain phrases for text
    processing, call the custom Python tool; otherwise use RAG.

    Updates the state with the chosen operation, tool name, tool input and
    plan.  Records the reasoning step in ``react_steps``.
    """
    q = state["user_query"].strip()
    ql = q.lower()

    # Reset fields for new plan
    state["react_steps"] = []
    state["error"] = ""
    state["retrieved_context"] = ""
    state["citations"] = []
    state["tool_result"] = {}
    state["tool_input"] = {}
    state["tool_name"] = ""

    # Routing rules
    if "weather" in ql or "temperature" in ql or "forecast" in ql:
        state["operation"] = "tool"
        state["tool_name"] = "weather"
        # Extract location after "in" if present, else default to Chennai
        if " in " in ql:
            loc = q.split(" in ", 1)[1].strip()
        else:
            loc = "Chennai"
        state["tool_input"] = {"location": loc, "days": 3}
        state["plan"] = "Call weather tool (no API key) using Open‑Meteo."

    elif "calculate" in ql or re.search(r"\d+\s*[\+\-\*\/\%\*\*]\s*\d+", q):
        state["operation"] = "tool"
        state["tool_name"] = "calculator"
        expr = re.sub(r"(?i)\bcalculate\b", "", q).strip()
        # Use the entire query if expression extraction yields empty string
        state["tool_input"] = {"expression": expr if expr else q}
        state["plan"] = "Call calculator tool to compute the expression."

    elif any(k in ql for k in [
        "word count", "count words",
        "extract numbers",
        "title case",
        "reverse text",
        "lower case",
        "upper case",
        "remove punctuation",
        "remove extra spaces",
        "sentence count",
        "unique words",
    ]):
        state["operation"] = "tool"
        state["tool_name"] = "python_custom"
        # Determine which operation to perform
        if "word count" in ql or "count words" in ql:
            op = "word_count"
        elif "extract numbers" in ql:
            op = "extract_numbers"
        elif "title case" in ql:
            op = "title_case"
        elif "reverse text" in ql:
            op = "reverse_text"
        elif "lower case" in ql:
            op = "lower_case"
        elif "upper case" in ql:
            op = "upper_case"
        elif "remove punctuation" in ql:
            op = "remove_punctuation"
        elif "remove extra spaces" in ql:
            op = "remove_extra_spaces"
        elif "sentence count" in ql:
            op = "sentence_count"
        elif "unique words" in ql:
            op = "unique_words"
        else:
            op = "word_count"
        # Extract text after the colon or use the whole query
        text_part = q.split(":", 1)[1].strip() if ":" in q else q
        state["tool_input"] = {"operation": op, "text": text_part}
        state["plan"] = f"Call python_custom tool for '{op}'."

    else:
        state["operation"] = "rag"
        state["plan"] = "Use RAG to retrieve context from PDF and answer grounded with citations."

    state["react_steps"].append({"reason": state["plan"]})
    return state


def retrieval_agent(state: WorkflowState) -> WorkflowState:
    """Retrieve relevant context from the vector store for the user's query."""
    if _vector_store is None:
        raise RuntimeError("Vector store is not initialised. Call set_vector_store() before invoking retrieval_agent.")
    q = state["user_query"]
    state["react_steps"].append({"act": "RAG.retrieve", "input": q})
    context, citations = retrieve_rag_chunks(_vector_store, q, k=3)
    state["retrieved_context"] = context
    state["citations"] = citations
    state["react_steps"].append({"observe": f"Retrieved {len(citations)} chunks"})
    return state


def tool_execution_agent(state: WorkflowState) -> WorkflowState:
    """Execute the appropriate tool based on ``state.tool_name`` and store the result."""
    tname = state.get("tool_name", "")
    tinp = state.get("tool_input", {})
    state["react_steps"].append({"act": "Tool.call", "tool": tname, "input": tinp})
    if tname == "weather":
        result = weather_tool_call(tinp)
    elif tname == "calculator":
        result = calculator_tool_call(tinp)
    elif tname == "python_custom":
        result = python_custom_tool_call(tinp)
    else:
        result = {"ok": False, "error": f"Unknown tool: {tname}"}
    state["tool_result"] = result
    state["react_steps"].append({"observe": result})
    return state


def synthesis_agent(state: WorkflowState) -> WorkflowState:
    """Synthesise the final answer based on the chosen operation (rag or tool)."""
    op = state.get("operation", "")
    if op == "rag":
        if _llm is None:
            raise RuntimeError("LLM is not initialised. Call set_llm() before invoking synthesis_agent.")
        q = state["user_query"]
        context = state.get("retrieved_context", "")
        state["react_steps"].append({"act": "LLM.generate_grounded_answer"})
        ans = generate_answer_from_context(_llm, context, q)
        state["final_answer"] = ans
        # Keep citations from retrieved docs (already in state)
    elif op == "tool":
        tname = state.get("tool_name", "")
        result = state.get("tool_result", {})
        # Tool output formatting
        if tname == "weather" and result.get("ok"):
            lines = [f"Weather Forecast for {result.get('location')}:"]
            for d in result.get("forecast", []):
                lines.append(
                    f"- {d['date']}: min {d['temp_min_c']}°C, max {d['temp_max_c']}°C, "
                    f"rain {d['precip_mm']}mm, wind {d['wind_max_kmh']}km/h"
                )
            state["final_answer"] = "\n".join(lines)
        elif tname == "calculator" and result.get("ok"):
            state["final_answer"] = f"Result: {result.get('result')}"
        elif tname == "python_custom" and result.get("ok"):
            state["final_answer"] = "Python Custom Tool Output:\n" + json.dumps(result, indent=2)
        else:
            state["final_answer"] = f"Tool '{tname}' failed: {json.dumps(result, indent=2)}"
        # Tools don't produce citations
        state["citations"] = []
    else:
        state["final_answer"] = "Unsupported operation."
        state["citations"] = []
    return state


def route_agent(state: WorkflowState) -> str:
    """Routing function used by LangGraph to decide next node."""
    return "tool" if state.get("operation") == "tool" else "rag"