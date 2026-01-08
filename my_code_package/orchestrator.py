"""
Orchestrator for the multi‑agent RAG system.

This module exposes functions to initialise the system (loading the PDF,
building the vector store and LLM), build the LangGraph workflow and handle
user queries.  It uses the agents defined in ``my_code_package.agents`` and
the RAG utilities from ``my_code_package.rag``.
"""

from __future__ import annotations

import os
from typing import Dict, Any

from langgraph.graph import StateGraph, END

from .rag import initialize_vector_store, create_local_llm
from .agents import (
    WorkflowState,
    set_vector_store,
    set_llm,
    planning_agent,
    retrieval_agent,
    tool_execution_agent,
    synthesis_agent,
    route_agent,
)


def init_system(
    pdf_path: str,
    faiss_directory: str = "./faiss_store",
    model_id: str = "google/flan-t5-base",
    max_new_tokens: int = 220,
) -> StateGraph:
    """Initialise the RAG system and return a compiled LangGraph.

    Args:
        pdf_path: Path to the PDF to ingest and build the vector store from.
        faiss_directory: Directory to store/load the FAISS index.
        model_id: Identifier of the HuggingFace model to use for the LLM.
        max_new_tokens: Maximum tokens to generate per answer.

    Returns:
        A compiled LangGraph ready to execute queries.
    """
    # Build or load FAISS index
    vs = initialize_vector_store(pdf_path=pdf_path, faiss_directory=faiss_directory)
    set_vector_store(vs)

    # Load local LLM
    llm = create_local_llm(model_id=model_id, max_new_tokens=max_new_tokens)
    set_llm(llm)

    # Build graph
    graph = StateGraph(WorkflowState)
    graph.add_node("planner", planning_agent)
    graph.add_node("rag", retrieval_agent)
    graph.add_node("tool", tool_execution_agent)
    graph.add_node("synth", synthesis_agent)
    graph.set_entry_point("planner")
    # Conditional routing from planner
    graph.add_conditional_edges("planner", route_agent, {"rag": "rag", "tool": "tool"})
    # Linear connections to synthesiser and end
    graph.add_edge("rag", "synth")
    graph.add_edge("tool", "synth")
    graph.add_edge("synth", END)
    app_graph = graph.compile()
    return app_graph


def handle_user_query(graph: StateGraph, user_query: str) -> Dict[str, Any]:
    """Invoke the graph for a single user query and return structured results."""
    init_state: WorkflowState = {
        "user_query": user_query,
        "operation": "",
        "plan": "",
        "react_steps": [],
        "retrieved_context": "",
        "citations": [],
        "tool_name": "",
        "tool_input": {},
        "tool_result": {},
        "final_answer": "",
        "error": "",
    }
    out = graph.invoke(init_state)
    return {
        "query": user_query,
        "operation": out.get("operation"),
        "plan": out.get("plan"),
        "answer": out.get("final_answer"),
        "citations": out.get("citations", []),
        "react_steps": out.get("react_steps", []),
        "tool_name": out.get("tool_name", ""),
        "tool_result": out.get("tool_result", {}),
    }