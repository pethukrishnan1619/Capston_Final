"""
User interface for the multi‑agent RAG system.

This module defines a Gradio interface that allows users to ask questions and
see the system's answer, citations and ReAct trace.  It relies on a graph
constructed by ``orchestrator.init_system`` and a query handler from
``orchestrator.handle_user_query``.  The interface can be launched in a
browser (for local development) or bound to a specific port (for cloud
deployment).
"""

from __future__ import annotations

import os
import json
from typing import Callable, Any

import gradio as gr


def build_interface(handle_query: Callable[[str], Any]) -> gr.Blocks:
    """Construct the Gradio Blocks interface given a query handler.

    The handler should accept a user query string and return a dict containing
    at least ``answer``, ``citations`` and ``react_steps`` keys.  Additional
    keys (e.g. ``tool_name`` or ``plan``) are ignored for display purposes.
    """
    def interface_function(user_query: str):
        res = handle_query(user_query)
        answer = res.get("answer", "")
        citations = res.get("citations", [])
        if citations:
            citation_text = "\n".join(
                [f"- {c['source']} (page {c['page']})" for c in citations]
            )
        else:
            citation_text = "No citations available."
        react_trace = json.dumps(res.get("react_steps", []), indent=2)
        return answer, citation_text, react_trace

    with gr.Blocks(title="Multi‑Agent RAG + Tools") as demo:
        gr.Markdown("## 🧠 Multi‑Agent RAG System")
        with gr.Row():
            with gr.Column(scale=2):
                user_query = gr.Textbox(
                    label="Ask your question",
                    placeholder="Examples: weather in Chennai | calculate (10+20)/2 | Applications of AI",
                    lines=3,
                )
                ask_btn = gr.Button("Ask")
                answer_box = gr.Textbox(label="Answer", lines=5, interactive=False)
                citation_box = gr.Textbox(label="Citations", lines=5, interactive=False)
            with gr.Column(scale=1):
                react_box = gr.Code(
                    label="ReAct Trace (Reason → Act → Observe)",
                    language="json",
                    lines=24,
                )
        ask_btn.click(
            fn=interface_function,
            inputs=user_query,
            outputs=[answer_box, citation_box, react_box],
        )
    return demo


def launch_demo(handle_query: Callable[[str], Any]) -> None:
    """Launch the Gradio interface using the provided query handler.

    The server binds to 0.0.0.0 and uses the port specified by the ``PORT``
    environment variable if present, otherwise 7860.  This enables both
    local development and hosting on platforms like Render, which expose the
    application on a specific port.
    """
    demo = build_interface(handle_query)
    port = int(os.environ.get("PORT", 7860))
    # Launch the demo; in a cloud environment (e.g. Render), `inbrowser` should be False
    demo.launch(server_name="0.0.0.0", server_port=port)
    # demo.launch(server_name="127.0.0.1", server_port=port, inbrowser=True)
