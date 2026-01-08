"""
Entry point for the multi‑agent RAG system.

This script allows you to run the system either in a command‑line loop or via
a Gradio user interface.  It expects a PDF file to build the knowledge base
and will create a FAISS vector index under the specified directory.  By
default, the UI is launched; pass ``--no-ui`` to run without a web interface.
"""

from __future__ import annotations

import argparse
import os

from my_code_package.orchestrator import init_system, handle_user_query
from my_code_package.interface import launch_demo


def run_cli(graph) -> None:
    """Run an interactive command‑line loop for user queries."""
    print("Welcome to the Multi‑Agent RAG System (CLI mode).  Type 'quit' to exit.")
    while True:
        try:
            q = input("\nEnter your question: ")
        except EOFError:
            break
        if q.lower() in {"quit", "exit"}:
            break
        result = handle_user_query(graph, q)
        print("\nAnswer:\n", result.get("answer", ""))
        citations = result.get("citations", [])
        if citations:
            print("\nCitations:")
            for c in citations:
                print(f"- {c['source']} (page {c['page']})")
        else:
            print("\nNo citations available.")
        print("\nReAct Trace:")
        for step in result.get("react_steps", []):
            print(step)


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi‑agent RAG system")
    parser.add_argument("--pdf", required=True, help="Path to the PDF file to ingest")
    parser.add_argument("--faiss-dir", default="./faiss_store", help="Directory to store/load FAISS index")
    parser.add_argument("--model-id", default="google/flan-t5-base", help="HuggingFace model identifier for the LLM")
    parser.add_argument("--max-tokens", type=int, default=220, help="Maximum number of tokens to generate per answer")
    parser.add_argument("--no-ui", action="store_true", help="Run in command‑line mode without launching the UI")
    args = parser.parse_args()

    pdf_path = args.pdf
    faiss_dir = args.faiss_dir
    model_id = args.model_id
    max_tokens = args.max_tokens

    # Ensure the FAISS directory exists
    os.makedirs(faiss_dir, exist_ok=True)

    # Initialise system and build graph
    graph = init_system(
        pdf_path=pdf_path,
        faiss_directory=faiss_dir,
        model_id=model_id,
        max_new_tokens=max_tokens,
    )
    if args.no_ui:
        run_cli(graph)
    else:
        # Wrap handler so interface only needs query string
        launch_demo(lambda q: handle_user_query(graph, q))


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    demo.launch(server_name="0.0.0.0", server_port=port)
