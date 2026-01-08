# Multi‑Agent Retrieval‑Augmented Generation (RAG) System

This repository contains a complete Python package and CLI application
implementing a multi‑agent RAG system with tool integration and a web UI.

The system ingests a PDF document, builds a vector store of its contents,
retrieves relevant passages to answer user questions and, when appropriate,
calls external tools (weather forecast, scientific calculator or custom text
processing functions) to generate the final answer.  It is orchestrated
using the [LangGraph](https://python.langchain.com/docs/guides/agents/langgraph/) framework for
deterministic state‑machine control flow.

## Project Structure

```
your‑project/
├── my_code_package/          # Python package containing modules
│   ├── __init__.py
│   ├── rag.py                # PDF ingestion, vector store and LLM helpers
│   ├── tools.py              # Weather, calculator and custom text tools
│   ├── agents.py             # Planner, retriever, tool executor and synthesiser
│   ├── orchestrator.py       # System initialiser and LangGraph builder
│   └── interface.py          # Gradio UI construction and launch helper
├── main.py                   # Entry point for CLI or UI
├── requirements.txt          # Python dependencies
├── README.md                 # Project overview and usage instructions
├── data/
│   └── artificial_intelligence_tutorial.pdf  # Knowledge base for RAG
├── faiss_store/
│   └── .gitkeep              # Placeholder for the FAISS index (generated at runtime)
└── .gitignore               # Exclude the FAISS index from version control
```

## Features

* **Multi‑Agent Design:** Separate agents handle planning, retrieval, tool
  execution and synthesis.  Responsibilities are clearly separated, making
  the system easier to maintain and extend.
* **ReAct Reasoning:** Uses a Reason → Act → Observe loop.  Every tool call
  is justified by the planner and logged in a ReAct trace for transparency.
* **Tool Integration:** Provides three tools:
  - **Weather Tool** using the Open‑Meteo API (no API key required).
  - **Scientific Calculator** supporting arithmetic and common mathematical
    functions (sin, cos, tan, sqrt, log, factorial, etc.).
  - **Custom Python Tool** with ten text processing operations (word count,
    extract numbers, title case, reverse text, lower/upper case, remove
    punctuation, remove extra spaces, sentence count and unique words).
* **Stateful Orchestration:** Orchestrated with LangGraph; a state machine
  determines which agent runs next and prevents infinite loops.
* **User Interface:** A Gradio web UI lets users ask questions and view
  answers, citations and the underlying ReAct trace.  The system can also
  run in a CLI loop for environments without a browser.

## Setup

1. **Clone the repository and navigate into it.**

2. **Create and activate a virtual environment (optional but recommended):**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```

3. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application:**

   The system requires a PDF to build its knowledge base.  The provided
   `artificial_intelligence_tutorial.pdf` is included under the `data/` folder.

   *To launch the web UI (Gradio):*

   ```bash
   python main.py --pdf data/artificial_intelligence_tutorial.pdf
   ```

   This command will initialise the vector store and local language model,
   then open the UI in your default web browser.  On first run the model and
   index may take a few moments to build; subsequent runs will reuse the
   cached FAISS index under `faiss_store/`.

   *To run in command‑line mode (no UI):*

   ```bash
   python main.py --pdf data/artificial_intelligence_tutorial.pdf --no-ui
   ```

   You can then type questions directly into the terminal.  Type `quit` or
   `exit` to leave.

5. **Deploying to the cloud:**

   The UI binds to the port specified by the `PORT` environment variable if
   present, which enables deployment on platforms like Render.  The Gradio
   server listens on `0.0.0.0`.  See the comments in `main.py` for details.

## Notes

* The FAISS index files (`index.faiss` and `index.pkl`) are created under
  `faiss_store/` at runtime.  They can be large and are excluded from Git via
  `.gitignore`.
* You can replace the PDF in `data/` with your own document to customise the
  knowledge base.  The system will automatically rebuild the index on first
  run.

## License

This project is provided for educational purposes.  Feel free to adapt it to
your own use cases.