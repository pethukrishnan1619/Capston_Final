"""
RAG utilities for the multi‑agent system.

This module provides helper functions to extract text from a PDF, split it into
chunks, build a FAISS vector store using sentence embeddings, and perform
similarity search.  It also includes a convenience function for creating a
local HuggingFace model pipeline and a function to generate a grounded answer
from retrieved context.
"""

from __future__ import annotations

import os
from typing import List, Tuple, Dict

from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.faiss import FAISS
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from langchain_core.documents import Document


def extract_pdf_pages(path: str) -> List[Document]:
    """Extract pages from a PDF and return them as langchain Document objects.

    Each document's metadata includes the source filename and page number.  If
    no text is extracted from a page, that page is skipped.
    """
    reader = PdfReader(path)
    docs: List[Document] = []
    for i, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            docs.append(Document(page_content=text, metadata={"source": os.path.basename(path), "page": i}))
    return docs


def initialize_vector_store(
    pdf_path: str,
    faiss_directory: str = "./faiss_store",
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    chunk_size: int = 500,
    chunk_overlap: int = 100,
) -> FAISS:
    """Initialise or load a FAISS vector store from the given PDF.

    If a FAISS index already exists in the specified directory, it is loaded
    (along with the stored embeddings).  Otherwise, the PDF is read, split
    into chunks and indexed using sentence embeddings.  The resulting vector
    store is persisted to disk so subsequent runs can reuse it.

    Args:
        pdf_path: Path to the PDF file to ingest.
        faiss_directory: Directory where the FAISS index will be stored.
        embedding_model: Name of the HuggingFace sentence transformer model.
        chunk_size: Maximum size of each text chunk.
        chunk_overlap: Number of overlapping characters between chunks.

    Returns:
        A FAISS vector store ready for similarity search.
    """
    os.makedirs(faiss_directory, exist_ok=True)
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model)

    # If existing index exists, load it
    index_path = os.path.join(faiss_directory, "index.faiss")
    if os.path.exists(index_path):
        # allow_dangerous_deserialization is required when loading untrusted data
        return FAISS.load_local(faiss_directory, embeddings, allow_dangerous_deserialization=True)

    # Build a new index from the PDF
    docs = extract_pdf_pages(pdf_path)
    if not docs:
        raise ValueError("No text extracted from PDF. If it is scanned (image-only), you need OCR.")

    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_documents(docs)

    vs = FAISS.from_documents(chunks, embeddings)
    vs.save_local(faiss_directory)
    return vs


def retrieve_rag_chunks(vector_store: FAISS, query: str, k: int = 3) -> Tuple[str, List[Dict[str, int]]]:
    """Retrieve the top-k most similar chunks from the vector store for a query.

    The returned context is a single string with the page content of each
    retrieved document separated by blank lines.  Citations are returned as a
    list of dictionaries containing the source filename and page number.

    Args:
        vector_store: A FAISS vector store previously initialised.
        query: The user query to search for.
        k: Number of chunks to retrieve.

    Returns:
        context: Concatenated page contents of retrieved documents.
        citations: List of metadata dictionaries for each document.
    """
    retrieved_docs = vector_store.similarity_search(query, k=k)
    context = "\n\n".join(
        f"(source={d.metadata.get('source')}, page={d.metadata.get('page')})\n{d.page_content}"
        for d in retrieved_docs
    )
    citations = [{"source": d.metadata.get("source"), "page": d.metadata.get("page")} for d in retrieved_docs]
    return context, citations


def create_local_llm(model_id: str = "google/flan-t5-base", max_new_tokens: int = 220) -> HuggingFacePipeline:
    """Create a local HuggingFace text2text generation pipeline.

    Loads the specified model and tokenizer and wraps it in a pipeline.  This
    function does not cache the model; repeated calls will reload the model.

    Args:
        model_id: HuggingFace model identifier (e.g., google/flan-t5-base).
        max_new_tokens: Maximum number of tokens to generate per answer.

    Returns:
        A HuggingFacePipeline compatible with LangChain.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
    gen_pipe = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=max_new_tokens,
    )
    return HuggingFacePipeline(pipeline=gen_pipe)


def generate_answer_from_context(llm: HuggingFacePipeline, context: str, query: str) -> str:
    """Generate a grounded answer given retrieved context and user query.

    The LLM is prompted to answer strictly using the provided context.  If the
    answer is not in the context, it should state that it doesn't know.

    Args:
        llm: The HuggingFace pipeline to use for generation.
        context: The concatenated context retrieved by the RAG component.
        query: The original user question.

    Returns:
        The generated answer as a string.
    """
    prompt = (
        "Answer the question strictly using the context below.\n"
        "If the answer is not present in the context, say: \"I don't know from the provided documents.\"\n\n"
        f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
    )
    return llm.invoke(prompt)