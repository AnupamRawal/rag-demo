"""
stores.py — embeddings, vector stores, and multi-source document loading.

Each source is loaded with a dedicated LangChain loader:
  PDF upload  -> PyPDFLoader
  TXT/MD      -> TextLoader
  Web URL     -> WebBaseLoader

Every Document gets `loader` and `source_name` added to its metadata so the
sources panel can show exactly where each chunk came from.
"""

import time
from dataclasses import dataclass, field

import streamlit as st
from langchain_chroma import Chroma
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    WebBaseLoader,
)
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


@dataclass
class SourceSpec:
    """One document source selected by the user in the sidebar."""

    source_type: str  # pdf | text | url
    loader_name: str  # human-readable loader class
    path_or_url: str  # temp file path or web URL
    display_name: str  # file name or URL for the UI


@dataclass
class SourceLoadResult:
    spec: SourceSpec
    docs: list[Document] = field(default_factory=list)
    load_seconds: float = 0.0
    error: str | None = None


def get_embeddings() -> HuggingFaceEmbeddings:
    # Cached by the caller (st.cache_resource) — one model instance is reused.
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


def load_source(spec: SourceSpec) -> SourceLoadResult:
    """Load a single source with the right LangChain loader. Never raises."""
    result = SourceLoadResult(spec=spec)
    t0 = time.perf_counter()
    try:
        if spec.source_type == "pdf":
            loader = PyPDFLoader(spec.path_or_url)
        elif spec.source_type == "text":
            loader = TextLoader(spec.path_or_url, encoding="utf-8")
        elif spec.source_type == "url":
            loader = WebBaseLoader(spec.path_or_url)
        else:
            raise ValueError(f"Unknown source type: {spec.source_type}")

        docs = loader.load()
        if not docs or all(not d.page_content.strip() for d in docs):
            raise ValueError("No extractable text found (empty file?)")

        # Tag every document with loader + source metadata
        for d in docs:
            d.metadata["loader"] = spec.loader_name
            d.metadata["source_name"] = spec.display_name
            d.metadata["source_type"] = spec.source_type

        result.docs = docs
    except Exception as e:  # one bad source must not break the others
        result.error = f"{type(e).__name__}: {e}"
    result.load_seconds = time.perf_counter() - t0
    return result


def build_vector_store(store_type: str, chunks: list[Document], embeddings):
    """Create an in-memory FAISS or Chroma index from chunks."""
    if store_type == "FAISS":
        return FAISS.from_documents(chunks, embeddings)
    if store_type == "Chroma":
        # Ephemeral in-memory Chroma — no on-disk persistence for a demo
        return Chroma.from_documents(chunks, embeddings)
    raise ValueError(f"Unknown vector store: {store_type}")
