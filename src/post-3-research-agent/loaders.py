"""loaders.py — Document loading, chunking, embeddings, and FAISS indexing.

Handles multi-source ingestion:
  - PDF uploads via PyPDFLoader
  - Text / Markdown files via TextLoader (UTF-8)
  - Web URLs via WebBaseLoader

Per-source error isolation ensures one corrupted file or unreachable URL does not
block other documents. Every chunk carries metadata ('loader', 'source_name', 'source',
'page') for clear attribution in search results and citation cards.
"""

import hashlib
import time
from dataclasses import dataclass, field

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    WebBaseLoader,
)
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


@dataclass
class SourceSpec:
    """Descriptor for a document source selected in the sidebar."""

    source_type: str  # 'pdf' | 'text' | 'url'
    loader_name: str  # human-readable loader name
    path_or_url: str  # temp file path or HTTP URL
    display_name: str  # human-friendly label (file name or URL)
    content_hash: str = ""  # sha256 or URL string for fingerprinting


@dataclass
class SourceLoadResult:
    """Telemetry and documents resulting from loading one source."""

    spec: SourceSpec
    docs: list[Document] = field(default_factory=list)
    load_seconds: float = 0.0
    error: str | None = None


def get_embeddings() -> HuggingFaceEmbeddings:
    """Return cached HuggingFace sentence-transformers embeddings."""
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


def hash_bytes(data: bytes) -> str:
    """Compute sha256 digest of file bytes for caching index state."""
    return hashlib.sha256(data).hexdigest()


def load_source(spec: SourceSpec) -> SourceLoadResult:
    """Load a single source. Never raises; captures exceptions cleanly."""
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
            raise ValueError(f"Unknown source type: '{spec.source_type}'")

        docs = loader.load()
        if not docs or all(not d.page_content.strip() for d in docs):
            raise ValueError("No extractable text found (file may be empty or unparseable).")

        for d in docs:
            d.metadata["loader"] = spec.loader_name
            d.metadata["source_name"] = spec.display_name
            d.metadata["source"] = spec.display_name
            d.metadata["source_type"] = spec.source_type

        result.docs = docs
    except Exception as e:
        result.error = f"{type(e).__name__}: {e}"
    result.load_seconds = time.perf_counter() - t0
    return result


def chunk_documents(docs: list[Document], chunk_size: int = 1000, chunk_overlap: int = 200) -> list[Document]:
    """Split documents into chunks preserving metadata."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return splitter.split_documents(docs)


def build_vector_store(chunks: list[Document], embeddings) -> FAISS:
    """Build an in-memory FAISS vector store from document chunks."""
    return FAISS.from_documents(chunks, embeddings)
