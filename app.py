"""
RAG Demo: Upload a PDF and ask questions about it.

Built with LangChain + Groq (open-source models) + Streamlit.
- PDF parsing: PyPDF
- Chunking: RecursiveCharacterTextSplitter
- Embeddings: HuggingFace sentence-transformers (free, local)
- Vector store: FAISS (in-memory)
- LLM: Groq open-source chat models (default: openai/gpt-oss-120b)
- The retriever is exposed as a LangChain *Tool* so the LLM's answer
  is grounded in retrieved chunks.
"""

import os

import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

# ---------- Page config ----------
st.set_page_config(page_title="PDF RAG Demo (Groq)", page_icon="📄", layout="wide")
st.title("📄 PDF RAG Demo")
st.caption(
    "Upload a PDF and ask questions. Powered by LangChain + Groq open-source models "
    "+ HuggingFace embeddings + FAISS."
)


# ---------- Cached resources ----------
@st.cache_resource(show_spinner=False)
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


@st.cache_resource(show_spinner=False)
def get_llm(model: str):
    return ChatGroq(
        model=model,
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0,
    )


PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            ("You are a helpful assistant answering questions about an uploaded "
            "PDF document. Use ONLY the following context from the document to "
            "answer. If the answer is not in the context, say you cannot find "
            "it in the document.\n\n"
            "Context:\n{context}"),
        ),
        ("human", "{question}"),
    ]
)


def format_docs(docs):
    return "\n\n---\n\n".join(d.page_content for d in docs)


# ---------- Sidebar: PDF upload & indexing ----------
with st.sidebar:
    st.header("1. Upload PDF")
    uploaded = st.file_uploader("Choose a PDF", type=["pdf"])

    vectorstore = None
    if uploaded is not None:
        with st.spinner("Indexing PDF... (parsing → chunking → embedding)"):
            # Persist the upload to a temp file so PyPDFLoader can read it
            tmp_path = os.path.join(st.session_state.get("tmp_dir", "."), uploaded.name)
            os.makedirs(os.path.dirname(tmp_path), exist_ok=True)
            with open(tmp_path, "wb") as f:
                f.write(uploaded.getvalue())

            docs = PyPDFLoader(tmp_path).load()
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000, chunk_overlap=200
            )
            chunks = splitter.split_documents(docs)
            vectorstore = FAISS.from_documents(chunks, get_embeddings())

        st.success(f"✅ Indexed {len(chunks)} chunks from {len(docs)} pages.")

        model = st.selectbox(
            "Groq model",
            ["openai/gpt-oss-120b", "qwen/qwen3.8-27b", "allam-2-7b"],
            index=0,
        )
        st.header("2. Ask questions")

# ---------- Main: Q&A ----------
if vectorstore is None:
    st.info("👈 Upload a PDF in the sidebar to get started.")
    st.stop()

st.header("3. Chat with your document")

# Expose the retriever as a LangChain tool so the pipeline is tool-based
@tool
def pdf_search(question: str) -> list:
    """Search the uploaded PDF and return the most relevant chunks for a question."""
    return vectorstore.as_retriever(search_kwargs={"k": 4}).invoke(question)


question = st.text_input("Your question about the PDF:", key="q")
ask = st.button("Ask", type="primary")

if ask and question:
    with st.spinner("Searching the document and generating an answer..."):
        context_docs = pdf_search.invoke({"question": question})
        chain = PROMPT | get_llm(model)
        answer = chain.invoke(
            {"context": format_docs(context_docs), "question": question}
        )

    st.markdown("### 💬 Answer")
    st.write(answer.content)

    with st.expander("🔎 Retrieved chunks (sources)"):
        for i, doc in enumerate(context_docs, 1):
            page = doc.metadata.get("page", "?")
            st.markdown(f"**Chunk {i} — page {int(page) + 1 if isinstance(page, int) else page}**")
            st.text(doc.page_content[:800])
elif ask:
    st.warning("Please enter a question first.")
