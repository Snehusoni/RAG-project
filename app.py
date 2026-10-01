import os
import time
from typing import List
import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage

from src.config import Config
from src.document_processor import DocumentProcessor
from src.vector_store import VectorStoreManager, get_embeddings
from src.rag_engine import RAGEngine, get_llm
from src.utils import export_chat_history_markdown, get_sample_document_text

# =====================================================================
# Page Configuration & Styling
# =====================================================================
st.set_page_config(
    page_title="DocuMind RAG — AI Document Intelligence",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom High-End Modern CSS
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }

    .main-header {
        background: linear-gradient(135deg, #1e1b4b 0%, #312e81 50%, #4338ca 100%);
        padding: 24px 28px;
        border-radius: 16px;
        color: white;
        margin-bottom: 24px;
        box-shadow: 0 10px 25px -5px rgba(49, 46, 129, 0.3);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    .main-header h1 {
        color: #ffffff;
        font-size: 2.1rem;
        font-weight: 700;
        margin: 0;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .main-header p {
        color: #c7d2fe;
        font-size: 0.95rem;
        margin-top: 8px;
        margin-bottom: 0;
    }

    .stat-badge {
        display: inline-block;
        padding: 4px 10px;
        background: rgba(255, 255, 255, 0.15);
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        color: #ffffff;
        margin-right: 6px;
    }

    .source-card {
        background: #0f172a;
        border: 1px solid #1e293b;
        border-left: 4px solid #6366f1;
        padding: 12px 16px;
        border-radius: 8px;
        margin-top: 8px;
        margin-bottom: 8px;
        font-size: 0.88rem;
    }

    .source-meta {
        font-weight: 600;
        color: #818cf8;
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 6px;
    }

    .confidence-pill {
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 700;
        background: #1e1b4b;
        color: #a5b4fc;
        border: 1px solid #3730a3;
    }

    .stChatMessage {
        border-radius: 12px;
        padding: 12px;
        margin-bottom: 8px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =====================================================================
# State Initialization
# =====================================================================
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # List of dicts: {'role', 'content', 'sources', 'query'}

if "langchain_messages" not in st.session_state:
    st.session_state.langchain_messages = []  # List of HumanMessage / AIMessage

if "vector_store_manager" not in st.session_state:
    st.session_state.vector_store_manager = None

if "rag_engine" not in st.session_state:
    st.session_state.rag_engine = None

if "processed_files" not in st.session_state:
    st.session_state.processed_files = []


# =====================================================================
# Helper: Lazy Engine Initializer
# =====================================================================
def initialize_rag_system(
    embedding_provider: str,
    llm_provider: str,
    llm_model: str,
    retriever_type: str,
    top_k: int,
    temperature: float,
    google_key: str,
    groq_key: str,
    openai_key: str,
):
    """Safely initialize or reconfigure Vector Store and RAG Engine."""
    try:
        # Resolve API keys
        if google_key:
            os.environ["GOOGLE_API_KEY"] = google_key
        if groq_key:
            os.environ["GROQ_API_KEY"] = groq_key
        if openai_key:
            os.environ["OPENAI_API_KEY"] = openai_key

        emb = get_embeddings(
            provider=embedding_provider,
            api_key=google_key if embedding_provider == "google" else openai_key,
        )

        vsm = VectorStoreManager(embeddings=emb)
        st.session_state.vector_store_manager = vsm

        # Active key for selected LLM
        active_llm_key = (
            google_key if llm_provider == "google"
            else groq_key if llm_provider == "groq"
            else openai_key
        )

        llm = get_llm(
            provider=llm_provider,
            model_name=llm_model,
            temperature=temperature,
            api_key=active_llm_key,
            streaming=True,
        )

        engine = RAGEngine(
            vector_store_manager=vsm,
            llm=llm,
            retriever_type=retriever_type,
            top_k=top_k,
        )
        st.session_state.rag_engine = engine
        return True, "RAG System successfully synchronized!"
    except Exception as e:
        return False, str(e)


# =====================================================================
# Sidebar: Configuration & Document Ingestion
# =====================================================================
with st.sidebar:
    st.markdown("### ⚙️ System Configuration")

    # Provider Selection
    llm_provider = st.selectbox(
        "🧠 LLM Provider",
        options=["google", "groq", "openai"],
        format_func=lambda x: {
            "google": "Google Gemini (Recommended)",
            "groq": "Groq Cloud (Ultra Fast)",
            "openai": "OpenAI (GPT-4o)",
        }[x],
        index=0,
    )

    # Dynamic Model Names
    if llm_provider == "google":
        model_name = st.selectbox("Model", ["gemini-1.5-flash", "gemini-1.5-pro"])
    elif llm_provider == "groq":
        model_name = st.selectbox("Model", ["llama-3.1-70b-versatile", "llama-3.1-8b-instant", "mixtral-8x7b-32768"])
    else:
        model_name = st.selectbox("Model", ["gpt-4o-mini", "gpt-4o"])

    # Embeddings Selection
    emb_provider = st.selectbox(
        "📐 Embeddings Engine",
        options=["google", "huggingface", "openai"],
        format_func=lambda x: {
            "google": "Google text-embedding-004",
            "huggingface": "HuggingFace Local (Free / Offline)",
            "openai": "OpenAI text-embedding-3-small",
        }[x],
        index=0 if llm_provider == "google" else 1,
    )

    # API Keys Section
    with st.expander("🔑 API Keys", expanded=not bool(Config.GOOGLE_API_KEY or Config.GROQ_API_KEY or Config.OPENAI_API_KEY)):
        google_api_key = st.text_input(
            "Google API Key",
            value=Config.GOOGLE_API_KEY or "",
            type="password",
            help="Free key at https://aistudio.google.com/app/apikey",
        )
        groq_api_key = st.text_input(
            "Groq API Key",
            value=Config.GROQ_API_KEY or "",
            type="password",
            help="Free key at https://console.groq.com/keys",
        )
        openai_api_key = st.text_input(
            "OpenAI API Key",
            value=Config.OPENAI_API_KEY or "",
            type="password",
        )

    # Retrieval Tuning
    with st.expander("🎯 Retrieval Hyperparameters"):
        retriever_type = st.selectbox(
            "Search Algorithm",
            options=["similarity", "mmr"],
            format_func=lambda x: "Similarity (Cosine/L2)" if x == "similarity" else "MMR (Maximal Marginal Relevance)",
        )
        top_k = st.slider("Top Chunks Retrieved (k)", min_value=1, max_value=8, value=4)
        chunk_size = st.slider("Chunk Size (Chars)", min_value=300, max_value=2000, value=1000, step=100)
        chunk_overlap = st.slider("Chunk Overlap (Chars)", min_value=50, max_value=400, value=200, step=25)
        temperature = st.slider("LLM Temperature", min_value=0.0, max_value=1.0, value=0.2, step=0.05)

    st.markdown("---")
    st.markdown("### 📂 Document Ingestion")

    # Document Uploader
    uploaded_files = st.file_uploader(
        "Upload PDF, TXT, or MD documents",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True,
    )

    col1, col2 = st.columns(2)
    with col1:
        process_btn = st.button("⚡ Ingest Files", use_container_width=True, type="primary")
    with col2:
        sample_btn = st.button("📄 Load Sample", use_container_width=True)

    # Document Processor instantiation
    doc_processor = DocumentProcessor(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    # Ingest Uploaded Files
    if process_btn and uploaded_files:
        with st.spinner("Processing & indexing documents..."):
            success, msg = initialize_rag_system(
                emb_provider, llm_provider, model_name, retriever_type, top_k, temperature,
                google_api_key, groq_api_key, openai_api_key
            )
            if not success:
                st.error(f"Initialization Failed: {msg}")
            else:
                total_chunks_added = 0
                for file in uploaded_files:
                    result = doc_processor.process_and_chunk(file.getvalue(), file.name)
                    added = st.session_state.vector_store_manager.add_documents(result["chunks"])
                    total_chunks_added += added
                    st.session_state.processed_files.append(file.name)

                st.success(f"Indexed {len(uploaded_files)} file(s) into {total_chunks_added} chunks!")
                st.rerun()

    # Load Sample File
    if sample_btn:
        with st.spinner("Loading sample Attention Is All You Need summary..."):
            success, msg = initialize_rag_system(
                emb_provider, llm_provider, model_name, retriever_type, top_k, temperature,
                google_api_key, groq_api_key, openai_api_key
            )
            if not success:
                st.error(f"Initialization Failed: {msg}")
            else:
                sample_text = get_sample_document_text()
                result = doc_processor.process_and_chunk(sample_text.encode("utf-8"), "attention_is_all_you_need_sample.txt")
                st.session_state.vector_store_manager.add_documents(result["chunks"])
                st.session_state.processed_files.append("attention_is_all_you_need_sample.txt")
                st.success(f"Loaded sample research paper ({result['chunks_count']} chunks indexed)!")
                st.rerun()

    # Database Status & Telemetry Card
    st.markdown("---")
    st.markdown("### 📊 Vector DB Telemetry")
    if st.session_state.vector_store_manager is None:
        st.info("System idle. Ingest a document or start a query.")
    else:
        stats = st.session_state.vector_store_manager.get_collection_stats()
        st.markdown(f"**Total Chunks:** `{stats['total_chunks']}`")
        st.markdown(f"**Indexed Files:** `{stats['document_count']}`")
        if stats["indexed_files"]:
            for f in stats["indexed_files"]:
                st.caption(f"• {f}")

        if st.button("🗑️ Reset Vector Database", use_container_width=True):
            st.session_state.vector_store_manager.clear_collection()
            st.session_state.chat_history = []
            st.session_state.langchain_messages = []
            st.session_state.processed_files = []
            st.success("Vector database wiped.")
            st.rerun()


# =====================================================================
# Main Body: Header & Tabs
# =====================================================================
st.markdown(
    """
    <div class="main-header">
        <h1>🧠 DocuMind RAG — Enterprise Document Intelligence</h1>
        <p>Production-grade Retrieval-Augmented Generation with LangChain LCEL, ChromaDB, and Multi-Provider LLMs.</p>
        <div style="margin-top: 12px;">
            <span class="stat-badge">⚡ Zero-Hallucination Grounding</span>
            <span class="stat-badge">🔍 MMR / Similarity Search</span>
            <span class="stat-badge">💬 Multi-turn Conversational Memory</span>
            <span class="stat-badge">📑 Source Citations & Confidence</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

tab_chat, tab_inspector, tab_architecture = st.tabs(
    ["💬 Conversational Q&A", "🔍 Document & Vector Inspector", "📐 Architecture & Resume Points"]
)

# ---------------------------------------------------------------------
# TAB 1: Conversational Q&A Chatbot
# ---------------------------------------------------------------------
with tab_chat:
    # Display message history
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            # Display source citations if available
            if msg.get("sources"):
                with st.expander(f"📑 Citations & Context ({len(msg['sources'])} chunks retrieved)"):
                    if msg.get("standalone_question"):
                        st.caption(f"🔍 *Rephrased Standalone Search Query:* **{msg['standalone_question']}**")
                    for i, source in enumerate(msg["sources"]):
                        conf = source.get("confidence_score", 0.0)
                        conf_pct = int(conf * 100)
                        st.markdown(
                            f"""
                            <div class="source-card">
                                <div class="source-meta">
                                    <span>📄 {source.get('source', 'Unknown')} (Page {source.get('page', 1)})</span>
                                    <span class="confidence-pill">Relevance: {conf_pct}%</span>
                                </div>
                                <div style="color: #cbd5e1; font-size: 0.85rem; line-height: 1.4;">
                                    {source.get('content', '')}
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

    # Chat Input
    user_prompt = st.chat_input("Ask a question grounded in your uploaded documents...")

    if user_prompt:
        # 1. Ensure engine is initialized
        if st.session_state.rag_engine is None:
            init_ok, init_err = initialize_rag_system(
                emb_provider, llm_provider, model_name, retriever_type, top_k, temperature,
                google_api_key, groq_api_key, openai_api_key
            )
            if not init_ok:
                st.error(f"Cannot run query: {init_err}")
                st.stop()

        # 2. Append and render user message
        st.session_state.chat_history.append({"role": "user", "content": user_prompt})
        with st.chat_message("user"):
            st.markdown(user_prompt)

        # 3. Stream or generate assistant answer
        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            full_response = ""
            metadata_info = {}

            with st.spinner("Retrieving grounded context & reasoning..."):
                try:
                    stream_gen = st.session_state.rag_engine.stream_query(
                        question=user_prompt,
                        chat_history=st.session_state.langchain_messages,
                    )

                    for chunk in stream_gen:
                        if chunk["type"] == "metadata":
                            metadata_info = chunk
                        elif chunk["type"] == "token":
                            full_response += chunk["token"]
                            response_placeholder.markdown(full_response + "▌")

                    response_placeholder.markdown(full_response)

                    # Render citations expander
                    sources = metadata_info.get("sources", [])
                    standalone_q = metadata_info.get("standalone_question", "")
                    if sources:
                        with st.expander(f"📑 Citations & Context ({len(sources)} chunks retrieved)"):
                            if standalone_q:
                                st.caption(f"🔍 *Rephrased Standalone Search Query:* **{standalone_q}**")
                            for s in sources:
                                conf = s.get("confidence_score", 0.0)
                                conf_pct = int(conf * 100)
                                st.markdown(
                                    f"""
                                    <div class="source-card">
                                        <div class="source-meta">
                                            <span>📄 {s.get('source', 'Unknown')} (Page {s.get('page', 1)})</span>
                                            <span class="confidence-pill">Relevance: {conf_pct}%</span>
                                        </div>
                                        <div style="color: #cbd5e1; font-size: 0.85rem;">
                                            {s.get('content', '')}
                                        </div>
                                    </div>
                                    """,
                                    unsafe_allow_html=True,
                                )

                    # Update session state history
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": full_response,
                        "sources": sources,
                        "standalone_question": standalone_q,
                    })

                    # Maintain LangChain BaseMessage memory
                    st.session_state.langchain_messages.append(HumanMessage(content=user_prompt))
                    st.session_state.langchain_messages.append(AIMessage(content=full_response))

                except Exception as e:
                    st.error(f"Error during execution: {str(e)}")

    # Footer controls
    if st.session_state.chat_history:
        st.markdown("---")
        col_c1, col_c2 = st.columns([1, 4])
        with col_c1:
            if st.button("🧹 Clear Chat"):
                st.session_state.chat_history = []
                st.session_state.langchain_messages = []
                st.rerun()
        with col_c2:
            md_transcript = export_chat_history_markdown(st.session_state.chat_history)
            st.download_button(
                "📥 Export Conversation (Markdown)",
                data=md_transcript,
                file_name="rag_chat_transcript.md",
                mime="text/markdown",
            )

# ---------------------------------------------------------------------
# TAB 2: Document & Vector DB Inspector
# ---------------------------------------------------------------------
with tab_inspector:
    st.markdown("### 🔬 Real-Time Vector Database & Retrieval Inspector")
    st.markdown("Test vector embeddings and cosine similarity search directly against indexed chunks.")

    if st.session_state.vector_store_manager is None:
        st.info("No vector store initialized yet. Ingest a document or click 'Load Sample' in the sidebar.")
    else:
        stats = st.session_state.vector_store_manager.get_collection_stats()
        c1, c2, c3 = st.columns(3)
        c1.metric("Indexed Chunks", stats["total_chunks"])
        c2.metric("Active Documents", stats["document_count"])
        c3.metric("Storage Type", "ChromaDB (Persistent)")

        test_query = st.text_input("Enter a query to inspect raw similarity search:", value="What is multi-head attention?")
        test_k = st.slider("Inspect Top-K Chunks", min_value=1, max_value=10, value=3)

        if st.button("Run Vector Similarity Inspection"):
            with st.spinner("Computing query embeddings & querying ChromaDB..."):
                results = st.session_state.vector_store_manager.similarity_search_with_score(test_query, k=test_k)
                if not results:
                    st.warning("No matches found or collection is empty.")
                else:
                    for rank, (doc, score) in enumerate(results, 1):
                        conf_pct = int(score * 100)
                        st.markdown(
                            f"""
                            <div class="source-card">
                                <div class="source-meta">
                                    <span>#{rank} • 📄 {doc.metadata.get('source', 'Unknown')} (Page {doc.metadata.get('page', 1)}) [Chunk ID: <code>{doc.metadata.get('chunk_id', 'N/A')}</code>]</span>
                                    <span class="confidence-pill">Similarity Score: {conf_pct}%</span>
                                </div>
                                <p style="color: #e2e8f0; margin-top: 8px;">{doc.page_content}</p>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

# ---------------------------------------------------------------------
# TAB 3: Architecture & Resume Bullet Points
# ---------------------------------------------------------------------
with tab_architecture:
    st.markdown("### 📐 End-to-End System Architecture")
    st.markdown(
        """
        ```mermaid
        flowchart LR
            PDF[User PDF / Document] --> Loader[Document Processor & Sanitizer]
            Loader --> Chunker[RecursiveCharacterTextSplitter]
            Chunker --> Embedder[Embeddings Engine (Google/OpenAI/HF)]
            Embedder --> VDB[(Chroma Vector Database)]
            
            Query[User Question] --> Rewriter[LCEL History-Aware Rewriter]
            VDB -.->|Cosine / MMR Retrieval| Rewriter
            Rewriter --> ContextAugment[Context-Augmented Prompt]
            ContextAugment --> LLM[LLM (Gemini / Groq / GPT-4o)]
            LLM --> Output[Streaming Answer + Citations]
        ```
        """
    )

    st.markdown("---")
    st.markdown("### 💼 Resume Points (Copy-Paste Ready for Your CV)")
    st.markdown(
        """
        **AI / LLM Engineer Resume Bullet Points:**
        * **Architected an Enterprise RAG Engine:** Built an end-to-end Retrieval-Augmented Generation (RAG) platform using **Python**, **LangChain (LCEL)**, and **ChromaDB**, achieving sub-second context retrieval across multi-page enterprise PDFs.
        * **Contextual Grounding & Hallucination Suppression:** Formulated strict zero-hallucination prompt pipelines with **history-aware query rewriting**, verifiable source citations (document name, page number, confidence score), and MMR (Maximal Marginal Relevance) retrieval.
        * **Multi-Provider LLM & Vector Pipeline:** Engineered a decoupled architecture supporting **Google Gemini 1.5 Flash/Pro**, **Groq Llama-3.1**, and **OpenAI GPT-4o**, along with local offline embeddings via **HuggingFace Sentence-Transformers**.
        * **Optimized Chunking & Document Ingestion:** Implemented `RecursiveCharacterTextSplitter` with semantic boundaries, dynamic overlap, and metadata preservation (page number, chunk ID, character metrics).
        """
    )
