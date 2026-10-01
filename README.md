# 🧠 DocuMind RAG — Enterprise Document Intelligence & Q&A Engine

[![Python 3.11](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![LangChain](https://img.shields.io/badge/LangChain-LCEL%200.2%2B-1C3C3C?logo=chainlink&logoColor=white)](https://python.langchain.com/)
[![ChromaDB](https://img.shields.io/badge/Vector%20DB-ChromaDB-FF6600)](https://www.trychroma.com/)
[![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> A production-grade **Retrieval-Augmented Generation (RAG)** platform designed to ingest multi-page enterprise documents (PDFs, TXT, Markdown), index them in a persistent vector database, and provide context-grounded, zero-hallucination conversational answers with verifiable source citations and page numbers.

---

## 🌟 Key Highlights & Engineering Features

* **⚡ Modern LangChain LCEL (LangChain Expression Language):** Built with declarative, composable runnable chains (`RunnablePassthrough`, `RunnableParallel`), avoiding deprecated legacy wrapper chains.
* **🛡️ Zero-Hallucination Prompt Architecture:** Enforces strict context-grounded answers. If the information is not present in the indexed documents, the model transparently declines to answer rather than fabricating facts.
* **💬 Conversational Memory & History-Aware Query Rewriting:** Automatically reformulates ambiguous follow-up questions (e.g., *"What are its advantages?"*) into standalone search queries before querying the vector store.
* **📐 Multi-Provider Flexibility:**
  * **LLMs:** Google Gemini (1.5 Flash / Pro), Groq Cloud (Llama-3.1 70B / 8B), and OpenAI (GPT-4o / GPT-4o-mini).
  * **Embeddings:** Google `text-embedding-004`, OpenAI `text-embedding-3-small`, and local **HuggingFace** (`all-MiniLM-L6-v2`) for 100% offline, free execution.
* **🔍 Advanced Retrieval Algorithms:**
  * Standard Cosine Similarity Retrieval
  * **MMR (Maximal Marginal Relevance)** to balance relevance with diversity and prevent duplicate context chunks.
* **📑 Transparent Attribution & Confidence Scoring:** Every answer includes an expandable citations panel displaying the exact document name, page number, confidence percentage, and snippet text.
* **🔬 Real-Time Vector Database Inspector:** A dedicated inspection sandbox to execute raw vector queries, inspect cosine distances, and debug retrieval performance.

---

## 🏗️ Architecture

```
User Document (PDF/TXT)
         │
         ▼
[ DocumentProcessor ]  ──> Metadata Extraction (page, source, chunk_id)
         │
         ▼
[ RecursiveTextSplitter ] (Chunk Size: 1000, Overlap: 200)
         │
         ▼
[ VectorStoreManager ] ──> Dense Embeddings ──> [ ChromaDB Persistent Store ]
                                                         │
─────────────────────────────────────────────────────────┼───────────
                                                         │
User Question + Chat History                             │
         │                                               │
         ▼                                               │
[ History-Aware Rewriter ] ──> Standalone Search Query   │
         │                                               │
         ▼                                               │
[ Vector Search / MMR ] <────────────────────────────────┘
         │
         ▼
[ Context-Augmented Prompt ] ──> Strict Anti-Hallucination Guardrails
         │
         ▼
[ LLM (Gemini / Groq / OpenAI) ] ──> Streaming Answer + Verbatim Page Citations
```

---

## 🚀 Quickstart Guide

### 1. Prerequisites
- Python 3.10+ (Recommended: Python 3.11)
- Google Gemini, Groq, or OpenAI API key (Google Gemini and Groq offer generous free tiers)

### 2. Installation

Clone or navigate to the project directory:
```bash
# Create and activate a virtual environment
python -m venv .venv

# On Windows (PowerShell):
.\.venv\Scripts\Activate.ps1

# On macOS/Linux:
source .venv/bin/activate
```

Install the dependencies:
```bash
pip install -r requirements.txt
```

### 3. Environment Configuration
Copy the template configuration file:
```bash
copy .env.example .env   # Windows
# or: cp .env.example .env (Linux/Mac)
```

Edit `.env` and add your preferred API key:
```env
GOOGLE_API_KEY=your_gemini_api_key_here
# Optional alternatives:
GROQ_API_KEY=your_groq_api_key_here
OPENAI_API_KEY=your_openai_api_key_here
```

### 4. Run the Automated Test Suite
Verify your installation, chunking, and ChromaDB vector store locally:
```bash
python test_rag.py
```

### 5. Launch the Web Interface
Start the interactive Streamlit application:
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 📂 Project Structure

```
├── app.py                      # Modern Streamlit UI with streaming, citations & inspector
├── requirements.txt            # Project dependencies
├── test_rag.py                 # Automated integration test suite
├── .env.example                # Sample environment variables template
├── RESUME_POINTS.md            # Resume bullets, interview Q&A, and system design notes
├── README.md                   # Complete documentation
├── src/
│   ├── __init__.py
│   ├── config.py               # Central configuration & API key validation
│   ├── document_processor.py   # PDF text extraction, sanitization, and recursive chunking
│   ├── vector_store.py         # ChromaDB manager, embeddings factory, and MMR retrieval
│   ├── rag_engine.py           # LCEL conversational chains, prompt templates & streaming
│   └── utils.py                # Formatting helpers, badge generators, and sample data
├── sample_documents/
│   └── sample_ai_paper.txt     # Sample evaluation document on Transformer architectures
└── data/
    └── chroma_db/              # Persistent vector store storage directory (auto-created)
```

---

## 📊 Retrieval & Chunking Strategy

| Parameter | Default | Description |
|---|---|---|
| `chunk_size` | `1000` chars | Balances semantic context with prompt token budgets |
| `chunk_overlap` | `200` chars | Ensures sentence boundaries aren't split across chunk edges |
| `retriever_type` | `similarity` / `mmr` | MMR uses $\lambda=0.7$ to penalize repetitive chunks |
| `top_k` | `4` chunks | Number of highest-ranking context segments passed to LLM |
| `temperature` | `0.2` | Low temperature ensures deterministic, grounded generation |

---

## 💼 Resume & Interview Readiness

For tips on how to describe this project in technical interviews, explain system tradeoffs (e.g. dense retrieval vs BM25, MMR lambda selection, hallucination mitigation), refer to:
👉 **[RESUME_POINTS.md](file:///e:/Best%203%20Ai%20project%20for%20my%20resume/RAG%20Document%20Q&A%20chatbot/RESUME_POINTS.md)**

---

## 📜 License
This project is licensed under the MIT License.
