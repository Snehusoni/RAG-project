from typing import List, Dict, Any, Optional, Generator
from operator import itemgetter

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.documents import Document

from src.config import Config
from src.vector_store import VectorStoreManager


def get_llm(
    provider: Optional[str] = None,
    model_name: Optional[str] = None,
    temperature: float = Config.TEMPERATURE,
    api_key: Optional[str] = None,
    streaming: bool = True,
) -> BaseChatModel:
    """
    Factory to construct LLM client across Google Gemini, Groq, and OpenAI.
    """
    provider = (provider or Config.DEFAULT_LLM_PROVIDER).lower()

    if provider == "google":
        key = api_key or Config.GOOGLE_API_KEY
        if not key:
            raise ValueError(
                "GOOGLE_API_KEY not found. Please provide an API key in your .env or the UI."
            )
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=model_name or "gemini-1.5-flash",
            google_api_key=key,
            temperature=temperature,
            streaming=streaming,
        )

    elif provider == "groq":
        key = api_key or Config.GROQ_API_KEY
        if not key:
            raise ValueError(
                "GROQ_API_KEY not found. Please provide an API key in your .env or the UI."
            )
        from langchain_groq import ChatGroq

        return ChatGroq(
            model=model_name or "llama-3.1-70b-versatile",
            groq_api_key=key,
            temperature=temperature,
            streaming=streaming,
        )

    elif provider == "openai":
        key = api_key or Config.OPENAI_API_KEY
        if not key:
            raise ValueError(
                "OPENAI_API_KEY not found. Please provide an API key in your .env or the UI."
            )
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=model_name or "gpt-4o-mini",
            openai_api_key=key,
            temperature=temperature,
            streaming=streaming,
        )

    else:
        raise ValueError(f"Unsupported LLM provider: '{provider}'. Choose 'google', 'groq', or 'openai'.")


# =====================================================================
# Prompt Templates (Zero-Hallucination & Source-Grounded)
# =====================================================================

CONTEXTUALIZE_Q_SYSTEM_PROMPT = """Given a chat history and the latest user question \
which might reference context in the chat history, formulate a standalone question \
which can be understood without the chat history. Do NOT answer the question, \
just reformulate it if needed and otherwise return it as is."""

QA_SYSTEM_PROMPT = """You are DocuMind, an elite enterprise document analysis and question-answering assistant.
Your mission is to provide accurate, concise, and context-grounded answers based strictly on the provided context retrieved from user-uploaded documents.

STRICT CONSTRAINTS & GUIDELINES:
1. ONLY answer using facts explicitly mentioned in the context below.
2. If the answer cannot be determined from the provided context, state clearly and politely:
   "I cannot find the answer to this question in the provided document(s)."
3. Do NOT make assumptions, extrapolate, or use outside training knowledge not present in the context.
4. When citing facts, mention the document source and page number when available in the context metadata.
5. Keep explanations structured, professional, and clear with bullet points where appropriate.

<CONTEXT>
{context}
</CONTEXT>
"""


def format_docs_with_metadata(docs: List[Document]) -> str:
    """Format retrieved documents with clean metadata header for LLM context grounding."""
    formatted = []
    for i, doc in enumerate(docs):
        source = doc.metadata.get("source", "Unknown Document")
        page = doc.metadata.get("page", "?")
        formatted.append(
            f"--- [Snippet {i+1} | Source: {source} (Page {page})] ---\n{doc.page_content.strip()}"
        )
    return "\n\n".join(formatted)


class RAGEngine:
    """
    Modern LCEL-powered Conversational Retrieval-Augmented Generation Engine.
    Features:
    - History-aware query contextualization (multi-turn memory)
    - Strict anti-hallucination prompt engineering
    - Precise citation attribution (source + page + snippet)
    - Token streaming and raw context inspection
    """

    def __init__(
        self,
        vector_store_manager: VectorStoreManager,
        llm: Optional[BaseChatModel] = None,
        retriever_type: str = Config.DEFAULT_RETRIEVER_SEARCH_TYPE,
        top_k: int = Config.DEFAULT_TOP_K,
    ):
        self.vector_store_manager = vector_store_manager
        self.llm = llm or get_llm()
        self.retriever_type = retriever_type
        self.top_k = top_k
        self.retriever = self.vector_store_manager.as_retriever(
            search_type=self.retriever_type,
            k=self.top_k,
        )

        # Contextualize query prompt (for conversation memory)
        self.contextualize_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", CONTEXTUALIZE_Q_SYSTEM_PROMPT),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{question}"),
            ]
        )

        # Main Answer prompt
        self.qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", QA_SYSTEM_PROMPT),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{question}"),
            ]
        )

        self._build_chains()

    def update_llm(self, llm: BaseChatModel):
        """Hot-swap the active LLM."""
        self.llm = llm
        self._build_chains()

    def update_retriever(self, retriever_type: str, top_k: int):
        """Update retriever hyperparameters on the fly."""
        self.retriever_type = retriever_type
        self.top_k = top_k
        self.retriever = self.vector_store_manager.as_retriever(
            search_type=self.retriever_type,
            k=self.top_k,
        )
        self._build_chains()

    def _build_chains(self):
        """Construct modern LangChain Expression Language (LCEL) chains."""
        # 1. Standalone Question Generator Chain
        self.question_rewriter_chain = (
            self.contextualize_prompt
            | self.llm
            | StrOutputParser()
        )

        # 2. Main Generation Chain
        self.rag_chain = (
            RunnablePassthrough.assign(
                context=lambda x: format_docs_with_metadata(x["context"])
            )
            | self.qa_prompt
            | self.llm
            | StrOutputParser()
        )

    def contextualize_question(self, question: str, chat_history: List[BaseMessage]) -> str:
        """Rewrite conversational question into standalone query if chat history exists."""
        if not chat_history:
            return question
        try:
            standalone = self.question_rewriter_chain.invoke(
                {"chat_history": chat_history, "question": question}
            )
            return standalone.strip() if standalone else question
        except Exception:
            return question

    def retrieve_context_with_scores(self, query: str) -> List[Dict[str, Any]]:
        """Retrieve relevant context chunks along with confidence scores and metadata."""
        scored_docs = self.vector_store_manager.similarity_search_with_score(
            query=query,
            k=self.top_k,
        )
        results = []
        for doc, score in scored_docs:
            results.append({
                "content": doc.page_content,
                "source": doc.metadata.get("source", "Unknown"),
                "page": doc.metadata.get("page", 1),
                "chunk_id": doc.metadata.get("chunk_id", "N/A"),
                "confidence_score": score,
                "metadata": doc.metadata,
            })
        return results

    def query(
        self,
        question: str,
        chat_history: Optional[List[BaseMessage]] = None,
    ) -> Dict[str, Any]:
        """
        Execute full RAG pipeline:
        1. Contextualize question against chat history
        2. Retrieve relevant document chunks from vector store
        3. Format context with source metadata
        4. Synthesize zero-hallucination answer
        5. Return answer with full source attributions and telemetry
        """
        history = chat_history or []

        # Step 1: Rephrase question if multi-turn
        standalone_question = self.contextualize_question(question, history)

        # Step 2: Retrieve relevant chunks
        retrieved_items = self.retrieve_context_with_scores(standalone_question)
        retrieved_docs = [
            Document(
                page_content=item["content"],
                metadata=item["metadata"]
            )
            for item in retrieved_items
        ]

        if not retrieved_docs:
            return {
                "answer": "No relevant documents found. Please upload a PDF or document first.",
                "sources": [],
                "standalone_question": standalone_question,
            }

        # Step 3: Generate grounded answer
        answer = self.rag_chain.invoke(
            {
                "context": retrieved_docs,
                "chat_history": history,
                "question": standalone_question,
            }
        )

        return {
            "answer": answer,
            "sources": retrieved_items,
            "standalone_question": standalone_question,
        }

    def stream_query(
        self,
        question: str,
        chat_history: Optional[List[BaseMessage]] = None,
    ) -> Generator[Dict[str, Any], None, None]:
        """
        Stream response tokens in real-time while preserving source telemetry.
        Yields {'type': 'token', 'token': '...'} or {'type': 'sources', 'sources': [...]}
        """
        history = chat_history or []
        standalone_question = self.contextualize_question(question, history)
        retrieved_items = self.retrieve_context_with_scores(standalone_question)
        retrieved_docs = [
            Document(
                page_content=item["content"],
                metadata=item["metadata"]
            )
            for item in retrieved_items
        ]

        # First yield the sources and rewritten query
        yield {
            "type": "metadata",
            "standalone_question": standalone_question,
            "sources": retrieved_items,
        }

        if not retrieved_docs:
            yield {
                "type": "token",
                "token": "No relevant documents found. Please upload a PDF or document first.",
            }
            return

        formatted_context = format_docs_with_metadata(retrieved_docs)
        stream_chain = (
            self.qa_prompt
            | self.llm
            | StrOutputParser()
        )

        for chunk in stream_chain.stream(
            {
                "context": formatted_context,
                "chat_history": history,
                "question": standalone_question,
            }
        ):
            yield {"type": "token", "token": chunk}
