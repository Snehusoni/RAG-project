import os
import shutil
from typing import List, Tuple, Dict, Any, Optional
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_chroma import Chroma

from src.config import Config


def get_embeddings(
    provider: Optional[str] = None,
    api_key: Optional[str] = None,
) -> Embeddings:
    """
    Factory to return the appropriate Embeddings model based on provider preference.
    Supports Google Gemini, OpenAI, and local HuggingFace (offline/free).
    """
    provider = (provider or Config.DEFAULT_EMBEDDING_PROVIDER).lower()

    if provider == "google":
        key = api_key or Config.GOOGLE_API_KEY
        if not key:
            raise ValueError(
                "GOOGLE_API_KEY not found. Please set it in .env or choose HuggingFace local embeddings."
            )
        from langchain_google_genai import GoogleGenerativeAIEmbeddings

        # text-embedding-004 is Google's state-of-the-art embedding model
        return GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004",
            google_api_key=key,
        )

    elif provider == "openai":
        key = api_key or Config.OPENAI_API_KEY
        if not key:
            raise ValueError(
                "OPENAI_API_KEY not found. Please set it in .env or choose another provider."
            )
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(
            model="text-embedding-3-small",
            openai_api_key=key,
        )

    elif provider in ["huggingface", "local"]:
        # Completely free and runs locally on CPU/GPU without any API keys
        from langchain_community.embeddings import HuggingFaceEmbeddings

        return HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

    else:
        raise ValueError(
            f"Unsupported embedding provider: '{provider}'. Choose 'google', 'openai', or 'huggingface'."
        )


class VectorStoreManager:
    """
    Production-grade Vector Database abstraction wrapping ChromaDB.
    Supports persistent indexing, MMR search, similarity scoring, and document filtering.
    """

    def __init__(
        self,
        embeddings: Optional[Embeddings] = None,
        persist_directory: Optional[str] = None,
        collection_name: Optional[str] = None,
    ):
        self.persist_directory = persist_directory or Config.CHROMA_PERSIST_DIR
        self.collection_name = collection_name or Config.COLLECTION_NAME
        self.embeddings = embeddings or get_embeddings()

        Config.ensure_directories()

        self._vector_store = Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embeddings,
            persist_directory=self.persist_directory,
        )

    @property
    def vector_store(self) -> Chroma:
        """Access the underlying LangChain Chroma instance."""
        return self._vector_store

    def add_documents(self, documents: List[Document]) -> int:
        """
        Index chunked documents into the vector store.
        Returns the number of documents indexed.
        """
        if not documents:
            return 0

        # Extract IDs from metadata if present for deterministic indexing
        ids = [doc.metadata.get("chunk_id") for doc in documents]
        # Only pass ids if all documents have non-empty unique chunk_ids
        if all(ids) and len(ids) == len(set(ids)):
            self._vector_store.add_documents(documents, ids=ids)
        else:
            self._vector_store.add_documents(documents)

        return len(documents)

    def similarity_search_with_score(
        self,
        query: str,
        k: int = Config.DEFAULT_TOP_K,
        filter_dict: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[Document, float]]:
        """
        Perform similarity search and return matching documents along with similarity scores.
        Chroma returns cosine/L2 distance where lower is closer. We normalize this for clarity.
        """
        results = self._vector_store.similarity_search_with_score(
            query=query,
            k=k,
            filter=filter_dict,
        )

        normalized_results: List[Tuple[Document, float]] = []
        for doc, raw_score in results:
            # Chroma score is distance. Distance = 0 means identical.
            # Convert distance to approximate similarity confidence (0.0 to 1.0)
            confidence = max(0.0, min(1.0, 1.0 / (1.0 + float(raw_score))))
            normalized_results.append((doc, round(confidence, 4)))

        return normalized_results

    def as_retriever(
        self,
        search_type: str = Config.DEFAULT_RETRIEVER_SEARCH_TYPE,
        k: int = Config.DEFAULT_TOP_K,
        score_threshold: Optional[float] = None,
        filter_dict: Optional[Dict[str, Any]] = None,
    ):
        """
        Return a LangChain Retriever object configured with MMR or similarity.
        MMR (Maximal Marginal Relevance) maximizes relevance while penalizing redundant chunks.
        """
        search_kwargs: Dict[str, Any] = {"k": k}
        if filter_dict:
            search_kwargs["filter"] = filter_dict

        if search_type == "mmr":
            search_kwargs["fetch_k"] = k * 3
            search_kwargs["lambda_mult"] = 0.7  # 1.0 = max relevance, 0.0 = max diversity
            return self._vector_store.as_retriever(
                search_type="mmr",
                search_kwargs=search_kwargs,
            )
        elif search_type == "similarity_score_threshold":
            search_kwargs["score_threshold"] = score_threshold or Config.SCORE_THRESHOLD
            return self._vector_store.as_retriever(
                search_type="similarity_score_threshold",
                search_kwargs=search_kwargs,
            )
        else:
            return self._vector_store.as_retriever(
                search_type="similarity",
                search_kwargs=search_kwargs,
            )

    def get_collection_stats(self) -> Dict[str, Any]:
        """Inspect vector database telemetry and return indexed files and chunk metrics."""
        try:
            collection = self._vector_store._collection
            count = collection.count()

            # Retrieve distinct source documents
            distinct_sources = set()
            if count > 0:
                metadata_sample = collection.get(limit=min(count, 1000), include=["metadatas"])
                for meta in metadata_sample.get("metadatas", []):
                    if meta and "source" in meta:
                        distinct_sources.add(meta["source"])

            return {
                "total_chunks": count,
                "document_count": len(distinct_sources),
                "indexed_files": sorted(list(distinct_sources)),
                "storage_directory": self.persist_directory,
                "collection_name": self.collection_name,
            }
        except Exception as e:
            return {
                "total_chunks": 0,
                "document_count": 0,
                "indexed_files": [],
                "error": str(e),
            }

    def clear_collection(self) -> bool:
        """Reset and wipe the vector store."""
        try:
            self._vector_store.delete_collection()
            # Recreate an empty Chroma store
            self._vector_store = Chroma(
                collection_name=self.collection_name,
                embedding_function=self.embeddings,
                persist_directory=self.persist_directory,
            )
            return True
        except Exception:
            # Fallback: remove directory on disk
            if os.path.exists(self.persist_directory):
                shutil.rmtree(self.persist_directory, ignore_errors=True)
            Config.ensure_directories()
            self._vector_store = Chroma(
                collection_name=self.collection_name,
                embedding_function=self.embeddings,
                persist_directory=self.persist_directory,
            )
            return True
