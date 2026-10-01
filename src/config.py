import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

# Automatically load .env file from project root
BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")


class Config:
    """Central configuration for DocuMind RAG System."""

    # Project Paths
    ROOT_DIR: Path = BASE_DIR
    DATA_DIR: Path = BASE_DIR / "data"
    CHROMA_PERSIST_DIR: str = os.getenv("CHROMA_PERSIST_DIRECTORY", str(BASE_DIR / "data" / "chroma_db"))
    COLLECTION_NAME: str = os.getenv("COLLECTION_NAME", "rag_document_collection")

    # API Keys
    GOOGLE_API_KEY: Optional[str] = os.getenv("GOOGLE_API_KEY")
    GROQ_API_KEY: Optional[str] = os.getenv("GROQ_API_KEY")
    OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")

    # Chunking Hyperparameters
    DEFAULT_CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", 1000))
    DEFAULT_CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", 200))

    # Retrieval Hyperparameters
    DEFAULT_RETRIEVER_SEARCH_TYPE: str = os.getenv("RETRIEVER_SEARCH_TYPE", "similarity")
    DEFAULT_TOP_K: int = int(os.getenv("RETRIEVER_TOP_K", 4))
    SCORE_THRESHOLD: float = float(os.getenv("SCORE_THRESHOLD", 0.35))

    # Model Defaults
    DEFAULT_LLM_PROVIDER: str = os.getenv("DEFAULT_LLM_PROVIDER", "google")
    DEFAULT_MODEL_NAME: str = os.getenv("DEFAULT_MODEL_NAME", "gemini-1.5-flash")
    DEFAULT_EMBEDDING_PROVIDER: str = os.getenv("DEFAULT_EMBEDDING_PROVIDER", "google")
    TEMPERATURE: float = float(os.getenv("TEMPERATURE", 0.2))

    @classmethod
    def ensure_directories(cls):
        """Ensure necessary storage directories exist."""
        cls.DATA_DIR.mkdir(parents=True, exist_ok=True)
        Path(cls.CHROMA_PERSIST_DIR).mkdir(parents=True, exist_ok=True)

    @classmethod
    def get_available_providers(cls) -> dict:
        """Return dict of available providers based on configured API keys."""
        return {
            "google": bool(cls.GOOGLE_API_KEY and cls.GOOGLE_API_KEY.strip()),
            "groq": bool(cls.GROQ_API_KEY and cls.GROQ_API_KEY.strip()),
            "openai": bool(cls.OPENAI_API_KEY and cls.OPENAI_API_KEY.strip()),
            "huggingface": True,  # Always available locally for embeddings!
        }
