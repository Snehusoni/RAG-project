import io
import os
import re
from typing import List, Dict, Any, Union, BinaryIO
from pathlib import Path
from pypdf import PdfReader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import Config


class DocumentProcessor:
    """Enterprise document ingestion, metadata preservation, and smart chunking engine."""

    def __init__(
        self,
        chunk_size: int = Config.DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = Config.DEFAULT_CHUNK_OVERLAP,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
            is_separator_regex=False,
        )

    @staticmethod
    def clean_text(text: str) -> str:
        """Sanitize raw extracted text by normalizing whitespace and non-printable characters."""
        if not text:
            return ""
        # Replace non-breaking spaces & control characters
        text = text.replace("\xa0", " ").replace("\r\n", "\n").replace("\r", "\n")
        # Collapse multiple horizontal whitespaces
        text = re.sub(r"[ \t]+", " ", text)
        # Collapse 3 or more consecutive newlines into 2
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def load_pdf_from_bytes(self, file_bytes: bytes, filename: str) -> List[Document]:
        """Extract text and metadata from in-memory PDF bytes (e.g. from Streamlit file uploader)."""
        reader = PdfReader(io.BytesIO(file_bytes))
        total_pages = len(reader.pages)
        documents: List[Document] = []

        for page_idx, page in enumerate(reader.pages):
            page_text = page.extract_text() or ""
            cleaned_text = self.clean_text(page_text)

            if cleaned_text:
                doc = Document(
                    page_content=cleaned_text,
                    metadata={
                        "source": filename,
                        "page": page_idx + 1,  # 1-indexed
                        "total_pages": total_pages,
                        "file_type": "pdf",
                        "char_count": len(cleaned_text),
                    },
                )
                documents.append(doc)

        return documents

    def load_pdf_from_path(self, file_path: Union[str, Path]) -> List[Document]:
        """Extract text and metadata from a local PDF file on disk."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"PDF file not found: {file_path}")

        with open(path, "rb") as f:
            return self.load_pdf_from_bytes(f.read(), path.name)

    def load_text_from_bytes(self, file_bytes: bytes, filename: str) -> List[Document]:
        """Extract text from in-memory TXT or Markdown file."""
        try:
            content = file_bytes.decode("utf-8")
        except UnicodeDecodeError:
            content = file_bytes.decode("latin-1", errors="replace")

        cleaned_text = self.clean_text(content)
        if not cleaned_text:
            return []

        doc = Document(
            page_content=cleaned_text,
            metadata={
                "source": filename,
                "page": 1,
                "total_pages": 1,
                "file_type": Path(filename).suffix.lstrip(".").lower() or "text",
                "char_count": len(cleaned_text),
            },
        )
        return [doc]

    def load_document(self, file_input: Union[bytes, BinaryIO, str, Path], filename: str) -> List[Document]:
        """Polymorphic loader for PDF, TXT, MD from file path or bytes."""
        extension = Path(filename).suffix.lower()

        if isinstance(file_input, (str, Path)):
            with open(file_input, "rb") as f:
                raw_bytes = f.read()
        elif hasattr(file_input, "read"):
            raw_bytes = file_input.read()
            if hasattr(file_input, "seek"):
                file_input.seek(0)
        elif isinstance(file_input, bytes):
            raw_bytes = file_input
        else:
            raise ValueError(f"Unsupported file input type: {type(file_input)}")

        if extension == ".pdf":
            return self.load_pdf_from_bytes(raw_bytes, filename)
        elif extension in [".txt", ".md", ".csv", ".json", ".log"]:
            return self.load_text_from_bytes(raw_bytes, filename)
        else:
            # Fallback text attempt
            return self.load_text_from_bytes(raw_bytes, filename)

    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split loaded documents into optimal chunks while maintaining page and source metadata.
        Adds unique chunk_id to each chunk for traceability.
        """
        if not documents:
            return []

        chunked_docs = self.splitter.split_documents(documents)

        # Enrich chunk metadata with chunk index and unique identifier
        for idx, chunk in enumerate(chunked_docs):
            source = chunk.metadata.get("source", "unknown")
            page = chunk.metadata.get("page", 1)
            chunk.metadata["chunk_index"] = idx
            chunk.metadata["chunk_id"] = f"{source}_p{page}_c{idx}"
            chunk.metadata["chunk_char_count"] = len(chunk.page_content)

        return chunked_docs

    def process_and_chunk(
        self, file_input: Union[bytes, BinaryIO, str, Path], filename: str
    ) -> Dict[str, Any]:
        """
        Convenience end-to-end pipeline: load, clean, chunk, and return structured telemetry.
        """
        raw_docs = self.load_document(file_input, filename)
        chunks = self.chunk_documents(raw_docs)

        total_chars = sum(len(c.page_content) for c in chunks)
        avg_chunk_size = (total_chars / len(chunks)) if chunks else 0

        return {
            "filename": filename,
            "raw_pages_count": len(raw_docs),
            "chunks_count": len(chunks),
            "total_characters": total_chars,
            "average_chunk_size": round(avg_chunk_size, 1),
            "chunks": chunks,
        }
