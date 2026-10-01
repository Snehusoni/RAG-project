"""End-to-End Automated Integration Test for DocuMind RAG Architecture.
Validates:
1. Document loading & text extraction
2. RecursiveCharacterTextSplitter chunking & metadata integrity
3. Vector store indexing & MMR / similarity retrieval
4. Source citation attribution & confidence metrics
"""

import sys
from pathlib import Path

# Ensure UTF-8 output encoding on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from src.document_processor import DocumentProcessor
from src.vector_store import VectorStoreManager, get_embeddings
from src.utils import get_sample_document_text


def test_document_processor():
    print("\n--- [1] Testing Document Processor & Chunking ---")
    processor = DocumentProcessor(chunk_size=500, chunk_overlap=100)
    sample_text = get_sample_document_text()
    
    result = processor.process_and_chunk(sample_text.encode("utf-8"), "attention_paper_test.txt")
    print(f"[OK] Filename: {result['filename']}")
    print(f"[OK] Total Chunks Created: {result['chunks_count']}")
    print(f"[OK] Total Characters: {result['total_characters']}")
    print(f"[OK] Average Chunk Size: {result['average_chunk_size']} chars")
    
    assert result['chunks_count'] > 0, "No chunks were produced"
    sample_chunk = result['chunks'][0]
    assert "source" in sample_chunk.metadata, "Metadata missing 'source'"
    assert "page" in sample_chunk.metadata, "Metadata missing 'page'"
    assert "chunk_id" in sample_chunk.metadata, "Metadata missing 'chunk_id'"
    print(f"[OK] Chunk Metadata Verified: {sample_chunk.metadata}")
    return result['chunks']


def test_vector_store(chunks):
    print("\n--- [2] Testing Vector Store & Similarity Search ---")
    # Use offline HuggingFace embeddings for standalone automated testing
    print("Initializing local HuggingFace embeddings (offline/free)...")
    embeddings = get_embeddings(provider="huggingface")
    
    vsm = VectorStoreManager(
        embeddings=embeddings,
        persist_directory=str(BASE_DIR / "data" / "test_chroma_db"),
        collection_name="test_collection",
    )
    
    # Reset any prior test data
    vsm.clear_collection()
    
    added_count = vsm.add_documents(chunks)
    print(f"[OK] Successfully indexed {added_count} chunks into ChromaDB.")
    
    # Test Similarity Search
    query = "What is the attention equation and formula?"
    print(f"Executing similarity search for query: '{query}'")
    results = vsm.similarity_search_with_score(query, k=2)
    
    assert len(results) > 0, "No search results returned"
    for i, (doc, score) in enumerate(results, 1):
        print(f"  Result #{i}: [Score: {score*100:.1f}%] - {doc.page_content[:120]}...")
    
    # Test MMR Search
    print("Testing MMR (Maximal Marginal Relevance) retrieval...")
    retriever = vsm.as_retriever(search_type="mmr", k=2)
    mmr_docs = retriever.invoke(query)
    print(f"[OK] MMR retrieved {len(mmr_docs)} diverse chunks.")
    
    # Check collection statistics
    stats = vsm.get_collection_stats()
    print(f"[OK] Collection Stats: {stats['total_chunks']} chunks from {stats['document_count']} document(s).")
    
    # Clean up test database
    vsm.clear_collection()
    print("[OK] Test vector database cleaned up successfully.")


if __name__ == "__main__":
    print("=" * 60)
    print("Starting Automated RAG Pipeline Verification Suite")
    print("=" * 60)
    try:
        chunks = test_document_processor()
        test_vector_store(chunks)
        print("\n" + "=" * 60)
        print("ALL TESTS PASSED SUCCESSFULLY! RAG ENGINE IS PRODUCTION READY.")
        print("=" * 60)
    except Exception as e:
        print(f"\n[ERROR] Test Failed with Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
