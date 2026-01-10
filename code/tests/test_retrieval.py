"""Test retrieval functionality."""

import pytest
import numpy as np
from code.src.indexing.bm25_indexer import BM25Indexer
from code.src.indexing.embedding_generator import EmbeddingGenerator
from code.src.indexing.vector_indexer import VectorIndexer
from code.src.preprocessing.preprocessor import SanskritPreprocessor
from code.src.utils.logger import setup_logger

logger = setup_logger(__name__)

def test_bm25_search():
    """Test BM25 keyword search."""
    
    # Create sample chunks
    chunks = [
        {'text_slp1': 'Darma', 'chunk_id': '1'},
        {'text_slp1': 'arTa', 'chunk_id': '2'},
        {'text_slp1': 'kAma', 'chunk_id': '3'},
        {'text_slp1': 'mokza', 'chunk_id': '4'},
    ]
    
    # Build index
    indexer = BM25Indexer(ngram_size=3)
    indexer.build_index(chunks)
    
    # Search
    results = indexer.search('Darma', top_k=2)
    
    # Should return chunk 1 first
    assert results[0][0] == 0  # Index 0 = "Darma"
    assert results[0][1] > 0  # Non-zero score

def test_vector_search():
    """Test vector similarity search."""
    
    # Create sample embeddings
    embeddings = np.random.rand(10, 384).astype('float32')
    
    # Normalize
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / norms
    
    chunks = [{'chunk_id': str(i)} for i in range(10)]
    
    # Build index
    indexer = VectorIndexer(embedding_dim=384)
    indexer.build_index(embeddings, chunks, index_type="FlatL2")
    
    # Search with first embedding (should return itself)
    query_embedding = embeddings[0]
    results = indexer.search(query_embedding, top_k=3)
    
    # First result should be index 0 with distance ~0
    assert results[0][0] == 0
    assert results[0][1] < 0.01  # Very small distance

def test_cross_script_retrieval():
    """Test that different scripts retrieve same chunks."""
    
    preprocessor = SanskritPreprocessor()
    
    # Same query in 3 scripts
    queries = [
        "धर्मः",      # Devanagari
        "dharmaḥ",    # IAST
        "dharmah"     # Loose
    ]
    
    # Preprocess all
    processed = [preprocessor.process(q).slp1 for q in queries]
    
    # All should be identical
    assert processed[0] == processed[1] == processed[2]
    
    logger.info(f"✓ All scripts convert to: {processed[0]}")

if __name__ == "__main__":
    pytest.main([__file__, "-v"])