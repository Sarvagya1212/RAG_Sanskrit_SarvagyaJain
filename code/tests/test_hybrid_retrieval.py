"""Tests for hybrid retrieval."""

import pytest
import numpy as np
from code.src.retrieval import weighted_score_fusion, HybridRetriever
from code.src.indexing import BM25Indexer, VectorIndexer, EmbeddingGenerator, MetadataStore
from code.src.preprocessing import SanskritPreprocessor


class TestWeightedScoreFusion:
    """Tests for weighted score fusion algorithm."""
    
    def test_fusion_basic(self):
        """Test basic weighted score fusion."""
        # Two rankings with overlap
        bm25_results = [(0, 10.0), (1, 8.0), (2, 6.0)]
        vector_results = [(1, 0.9), (0, 0.8), (3, 0.7)]
        
        fused = weighted_score_fusion(bm25_results, vector_results)
        
        # Both docs 0 and 1 appear highly ranked
        result_ids = [doc_id for doc_id, _ in fused]
        assert 0 in result_ids[:3]
        assert 1 in result_ids[:3]
    
    def test_fusion_empty_results(self):
        """Fusion with empty results should handle gracefully."""
        bm25_results = [(0, 5.0), (1, 3.0)]
        vector_results = []
        
        fused = weighted_score_fusion(bm25_results, vector_results)
        
        # Should still return BM25 results
        assert len(fused) >= 2
    
    def test_fusion_weights(self):
        """Custom weights should affect ranking."""
        bm25_results = [(0, 10.0)]  # Doc 0 only in BM25
        vector_results = [(1, 1.0)]  # Doc 1 only in vector
        
        # With high BM25 weight, doc 0 should rank higher
        fused_bm25_heavy = weighted_score_fusion(bm25_results, vector_results, bm25_weight=0.9, vector_weight=0.1)
        
        # Doc 0 should be first with heavy BM25 weight
        assert fused_bm25_heavy[0][0] == 0


class TestHybridRetriever:
    """Tests for hybrid retriever."""
    
    @pytest.fixture
    def mock_components(self):
        """Create mock components for testing."""
        # Would need actual implementation or mocks
        # This is a placeholder
        pass
    
    def test_query_preprocessing(self):
        """Query should be preprocessed to SLP1."""
        preprocessor = SanskritPreprocessor()
        
        queries = ["धर्मः", "dharmaḥ", "dharmah"]
        
        for query in queries:
            result = preprocessor.process(query)
            assert result.slp1 == "DarmaH"
    
    def test_top_k_limit(self):
        """Should return exactly top_k results."""
        # Placeholder - would need actual retriever
        pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
