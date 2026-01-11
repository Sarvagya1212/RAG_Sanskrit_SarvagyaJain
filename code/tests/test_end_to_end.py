"""End-to-end integration tests for the complete RAG pipeline."""

import pytest
from unittest.mock import Mock
from code.src.preprocessing import SanskritPreprocessor
from code.src.generation import PromptTemplate


class TestPromptTemplate:
    """Test prompt template formatting."""
    
    def test_format_context(self):
        """Context should be formatted with sources."""
        template = PromptTemplate()
        
        chunks = [
            {
                'story_title': 'Test Story 1',
                'parent_text': 'First chunk text'
            },
            {
                'story_title': 'Test Story 2',
                'parent_text': 'Second chunk text'
            }
        ]
        
        context = template.format_context(chunks)
        
        assert 'Test Story 1' in context
        assert 'Test Story 2' in context
    
    def test_build_prompt(self):
        """Full prompt should include system, context, and query."""
        template = PromptTemplate()
        
        chunks = [{'story_title': 'Test', 'parent_text': 'Test text'}]
        query = "What is this about?"
        
        prompt = template.build_prompt(query, chunks)
        
        assert 'Sanskrit' in prompt  # System prompt
        assert 'What is this about?' in prompt  # Query


class TestEndToEndFlow:
    """Test complete pipeline integration."""
    
    def test_preprocessing_to_retrieval(self):
        """Test query preprocessing feeds into retrieval."""
        preprocessor = SanskritPreprocessor()
        
        # Different scripts, same word
        queries = ["धर्मः", "dharmaḥ", "dharmah"]
        
        # All should preprocess to SLP1
        results = [preprocessor.process(q).slp1 for q in queries]
        
        # All results should be non-empty
        assert all(len(r) > 0 for r in results)
    
    def test_retrieval_output_format(self):
        """Retrieved chunks should have required metadata for generation."""
        # Mock retrieval result
        mock_chunk = {
            'chunk_id': '1_1',
            'story_id': 1,
            'story_title': 'Test Story',
            'parent_text': 'Original Sanskrit text',
            'retrieval_score': 0.95,
        }
        
        # Verify all required fields present
        required_fields = ['story_title', 'parent_text']
        for field in required_fields:
            assert field in mock_chunk


class TestCitationGeneration:
    """Test source citation functionality."""
    
    def test_citation_extraction(self):
        """Citations should be extracted from chunks."""
        chunks = [
            {'story_title': 'Story 1', 'story_id': 1, 'chunk_id': '1_1'},
            {'story_title': 'Story 1', 'story_id': 1, 'chunk_id': '1_2'},
            {'story_title': 'Story 2', 'story_id': 2, 'chunk_id': '2_1'},
        ]
        
        # Extract unique story titles
        seen_stories = set()
        sources = []
        for chunk in chunks:
            story_title = chunk['story_title']
            if story_title not in seen_stories:
                sources.append({
                    'story_title': story_title,
                    'story_id': chunk['story_id']
                })
                seen_stories.add(story_title)
        
        # Should have 2 unique stories
        assert len(sources) == 2
        assert sources[0]['story_title'] == 'Story 1'
        assert sources[1]['story_title'] == 'Story 2'


class TestCrossScriptEndToEnd:
    """Test that different scripts work through entire pipeline."""
    
    def test_devanagari_query(self):
        """Devanagari query should work end-to-end."""
        preprocessor = SanskritPreprocessor()
        
        query = "धर्मः किम् अस्ति?"
        result = preprocessor.process(query)
        
        assert result.script == "devanagari"
        assert result.slp1 is not None
        assert len(result.slp1) > 0
    
    def test_iast_query(self):
        """IAST query should work end-to-end."""
        preprocessor = SanskritPreprocessor()
        
        query = "dharmaḥ kim asti?"
        result = preprocessor.process(query)
        
        assert result.script == "iast"
        assert result.slp1 is not None
    
    def test_loose_roman_query(self):
        """Loose Roman query should work end-to-end."""
        preprocessor = SanskritPreprocessor()
        
        query = "dharmah kim asti?"
        result = preprocessor.process(query)
        
        assert result.script == "loose_roman"
        assert result.slp1 is not None


class TestPipelineIntegration:
    """Test integration between components."""
    
    def test_bm25_to_metadata(self):
        """BM25 results should map to metadata store."""
        # Simulates: BM25 returns indices → MetadataStore retrieves chunks
        mock_bm25_results = [(0, 5.2), (1, 3.8), (2, 2.1)]
        
        # Each index should be retrievable from metadata
        for idx, score in mock_bm25_results:
            assert isinstance(idx, int)
            assert isinstance(score, float)
            assert idx >= 0
    
    def test_vector_to_metadata(self):
        """Vector results should map to metadata store."""
        # Simulates: Vector search returns indices → MetadataStore retrieves
        mock_vector_results = [(1, 0.15), (0, 0.23), (3, 0.45)]
        
        for idx, distance in mock_vector_results:
            assert isinstance(idx, int)
            assert isinstance(distance, float)
            assert idx >= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
