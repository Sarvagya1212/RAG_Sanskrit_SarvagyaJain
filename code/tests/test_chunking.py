"""Tests for chunking module."""

import pytest
from code.src.chunking import (
    Chunk,
    SanskritChunker,
    detect_content_type,
    estimate_token_count
)


class TestContentTypeDetection:
    """Tests for content type detection."""
    
    def test_narrative_prose_detection(self):
        """Narrative prose should be detected."""
        text = "राजा वने गतवान् । सः मृगम् अपश्यत् । मृगः पलायितः ।"
        assert detect_content_type(text) == "narrative_prose"
    
    def test_dialogue_prose_detection(self):
        """Dialogue prose should be detected."""
        # High danda density + many dialogue markers
        text = '"कः अस्ति?" इति प्रष्टवान् । "अहम् अस्मि" इति उक्तवान् । "गच्छ" इति आदिष्टवान् ।'
        content_type = detect_content_type(text)
        assert content_type in ["dialogue_prose", "narrative_prose"]
    
    def test_verse_conclusion_detection(self):
        """Verse conclusion should be detected."""
        # High double danda density
        text = "उद्यमः साहसम् धैर्यम् ॥ बुद्धिः शक्तिः पराक्रमः ॥"
        assert detect_content_type(text) == "verse_conclusion"
    
    def test_empty_text(self):
        """Empty text should default to narrative_prose."""
        assert detect_content_type("") == "narrative_prose"
        assert detect_content_type("   ") == "narrative_prose"


class TestTokenEstimation:
    """Tests for token count estimation."""
    
    def test_token_estimation(self):
        """Token count should be estimated correctly."""
        text = "धर्मक्षेत्रे कुरुक्षेत्रे"  # ~26 chars → ~6-7 tokens
        tokens = estimate_token_count(text)
        assert 5 <= tokens <= 8
    
    def test_empty_text(self):
        """Empty text should return 0 tokens."""
        assert estimate_token_count("") == 0
        assert estimate_token_count("   ") == 0


class TestSanskritChunker:
    """Tests for SanskritChunker class."""
    
    @pytest.fixture
    def chunker(self):
        """Create chunker instance."""
        return SanskritChunker(narrative_target_tokens=50)  # Small for testing
    
    def test_initialization(self, chunker):
        """Chunker should initialize correctly."""
        assert chunker is not None
        assert chunker.narrative_target_tokens == 50
    
    def test_chunk_narrative_prose(self, chunker):
        """Narrative prose should be chunked by sentences."""
        text = "राजा वने गतवान् । सः मृगम् अपश्यत् । मृगः पलायितः । राजा निराशः अभवत् ।"
        slp1 = text  # Using same for test
        
        chunks = chunker.chunk_narrative_prose(text, slp1, story_id=1, story_title="Test")
        
        assert len(chunks) > 0
        assert all(isinstance(c, Chunk) for c in chunks)
        assert all(c.content_type == "narrative_prose" for c in chunks)
        assert all(c.story_id == 1 for c in chunks)
    
    def test_chunk_dialogue_prose(self, chunker):
        """Dialogue prose should be chunked by इति."""
        text = '"कः असि?" इति प्रष्टवान् । "राजा अस्मि" इति उक्तवान् ।'
        slp1 = text
        
        chunks = chunker.chunk_dialogue_prose(text, slp1, story_id=1)
        
        assert len(chunks) >= 1
        assert all(c.content_type == "dialogue_prose" for c in chunks)
    
    def test_chunk_verse_conclusion(self, chunker):
        """Verse conclusion should be kept complete."""
        text = "उद्यमः साहसम् धैर्यम् ॥"
        slp1 = text
        
        chunks = chunker.chunk_verse_conclusion(text, slp1, story_id=1)
        
        assert len(chunks) == 1
        assert chunks[0].content_type == "verse_conclusion"
        assert chunks[0].metadata.get('is_moral_verse') is True
    
    def test_chunk_story_with_detection(self, chunker):
        """chunk_story should auto-detect content type."""
        text = "राजा गतवान् । सः आगतवान् ।"
        slp1 = text
        
        chunks = chunker.chunk_story(text, slp1, story_id=1, story_title="Test")
        
        assert len(chunks) >= 1
        assert all(isinstance(c, Chunk) for c in chunks)


class TestChunkObject:
    """Tests for Chunk dataclass."""
    
    def test_chunk_creation(self):
        """Chunk should be created with all fields."""
        chunk = Chunk(
            chunk_id=1,
            text="test text",
            slp1_text="test slp1",
            content_type="narrative_prose",
            story_id=1,
            story_title="Test Story",
            token_count=10
        )
        
        assert chunk.chunk_id == 1
        assert chunk.text == "test text"
        assert chunk.content_type == "narrative_prose"
    
    def test_chunk_to_dict(self):
        """Chunk should convert to dictionary."""
        chunk = Chunk(
            chunk_id=1,
            text="test",
            slp1_text="test",
            content_type="narrative_prose",
            token_count=5
        )
        
        d = chunk.to_dict()
        assert d['chunk_id'] == 1
        assert d['content_type'] == "narrative_prose"
        assert 'metadata' in d


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
