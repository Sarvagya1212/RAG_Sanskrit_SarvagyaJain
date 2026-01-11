"""Tests for chunking module."""

import pytest
from code.src.chunking import (
    HierarchicalChunker,
    ParentChunk,
    ChildChunk
)


class TestHierarchicalChunker:
    """Tests for HierarchicalChunker class."""
    
    @pytest.fixture
    def chunker(self):
        """Create chunker instance."""
        return HierarchicalChunker()
    
    def test_initialization(self, chunker):
        """Chunker should initialize correctly."""
        assert chunker is not None
        assert chunker.PARENT_TARGET_TOKENS == 700
        assert chunker.CHILD_TARGET_TOKENS == 175
    
    def test_chunk_story(self, chunker):
        """Should create parent and child chunks."""
        story_id = 'test_1'
        story_title = 'Test Story'
        story_text = 'राजा वने गतवान् । सः मृगम् अपश्यत् । मृगः पलायितः । राजा निराशः अभवत् ।'
        
        parents, children = chunker.chunk_story(story_id, story_title, story_text)
        
        assert len(parents) >= 1
        assert len(children) >= 1
        assert all(isinstance(p, ParentChunk) for p in parents)
        assert all(isinstance(c, ChildChunk) for c in children)
    
    def test_parent_child_relationship(self, chunker):
        """Children should reference their parent."""
        parents, children = chunker.chunk_story(
            'test_2', 'Relationship Test', 
            'Text for testing parent-child relationship ।'
        )
        
        # Each child should have a valid parent_id
        parent_ids = {p.parent_id for p in parents}
        for child in children:
            assert child.parent_id in parent_ids


class TestParentChunk:
    """Tests for ParentChunk dataclass."""
    
    def test_creation(self):
        """ParentChunk should be created correctly."""
        parent = ParentChunk(
            parent_id='p1',
            story_id='s1',
            story_title='Test',
            text='Original text',
            preprocessed_text='preprocessed text',
            start_char=0,
            end_char=13,
            token_count=10
        )
        
        assert parent.parent_id == 'p1'
        assert parent.story_title == 'Test'


class TestChildChunk:
    """Tests for ChildChunk dataclass."""
    
    def test_creation(self):
        """ChildChunk should be created correctly."""
        child = ChildChunk(
            chunk_id='c1',
            parent_id='p1',
            story_id='s1',
            story_title='Test',
            text='Child text',
            preprocessed_text='child preprocessed',
            parent_text='Parent text',
            parent_preprocessed='parent preprocessed',
            child_index=0,
            total_children=2,
            start_char=0,
            end_char=10,
            token_count=5
        )
        
        assert child.chunk_id == 'c1'
        assert child.parent_id == 'p1'
        assert child.child_index == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
