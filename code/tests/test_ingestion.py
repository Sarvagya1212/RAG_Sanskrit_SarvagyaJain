import pytest
from pathlib import Path
from code.src.ingestion.document_loader import DocumentLoader
from code.src.ingestion.story_segmenter import StorySegmenter

def test_document_loader_initialization():
    """Test DocumentLoader initializes correctly."""
    loader = DocumentLoader()
    assert loader is not None
    assert loader.stats['total_chars'] == 0

def test_devanagari_validation():
    """Test Devanagari character detection."""
    loader = DocumentLoader()
    
    # Valid Devanagari text
    valid_text = "धर्मक्षेत्रे कुरुक्षेत्रे"
    loader._validate_unicode(valid_text)  # Should not raise
    
    # Collect stats
    loader._collect_statistics(valid_text)
    assert loader.stats['devanagari_chars'] > 0

def test_danda_counting():
    """Test danda character counting."""
    loader = DocumentLoader()
    
    text = "धर्मः। अर्थः। कामः। मोक्षः॥"
    loader._collect_statistics(text)
    
    assert loader.stats['danda_count'] == 3
    assert loader.stats['double_danda_count'] == 1

def test_story_segmenter_initialization():
    """Test StorySegmenter initializes correctly."""
    segmenter = StorySegmenter()
    assert segmenter is not None
    assert segmenter.title_pattern is not None

def test_title_detection():
    """Test line-based title detection."""
    segmenter = StorySegmenter()
    
    # Titles are standalone Sanskrit lines (< 50 chars, no dandas, 5+ Devanagari)
    text = """मूर्खभृत्यस्य

अथैकदा कश्चित् मूर्खः आसीत् । सः गृहे गतवान् ।

कालीदासस्य चातुर्यम्

कालीदासः महाकविः आसीत् । सः काव्यम् अलिखत् ।
"""
    
    titles = segmenter._detect_story_titles(text)
    assert len(titles) == 2
    assert titles[0][0] == "मूर्खभृत्यस्य"
    assert titles[1][0] == "कालीदासस्य चातुर्यम्"

def test_title_rejects_prose_sentences():
    """Test that prose sentences are not detected as titles."""
    segmenter = StorySegmenter()
    
    # These should NOT be detected as titles
    assert not segmenter._is_sanskrit_title("एकः परमः देवभक्तः अस्ति")  # starts with एकः
    assert not segmenter._is_sanskrit_title("सः गृहे गतवान् ।")  # contains danda
    assert not segmenter._is_sanskrit_title("अथ कश्चित् राजा आसीत्")  # starts with अथ
    
    # These SHOULD be detected as titles
    assert segmenter._is_sanskrit_title("मूर्खभृत्यस्य")
    assert segmenter._is_sanskrit_title("वृद्धायाः चार्तुयम्")
    assert segmenter._is_sanskrit_title("कालीदासस्य चातुर्यम्")

def test_content_type_detection():
    """Test content type classification."""
    segmenter = StorySegmenter()
    
    # Narrative prose
    metadata1 = {
        'danda_density': 0.05,
        'double_danda_density': 0.01,
        'dialogue_markers': 2
    }
    assert segmenter._detect_content_type(metadata1) == 'narrative_prose'
    
    # Dialogue prose
    metadata2 = {
        'danda_density': 0.10,
        'double_danda_density': 0.01,
        'dialogue_markers': 8
    }
    assert segmenter._detect_content_type(metadata2) == 'dialogue_prose'
    
    # Verse conclusion
    metadata3 = {
        'danda_density': 0.05,
        'double_danda_density': 0.03,
        'dialogue_markers': 1
    }
    assert segmenter._detect_content_type(metadata3) == 'verse_conclusion'

if __name__ == "__main__":
    pytest.main([__file__, "-v"])