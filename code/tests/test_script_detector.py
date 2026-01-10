"""Tests for script detection module."""

import pytest
from code.src.preprocessing.script_detector import (
    detect_script,
    is_devanagari,
    is_iast,
    has_mixed_scripts,
    get_script_stats
)


class TestIsDevanagari:
    """Tests for is_devanagari function."""
    
    def test_pure_devanagari(self):
        """Pure Devanagari text should return True."""
        assert is_devanagari("धर्मक्षेत्रे") is True
        assert is_devanagari("कुरुक्षेत्रे") is True
        assert is_devanagari("संजय उवाच") is True
    
    def test_empty_text(self):
        """Empty text should return False."""
        assert is_devanagari("") is False
        assert is_devanagari("   ") is False
    
    def test_roman_only(self):
        """Roman text should return False."""
        assert is_devanagari("dharma") is False
        assert is_devanagari("hello world") is False
    
    def test_majority_devanagari(self):
        """Text with >50% Devanagari should return True."""
        assert is_devanagari("धर्म a") is True  # 4 devanagari, 1 roman


class TestIsIast:
    """Tests for is_iast function."""
    
    def test_iast_with_diacritics(self):
        """IAST text with diacritics should return True."""
        assert is_iast("dharmaḥ") is True
        assert is_iast("dharmakṣetre") is True
        assert is_iast("śānti") is True
        assert is_iast("yogī") is True
        assert is_iast("ṛṣi") is True
    
    def test_plain_roman(self):
        """Plain Roman text should return False."""
        assert is_iast("dharma") is False
        assert is_iast("yoga") is False
        assert is_iast("hello") is False
    
    def test_empty_text(self):
        """Empty text should return False."""
        assert is_iast("") is False


class TestHasMixedScripts:
    """Tests for has_mixed_scripts function."""
    
    def test_mixed_devanagari_roman(self):
        """Mixed Devanagari and Roman should return True."""
        assert has_mixed_scripts("dharma धर्म") is True
        assert has_mixed_scripts("धर्म dharma") is True
    
    def test_pure_devanagari(self):
        """Pure Devanagari should return False."""
        assert has_mixed_scripts("धर्मक्षेत्रे") is False
    
    def test_pure_roman(self):
        """Pure Roman should return False."""
        assert has_mixed_scripts("dharma yoga") is False
    
    def test_empty_text(self):
        """Empty text should return False."""
        assert has_mixed_scripts("") is False


class TestDetectScript:
    """Tests for main detect_script function."""
    
    def test_devanagari_detection(self):
        """Devanagari text should be detected."""
        assert detect_script("धर्मक्षेत्रे") == "devanagari"
        assert detect_script("संस्कृत भाषा") == "devanagari"
    
    def test_iast_detection(self):
        """IAST text should be detected."""
        assert detect_script("dharmakṣetre") == "iast"
        assert detect_script("śāntiḥ") == "iast"
        assert detect_script("yogī") == "iast"
    
    def test_loose_roman_detection(self):
        """Plain Roman text should be detected as loose_roman."""
        assert detect_script("dharmakshetre") == "loose_roman"
        assert detect_script("yoga") == "loose_roman"
        assert detect_script("hello world") == "loose_roman"
    
    def test_empty_text(self):
        """Empty text should default to loose_roman."""
        assert detect_script("") == "loose_roman"
        assert detect_script("   ") == "loose_roman"
    
    def test_mixed_scripts(self):
        """Mixed scripts should detect based on majority."""
        # Majority Devanagari
        result = detect_script("धर्म a")
        assert result == "devanagari"


class TestGetScriptStats:
    """Tests for get_script_stats function."""
    
    def test_devanagari_stats(self):
        """Stats for Devanagari text."""
        stats = get_script_stats("धर्मः")
        assert stats['devanagari_chars'] > 0
        assert stats['script_type'] == "devanagari"
        assert stats['is_mixed'] is False
    
    def test_mixed_stats(self):
        """Stats for mixed text."""
        stats = get_script_stats("dharma धर्म")
        assert stats['is_mixed'] is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
