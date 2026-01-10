"""Tests for transliteration module."""

import pytest
from code.src.preprocessing.transliterator import (
    to_slp1,
    from_slp1,
    fix_word_final_h,
    normalize_to_slp1,
    are_equivalent
)


class TestFixWordFinalH:
    """Tests for fix_word_final_h function."""
    
    def test_word_final_h_at_end(self):
        """Word-final h should become H."""
        assert fix_word_final_h("dharmah") == "dharmaH"
        assert fix_word_final_h("narah") == "naraH"
        assert fix_word_final_h("devah") == "devaH"
    
    def test_word_final_h_before_space(self):
        """Word-final h before space should become H."""
        assert fix_word_final_h("narah gacchati") == "naraH gacchati"
        assert fix_word_final_h("dharmah ca") == "dharmaH ca"
    
    def test_h_in_middle_unchanged(self):
        """h in middle of word should remain unchanged."""
        assert fix_word_final_h("saha") == "saha"
        assert fix_word_final_h("moha") == "moha"
        assert fix_word_final_h("graha") == "graha"
    
    def test_consonant_h_unchanged(self):
        """h after consonant should remain unchanged."""
        # These are not visarga patterns
        result = fix_word_final_h("path")
        assert result == "path"  # No vowel before h
    
    def test_empty_text(self):
        """Empty text should return empty."""
        assert fix_word_final_h("") == ""
        assert fix_word_final_h(None) is None


class TestToSlp1:
    """Tests for to_slp1 function."""
    
    def test_devanagari_to_slp1(self):
        """Devanagari should convert to SLP1."""
        result = to_slp1("धर्मः", "devanagari")
        assert "Darma" in result or "dharma" in result.lower()
    
    def test_iast_to_slp1(self):
        """IAST should convert to SLP1."""
        result = to_slp1("dharmaḥ", "iast")
        assert result is not None
    
    def test_loose_roman_to_slp1(self):
        """Loose Roman with h fix should convert to SLP1."""
        result = to_slp1("dharmah", "loose_roman")
        assert "H" in result  # Should have visarga
    
    def test_auto_detect_devanagari(self):
        """Auto-detection should work for Devanagari."""
        result = to_slp1("धर्मः")
        assert result is not None
        assert len(result) > 0
    
    def test_empty_text(self):
        """Empty text should return empty."""
        assert to_slp1("") == ""
        assert to_slp1("   ") == "   "


class TestFromSlp1:
    """Tests for from_slp1 function."""
    
    def test_slp1_to_devanagari(self):
        """SLP1 should convert to Devanagari."""
        result = from_slp1("DarmaH", "devanagari")
        assert result is not None
        # Should contain Devanagari characters
        has_devanagari = any(0x0900 <= ord(c) <= 0x097F for c in result)
        assert has_devanagari
    
    def test_slp1_to_iast(self):
        """SLP1 should convert to IAST."""
        result = from_slp1("DarmaH", "iast")
        assert result is not None
    
    def test_empty_text(self):
        """Empty text should return empty."""
        assert from_slp1("") == ""


class TestCrossScriptEquivalence:
    """Critical tests: all scripts should normalize to same SLP1."""
    
    def test_dharma_equivalence(self):
        """'dharma' in all scripts should be equivalent."""
        devanagari = "धर्म"
        iast = "dharma"
        loose = "dharma"
        
        slp1_dev = normalize_to_slp1(devanagari)
        slp1_iast = normalize_to_slp1(iast)
        slp1_loose = normalize_to_slp1(loose)
        
        # All should produce same result
        assert slp1_dev == slp1_iast == slp1_loose
    
    def test_dharmah_with_visarga(self):
        """'dharmaḥ' with visarga should be equivalent across scripts."""
        devanagari = "धर्मः"
        iast = "dharmaḥ"
        loose = "dharmah"  # User types h at end
        
        slp1_dev = normalize_to_slp1(devanagari)
        slp1_iast = normalize_to_slp1(iast)
        slp1_loose = normalize_to_slp1(loose)
        
        # All should have visarga (H)
        assert "H" in slp1_dev
        assert "H" in slp1_iast
        assert "H" in slp1_loose
    
    def test_are_equivalent_function(self):
        """are_equivalent should detect cross-script matches."""
        assert are_equivalent("धर्म", "dharma") is True
        assert are_equivalent("योग", "yoga") is True
    
    def test_non_equivalent_texts(self):
        """Different words should not be equivalent."""
        assert are_equivalent("धर्म", "karma") is False


class TestNormalizeToSlp1:
    """Tests for normalize_to_slp1 convenience function."""
    
    def test_auto_normalize(self):
        """Should auto-detect and normalize any script."""
        result1 = normalize_to_slp1("धर्मक्षेत्रे")
        result2 = normalize_to_slp1("dharmakṣetre")
        
        assert result1 is not None
        assert result2 is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
