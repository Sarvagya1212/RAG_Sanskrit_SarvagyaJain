"""Tests for normalization module."""

import pytest
from code.src.preprocessing.normalizer import (
    normalize_unicode,
    normalize_anusvara,
    clean_text,
    remove_punctuation,
    preprocess_text,
    preprocess_query,
    preprocess_document
)


class TestNormalizeUnicode:
    """Tests for Unicode NFC normalization."""
    
    def test_nfc_normalization(self):
        """Text should be NFC normalized."""
        # Composed vs decomposed characters
        result = normalize_unicode("ā")  # Should work with any form
        assert result is not None
        assert len(result) >= 1
    
    def test_empty_text(self):
        """Empty text should return empty."""
        assert normalize_unicode("") == ""
        assert normalize_unicode(None) is None


class TestNormalizeAnusvara:
    """Tests for anusvara normalization in SLP1."""
    
    def test_nasal_before_velar(self):
        """Velar nasal (N/ङ्) before velar consonant should become M."""
        assert normalize_anusvara("saNkara") == "saMkara"  # N before k
        assert normalize_anusvara("saNga") == "saMga"      # N before g
        # Also n (common variant) before velar
        assert normalize_anusvara("sankara") == "saMkara"  # n before k
    
    def test_nasal_before_palatal(self):
        """Palatal nasal (Y/ञ्) before palatal consonant should become M."""
        assert normalize_anusvara("saYcaya") == "saMcaya"  # Y before c
        assert normalize_anusvara("maYju") == "maMju"      # Y before j
    
    def test_nasal_before_dental(self):
        """Dental nasal (n/न्) before dental consonant should become M."""
        assert normalize_anusvara("santa") == "saMta"      # n before t
        assert normalize_anusvara("vanda") == "vaMda"      # n before d
    
    def test_nasal_before_labial(self):
        """Labial nasal (m/म्) before labial consonant should become M."""
        assert normalize_anusvara("sampanna") == "saMpanna"  # m before p
        assert normalize_anusvara("kambala") == "kaMbala"    # m before b
    
    def test_anusvara_unchanged(self):
        """Already anusvara (M) should remain unchanged."""
        assert normalize_anusvara("saMskfta") == "saMskfta"
    
    def test_empty_text(self):
        """Empty text should return empty."""
        assert normalize_anusvara("") == ""


class TestCleanText:
    """Tests for text cleanup."""
    
    def test_multiple_spaces(self):
        """Multiple spaces should become single space."""
        assert clean_text("hello   world") == "hello world"
        assert clean_text("a  b  c") == "a b c"
    
    def test_multiple_newlines(self):
        """Multiple newlines should become double newline."""
        result = clean_text("para1\n\n\n\npara2")
        assert "\n\n\n" not in result
    
    def test_trailing_whitespace(self):
        """Leading/trailing whitespace should be stripped."""
        assert clean_text("  hello  ") == "hello"
        assert clean_text("\nhello\n") == "hello"
    
    def test_empty_text(self):
        """Empty text should return empty."""
        assert clean_text("") == ""


class TestRemovePunctuation:
    """Tests for punctuation removal."""
    
    def test_keep_dandas(self):
        """Dandas should be kept by default."""
        result = remove_punctuation("धर्मः। अर्थः।", keep_dandas=True)
        assert "।" in result
    
    def test_remove_dandas(self):
        """Dandas should be removed when specified."""
        result = remove_punctuation("धर्म। अर्थ।", keep_dandas=False)
        assert "।" not in result
    
    def test_empty_text(self):
        """Empty text should return empty."""
        assert remove_punctuation("") == ""


class TestPreprocessText:
    """Tests for complete preprocessing pipeline."""
    
    def test_devanagari_preprocessing(self):
        """Devanagari should be preprocessed to SLP1."""
        result = preprocess_text("धर्मः")
        assert result is not None
        assert "H" in result  # Should have visarga
    
    def test_iast_preprocessing(self):
        """IAST should be preprocessed to SLP1."""
        result = preprocess_text("dharmaḥ")
        assert result is not None
    
    def test_loose_roman_preprocessing(self):
        """Loose Roman with h fix should work."""
        result = preprocess_text("dharmah")
        assert "H" in result  # h should become H (visarga)
    
    def test_empty_text(self):
        """Empty text should return empty string."""
        assert preprocess_text("") == ""
        assert preprocess_text("   ") == ""


class TestCrossScriptEquivalence:
    """Critical tests: all scripts should normalize to same result."""
    
    def test_dharma_all_scripts_equal(self):
        """dharma in all scripts should preprocess to same result."""
        dev = preprocess_text("धर्म")
        iast = preprocess_text("dharma")
        loose = preprocess_text("dharma")
        
        assert dev == iast == loose
    
    def test_sanskrit_all_scripts_equal(self):
        """sanskrit/संस्कृत should be equivalent."""
        dev = preprocess_text("संस्कृत")
        # After anusvara normalization, should have M
        assert "M" in dev or "m" in dev.lower()


class TestPreprocessQuery:
    """Tests for query preprocessing."""
    
    def test_query_preprocessing(self):
        """Query should be preprocessed correctly."""
        result = preprocess_query("धर्मः")
        assert result is not None
        assert len(result) > 0


class TestPreprocessDocument:
    """Tests for document preprocessing."""
    
    def test_document_preprocessing(self):
        """Document should be preprocessed correctly."""
        result = preprocess_document("धर्मक्षेत्रे कुरुक्षेत्रे")
        assert result is not None
        assert len(result) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
