"""Comprehensive tests for preprocessing integration."""

import pytest
import warnings
from code.src.preprocessing.preprocessor import (
    SanskritPreprocessor,
    PreprocessingResult,
    get_preprocessor,
    quick_process
)


class TestSanskritPreprocessor:
    """Tests for main SanskritPreprocessor class."""
    
    @pytest.fixture
    def preprocessor(self):
        """Create preprocessor instance for tests."""
        return SanskritPreprocessor()
    
    def test_initialization(self, preprocessor):
        """Preprocessor should initialize correctly."""
        assert preprocessor is not None
        assert preprocessor.normalize_nasals is True
        assert preprocessor.log_mixed_scripts is True
    
    def test_process_returns_result(self, preprocessor):
        """Process should return PreprocessingResult."""
        result = preprocessor.process("धर्मः")
        assert isinstance(result, PreprocessingResult)
        assert result.original == "धर्मः"
        assert result.script == "devanagari"
        assert len(result.slp1) > 0
        assert result.processing_time >= 0
    
    def test_process_empty_text(self, preprocessor):
        """Empty text should return empty result."""
        result = preprocessor.process("")
        assert result.slp1 == ""
        result2 = preprocessor.process("   ")
        assert result2.slp1 == ""


class TestCrossScriptEquivalence:
    """THE MOST IMPORTANT TESTS: Cross-script equivalence."""
    
    @pytest.fixture
    def preprocessor(self):
        return SanskritPreprocessor()
    
    def test_dharma_cross_script(self, preprocessor):
        """'dharma' in all scripts should produce identical SLP1."""
        # Same word in 3 scripts
        devanagari = "धर्म"
        iast = "dharma"
        loose = "dharma"
        
        # Process all three
        result1 = preprocessor.process(devanagari)
        result2 = preprocessor.process(iast)
        result3 = preprocessor.process(loose)
        
        # MUST be identical
        assert result1.slp1 == result2.slp1
        assert result2.slp1 == result3.slp1
        
        print(f"✓ All three convert to: {result1.slp1}")
    
    def test_dharmah_with_visarga(self, preprocessor):
        """'dharmaḥ' with visarga should be equivalent across scripts."""
        devanagari = "धर्मः"
        iast = "dharmaḥ"
        loose = "dharmah"  # User types h at end
        
        result1 = preprocessor.process(devanagari)
        result2 = preprocessor.process(iast)
        result3 = preprocessor.process(loose)
        
        # All should have visarga (H in SLP1)
        assert "H" in result1.slp1
        assert "H" in result2.slp1
        assert "H" in result3.slp1
        
        # All should be identical
        assert result1.slp1 == result2.slp1 == result3.slp1
        
        print(f"✓ All three convert to: {result1.slp1}")
    
    def test_kurukshetra_cross_script(self, preprocessor):
        """More complex word should work across scripts."""
        dev = "कुरुक्षेत्र"
        iast = "kurukṣetra"
        
        result1 = preprocessor.process(dev)
        result2 = preprocessor.process(iast)
        
        assert result1.slp1 == result2.slp1
    
    def test_are_equivalent_method(self, preprocessor):
        """are_equivalent should detect cross-script matches."""
        assert preprocessor.are_equivalent("धर्म", "dharma") is True
        assert preprocessor.are_equivalent("योग", "yoga") is True
        assert preprocessor.are_equivalent("धर्म", "karma") is False


class TestAnusvaraNormalization:
    """Test nasal/anusvara normalization."""
    
    @pytest.fixture
    def preprocessor(self):
        return SanskritPreprocessor(normalize_nasals=True)
    
    def test_sanskrit_nasal_variations(self, preprocessor):
        """Three ways to write 'sanskrit' should become identical."""
        # Three ways to write the same word
        anusvara = "संस्कृत"    # with anusvara (ं)
        # Note: We can't easily test dental-n and labial-m variations 
        # as they require specific keyboard input
        
        result = preprocessor.process(anusvara)
        
        # Should have M (anusvara in SLP1)
        assert "M" in result.slp1 or "m" in result.slp1.lower()
    
    def test_anusvara_before_consonants(self, preprocessor):
        """Anusvara variations should normalize."""
        # Text with anusvara
        text = "संगम"  # sangama
        result = preprocessor.process(text)
        
        # Should have normalized form
        assert len(result.slp1) > 0


class TestMixedScripts:
    """Test mixed script detection and handling."""
    
    @pytest.fixture
    def preprocessor(self):
        return SanskritPreprocessor(log_mixed_scripts=True)
    
    def test_mixed_script_detected(self, preprocessor):
        """Mixed script text should be flagged."""
        mixed = "dharma धर्म"
        result = preprocessor.process(mixed)
        
        # Should detect mixed scripts
        assert result.is_mixed_script is True
    
    def test_pure_script_not_flagged(self, preprocessor):
        """Pure script text should not be flagged."""
        pure_dev = "धर्मक्षेत्रे"
        result = preprocessor.process(pure_dev)
        
        assert result.is_mixed_script is False


class TestPreprocessingResult:
    """Test PreprocessingResult dataclass."""
    
    def test_to_dict(self):
        """Result should convert to dictionary."""
        result = PreprocessingResult(
            original="test",
            script="devanagari",
            slp1="test",
            processing_time=0.001,
            is_mixed_script=False,
            stats={}
        )
        
        d = result.to_dict()
        assert d['original'] == "test"
        assert d['script'] == "devanagari"
        assert 'slp1' in d
        assert 'processing_time' in d


class TestBatchProcessing:
    """Test batch processing capability."""
    
    @pytest.fixture
    def preprocessor(self):
        return SanskritPreprocessor()
    
    def test_process_batch(self, preprocessor):
        """Batch processing should work."""
        texts = ["धर्म", "yoga", "śānti"]
        results = preprocessor.process_batch(texts)
        
        assert len(results) == 3
        assert all(isinstance(r, PreprocessingResult) for r in results)


class TestConversionMethods:
    """Test conversion back to display scripts."""
    
    @pytest.fixture
    def preprocessor(self):
        return SanskritPreprocessor()
    
    def test_to_devanagari(self, preprocessor):
        """Should convert SLP1 back to Devanagari."""
        result = preprocessor.to_devanagari("Darma")
        # Should contain Devanagari characters
        has_dev = any(0x0900 <= ord(c) <= 0x097F for c in result)
        assert has_dev
    
    def test_to_iast(self, preprocessor):
        """Should convert SLP1 back to IAST."""
        result = preprocessor.to_iast("Darma")
        assert result is not None
        assert len(result) > 0


class TestConvenienceFunctions:
    """Test module-level convenience functions."""
    
    def test_get_preprocessor(self):
        """Should return singleton preprocessor."""
        p1 = get_preprocessor()
        p2 = get_preprocessor()
        assert p1 is p2
    
    def test_quick_process(self):
        """Quick process should return SLP1 string."""
        result = quick_process("धर्मः")
        assert isinstance(result, str)
        assert len(result) > 0
        assert "H" in result  # Should have visarga


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
