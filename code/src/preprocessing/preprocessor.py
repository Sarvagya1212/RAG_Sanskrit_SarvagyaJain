"""Main preprocessor class coordinating all preprocessing modules."""

import time
from typing import Dict, Any, Optional
from dataclasses import dataclass, field

from code.src.preprocessing.script_detector import (
    detect_script,
    is_devanagari,
    has_mixed_scripts,
    get_script_stats
)
from code.src.preprocessing.transliterator import (
    to_slp1,
    from_slp1,
    fix_word_final_h
)
from code.src.preprocessing.normalizer import (
    normalize_unicode,
    normalize_anusvara,
    clean_text,
    preprocess_text
)
from code.src.utils.logger import setup_logger

logger = setup_logger(__name__)


@dataclass
class PreprocessingResult:
    """Result of preprocessing operation."""
    original: str
    script: str
    slp1: str
    processing_time: float
    is_mixed_script: bool = False
    stats: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'original': self.original,
            'script': self.script,
            'slp1': self.slp1,
            'processing_time': self.processing_time,
            'is_mixed_script': self.is_mixed_script,
            'stats': self.stats
        }


class SanskritPreprocessor:
    """
    Main preprocessing coordinator for Sanskrit text.
    
    Combines script detection, transliteration, and normalization
    into a single unified interface.
    
    Usage:
        >>> preprocessor = SanskritPreprocessor()
        >>> result = preprocessor.process("धर्मः")
        >>> print(result.slp1)
        'DarmaH'
    """
    
    def __init__(self, normalize_nasals: bool = True, log_mixed_scripts: bool = True):
        """
        Initialize preprocessor.
        
        Args:
            normalize_nasals: If True, normalize anusvara variations
            log_mixed_scripts: If True, log warnings for mixed script text
        """
        self.normalize_nasals = normalize_nasals
        self.log_mixed_scripts = log_mixed_scripts
        logger.info("SanskritPreprocessor initialized")
    
    def process(self, text: str, script_override: str = None) -> PreprocessingResult:
        """
        Full preprocessing pipeline with logging.
        
        Steps:
        1. Unicode NFC normalization
        2. Script detection
        3. Mixed script warning
        4. Word-final h fix (for loose Roman)
        5. Convert to SLP1
        6. Anusvara normalization
        7. Text cleanup
        
        Args:
            text: Input text in any script
            script_override: Optional script name to force (e.g. 'devanagari')
            
        Returns:
            PreprocessingResult with original, script, slp1, timing
        """
        start_time = time.perf_counter()
        
        if not text or not text.strip():
            return PreprocessingResult(
                original=text or "",
                script="unknown",
                slp1="",
                processing_time=0.0
            )
        
        # Step 1: Unicode normalization
        normalized_text = normalize_unicode(text)
        
        # Step 2: Detect script
        if script_override:
            script = script_override
            logger.debug(f"Using overridden script: {script}")
        else:
            script = detect_script(normalized_text)
            logger.debug(f"Detected script: {script}")
        
        # Step 3: Check for mixed scripts
        is_mixed = has_mixed_scripts(normalized_text)
        if is_mixed and self.log_mixed_scripts:
            logger.warning(f"Mixed scripts detected in: '{text[:50]}...'")
        
        # Step 4: Fix word-final h for loose Roman
        if script == "loose_roman":
            normalized_text = fix_word_final_h(normalized_text)
        
        # Step 5: Convert to SLP1
        slp1_text = to_slp1(normalized_text, script)
        
        # Step 6: Anusvara normalization
        if self.normalize_nasals:
            slp1_text = normalize_anusvara(slp1_text)
        
        # Step 7: Vowel length normalization (for fuzzy matching)
        from code.src.preprocessing.normalizer import normalize_vowel_length
        slp1_text = normalize_vowel_length(slp1_text)
        
        # Step 8: Clean up
        slp1_text = clean_text(slp1_text)
        
        # Calculate timing
        processing_time = time.perf_counter() - start_time
        
        # Get stats
        stats = get_script_stats(text)
        
        result = PreprocessingResult(
            original=text,
            script=script,
            slp1=slp1_text,
            processing_time=processing_time,
            is_mixed_script=is_mixed,
            stats=stats
        )
        
        logger.debug(
            f"Processed: '{text[:30]}...' → '{slp1_text[:30]}...' "
            f"in {processing_time*1000:.2f}ms"
        )
        
        return result
    
    def process_batch(self, texts: list) -> list:
        """
        Process multiple texts.
        
        Args:
            texts: List of input texts
            
        Returns:
            List of PreprocessingResult objects
        """
        results = []
        for text in texts:
            results.append(self.process(text))
        return results
    
    def to_devanagari(self, slp1_text: str) -> str:
        """
        Convert SLP1 back to Devanagari for display.
        
        Args:
            slp1_text: SLP1 encoded text
            
        Returns:
            Devanagari text
        """
        return from_slp1(slp1_text, "devanagari")
    
    def to_iast(self, slp1_text: str) -> str:
        """
        Convert SLP1 back to IAST for display.
        
        Args:
            slp1_text: SLP1 encoded text
            
        Returns:
            IAST text
        """
        return from_slp1(slp1_text, "iast")
    
    def are_equivalent(self, text1: str, text2: str) -> bool:
        """
        Check if two texts are equivalent when normalized.
        
        Args:
            text1: First text (any script)
            text2: Second text (any script)
            
        Returns:
            True if both normalize to same SLP1
        """
        result1 = self.process(text1)
        result2 = self.process(text2)
        return result1.slp1 == result2.slp1


# Module-level convenience instance
_default_preprocessor: Optional[SanskritPreprocessor] = None


def get_preprocessor() -> SanskritPreprocessor:
    """Get or create default preprocessor instance."""
    global _default_preprocessor
    if _default_preprocessor is None:
        _default_preprocessor = SanskritPreprocessor()
    return _default_preprocessor


def quick_process(text: str) -> str:
    """
    Convenience function for quick preprocessing.
    
    Args:
        text: Input text
        
    Returns:
        Normalized SLP1 text
    """
    return get_preprocessor().process(text).slp1
