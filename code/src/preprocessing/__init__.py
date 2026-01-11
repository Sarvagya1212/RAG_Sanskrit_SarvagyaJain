"""Preprocessing module for Sanskrit text processing."""

from code.src.preprocessing.script_detector import (
    detect_script,
    is_devanagari,
    is_iast,
    has_mixed_scripts,
    get_script_stats
)

from code.src.preprocessing.transliterator import (
    to_slp1,
    from_slp1,
    fix_word_final_h,
    normalize_to_slp1,
    are_equivalent
)

from code.src.preprocessing.normalizer import (
    normalize_unicode,
    normalize_anusvara,
    normalize_vowel_length,
    clean_text,
    remove_punctuation,
    preprocess_text,
    preprocess_query,
    preprocess_document
)

from code.src.preprocessing.preprocessor import (
    SanskritPreprocessor,
    PreprocessingResult,
    get_preprocessor,
    quick_process
)

__all__ = [
    # Script detection
    'detect_script',
    'is_devanagari',
    'is_iast',
    'has_mixed_scripts',
    'get_script_stats',
    # Transliteration
    'to_slp1',
    'from_slp1',
    'fix_word_final_h',
    'normalize_to_slp1',
    'are_equivalent',
    # Normalization
    'normalize_unicode',
    'normalize_anusvara',
    'normalize_vowel_length',
    'clean_text',
    'preprocess_text',
    'preprocess_query',
    'preprocess_document',
    # Main preprocessor
    'SanskritPreprocessor',
    'PreprocessingResult',
    'get_preprocessor',
    'quick_process'
]
