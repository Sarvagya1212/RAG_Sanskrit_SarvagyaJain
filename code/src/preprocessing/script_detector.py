"""Script detection module for Sanskrit text processing."""

from typing import Set
from code.src.utils.logger import setup_logger

logger = setup_logger(__name__)

# Devanagari Unicode range
DEVANAGARI_START = 0x0900
DEVANAGARI_END = 0x097F

# IAST diacritics for Sanskrit romanization
IAST_DIACRITICS: Set[str] = {
    'ā', 'ī', 'ū', 'ṛ', 'ṝ', 'ḷ', 'ḹ',  # Vowels
    'ṃ', 'ḥ',                             # Anusvara, Visarga
    'ṅ', 'ñ',                             # Nasals
    'ṭ', 'ḍ', 'ṇ',                        # Retroflex
    'ś', 'ṣ',                             # Sibilants
    # Uppercase versions
    'Ā', 'Ī', 'Ū', 'Ṛ', 'Ṝ', 'Ḷ', 'Ḹ',
    'Ṃ', 'Ḥ', 'Ṅ', 'Ñ', 'Ṭ', 'Ḍ', 'Ṇ', 'Ś', 'Ṣ'
}


def is_devanagari(text: str) -> bool:
    """
    Check if text is primarily Devanagari script.
    
    Args:
        text: Input text to analyze
        
    Returns:
        True if >50% of alphabetic characters are Devanagari
    """
    if not text:
        return False
    
    # Count Devanagari and total alphabetic characters
    devanagari_count = 0
    alpha_count = 0
    
    for char in text:
        code = ord(char)
        if DEVANAGARI_START <= code <= DEVANAGARI_END:
            devanagari_count += 1
            alpha_count += 1
        elif char.isalpha():
            alpha_count += 1
    
    if alpha_count == 0:
        return False
    
    return (devanagari_count / alpha_count) > 0.5


def is_iast(text: str) -> bool:
    """
    Check if text contains IAST (International Alphabet of Sanskrit Transliteration).
    
    IAST uses specific diacritics like ā, ī, ū, ṛ, ḥ, ṃ, ś, ṣ to represent
    Sanskrit sounds in Roman script.
    
    Args:
        text: Input text to analyze
        
    Returns:
        True if text contains IAST diacritics
    """
    if not text:
        return False
    
    for char in text:
        if char in IAST_DIACRITICS:
            return True
    
    return False


def has_mixed_scripts(text: str) -> bool:
    """
    Detect if text mixes Devanagari and Roman scripts.
    
    Args:
        text: Input text to analyze
        
    Returns:
        True if both Devanagari and Roman characters are present
    """
    if not text:
        return False
    
    has_devanagari = False
    has_roman = False
    
    for char in text:
        code = ord(char)
        if DEVANAGARI_START <= code <= DEVANAGARI_END:
            has_devanagari = True
        elif char.isalpha():  # Roman/Latin alphabet
            has_roman = True
        
        # Early exit if both found
        if has_devanagari and has_roman:
            logger.warning(
                f"Mixed scripts detected in text: '{text[:50]}...'"
                if len(text) > 50 else f"Mixed scripts detected in text: '{text}'"
            )
            return True
    
    return False


def detect_script(text: str) -> str:
    """
    Automatically detect the script type of input text.
    
    Detection priority:
    1. Check for Devanagari characters (Unicode range U+0900-U+097F)
    2. Check for IAST diacritics (ā, ī, ū, ṛ, ḥ, ṃ, ś, ṣ, etc.)
    3. Default to loose_roman for plain ASCII
    
    Args:
        text: Input text to analyze
        
    Returns:
        Script type: "devanagari", "iast", or "loose_roman"
        
    Examples:
        >>> detect_script("धर्मः")
        'devanagari'
        >>> detect_script("dharmaḥ")
        'iast'
        >>> detect_script("dharmah")
        'loose_roman'
    """
    if not text or not text.strip():
        logger.warning("Empty text provided for script detection")
        return "loose_roman"
    
    # Check for mixed scripts first (logs warning)
    if has_mixed_scripts(text):
        # Return based on majority script
        if is_devanagari(text):
            return "devanagari"
        elif is_iast(text):
            return "iast"
        return "loose_roman"
    
    # Priority 1: Check for Devanagari
    if is_devanagari(text):
        logger.debug(f"Detected Devanagari script: '{text[:30]}...'")
        return "devanagari"
    
    # Priority 2: Check for IAST diacritics
    if is_iast(text):
        logger.debug(f"Detected IAST script: '{text[:30]}...'")
        return "iast"
    
    # Default: Plain ASCII romanization
    logger.debug(f"Detected loose Roman script: '{text[:30]}...'")
    return "loose_roman"


def get_script_stats(text: str) -> dict:
    """
    Get detailed statistics about script composition of text.
    
    Args:
        text: Input text to analyze
        
    Returns:
        Dictionary with script statistics
    """
    stats = {
        'total_chars': len(text),
        'devanagari_chars': 0,
        'iast_chars': 0,
        'roman_chars': 0,
        'other_chars': 0,
        'script_type': None,
        'is_mixed': False
    }
    
    for char in text:
        code = ord(char)
        if DEVANAGARI_START <= code <= DEVANAGARI_END:
            stats['devanagari_chars'] += 1
        elif char in IAST_DIACRITICS:
            stats['iast_chars'] += 1
        elif char.isalpha():
            stats['roman_chars'] += 1
        elif not char.isspace():
            stats['other_chars'] += 1
    
    stats['script_type'] = detect_script(text)
    stats['is_mixed'] = has_mixed_scripts(text)
    
    return stats
