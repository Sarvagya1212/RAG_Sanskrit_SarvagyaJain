"""Transliteration module for Sanskrit text processing.

Converts between Devanagari, IAST, and SLP1 (internal encoding).
Uses indic-transliteration library for accurate conversions.
"""

import re
from typing import Optional
from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate
from code.src.preprocessing.script_detector import detect_script
from code.src.utils.logger import setup_logger

logger = setup_logger(__name__)

# Script constants for sanscript library
DEVANAGARI = sanscript.DEVANAGARI
IAST = sanscript.IAST
SLP1 = sanscript.SLP1
HK = sanscript.HK  # Harvard-Kyoto (similar to loose roman)


def fix_word_final_h(text: str) -> str:
    """
    Fix word-final 'h' to visarga 'H' in loose Roman text.
    
    In loose romanization, users often type 'h' at word end
    when they mean visarga (ḥ). This function corrects that.
    
    Pattern: vowel + h at word boundary → vowel + H (visarga)
    
    Args:
        text: Loose Roman Sanskrit text
        
    Returns:
        Text with word-final 'h' after vowels converted to 'H'
        
    Examples:
        >>> fix_word_final_h("dharmah")
        'dharmaH'
        >>> fix_word_final_h("narah gacchati")
        'naraH gacchati'
        >>> fix_word_final_h("saha")  # Not at word end
        'saha'
    """
    if not text:
        return text
    
    # Pattern: vowel (a, A, i, I, u, U, e, o, R, L) + h at word boundary
    # Word boundary: end of string, whitespace, or punctuation
    pattern = r'([aAiIuUeEoORLÁáÍíÚú])h(?=\s|$|[.,;:!?।॥\'\"])'
    
    result = re.sub(pattern, r'\1H', text)
    
    if result != text:
        logger.debug(f"Fixed word-final h: '{text}' → '{result}'")
    
    return result


def to_slp1(text: str, source_script: Optional[str] = None) -> str:
    """
    Convert text from any script to SLP1 (internal encoding).
    
    SLP1 is used internally because:
    - Uses only ASCII characters (a-zA-Z)
    - One-to-one mapping from Devanagari
    - Easier for indexing and searching
    
    Args:
        text: Input text in any Sanskrit script
        source_script: Optional script type ("devanagari", "iast", "loose_roman")
                      If None, auto-detects using detect_script()
    
    Returns:
        Text converted to SLP1 encoding
        
    Examples:
        >>> to_slp1("धर्मः")
        'DarmaH'
        >>> to_slp1("dharmaḥ")
        'DarmaH'
        >>> to_slp1("dharmah")
        'DarmaH'
    """
    if not text or not text.strip():
        return text
    
    # Auto-detect script if not provided
    if source_script is None:
        source_script = detect_script(text)
        logger.debug(f"Auto-detected script: {source_script}")
    
    try:
        if source_script == "devanagari":
            result = transliterate(text, DEVANAGARI, SLP1)
            
        elif source_script == "iast":
            result = transliterate(text, IAST, SLP1)
            
        elif source_script == "loose_roman":
            # First fix word-final h to H (visarga)
            fixed_text = fix_word_final_h(text)
            # Then transliterate from Harvard-Kyoto (closest to loose roman)
            result = transliterate(fixed_text, HK, SLP1)
            
        else:
            logger.warning(f"Unknown script '{source_script}', treating as loose_roman")
            fixed_text = fix_word_final_h(text)
            result = transliterate(fixed_text, HK, SLP1)
        
        logger.debug(f"Transliterated to SLP1: '{text[:30]}...' → '{result[:30]}...'")
        return result
        
    except Exception as e:
        logger.error(f"Transliteration error: {e}")
        return text  # Return original on error


def from_slp1(text: str, target_script: str = "devanagari") -> str:
    """
    Convert SLP1 text back to human-readable script.
    
    Used when displaying results to users.
    
    Args:
        text: SLP1 encoded text
        target_script: Target script ("devanagari" or "iast")
        
    Returns:
        Text converted to target script
        
    Examples:
        >>> from_slp1("DarmaH", "devanagari")
        'धर्मः'
        >>> from_slp1("DarmaH", "iast")
        'dharmaḥ'
    """
    if not text or not text.strip():
        return text
    
    try:
        if target_script == "devanagari":
            result = transliterate(text, SLP1, DEVANAGARI)
        elif target_script == "iast":
            result = transliterate(text, SLP1, IAST)
        else:
            logger.warning(f"Unknown target script '{target_script}', using devanagari")
            result = transliterate(text, SLP1, DEVANAGARI)
        
        logger.debug(f"Transliterated from SLP1: '{text[:30]}...' → '{result[:30]}...'")
        return result
        
    except Exception as e:
        logger.error(f"Transliteration error: {e}")
        return text


def normalize_to_slp1(text: str) -> str:
    """
    Convenience function: auto-detect script and convert to SLP1.
    
    This is the main entry point for normalizing user queries
    or document text for indexing.
    
    Args:
        text: Input text in any script
        
    Returns:
        Normalized SLP1 text
    """
    return to_slp1(text, source_script=None)


def are_equivalent(text1: str, text2: str) -> bool:
    """
    Check if two texts are equivalent when normalized to SLP1.
    
    Useful for cross-script matching.
    
    Args:
        text1: First text (any script)
        text2: Second text (any script)
        
    Returns:
        True if both normalize to the same SLP1 string
        
    Examples:
        >>> are_equivalent("धर्मः", "dharmaḥ")
        True
        >>> are_equivalent("धर्मः", "dharmah")
        True
    """
    slp1_1 = normalize_to_slp1(text1)
    slp1_2 = normalize_to_slp1(text2)
    
    return slp1_1 == slp1_2
