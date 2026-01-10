"""Text normalization module for Sanskrit preprocessing.

Handles Unicode normalization, anusvara normalization, and text cleanup
to ensure consistent matching across different input variations.
"""

import re
import unicodedata
from typing import Optional
from code.src.preprocessing.script_detector import detect_script
from code.src.preprocessing.transliterator import to_slp1, fix_word_final_h
from code.src.utils.logger import setup_logger

logger = setup_logger(__name__)


def normalize_unicode(text: str) -> str:
    """
    Apply NFC Unicode normalization.
    
    Different keyboards/sources may encode the same character differently:
    - ā as single character (U+0101)
    - ā as 'a' + combining macron (U+0061 + U+0304)
    
    NFC normalization makes them identical.
    
    Args:
        text: Input text
        
    Returns:
        NFC-normalized text
    """
    if not text:
        return text
    
    normalized = unicodedata.normalize('NFC', text)
    
    if normalized != text:
        logger.debug("Applied NFC normalization")
    
    return normalized


def normalize_anusvara(text: str) -> str:
    """
    Convert all homorganic nasals before consonants to anusvara 'M'.
    
    Solves matching problem where same word has different nasal representations:
    - "saMskfta" (with anusvara ं) 
    - "sanskfta" (with explicit dental nasal न्)
    Both normalize to → "saMskfta"
    
    SLP1 Nasal-Consonant Classes:
    - N (ङ्) velar nasal before k, K, g, G (velars)
    - Y (ञ्) palatal nasal before c, C, j, J (palatals)  
    - R (ण्) retroflex nasal before w, W, q, Q (retroflexes)
    - n (न्) dental nasal before t, T, d, D (dentals)
    - m (म्) labial nasal before p, P, b, B (labials)
    
    Args:
        text: SLP1-encoded text
        
    Returns:
        Text with all homorganic nasals normalized to anusvara 'M'
        
    Examples:
        >>> normalize_anusvara("sanskfta")  # dental n before s
        'saMskfta'
        >>> normalize_anusvara("saNgIta")   # velar N before g
        'saMgIta'
    """
    if not text:
        return text
    
    # Velar nasals (N/ङ्) before velar consonants (k, K, g, G)
    text = re.sub(r'N([kKgG])', r'M\1', text)
    
    # Palatal nasals (Y/ञ्) before palatal consonants (c, C, j, J)
    text = re.sub(r'Y([cCjJ])', r'M\1', text)
    
    # Retroflex nasals (R/ण्) before retroflex consonants (w, W, q, Q)
    # Note: SLP1 uses w/W/q/Q for retroflex stops ṭ/ṭh/ḍ/ḍh
    text = re.sub(r'R([wWqQ])', r'M\1', text)
    
    # Dental nasals (n/न्) before dental consonants (t, T, d, D)
    text = re.sub(r'n([tTdD])', r'M\1', text)
    
    # Labial nasals (m/म्) before labial consonants (p, P, b, B)
    text = re.sub(r'm([pPbB])', r'M\1', text)
    
    # Nasals before sibilants (S=ś, z=ṣ, s) and h - these keep anusvara
    text = re.sub(r'[nRYNm]([Szsh])', r'M\1', text)
    
    # Also normalize n before k/g (common variant spelling)
    text = re.sub(r'n([kKgG])', r'M\1', text)
    
    logger.debug(f"Normalized anusvara in text")
    return text


def clean_text(text: str) -> str:
    """
    Clean and normalize whitespace in text.
    
    - Removes extra whitespace
    - Normalizes line breaks
    - Strips leading/trailing spaces
    
    Args:
        text: Input text
        
    Returns:
        Cleaned text
    """
    if not text:
        return text
    
    # Normalize line breaks to single newline
    text = re.sub(r'\r\n', '\n', text)
    text = re.sub(r'\r', '\n', text)
    
    # Replace multiple spaces with single space
    text = re.sub(r'[ \t]+', ' ', text)
    
    # Replace multiple newlines with double newline (paragraph break)
    text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
    
    # Strip leading/trailing whitespace from each line
    lines = text.split('\n')
    lines = [line.strip() for line in lines]
    text = '\n'.join(lines)
    
    # Strip overall leading/trailing whitespace
    text = text.strip()
    
    return text


def remove_punctuation(text: str, keep_dandas: bool = True) -> str:
    """
    Remove punctuation from text.
    
    Args:
        text: Input text
        keep_dandas: If True, keep Sanskrit dandas (। and ॥)
        
    Returns:
        Text with punctuation removed
    """
    if not text:
        return text
    
    if keep_dandas:
        # Remove all punctuation except dandas
        text = re.sub(r'[^\w\s।॥|]', '', text)
    else:
        # Remove all punctuation including dandas
        text = re.sub(r'[^\w\s]', '', text)
    
    return text


def preprocess_text(text: str, for_indexing: bool = True) -> str:
    """
    Complete preprocessing pipeline for Sanskrit text.
    
    Applies all normalization steps to prepare text for indexing or search.
    
    Pipeline:
    1. Unicode NFC normalization
    2. Script detection
    3. Fix word-final h (for loose Roman)
    4. Convert to SLP1
    5. Normalize anusvara
    6. Clean whitespace
    
    Args:
        text: Input text in any script
        for_indexing: If True, applies more aggressive normalization
        
    Returns:
        Fully normalized SLP1 text ready for indexing/search
        
    Examples:
        >>> preprocess_text("धर्मः")
        'DarmaH'
        >>> preprocess_text("dharmah")
        'DarmaH'
        >>> preprocess_text("संस्कृतम्")
        'saMskftam'
    """
    if not text or not text.strip():
        return ""
    
    # Step 1: Unicode normalization
    text = normalize_unicode(text)
    
    # Step 2: Detect script
    script = detect_script(text)
    logger.debug(f"Preprocessing text with detected script: {script}")
    
    # Step 3: Apply fixes based on script
    if script == "loose_roman":
        text = fix_word_final_h(text)
    
    # Step 4: Convert to SLP1
    slp1_text = to_slp1(text, script)
    
    # Step 5: Normalize anusvara (for better matching)
    if for_indexing:
        slp1_text = normalize_anusvara(slp1_text)
    
    # Step 6: Clean whitespace
    slp1_text = clean_text(slp1_text)
    
    logger.debug(f"Preprocessed: '{text[:30]}...' → '{slp1_text[:30]}...'")
    return slp1_text


def preprocess_query(query: str) -> str:
    """
    Preprocess a search query.
    
    Same as preprocess_text but optimized for queries.
    
    Args:
        query: User search query in any script
        
    Returns:
        Normalized SLP1 query
    """
    return preprocess_text(query, for_indexing=True)


def preprocess_document(text: str) -> str:
    """
    Preprocess document text for indexing.
    
    Args:
        text: Document text in any script
        
    Returns:
        Normalized SLP1 text for indexing
    """
    return preprocess_text(text, for_indexing=True)
