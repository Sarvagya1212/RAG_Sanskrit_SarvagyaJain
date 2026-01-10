"""Document loading and validation for Sanskrit texts."""

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import unicodedata
from code.src.utils.logger import setup_logger

logger = setup_logger(__name__)

@dataclass
class Document:
    """Represents a loaded document with metadata."""
    text: str
    source: str
    metadata: Dict
    encoding: str = "utf-8"
    
class DocumentLoader:
    """Load and validate Sanskrit text documents."""
    
    # Devanagari Unicode range
    DEVANAGARI_RANGE = (0x0900, 0x097F)
    
    def __init__(self):
        """Initialize document loader."""
        self.stats = {
            'total_chars': 0,
            'devanagari_chars': 0,
            'danda_count': 0,
            'double_danda_count': 0
        }
    
    def load_text_file(self, file_path: str) -> Document:
        """
        Load a plain text file with UTF-8 encoding.
        
        Args:
            file_path: Path to .txt file
            
        Returns:
            Document object with text and metadata
            
        Raises:
            FileNotFoundError: If file doesn't exist
            UnicodeDecodeError: If file isn't valid UTF-8
        """
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        logger.info(f"Loading text file: {path.name}")
        
        # Read with explicit UTF-8 encoding
        with open(path, 'r', encoding='utf-8-sig') as f:  # utf-8-sig handles BOM
            text = f.read()
        
        # Validate content
        self._validate_unicode(text)
        self._collect_statistics(text)
        
        metadata = {
            'filename': path.name,
            'source_path': str(path.absolute()),
            'file_size': path.stat().st_size,
            'char_count': len(text)
        }
        
        logger.info(f"Loaded {len(text)} characters from {path.name}")
        
        return Document(
            text=text,
            source=path.name,
            metadata=metadata
        )
    
    def _validate_unicode(self, text: str) -> None:
        """
        Validate text contains valid Devanagari Unicode.
        
        Args:
            text: Text to validate
            
        Raises:
            ValueError: If text appears corrupted
        """
        # Apply NFC normalization
        normalized = unicodedata.normalize('NFC', text)
        
        # Check for mojibake patterns (common corruption)
        mojibake_patterns = ['à¤', 'à¥', 'Ã¤', 'Ã¥']
        for pattern in mojibake_patterns:
            if pattern in text:
                raise ValueError(
                    f"Detected mojibake (corrupted Unicode). "
                    f"File may not be properly UTF-8 encoded."
                )
        
        # Verify Devanagari characters present
        devanagari_chars = sum(
            1 for char in normalized
            if self.DEVANAGARI_RANGE[0] <= ord(char) <= self.DEVANAGARI_RANGE[1]
        )
        
        if devanagari_chars == 0:
            logger.warning("No Devanagari characters detected in text")
        else:
            logger.debug(f"Found {devanagari_chars} Devanagari characters")
    
    def _collect_statistics(self, text: str) -> None:
        """
        Collect statistics about the text.
        
        Args:
            text: Text to analyze
        """
        self.stats['total_chars'] = len(text)
        self.stats['devanagari_chars'] = sum(
            1 for char in text
            if self.DEVANAGARI_RANGE[0] <= ord(char) <= self.DEVANAGARI_RANGE[1]
        )
        self.stats['danda_count'] = text.count('।')
        self.stats['double_danda_count'] = text.count('॥')
        
        logger.debug(f"Text statistics: {self.stats}")
    
    def get_statistics(self) -> Dict:
        """Return collected statistics."""
        return self.stats.copy()