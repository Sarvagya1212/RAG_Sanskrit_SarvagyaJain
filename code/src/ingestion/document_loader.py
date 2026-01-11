"""Document loading with support for text and PDF files."""

from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Any
from code.src.utils.logger import setup_logger

logger = setup_logger(__name__)

@dataclass
class Document:
    """Container for loaded document."""
    text: str
    source: str
    metadata: Dict[str, Any]

class DocumentLoader:
    """Loads documents from text and PDF files."""
    
    def __init__(self):
        """Initialize document loader."""
        self.stats = {
            'total_chars': 0,
            'total_files': 0
        }
    
    def load_text_file(self, file_path: str) -> Document:
        """
        Load plain text file.
        
        Args:
            file_path: Path to text file
            
        Returns:
            Document object
        """
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Read file
        with open(path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # Validate and collect stats
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
    
    def load_pdf(self, file_path: str) -> Document:
        """
        Load PDF file using pdfplumber for superior text extraction.
        
        pdfplumber advantages:
        - Better layout preservation
        - Superior multilingual text handling (Devanagari)
        - Cleaner extraction without formatting artifacts
        
        Args:
            file_path: Path to PDF file
            
        Returns:
            Document object
        """
        try:
            import pdfplumber
        except ImportError:
            raise ImportError(
                "pdfplumber is required for PDF loading. "
                "Install with: pip install pdfplumber"
            )
        
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        logger.info(f"Loading PDF file: {path.name}")
        
        # Extract text from PDF
        text_parts = []
        
        with pdfplumber.open(str(path)) as pdf:
            page_count = len(pdf.pages)
            
            for page_num, page in enumerate(pdf.pages, 1):
                # Extract text with layout preservation
                page_text = page.extract_text(layout=True)
                
                if page_text and page_text.strip():
                    # Clean up excessive whitespace while preserving structure
                    lines = [line.rstrip() for line in page_text.split('\n')]
                    
                    # Remove empty lines but keep paragraph breaks
                    cleaned_lines = []
                    prev_empty = False
                    for line in lines:
                        if line.strip():
                            cleaned_lines.append(line)
                            prev_empty = False
                        elif not prev_empty:
                            cleaned_lines.append('')  # Keep one empty line
                            prev_empty = True
                    
                    page_text_cleaned = '\n'.join(cleaned_lines)
                    text_parts.append(page_text_cleaned)
        
        # Combine all pages
        text = "\n\n".join(text_parts)
        
        # Validate content
        self._validate_unicode(text)
        self._collect_statistics(text)
        
        metadata = {
            'filename': path.name,
            'source_path': str(path.absolute()),
            'file_size': path.stat().st_size,
            'char_count': len(text),
            'page_count': page_count
        }
        
        logger.info(f"Loaded {len(text)} characters from {path.name} ({page_count} pages)")
        
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
        """
        try:
            # Try encoding as UTF-8
            text.encode('utf-8')
        except UnicodeEncodeError as e:
            logger.warning(f"Unicode encoding issues detected: {e}")
    
    def _collect_statistics(self, text: str) -> None:
        """
        Collect statistics about loaded text.
        
        Args:
            text: Text to analyze
        """
        self.stats['total_chars'] += len(text)
        self.stats['total_files'] += 1