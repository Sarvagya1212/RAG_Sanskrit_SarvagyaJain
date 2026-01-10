"""Intelligent chunking module for Sanskrit text.

Implements content-aware chunking strategies for:
- Narrative prose (story text)
- Dialogue sections
- Verse conclusions (moral shlokas)
- Pure verse/poetry
"""

import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from code.src.utils.logger import setup_logger

logger = setup_logger(__name__)


@dataclass
class Chunk:
    """Represents a single text chunk."""
    chunk_id: int
    text: str
    slp1_text: str
    content_type: str
    story_id: Optional[int] = None
    story_title: Optional[str] = None
    start_char: int = 0
    end_char: int = 0
    token_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'chunk_id': self.chunk_id,
            'text': self.text,
            'slp1_text': self.slp1_text,
            'content_type': self.content_type,
            'story_id': self.story_id,
            'story_title': self.story_title,
            'start_char': self.start_char,
            'end_char': self.end_char,
            'token_count': self.token_count,
            'metadata': self.metadata
        }


def detect_content_type(text: str) -> str:
    """
    Detect content type for chunking strategy.
    
    Types:
    - narrative_prose: Story narration (80% of content)
    - dialogue_prose: Conversation-heavy sections
    - verse_conclusion: Moral shlokas at story end
    
    Args:
        text: Input text (preferably SLP1)
        
    Returns:
        Content type string
    """
    if not text or len(text) < 10:
        return "narrative_prose"
    
    total_chars = len(text)
    
    # Count dandas and dialogue markers
    single_danda_count = text.count('।')
    double_danda_count = text.count('॥')
    dialogue_markers = text.count('"') + text.count('इति')
    
    # Calculate densities
    danda_density = single_danda_count / total_chars if total_chars > 0 else 0
    double_danda_density = double_danda_count / total_chars if total_chars > 0 else 0
    
    # Decision logic
    if double_danda_density > 0.02:
        return "verse_conclusion"
    elif danda_density > 0.08 and dialogue_markers > 5:
        return "dialogue_prose"
    else:
        return "narrative_prose"


def estimate_token_count(text: str) -> int:
    """
    Estimate token count for Sanskrit text.
    
    Rough estimate: characters / 4 for Sanskrit
    
    Args:
        text: Input text
        
    Returns:
        Estimated token count
    """
    # Remove whitespace for better estimate
    text_stripped = text.strip()
    if not text_stripped:
        return 0
    
    # Sanskrit tokens are typically 4-5 characters
    return len(text_stripped) // 4


class SanskritChunker:
    """
    Intelligent chunker for Sanskrit narrative text.
    
    Implements content-aware strategies for different text types.
    """
    
    def __init__(
        self,
        narrative_target_tokens: int = 175,  # 150-200 range
        verse_group_size: int = 4,
        overlap_sentences: int = 1
    ):
        """
        Initialize chunker.
        
        Args:
            narrative_target_tokens: Target tokens for narrative chunks
            verse_group_size: Number of verses to group together
            overlap_sentences: Sentences to overlap between chunks
        """
        self.narrative_target_tokens = narrative_target_tokens
        self.verse_group_size = verse_group_size
        self.overlap_sentences = overlap_sentences
        
        logger.info(f"SanskritChunker initialized (target: {narrative_target_tokens} tokens)")
    
    def chunk_narrative_prose(
        self,
        text: str,
        slp1_text: str,
        story_id: Optional[int] = None,
        story_title: Optional[str] = None
    ) -> List[Chunk]:
        """
        Chunk narrative prose by sentence boundaries.
        
        Strategy:
        - Split by single danda (।)
        - Target: 150-200 tokens per chunk
        - Overlap: Last sentence from previous chunk
        - Keep complete sentences only
        
        Args:
            text: Original text
            slp1_text: SLP1-normalized text
            story_id: Story identifier
            story_title: Story title
            
        Returns:
            List of Chunk objects
        """
        chunks = []
        
        # Split BOTH original and SLP1 text by danda (sentence boundary)
        # Original text has devanagari dandas (। or ॥)
        sentences_orig = re.split(r'([।॥])', text)
        
        # SLP1 text often uses '.' or '|' for dandas depending on transliteration
        # We need to be careful not to split on abbreviations if possible, but for Sanskrit SLP1 '.' is danda.
        # Use a regex that captures '.' or '|' or '||' (if preserved)
        sentences_slp1 = re.split(r'([.\|\।॥])', slp1_text)
        
        # Reconstruct sentences with danda - original text
        reconstructed_orig = []
        for i in range(0, len(sentences_orig) - 1, 2):
            if i + 1 < len(sentences_orig):
                sent = sentences_orig[i] + sentences_orig[i + 1]
                reconstructed_orig.append(sent.strip())
        if len(sentences_orig) % 2 == 1 and sentences_orig[-1].strip():
            reconstructed_orig.append(sentences_orig[-1].strip())
        
        # Reconstruct sentences with danda - SLP1 text
        reconstructed_slp1 = []
        for i in range(0, len(sentences_slp1) - 1, 2):
            if i + 1 < len(sentences_slp1):
                sent = sentences_slp1[i] + sentences_slp1[i + 1]
                reconstructed_slp1.append(sent.strip())
        if len(sentences_slp1) % 2 == 1 and sentences_slp1[-1].strip():
            reconstructed_slp1.append(sentences_slp1[-1].strip())
        
        if not reconstructed_orig:
            return chunks
        
        # Ensure parallel arrays have same length (use original as reference)
        # If SLP1 has fewer sentences, pad with empty; if more, truncate
        while len(reconstructed_slp1) < len(reconstructed_orig):
            reconstructed_slp1.append('')
        reconstructed_slp1 = reconstructed_slp1[:len(reconstructed_orig)]
        
        # Accumulate sentences into chunks (track both orig and slp1)
        current_orig_sentences = []
        current_slp1_sentences = []
        current_token_count = 0
        chunk_id = 1
        start_char = 0
        
        for i, (sentence_orig, sentence_slp1) in enumerate(zip(reconstructed_orig, reconstructed_slp1)):
            sentence_tokens = estimate_token_count(sentence_orig)
            
            # Check if adding this sentence exceeds target
            if current_token_count + sentence_tokens > self.narrative_target_tokens and current_orig_sentences:
                # Create chunk from accumulated sentences
                chunk_text = ' '.join(current_orig_sentences)
                chunk_slp1 = ' '.join(current_slp1_sentences)
                
                chunk = Chunk(
                    chunk_id=chunk_id,
                    text=chunk_text,
                    slp1_text=chunk_slp1,
                    content_type="narrative_prose",
                    story_id=story_id,
                    story_title=story_title,
                    start_char=start_char,
                    end_char=start_char + len(chunk_text),
                    token_count=current_token_count,
                    metadata={
                        'sentence_count': len(current_orig_sentences),
                        'has_overlap': chunk_id > 1
                    }
                )
                chunks.append(chunk)
                
                # Start new chunk with overlap (last sentence)
                if self.overlap_sentences > 0 and current_orig_sentences:
                    overlap_orig = current_orig_sentences[-self.overlap_sentences:]
                    overlap_slp1 = current_slp1_sentences[-self.overlap_sentences:]
                    current_orig_sentences = overlap_orig
                    current_slp1_sentences = overlap_slp1
                    current_token_count = sum(estimate_token_count(s) for s in overlap_orig)
                else:
                    current_orig_sentences = []
                    current_slp1_sentences = []
                    current_token_count = 0
                
                chunk_id += 1
                start_char += len(chunk_text)
            
            # Add sentence to current chunk
            current_orig_sentences.append(sentence_orig)
            current_slp1_sentences.append(sentence_slp1)
            current_token_count += sentence_tokens
        
        # Add final chunk
        if current_orig_sentences:
            chunk_text = ' '.join(current_orig_sentences)
            chunk_slp1 = ' '.join(current_slp1_sentences)
            
            chunk = Chunk(
                chunk_id=chunk_id,
                text=chunk_text,
                slp1_text=chunk_slp1,
                content_type="narrative_prose",
                story_id=story_id,
                story_title=story_title,
                start_char=start_char,
                end_char=start_char + len(chunk_text),
                token_count=current_token_count,
                metadata={
                    'sentence_count': len(current_orig_sentences),
                    'has_overlap': chunk_id > 1
                }
            )
            chunks.append(chunk)
        
        logger.info(f"Created {len(chunks)} narrative prose chunks")
        return chunks
    
    def chunk_dialogue_prose(
        self,
        text: str,
        slp1_text: str,
        story_id: Optional[int] = None,
        story_title: Optional[str] = None
    ) -> List[Chunk]:
        """
        Chunk dialogue sections keeping Q&A pairs together.
        
        Strategy:
        - Split by "इति" marker (speech end)
        - Keep question-answer pairs together
        
        Args:
            text: Original text
            slp1_text: SLP1-normalized text
            story_id: Story identifier
            story_title: Story title
            
        Returns:
            List of Chunk objects
        """
        chunks = []
        
        # Split BOTH original and SLP1 by इति (end of speech)
        dialogue_units_orig = re.split(r'(इति)', text)
        dialogue_units_slp1 = re.split(r'(iti)', slp1_text, flags=re.IGNORECASE)
        
        # Fallback if SLP1 'iti' not found (maybe text wasn't converted?)
        if len(dialogue_units_slp1) == 1 and len(dialogue_units_orig) > 1:
             # Try Devanagari split on SLP1 text just in case
             dialogue_units_slp1 = re.split(r'(इति)', slp1_text)
        
        # Reconstruct dialogue units - original
        reconstructed_orig = []
        for i in range(0, len(dialogue_units_orig) - 1, 2):
            if i + 1 < len(dialogue_units_orig):
                unit = dialogue_units_orig[i] + dialogue_units_orig[i + 1]
                reconstructed_orig.append(unit.strip())
        if len(dialogue_units_orig) % 2 == 1 and dialogue_units_orig[-1].strip():
            reconstructed_orig.append(dialogue_units_orig[-1].strip())
        
        # Reconstruct dialogue units - SLP1
        reconstructed_slp1 = []
        for i in range(0, len(dialogue_units_slp1) - 1, 2):
            if i + 1 < len(dialogue_units_slp1):
                unit = dialogue_units_slp1[i] + dialogue_units_slp1[i + 1]
                reconstructed_slp1.append(unit.strip())
        if len(dialogue_units_slp1) % 2 == 1 and dialogue_units_slp1[-1].strip():
            reconstructed_slp1.append(dialogue_units_slp1[-1].strip())
        
        # Ensure parallel arrays
        while len(reconstructed_slp1) < len(reconstructed_orig):
            reconstructed_slp1.append('')
        reconstructed_slp1 = reconstructed_slp1[:len(reconstructed_orig)]
        
        # Create chunks from dialogue units
        for idx, (unit_orig, unit_slp1) in enumerate(zip(reconstructed_orig, reconstructed_slp1), 1):
            chunk = Chunk(
                chunk_id=idx,
                text=unit_orig,
                slp1_text=unit_slp1,
                content_type="dialogue_prose",
                story_id=story_id,
                story_title=story_title,
                token_count=estimate_token_count(unit_orig),
                metadata={'dialogue_unit': True}
            )
            chunks.append(chunk)
        
        logger.info(f"Created {len(chunks)} dialogue chunks")
        return chunks
    
    def chunk_verse_conclusion(
        self,
        text: str,
        slp1_text: str,
        story_id: Optional[int] = None,
        story_title: Optional[str] = None
    ) -> List[Chunk]:
        """
        Keep verse conclusions (moral shlokas) as complete units.
        
        Strategy:
        - Don't split verses - keep complete
        - Each verse = one chunk
        
        Args:
            text: Original text
            slp1_text: SLP1-normalized text
            story_id: Story identifier
            story_title: Story title
            
        Returns:
            List of Chunk objects (typically 1 chunk)
        """
        chunk = Chunk(
            chunk_id=1,
            text=text.strip(),
            slp1_text=slp1_text.strip(),
            content_type="verse_conclusion",
            story_id=story_id,
            story_title=story_title,
            token_count=estimate_token_count(text),
            metadata={'is_moral_verse': True}
        )
        
        logger.info("Created 1 verse conclusion chunk")
        return [chunk]
    
    def chunk_story(
        self,
        text: str,
        slp1_text: str,
        story_id: Optional[int] = None,
        story_title: Optional[str] = None
    ) -> List[Chunk]:
        """
        Chunk a complete story with content-aware strategy.
        
        Args:
            text: Original story text
            slp1_text: SLP1-normalized text
            story_id: Story identifier
            story_title: Story title
            
        Returns:
            List of Chunk objects
        """
        # Detect content type
        content_type = detect_content_type(text)
        logger.info(f"Detected content type: {content_type}")
        
        # Apply appropriate chunking strategy
        if content_type == "narrative_prose":
            return self.chunk_narrative_prose(text, slp1_text, story_id, story_title)
        elif content_type == "dialogue_prose":
            return self.chunk_dialogue_prose(text, slp1_text, story_id, story_title)
        elif content_type == "verse_conclusion":
            return self.chunk_verse_conclusion(text, slp1_text, story_id, story_title)
        else:
            # Default to narrative
            return self.chunk_narrative_prose(text, slp1_text, story_id, story_title)
