"""Hierarchical chunking for Sanskrit texts using parent-child strategy.

Parent chunks: Large sections (600-800 tokens) providing full context
Child chunks: Smaller segments (150-200 tokens) for precise retrieval
"""

from dataclasses import dataclass
from typing import List, Tuple, Dict
import re
from code.src.preprocessing.preprocessor import SanskritPreprocessor
from code.src.utils.logger import setup_logger

logger = setup_logger(__name__)

@dataclass
class ParentChunk:
    """Represents a parent chunk (large context section)."""
    parent_id: str
    story_id: str
    story_title: str
    text: str
    preprocessed_text: str
    start_char: int
    end_char: int
    token_count: int

@dataclass
class ChildChunk:
    """Represents a child chunk (indexed for search)."""
    chunk_id: str
    parent_id: str
    story_id: str
    story_title: str
    text: str
    preprocessed_text: str
    parent_text: str
    parent_preprocessed: str
    child_index: int
    total_children: int
    start_char: int
    end_char: int
    token_count: int

class HierarchicalChunker:
    """
    Hierarchical chunking strategy for Sanskrit texts.
    
    Creates two-level hierarchy:
    - Parent chunks: ~600-800 tokens (full context)
    - Child chunks: ~150-200 tokens (precise retrieval)
    """
    
    # Chunking parameters
    PARENT_TARGET_TOKENS = 700  # Target for parent chunks
    PARENT_MIN_TOKENS = 600
    PARENT_MAX_TOKENS = 800
    
    CHILD_TARGET_TOKENS = 175   # Target for child chunks
    CHILD_MIN_TOKENS = 150
    CHILD_MAX_TOKENS = 200
    
    # Sanskrit sentence boundaries
    DANDA_PATTERN = r'[।॥]'
    
    def __init__(self):
        """Initialize hierarchical chunker."""
        self.preprocessor = SanskritPreprocessor()
        logger.info("HierarchicalChunker initialized")
    
    def chunk_story(self, story_id: str, story_title: str, story_text: str) -> Tuple[List[ParentChunk], List[ChildChunk]]:
        """
        Chunk a story into parent-child hierarchy.
        
        Args:
            story_id: Unique story identifier
            story_title: Story title
            story_text: Full story text
            
        Returns:
            Tuple of (parent_chunks, child_chunks)
        """
        logger.info(f"Chunking story: {story_title}")
        
        # Step 1: Create parent chunks (large sections)
        parents = self.create_parent_chunks(story_id, story_title, story_text)
        logger.info(f"Created {len(parents)} parent chunks")
        
        # Step 2: Split each parent into child chunks
        all_children = []
        for parent in parents:
            children = self.create_child_chunks(parent)
            all_children.extend(children)
        
        logger.info(f"Created {len(all_children)} child chunks from {len(parents)} parents")
        
        return parents, all_children
    
    def create_parent_chunks(self, story_id: str, story_title: str, text: str) -> List[ParentChunk]:
        """
        Create parent chunks (600-800 tokens).
        
        Strategy:
        - Split on major boundaries (double danda, large gaps)
        - Group sentences to reach target size
        - Preserve semantic coherence
        
        Args:
            story_id: Story identifier
            story_title: Story title
            text: Full story text
            
        Returns:
            List of ParentChunk objects
        """
        # Split into sentences
        sentences = self._split_into_sentences(text)
        
        if not sentences:
            return []
        
        parents = []
        current_sentences = []
        current_tokens = 0
        parent_index = 1
        char_offset = 0
        
        for sentence in sentences:
            sentence_tokens = self._estimate_tokens(sentence)
            
            # Check if adding this sentence would exceed max
            if current_tokens + sentence_tokens > self.PARENT_MAX_TOKENS and current_sentences:
                # Create parent chunk
                parent = self._create_parent_from_sentences(
                    story_id, story_title, current_sentences, parent_index, char_offset
                )
                parents.append(parent)
                
                # Update offset
                char_offset = parent.end_char
                
                # Start new parent
                current_sentences = [sentence]
                current_tokens = sentence_tokens
                parent_index += 1
            else:
                # Add to current parent
                current_sentences.append(sentence)
                current_tokens += sentence_tokens
        
        # Create final parent chunk
        if current_sentences:
            parent = self._create_parent_from_sentences(
                story_id, story_title, current_sentences, parent_index, char_offset
            )
            parents.append(parent)
        
        return parents
    
    def create_child_chunks(self, parent: ParentChunk) -> List[ChildChunk]:
        """
        Split a parent chunk into child chunks (150-200 tokens).
        
        Args:
            parent: Parent chunk to split
            
        Returns:
            List of ChildChunk objects
        """
        # Split parent text into sentences
        sentences = self._split_into_sentences(parent.text)
        
        if not sentences:
            return []
        
        children = []
        current_sentences = []
        current_tokens = 0
        child_index = 1
        char_offset = 0
        
        for sentence in sentences:
            sentence_tokens = self._estimate_tokens(sentence)
            
            # Check if adding would exceed max
            if current_tokens + sentence_tokens > self.CHILD_MAX_TOKENS and current_sentences:
                # Create child chunk
                child = self._create_child_from_sentences(
                    parent, current_sentences, child_index, char_offset
                )
                children.append(child)
                
                # Update offset
                char_offset = child.end_char - parent.start_char
                
                # Start new child
                current_sentences = [sentence]
                current_tokens = sentence_tokens
                child_index += 1
            else:
                current_sentences.append(sentence)
                current_tokens += sentence_tokens
        
        # Create final child chunk
        if current_sentences:
            child = self._create_child_from_sentences(
                parent, current_sentences, child_index, char_offset
            )
            children.append(child)
        
        # Update total_children count for all children
        total = len(children)
        for child in children:
            child.total_children = total
        
        return children
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences using Sanskrit punctuation.
        
        Args:
            text: Text to split
            
        Returns:
            List of sentences
        """
        # Split on danda markers
        parts = re.split(self.DANDA_PATTERN, text)
        
        sentences = []
        for part in parts:
            part = part.strip()
            if part:
                # Restore the danda at the end
                if not part.endswith('।') and not part.endswith('॥'):
                    part = part + ' ।'
                sentences.append(part)
        
        return sentences
    
    def _estimate_tokens(self, text: str) -> int:
        """
        Estimate token count for text.
        
        Simple heuristic: ~4 characters per token for Sanskrit.
        
        Args:
            text: Text to estimate
            
        Returns:
            Estimated token count
        """
        return len(text) // 4
    
    def _create_parent_from_sentences(
        self, 
        story_id: str, 
        story_title: str, 
        sentences: List[str], 
        parent_index: int,
        char_offset: int
    ) -> ParentChunk:
        """Create a ParentChunk from sentences."""
        text = ' '.join(sentences)
        
        # Preprocess text to SLP1
        result = self.preprocessor.process(text)
        preprocessed_text = result.slp1  # Fixed: use 'slp1' not 'slp1_text'
        
        parent_id = f"{story_id}_p{parent_index}"
        
        return ParentChunk(
            parent_id=parent_id,
            story_id=story_id,
            story_title=story_title,
            text=text,
            preprocessed_text=preprocessed_text,
            start_char=char_offset,
            end_char=char_offset + len(text),
            token_count=self._estimate_tokens(text)
        )
    
    def _create_child_from_sentences(
        self,
        parent: ParentChunk,
        sentences: List[str],
        child_index: int,
        char_offset: int
    ) -> ChildChunk:
        """Create a ChildChunk from sentences within a parent."""
        text = ' '.join(sentences)
        
        # Preprocess text to SLP1
        result = self.preprocessor.process(text)
        preprocessed_text = result.slp1  # Fixed: use 'slp1' not 'slp1_text'
        
        chunk_id = f"{parent.parent_id}_c{child_index}"
        
        return ChildChunk(
            chunk_id=chunk_id,
            parent_id=parent.parent_id,
            story_id=parent.story_id,
            story_title=parent.story_title,
            text=text,
            preprocessed_text=preprocessed_text,
            parent_text=parent.text,
            parent_preprocessed=parent.preprocessed_text,
            child_index=child_index,
            total_children=0,  # Will be updated later
            start_char=parent.start_char + char_offset,
            end_char=parent.start_char + char_offset + len(text),
            token_count=self._estimate_tokens(text)
        )
