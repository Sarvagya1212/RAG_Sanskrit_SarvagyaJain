"""Story boundary detection and segmentation for narrative texts."""

import re
from typing import List, Dict, Tuple
from dataclasses import dataclass
from code.src.utils.logger import setup_logger

logger = setup_logger(__name__)

@dataclass
class Story:
    """Represents a single story with metadata."""
    id: int
    title: str
    text: str
    start_char: int
    end_char: int
    metadata: Dict
    
class StorySegmenter:
    """Detect and segment individual stories from collection."""
    
    def __init__(self):
        """Initialize story segmenter."""
        # Pattern for standalone title lines (2-80 chars on their own line)
        self.title_pattern = re.compile(
            r'^\s*([^\n]{2,80})\s*$',
            re.MULTILINE
        )
        
        # English metadata pattern
        self.english_pattern = re.compile(r'[a-zA-Z@.:]+')

    def clean_story_title(self, raw_title: str) -> str:
        """
        Remove English annotations and metadata from story titles.
        
        Example:
            Input: "शीतं बहु बाधति । The cold hurts very much by: Kedar Naphade"
            Output: "शीतं बहु बाधति"
        """
        # Remove everything after "by:" (author annotations)
        title = re.sub(r'\s*by:.*$', '', raw_title, flags=re.IGNORECASE)
        
        # Remove English translations in parentheses
        title = re.sub(r'\(.*?\)', '', title)
        
        # Remove "The cold hurts" style English text
        # Keep only Devanagari and punctuation
        parts = title.split('।')
        if len(parts) > 1:
            # Take only the first part (before danda)
            title = parts[0] + '।'
        
        # Remove extra whitespace
        title = title.strip()
        
        return title
        
    def segment_stories(self, text: str) -> List[Story]:
        """
        Segment text into individual stories.
        
        Args:
            text: Complete text containing multiple stories
            
        Returns:
            List of Story objects
        """
        logger.info("Starting story segmentation")
        
        # Find all story titles
        titles = self._detect_story_titles(text)
        
        if not titles:
            logger.warning("No story titles detected, treating as single document")
            return [self._create_single_story(text)]
        
        # Create story boundaries
        stories = self._create_story_objects(text, titles)
        
        logger.info(f"Segmented into {len(stories)} stories")
        return stories
    
    def _detect_story_titles(self, text: str) -> List[Tuple[str, int]]:
        """
        Detect story titles in text using line-based detection.
        
        Args:
            text: Text to search
            
        Returns:
            List of (title, position) tuples
        """
        titles = []
        lines = text.splitlines(keepends=True)
        current_pos = 0
        
        for i, line in enumerate(lines):
            stripped = line.strip()
            
            # Check if this line could be a title
            if stripped and self._is_sanskrit_title(stripped):
                # Additional check: title lines are typically followed by blank line or content
                # and preceded by blank line (except first line)
                is_preceded_by_blank = (i == 0) or (i > 0 and not lines[i-1].strip())
                
                if is_preceded_by_blank:
                    titles.append((stripped, current_pos))
                    logger.debug(f"Found story title: {stripped[:40]}... at position {current_pos}")
            
            current_pos += len(line)
        
        return titles
    
    def _is_sanskrit_title(self, line: str) -> bool:
        """
        Check if a line is likely a Sanskrit story title.
        
        Criteria:
        - Line length between 2-50 characters (titles are typically short)
        - Contains at least 5 Devanagari characters
        - Does NOT start with common prose sentence starters
        - Not a long prose paragraph
        
        Args:
            line: Line text to check
            
        Returns:
            True if appears to be a Sanskrit title
        """
        # Reject long prose lines - but allow longer titles with metadata
        if len(line) > 120:
            return False
        
        # Reject very short lines
        if len(line) < 2:
            return False
        
        # Common prose sentence starters - these indicate story content, not titles
        prose_starters = [
            'एकः', 'एका', 'एकस्मिन्', 'अथ', 'कदाचित्', 'तदा', 'ततः',
            'सः', 'सा', 'तेन', 'इदानीम्', 'अपि', 'यदा', 'किंचित्'
        ]
        for starter in prose_starters:
            if line.startswith(starter):
                return False
        
        # Count Devanagari characters
        devanagari_chars = sum(
            1 for char in line
            if 0x0900 <= ord(char) <= 0x097F
        )
        
        # Must have at least 5 Devanagari characters to be a title
        if devanagari_chars >= 5:
            return True
        
        return False
    
    def _create_story_objects(
        self,
        text: str,
        titles: List[Tuple[str, int]]
    ) -> List[Story]:
        """
        Create Story objects from title boundaries.
        
        Args:
            text: Complete text
            titles: List of (title, position) tuples
            
        Returns:
            List of Story objects
        """
        stories = []
        
        for i, (title, start_pos) in enumerate(titles):
            # Determine end position (start of next story or end of text)
            if i < len(titles) - 1:
                end_pos = titles[i + 1][1]
            else:
                end_pos = len(text)
            
            # Extract story text
            story_text = text[start_pos:end_pos]
            
            # Clean the title
            cleaned_title = self.clean_story_title(title)
            
            # Clean the text (using original title for matching removal if needed)
            cleaned_text = self._clean_story_text(story_text, title)
            
            # Create metadata
            metadata = self._extract_metadata(story_text)
            
            story = Story(
                id=i + 1,
                title=cleaned_title,
                text=cleaned_text,
                start_char=start_pos,
                end_char=end_pos,
                metadata=metadata
            )
            
            stories.append(story)
            logger.info(
                f"Story {i+1}: '{title}' "
                f"({len(cleaned_text)} chars, "
                f"verses: {metadata.get('verse_count', 0)})"
            )
        
        return stories
    
    def _clean_story_text(self, text: str, title: str) -> str:
        """
        Clean story text by removing formatting and metadata.
        
        Args:
            text: Raw story text
            title: Story title
            
        Returns:
            Cleaned text
        """
        # Remove markdown bold markers
        cleaned = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
        
        # Remove the title line itself (it's in metadata)
        cleaned = cleaned.replace(title, '', 1)
        
        # Remove excessive whitespace
        cleaned = re.sub(r'\n\s*\n\s*\n+', '\n\n', cleaned)
        cleaned = cleaned.strip()
        
        return cleaned
    
    def _extract_metadata(self, text: str) -> Dict:
        """
        Extract metadata from story text.
        
        Args:
            text: Story text
            
        Returns:
            Metadata dictionary
        """
        metadata = {}
        
        # Count structural elements
        metadata['danda_count'] = text.count('।')
        metadata['double_danda_count'] = text.count('॥')
        metadata['dialogue_markers'] = text.count('इति') + text.count('"')
        
        # Estimate verse count (double dandas indicate verse endings)
        metadata['verse_count'] = text.count('॥')
        
        # Calculate densities
        if len(text) > 0:
            metadata['danda_density'] = metadata['danda_count'] / len(text)
            metadata['double_danda_density'] = metadata['double_danda_count'] / len(text)
        else:
            metadata['danda_density'] = 0.0
            metadata['double_danda_density'] = 0.0
        
        # Detect content type
        metadata['content_type'] = self._detect_content_type(metadata)
        
        # Extract English metadata (author, email) if present
        english_matches = self.english_pattern.findall(text)
        if any('@' in match for match in english_matches):
            # Found email, likely author metadata
            metadata['has_english_metadata'] = True
        
        return metadata
    
    def _detect_content_type(self, metadata: Dict) -> str:
        """
        Detect primary content type of story.
        
        Args:
            metadata: Story metadata
            
        Returns:
            Content type: 'verse_conclusion', 'dialogue_prose', or 'narrative_prose'
        """
        danda_density = metadata.get('danda_density', 0)
        double_danda_density = metadata.get('double_danda_density', 0)
        dialogue_markers = metadata.get('dialogue_markers', 0)
        
        if double_danda_density > 0.02:
            return 'verse_conclusion'
        elif danda_density > 0.08 and dialogue_markers > 5:
            return 'dialogue_prose'
        else:
            return 'narrative_prose'
    
    def _create_single_story(self, text: str) -> Story:
        """
        Create single story when no titles detected.
        
        Args:
            text: Complete text
            
        Returns:
            Single Story object
        """
        metadata = self._extract_metadata(text)
        
        story = Story(
            id=1,
            title="Untitled Story",
            text=text.strip(),
            start_char=0,
            end_char=len(text),
            metadata=metadata
        )
        
        return story