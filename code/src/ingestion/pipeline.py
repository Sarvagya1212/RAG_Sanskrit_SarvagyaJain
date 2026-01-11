"""Data ingestion pipeline orchestration."""

import json
from pathlib import Path
from typing import List, Dict
from code.src.ingestion.document_loader import DocumentLoader, Document
from code.src.ingestion.story_segmenter import StorySegmenter, Story
from code.src.utils.logger import setup_logger

logger = setup_logger(__name__)

class IngestionPipeline:
    """Orchestrate document loading and story segmentation."""
    
    def __init__(self, output_dir: str = "data/processed"):
        """
        Initialize ingestion pipeline.
        
        Args:
            output_dir: Directory to save processed stories
        """
        self.loader = DocumentLoader()
        self.segmenter = StorySegmenter()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def process_file(self, file_path: str) -> List[Story]:
        """
        Process a single file (text or PDF).
        
        Args:
            file_path: Path to the file
            
        Returns:
            List of Story objects
        """
        logger.info(f"Processing file: {file_path}")
        
        # Auto-detect file type and use appropriate loader
        if file_path.lower().endswith('.pdf'):
            document = self.loader.load_pdf(file_path)
        else:
            document = self.loader.load_text_file(file_path)
        
        # Segment into stories
        stories = self.segmenter.segment_stories(document.text)
        
        # Add source metadata to each story
        for story in stories:
            story.metadata['source_file'] = document.source
            story.metadata['source_path'] = document.metadata['source_path']
        
        return stories
    
    def process_directory(self, directory: str) -> List[Story]:
        """
        Process all text and PDF files in a directory.
        
        Args:
            directory: Path to directory containing .txt or .pdf files
            
        Returns:
            List of all stories from all files
        """
        dir_path = Path(directory)
        all_stories = []
        
        # Find all .txt and .pdf files
        txt_files = list(dir_path.glob("*.txt"))
        pdf_files = list(dir_path.glob("*.pdf"))
        all_files = txt_files + pdf_files
        
        if not all_files:
            logger.warning(f"No .txt or .pdf files found in {directory}")
            return []
        
        logger.info(f"Found {len(all_files)} files to process ({len(txt_files)} txt, {len(pdf_files)} pdf)")
        
        # Process each file
        for file in all_files:
            try:
                stories = self.process_file(str(file))
                all_stories.extend(stories)
                logger.info(f"Processed {file.name}: {len(stories)} stories")
            except Exception as e:
                logger.error(f"Error processing {file.name}: {e}")
        
        return all_stories
    
    def save_stories(self, stories: List[Story], output_name: str = "stories.json"):
        """
        Save processed stories to JSON file.
        
        Args:
            stories: List of Story objects
            output_name: Output filename
        """
        output_path = self.output_dir / output_name
        
        # Convert stories to serializable format
        stories_data = []
        for story in stories:
            story_dict = {
                'id': story.id,
                'title': story.title,
                'text': story.text,
                'start_char': story.start_char,
                'end_char': story.end_char,
                'metadata': story.metadata
            }
            stories_data.append(story_dict)
        
        # Save to JSON
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(stories_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Saved {len(stories)} stories to {output_path}")
    
    def generate_report(self, stories: List[Story]) -> Dict:
        """
        Generate statistics report for processed stories.
        
        Args:
            stories: List of Story objects
            
        Returns:
            Statistics dictionary
        """
        report = {
            'total_stories': len(stories),
            'total_characters': sum(len(s.text) for s in stories),
            'loader_stats': self.loader.get_statistics(),
            'stories': []
        }
        
        for story in stories:
            story_stats = {
                'id': story.id,
                'title': story.title,
                'char_count': len(story.text),
                'content_type': story.metadata.get('content_type'),
                'danda_count': story.metadata.get('danda_count'),
                'verse_count': story.metadata.get('verse_count'),
                'dialogue_markers': story.metadata.get('dialogue_markers')
            }
            report['stories'].append(story_stats)
        
        return report