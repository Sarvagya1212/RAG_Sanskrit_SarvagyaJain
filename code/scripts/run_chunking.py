"""
Run complete ingestion + chunking pipeline on stories.txt

Demonstrates:
1. Load and segment stories
2. Chunk each story with content-aware strategies
3. Save chunks to JSON
"""

import sys
import json
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from code.src.ingestion.pipeline import IngestionPipeline
from code.src.chunking import SanskritChunker
from code.src.preprocessing import SanskritPreprocessor
from code.src.utils.logger import setup_logger

logger = setup_logger(__name__)


def main():
    """Run complete pipeline."""
    print("\n" + "=" * 70)
    print("SANSKRIT RAG PIPELINE - INGESTION + CHUNKING")
    print("=" * 70)
    
    # Step 1: Ingest stories
    print("\n[Step 1] Loading and segmenting stories...")
    pipeline = IngestionPipeline()
    stories = pipeline.process_file("data/raw/stories.txt")
    print(f"✓ Loaded {len(stories)} stories")
    
    # Step 2: Chunk stories
    print("\n[Step 2] Chunking stories with content-aware strategies...")
    chunker = SanskritChunker(
        narrative_target_tokens=175,  # 150-200 range
        overlap_sentences=1
    )
    
    # Initialize preprocessor for SLP1 conversion
    preprocessor = SanskritPreprocessor()
    
    all_chunks = []
    chunk_stats = {
        'narrative_prose': 0,
        'dialogue_prose': 0,
        'verse_conclusion': 0
    }
    
    for story in stories:
        print(f"\n  Story {story.id}: '{story.title}'")
        print(f"    Characters: {len(story.text)}")
        
        # Preprocess story text to SLP1 (CRITICAL for index-query consistency)
        # Check for Devanagari characters to force correct script detection for mixed content
        has_devanagari = any(0x0900 <= ord(c) <= 0x097F for c in story.text)
        script_override = "devanagari" if has_devanagari else None
        
        preprocessed = preprocessor.process(story.text, script_override=script_override)
        slp1_text = preprocessed.slp1
        print(f"    Script detected: {preprocessed.script} (Override: {script_override})")
        
        # Chunk the story with properly preprocessed SLP1 text
        chunks = chunker.chunk_story(
            text=story.text,
            slp1_text=slp1_text,  # Now using actual SLP1!
            story_id=story.id,
            story_title=story.title
        )
        
        print(f"    Chunks created: {len(chunks)}")
        for chunk in chunks:
            print(f"      - Chunk {chunk.chunk_id}: {chunk.token_count} tokens ({chunk.content_type})")
            chunk_stats[chunk.content_type] += 1
        
        all_chunks.extend(chunks)
    
    # Step 3: Save results
    print("\n[Step 3] Saving chunks...")
    output_path = "data/processed/chunks.json"
    
    chunks_data = {
        'total_chunks': len(all_chunks),
        'chunk_stats': chunk_stats,
        'chunks': [chunk.to_dict() for chunk in all_chunks]
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(chunks_data, f, ensure_ascii=False, indent=2)
    
    print(f"✓ Saved {len(all_chunks)} chunks to {output_path}")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Stories processed: {len(stories)}")
    print(f"Total chunks: {len(all_chunks)}")
    print(f"\nChunk distribution:")
    for content_type, count in chunk_stats.items():
        print(f"  {content_type}: {count}")
    print()


if __name__ == "__main__":
    main()
