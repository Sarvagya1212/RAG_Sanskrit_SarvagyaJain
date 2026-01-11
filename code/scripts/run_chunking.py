"""
Run ingestion + hierarchical chunking pipeline with parent-child strategy

Demonstrates:
1. Load and segment stories
2. Create parent chunks (600-800 tokens) for context
3. Create child chunks (150-200 tokens) for search
4. Save both hierarchies to JSON
"""

import sys
import json
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from code.src.ingestion.pipeline import IngestionPipeline
from code.src.chunking.hierarchical_chunker import HierarchicalChunker
from code.src.utils.logger import setup_logger

logger = setup_logger(__name__)


def main():
    """Run complete hierarchical chunking pipeline."""
    print("\n" + "=" * 70)
    print("SANSKRIT RAG PIPELINE - HIERARCHICAL CHUNKING")
    print("=" * 70)
    
    # Step 1: Ingest stories
    print("\n[Step 1] Loading and segmenting stories...")
    pipeline = IngestionPipeline()
    
    # Auto-detect available file (try .txt first, then .pdf)
    data_dir = Path("data/raw")
    txt_files = list(data_dir.glob("*.txt"))
    pdf_files = list(data_dir.glob("*.pdf"))
    
    if txt_files:
        input_file = str(txt_files[0])
        print(f"Found text file: {txt_files[0].name}")
    elif pdf_files:
        input_file = str(pdf_files[0])
        print(f"Found PDF file: {pdf_files[0].name}")
    else:
        raise FileNotFoundError("No .txt or .pdf files found in data/raw/")
    
    stories = pipeline.process_file(input_file)
    print(f"✓ Loaded {len(stories)} stories")
    
    # Step 2: Create hierarchical chunks
    print("\n[Step 2] Creating parent-child chunk hierarchy...")
    chunker = HierarchicalChunker()
    
    all_parents = []
    all_children = []
    
    for i, story in enumerate(stories, 1):
        story_id = f"s{i}"
        print(f"\n  Story {i}/{len(stories)}: {story.title[:50]}...")
        
        # Create parent-child hierarchy
        parents, children = chunker.chunk_story(
            story_id=story_id,
            story_title=story.title,
            story_text=story.text
        )
        
        all_parents.extend(parents)
        all_children.extend(children)
        
        print(f"    → {len(parents)} parents, {len(children)} children")
    
    print(f"\n✓ Created {len(all_parents)} parent chunks, {len(all_children)} child chunks")
    
    # Step 3: Save to JSON
    print("\n[Step 3] Saving chunks...")
    output_dir = Path("data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save parent chunks
    parent_data = []
    for parent in all_parents:
        parent_data.append({
            'parent_id': parent.parent_id,
            'story_id': parent.story_id,
            'story_title': parent.story_title,
            'text': parent.text,
            'preprocessed_text': parent.preprocessed_text,
            'token_count': parent.token_count,
            'start_char': parent.start_char,
            'end_char': parent.end_char
        })
    
    with open(output_dir / "parent_chunks.json", 'w', encoding='utf-8') as f:
        json.dump(parent_data, f, ensure_ascii=False, indent=2)
    print(f"✓ Saved {len(parent_data)} parent chunks to parent_chunks.json")
    
    # Save child chunks
    child_data = []
    for child in all_children:
        child_data.append({
            'chunk_id': child.chunk_id,
            'parent_id': child.parent_id,
            'story_id': child.story_id,
            'story_title': child.story_title,
            'text': child.text,
            'preprocessed_text': child.preprocessed_text,
            'parent_text': child.parent_text,
            'parent_preprocessed': child.parent_preprocessed,
            'child_index': child.child_index,
            'total_children': child.total_children,
            'token_count': child.token_count,
            'start_char': child.start_char,
            'end_char': child.end_char
        })
    
    with open(output_dir / "child_chunks.json", 'w', encoding='utf-8') as f:
        json.dump(child_data, f, ensure_ascii=False, indent=2)
    print(f"✓ Saved {len(child_data)} child chunks to child_chunks.json")
    
    # Step 4: Generate statistics
    print("\n" + "=" * 70)
    print("CHUNKING STATISTICS")
    print("=" * 70)
    
    print(f"\nStories processed: {len(stories)}")
    print(f"Parent chunks: {len(all_parents)}")
    print(f"Child chunks: {len(all_children)}")
    print(f"Avg children per parent: {len(all_children) / len(all_parents):.1f}")
    
    # Parent token distribution
    parent_tokens = [p.token_count for p in all_parents]
    print(f"\nParent chunk tokens:")
    print(f"  Min: {min(parent_tokens)}")
    print(f"  Max: {max(parent_tokens)}")
    print(f"  Avg: {sum(parent_tokens) / len(parent_tokens):.0f}")
    
    # Child token distribution
    child_tokens = [c.token_count for c in all_children]
    print(f"\nChild chunk tokens:")
    print(f"  Min: {min(child_tokens)}")
    print(f"  Max: {max(child_tokens)}")
    print(f"  Avg: {sum(child_tokens) / len(child_tokens):.0f}")
    
    print("\n" + "=" * 70)
    print("✓ HIERARCHICAL CHUNKING COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
