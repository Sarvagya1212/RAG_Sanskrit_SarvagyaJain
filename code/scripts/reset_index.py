"""
Reset Index Script
File: code/scripts/reset_index.py

Clears all data from MetadataStore and deletes index files.
"""

import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from code.src.indexing.metadata_store import MetadataStore
from code.src.utils.logger import setup_logger

logger = setup_logger(__name__)

def reset_index():
    print("\n" + "="*60)
    print("RESETTING RAG INDEX")
    print("="*60)
    
    # 1. Clear Database
    print("\n1. Clearing Metadata Database...")
    try:
        store = MetadataStore()
        store.clear_all()
        store.close()
        print("   ✓ Database chunks cleared")
    except Exception as e:
        print(f"   ✗ Failed to clear database: {e}")
    
    # 2. Delete Index Files
    print("\n2. Deleting Index Files...")
    files_to_delete = [
        "data/processed/faiss_index.bin",
        "data/processed/bm25_index.pkl",
        # "data/processed/chunks.json" # Optional: keep intermediate chunks? Usually yes to skip re-chunking.
        # But if user wants to delete chunks from DB, maybe they want to re-chunk too?
        # Let's keep chunks.json as it's output of ingestion, not indexing.
        # But wait, user said "chunks from db".
    ]
    
    for file_path in files_to_delete:
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
                print(f"   ✓ Deleted {file_path}")
            except Exception as e:
                print(f"   ✗ Failed to delete {file_path}: {e}")
        else:
            print(f"   - File not found: {file_path}")

    print("\n" + "="*60)
    print("RESET COMPLETE")
    print("Run `python code/main.py --mode index --data ./data/raw` to re-index.")
    print("="*60)

if __name__ == "__main__":
    confirm = input("Are you sure you want to delete all embeddings and chunks? (y/n): ")
    if confirm.lower() == 'y':
        reset_index()
    else:
        print("Operation cancelled.")
