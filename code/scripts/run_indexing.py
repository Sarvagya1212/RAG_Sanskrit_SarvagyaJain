"""Script to build all indexes."""

import sys
import argparse
import time
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from code.src.indexing.indexing_pipeline import IndexingPipeline
from code.src.utils.config_loader import load_config
from code.src.utils.logger import setup_logger

logger = setup_logger(__name__, log_file="logs/indexing.log")

def main():
    """Run indexing pipeline."""
    parser = argparse.ArgumentParser(description="Build BM25 and vector indexes")
    parser.add_argument(
        '--chunks',
        type=str,
        default='data/processed/chunks.json',
        help='Path to chunks JSON file'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='code/config/config.yaml',
        help='Path to configuration file'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Initialize pipeline
    pipeline = IndexingPipeline(config)
    
    # Build indexes
    start_time = time.time()
    
    try:
        pipeline.build_indexes(chunks_path=args.chunks)
        
        elapsed_time = time.time() - start_time
        logger.info(f"\nTotal indexing time: {elapsed_time:.2f} seconds")
        
    except Exception as e:
        logger.error(f"Indexing failed: {e}", exc_info=True)
        raise
    
    finally:
        # Cleanup
        pipeline.metadata_store.close()

if __name__ == "__main__":
    main()