"""Indexing pipeline for parent-child chunks."""

from code.src.indexing.bm25_indexer import BM25Indexer
from code.src.indexing.embedding_generator import EmbeddingGenerator
from code.src.indexing.vector_indexer import VectorIndexer
from code.src.indexing.metadata_store import MetadataStore
from code.src.utils.logger import setup_logger
import json
import numpy as np
from typing import List, Dict
from pathlib import Path

logger = setup_logger(__name__)

class IndexingPipeline:
    """Indexing pipeline for parent-child hierarchical chunks."""
    
    def __init__(self, config: dict):
        """
        Initialize indexing pipeline.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        
        # Initialize components
        self.bm25_indexer = BM25Indexer(
            ngram_size=config['indexing']['bm25']['ngram_size']
        )
        
        self.embedding_generator = EmbeddingGenerator(
            model_name=config['models']['embedding']['name']
        )
        
        self.vector_indexer = VectorIndexer(
            embedding_dim=384  # multilingual-e5-small dimension
        )
        
        self.metadata_store = MetadataStore()
    
    def build_indexes(
        self, 
        parent_chunks_path: str = "data/processed/parent_chunks.json",
        child_chunks_path: str = "data/processed/child_chunks.json"
    ):
        """
        Build indexes from parent-child chunks.
        
        Strategy:
        - Index CHILD chunks in BM25 and FAISS (for precise search)
        - Store PARENT chunks in metadata (for context)
        - Link children to parents via parent_id
        
        Args:
            parent_chunks_path: Path to parent chunks JSON
            child_chunks_path: Path to child chunks JSON
        """
        logger.info("=" * 60)
        logger.info("INDEXING PIPELINE - PARENT-CHILD STRATEGY")
        logger.info("=" * 60)
        
        # Step 1: Load parent chunks
        logger.info(f"\n[Step 1] Loading parent chunks from {parent_chunks_path}")
        with open(parent_chunks_path, 'r', encoding='utf-8') as f:
            parents = json.load(f)
        logger.info(f"✓ Loaded {len(parents)} parent chunks")
        
        # Step 2: Load child chunks
        logger.info(f"\n[Step 2] Loading child chunks from {child_chunks_path}")
        with open(child_chunks_path, 'r', encoding='utf-8') as f:
            children = json.load(f)
        logger.info(f"✓ Loaded {len(children)} child chunks")
        
        # Step 3: Store parent chunks in metadata
        logger.info(f"\n[Step 3] Storing parent chunks in metadata database...")
        
        # Update parent child_count
        parent_counts = {}
        for child in children:
            parent_id = child['parent_id']
            parent_counts[parent_id] = parent_counts.get(parent_id, 0) + 1
        
        for parent in parents:
            parent['child_count'] = parent_counts.get(parent['parent_id'], 0)
        
        self.metadata_store.insert_parent_chunks(parents)
        logger.info(f"✓ Stored {len(parents)} parent chunks")
        
        # Step 4: Index CHILD chunks in BM25
        logger.info(f"\n[Step 4] Building BM25 index from child chunks...")
        self.bm25_indexer.build_index(children)  # Pass full child dictionaries
        self.bm25_indexer.save("data/processed/bm25_index.pkl")  # Fixed: use 'save' not 'save_index'
        logger.info(f"✓ BM25 index built with {len(children)} child chunks")
        
        # Step 5: Generate embeddings for CHILD chunks
        logger.info(f"\n[Step 5] Generating embeddings for child chunks...")
        embeddings = self.embedding_generator.generate_embeddings(children)  # Pass full dictionaries
        logger.info(f"✓ Generated embeddings: {embeddings.shape}")
        
        # Step 6: Build FAISS index from child embeddings
        logger.info(f"\n[Step 6] Building FAISS index...")
        self.vector_indexer.build_index(embeddings, children)  # Pass both embeddings and chunks
        self.vector_indexer.save("data/processed/faiss_index.bin")  # Fixed: use 'save'
        logger.info(f"✓ FAISS index saved with {len(embeddings)} child vectors")
        
        # Step 7: Store child chunks in metadata
        logger.info(f"\n[Step 7] Storing child chunks in metadata database...")
        self.metadata_store.insert_child_chunks(children)
        logger.info(f"✓ Stored {len(children)} child chunks with parent links")
        
        # Step 8: Generate statistics
        logger.info(f"\n[Step 8] Generating statistics...")
        stats = self._generate_statistics(parents, children, embeddings)
        
        # Save stats
        stats_path = Path("data/processed/indexing_stats.json")
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2)
        
        logger.info("=" * 60)
        logger.info("INDEXING COMPLETE")
        logger.info("=" * 60)
        logger.info(f"\nIndexed components:")
        logger.info(f"  - Parent chunks: {stats['parent_count']}")
        logger.info(f"  - Child chunks: {stats['child_count']}")
        logger.info(f"  - BM25 index size: {stats['bm25_index_size_mb']:.2f} MB")
        logger.info(f"  - FAISS index size: {stats['faiss_index_size_mb']:.2f} MB")
        logger.info(f"  - Embedding dimension: {stats['embedding_dim']}")
        logger.info(f"  - Avg children per parent: {stats['avg_children_per_parent']:.1f}")
        
        return stats
    
    def _generate_statistics(self, parents: List[Dict], children: List[Dict], embeddings: np.ndarray) -> Dict:
        """Generate indexing statistics."""
        
        # Calculate index sizes
        bm25_index_path = Path("data/processed/bm25_index.pkl")
        faiss_index_path = Path("data/processed/faiss_index.bin")
        
        bm25_size_mb = bm25_index_path.stat().st_size / (1024 * 1024) if bm25_index_path.exists() else 0
        faiss_size_mb = faiss_index_path.stat().st_size / (1024 * 1024) if faiss_index_path.exists() else 0
        
        # Token statistics
        parent_tokens = [p['token_count'] for p in parents]
        child_tokens = [c['token_count'] for c in children]
        
        stats = {
            'parent_count': len(parents),
            'child_count': len(children),
            'avg_children_per_parent': len(children) / len(parents) if parents else 0,
            'embedding_dim': embeddings.shape[1] if len(embeddings.shape) > 1 else 0,
            'bm25_index_size_mb': bm25_size_mb,
            'faiss_index_size_mb': faiss_size_mb,
            'embeddings_size_mb': embeddings.nbytes / (1024 * 1024),
            'parent_token_stats': {
                'min': min(parent_tokens) if parent_tokens else 0,
                'max': max(parent_tokens) if parent_tokens else 0,
                'avg': sum(parent_tokens) / len(parent_tokens) if parent_tokens else 0
            },
            'child_token_stats': {
                'min': min(child_tokens) if child_tokens else 0,
                'max': max(child_tokens) if child_tokens else 0,
                'avg': sum(child_tokens) / len(child_tokens) if child_tokens else 0
            }
        }
        
        return stats
    
    def close(self):
        """Close resources."""
        if self.metadata_store:
            self.metadata_store.close()