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
    """Master pipeline for building all indexes."""
    
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
            embedding_dim=384  # MiniLM-L6-v2 dimension
        )
        
        self.metadata_store = MetadataStore()
    
    def build_indexes(self, chunks_path: str = "data/processed/chunks.json"):
        """
        Build all indexes from chunks.
        
        Args:
            chunks_path: Path to chunks JSON file
        """
        logger.info("=" * 60)
        logger.info("STARTING INDEXING PIPELINE")
        logger.info("=" * 60)
        
        # Load chunks
        logger.info(f"Loading chunks from {chunks_path}")
        with open(chunks_path, 'r', encoding='utf-8') as f:
            chunks_data = json.load(f)
        
        # Handle both nested format (from run_chunking.py) and flat list format
        if isinstance(chunks_data, dict) and 'chunks' in chunks_data:
            chunks = chunks_data['chunks']
            logger.info(f"Loaded {len(chunks)} chunks from nested format")
        else:
            chunks = chunks_data
            logger.info(f"Loaded {len(chunks)} chunks from flat format")
        
        # Step 1: Build BM25 index
        logger.info("\n[1/4] Building BM25 index...")
        self.bm25_indexer.build_index(chunks)
        bm25_path = "data/processed/bm25_index.pkl"
        self.bm25_indexer.save(bm25_path)
        
        # Step 2: Generate embeddings
        logger.info("\n[2/4] Generating vector embeddings...")
        embeddings = self.embedding_generator.generate_embeddings(
            chunks,
            batch_size=self.config['models']['embedding']['batch_size'],
            normalize=self.config['models']['embedding']['normalize']
        )
        
        # Save embeddings
        embeddings_path = "data/processed/embeddings.npy"
        self.embedding_generator.save_embeddings(embeddings, embeddings_path)
        
        # Step 3: Build FAISS index
        logger.info("\n[3/4] Building FAISS vector index...")
        self.vector_indexer.build_index(
            embeddings,
            chunks,
            index_type=self.config['indexing']['vector']['index_type']
        )
        
        # Save FAISS index
        faiss_path = "data/processed/faiss_index.bin"
        self.vector_indexer.save(faiss_path)
        
        # Step 4: Store metadata
        logger.info("\n[4/4] Storing chunk metadata...")
        self.metadata_store.insert_chunks(chunks)
        
        # Generate statistics
        self._generate_statistics(chunks, embeddings)
        
        logger.info("\n" + "=" * 60)
        logger.info("INDEXING PIPELINE COMPLETE")
        logger.info("=" * 60)
    
    def _generate_statistics(self, chunks: List[Dict], embeddings: np.ndarray):
        """Generate and save indexing statistics."""
        
        stats = {
            'total_chunks': len(chunks),
            'embedding_dimension': embeddings.shape[1],
            'embeddings_size_mb': embeddings.nbytes / (1024 * 1024),
            'content_type_distribution': {},
            'story_distribution': {},
            'avg_chunk_length': 0,
        }
        
        # Count by content type
        for chunk in chunks:
            ctype = chunk.get('type', 'unknown')
            stats['content_type_distribution'][ctype] = \
                stats['content_type_distribution'].get(ctype, 0) + 1
            
            story_id = chunk.get('story_id', 0)
            stats['story_distribution'][story_id] = \
                stats['story_distribution'].get(story_id, 0) + 1
        
        # Average chunk length
        total_length = sum(len(c.get('text_slp1', '')) for c in chunks)
        stats['avg_chunk_length'] = total_length / len(chunks) if chunks else 0
        
        # Save statistics
        stats_path = "data/processed/indexing_stats.json"
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        
        # Print summary
        logger.info("\n=== Indexing Statistics ===")
        logger.info(f"Total chunks: {stats['total_chunks']}")
        logger.info(f"Embedding dimension: {stats['embedding_dimension']}")
        logger.info(f"Embeddings size: {stats['embeddings_size_mb']:.2f} MB")
        logger.info(f"Avg chunk length: {stats['avg_chunk_length']:.0f} chars")
        logger.info("\nContent type distribution:")
        for ctype, count in stats['content_type_distribution'].items():
            logger.info(f"  {ctype}: {count}")