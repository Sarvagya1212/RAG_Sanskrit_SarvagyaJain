"""Hybrid retrieval for parent-child chunks with BM25 and vector search."""

from typing import List, Dict, Tuple
import numpy as np
from code.src.utils.logger import setup_logger

logger = setup_logger(__name__)

# Score threshold for filtering low-confidence results
SCORE_THRESHOLD = 0.1  # Weighted fusion scores are 0-1 range


def weighted_score_fusion(
    bm25_results: List[Tuple[int, float]],
    vector_results: List[Tuple[int, float]],
    bm25_weight: float = 0.7,
    vector_weight: float = 0.3
) -> List[Tuple[int, float]]:
    """
    Combine BM25 and vector scores using weighted fusion.
    
    Normalizes scores to [0,1] range then combines with weights.
    Final score = bm25_weight * norm_bm25 + vector_weight * norm_vector
    
    Args:
        bm25_results: List of (doc_id, bm25_score) tuples
        vector_results: List of (doc_id, distance) tuples (lower = better)
        bm25_weight: Weight for BM25 scores (default: 0.7)
        vector_weight: Weight for vector scores (default: 0.3)
        
    Returns:
        Combined ranking as list of (doc_id, combined_score) tuples
    """
    # Normalize BM25 scores to [0, 1]
    bm25_scores = {}
    if bm25_results:
        max_bm25 = max(score for _, score in bm25_results) if bm25_results else 1.0
        max_bm25 = max(max_bm25, 0.001)  # Avoid division by zero
        for doc_id, score in bm25_results:
            bm25_scores[doc_id] = score / max_bm25
    
    # Normalize vector distances to [0, 1] (invert: lower distance = higher score)
    vector_scores = {}
    if vector_results:
        max_dist = max(dist for _, dist in vector_results) if vector_results else 1.0
        max_dist = max(max_dist, 0.001)
        for doc_id, dist in vector_results:
            # Convert distance to similarity (1 - normalized_distance)
            vector_scores[doc_id] = 1.0 - (dist / max_dist)
    
    # Combine scores
    all_doc_ids = set(bm25_scores.keys()) | set(vector_scores.keys())
    combined_scores = {}
    
    for doc_id in all_doc_ids:
        bm25_score = bm25_scores.get(doc_id, 0.0)
        vector_score = vector_scores.get(doc_id, 0.0)
        combined_scores[doc_id] = (bm25_weight * bm25_score) + (vector_weight * vector_score)
    
    # Sort by combined score (descending)
    sorted_results = sorted(
        combined_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )
    
    return sorted_results


class HybridRetriever:
    """
    Hybrid retrieval for parent-child chunks.
    
    Strategy:
    1. Search CHILD chunks using BM25 + Vector
    2. Deduplicate by parent_id (avoid duplicate parents)
    3. Return parent context for each unique match
    """
    
    def __init__(
        self,
        bm25_indexer,
        vector_indexer,
        embedding_generator,
        metadata_store,
        preprocessor,
        config: dict
    ):
        """
        Initialize hybrid retriever.
        
        Args:
            bm25_indexer: BM25 index
            vector_indexer: FAISS vector index
            embedding_generator: Embedding model
            metadata_store: Metadata database
            preprocessor: Text preprocessor
            config: Configuration dictionary
        """
        self.bm25_indexer = bm25_indexer
        self.vector_indexer = vector_indexer
        self.embedding_generator = embedding_generator
        self.metadata_store = metadata_store
        self.preprocessor = preprocessor
        self.config = config
        
        # Retrieval parameters
        self.bm25_top_k = config['retrieval']['bm25_top_k']
        self.vector_top_k = config['retrieval']['vector_top_k']
        self.fusion_weights = {
            'bm25': config['retrieval'].get('bm25_weight', 0.7),
            'vector': config['retrieval'].get('vector_weight', 0.3)
        }
        
        logger.info("HybridRetriever initialized for parent-child chunks")
    
    def search(
        self,
        query: str,
        top_k: int = 2,
        use_bm25: bool = True,
        use_vector: bool = True
    ) -> List[Dict]:
        """
        Search for relevant parent-child pairs.
        
        Strategy:
        1. Search CHILD chunks (precise retrieval)
        2. Deduplicate by parent_id
        3. Return parent context (rich context)
        
        Args:
            query: Search query
            top_k: Number of unique PARENT chunks to return
            use_bm25: Whether to use BM25 search
            use_vector: Whether to use vector search
            
        Returns:
            List of parent-child result dictionaries
        """
        logger.info(f"Searching for: '{query}' (top_k={top_k})")
        
        # Preprocess query to SLP1
        query_result = self.preprocessor.process(query)
        query_slp1 = query_result.slp1
        
        bm25_results = []
        vector_results = []
        
        # BM25 Search on child chunks
        if use_bm25:
            bm25_results = self.bm25_indexer.search(query_slp1, top_k=self.bm25_top_k)
        
        # Vector Search on child chunks
        if use_vector:
            # Use generate_query_embedding (adds "query:" prefix for E5)
            query_embedding = self.embedding_generator.generate_query_embedding(query_slp1)
            vector_results = self.vector_indexer.search(
                query_embedding,
                top_k=self.vector_top_k
            )
        
        # Fuse using weighted score fusion (scores in 0-1 range)
        if bm25_results or vector_results:
            fused_results = weighted_score_fusion(
                bm25_results, 
                vector_results,
                bm25_weight=self.fusion_weights['bm25'],
                vector_weight=self.fusion_weights['vector']
            )
        else:
            return []
        
        # Get all child chunks from metadata
        all_children = self.metadata_store.get_all_child_chunks()
        
        # Enrich results with parent-child data
        enriched_results = []
        seen_parents = set()
        
        for child_idx, fused_score in fused_results:
            # Stop if we have enough unique parents
            if len(enriched_results) >= top_k:
                break
            
            # Filter low scores
            if fused_score < SCORE_THRESHOLD:
                continue
            
            # Get child chunk
            if child_idx >= len(all_children):
                continue
            
            child = all_children[child_idx]
            parent_id = child['parent_id']
            
            # Skip if we've already seen this parent
            if parent_id in seen_parents:
                continue
            
            # Fetch parent chunk
            parent = self.metadata_store.get_parent_by_id(parent_id)
            if not parent:
                logger.warning(f"Parent {parent_id} not found for child {child['chunk_id']}")
                continue
            
            # Mark parent as seen
            seen_parents.add(parent_id)
            
            # Create enriched result
            enriched_results.append({
                'chunk_id': child['chunk_id'],
                'parent_id': parent_id,
                'score': float(fused_score),
                
                # Child content (matched section)
                'child_text': child['text'],
                'child_preprocessed': child['preprocessed_text'],
                
                # Parent content (full context)
                'parent_text': parent['text'],
                'parent_preprocessed': parent['preprocessed_text'],
                
                # Metadata
                'story_id': child['story_id'],
                'story_title': child['story_title'],
                'child_index': child['child_index'],
                'total_children': child['total_children']
            })
        
        logger.info(f"Retrieved {len(enriched_results)} unique parent contexts")
        
        return enriched_results
