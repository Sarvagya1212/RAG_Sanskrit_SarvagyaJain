"""Hybrid retrieval combining BM25 and vector search with Reciprocal Rank Fusion."""

from typing import List, Dict, Tuple
import numpy as np
from code.src.utils.logger import setup_logger

logger = setup_logger(__name__)


def reciprocal_rank_fusion(
    rankings: List[List[Tuple[int, float]]],
    k: int = 60
) -> List[Tuple[int, float]]:
    """
    Combine multiple rankings using Reciprocal Rank Fusion (RRF).
    
    RRF formula: score(d) = Σ 1/(k + rank(d))
    where k is a constant (typically 60) and rank starts at 1.
    
    Args:
        rankings: List of ranked lists, each containing (doc_id, score) tuples
        k: RRF constant (default: 60)
        
    Returns:
        Combined ranking as list of (doc_id, rrf_score) tuples
        
    Examples:
        >>> bm25_results = [(0, 5.2), (1, 3.1), (2, 2.0)]
        >>> vector_results = [(1, 0.95), (0, 0.89), (3, 0.75)]
        >>> fused = reciprocal_rank_fusion([bm25_results, vector_results])
        # Ranks: BM25=[0:1, 1:2, 2:3], Vector=[1:1, 0:2, 3:3]
        # Doc 0: 1/(60+1) + 1/(60+2) = 0.0164 + 0.0161 = 0.0325
        # Doc 1: 1/(60+2) + 1/(60+1) = 0.0161 + 0.0164 = 0.0325
    """
    # Collect all unique document IDs
    all_doc_ids = set()
    for ranking in rankings:
        for doc_id, _ in ranking:
            all_doc_ids.add(doc_id)
    
    # Calculate RRF score for each document
    rrf_scores = {}
    for doc_id in all_doc_ids:
        rrf_score = 0.0
        
        # Add contribution from each ranking
        for ranking in rankings:
            # Find rank of this document (1-indexed)
            rank = None
            for idx, (did, _) in enumerate(ranking, start=1):
                if did == doc_id:
                    rank = idx
                    break
            
            # If document appears in this ranking, add its RRF score
            if rank is not None:
                rrf_score += 1.0 / (k + rank)
        
        rrf_scores[doc_id] = rrf_score
    
    # Sort by RRF score (descending)
    sorted_results = sorted(
        rrf_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )
    
    return sorted_results


class HybridRetriever:
    """
    Hybrid retrieval combining BM25 (keyword) and Vector (semantic) search.
    
    Uses Reciprocal Rank Fusion to combine rankings.
    """
    
    def __init__(
        self,
        bm25_indexer,
        vector_indexer,
        embedding_generator,
        metadata_store,
        preprocessor
    ):
        """
        Initialize hybrid retriever.
        
        Args:
            bm25_indexer: BM25Indexer instance
            vector_indexer: VectorIndexer instance
            embedding_generator: EmbeddingGenerator instance
            metadata_store: MetadataStore instance
            preprocessor: SanskritPreprocessor instance
        """
        self.bm25_indexer = bm25_indexer
        self.vector_indexer = vector_indexer
        self.embedding_generator = embedding_generator
        self.metadata_store = metadata_store
        self.preprocessor = preprocessor
        
        logger.info("HybridRetriever initialized")
    
    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        mode: str = "bm25_primary",
        bm25_k: int = 50,
        vector_k: int = 50,
        rrf_k: int = 60
    ) -> List[Dict]:
        """
        Retrieve top-k chunks using configurable retrieval strategy.
        
        Modes:
        - "bm25_primary": BM25 retrieves candidates, vector reranks (default)
          Recommended for Sanskrit due to limited embedding model support.
        - "hybrid_rrf": Equal-weight RRF fusion of BM25 + Vector
        
        Args:
            query: User query (any script)
            top_k: Final number of results to return
            mode: Retrieval strategy ("bm25_primary" or "hybrid_rrf")
            bm25_k: Number of results from BM25
            vector_k: Number of results from vector search  
            rrf_k: RRF constant (only used in hybrid_rrf mode)
            
        Returns:
            List of chunk dictionaries with metadata
        """
        logger.info(f"Retrieving for query: '{query}' (mode={mode})")
        
        if mode == "bm25_primary":
            results = self._retrieve_bm25_primary(query, top_k, bm25_k)
        else:
            results = self._retrieve_hybrid_rrf(query, top_k, bm25_k, vector_k, rrf_k)
            
        # NEW: Filter by score threshold
        SCORE_THRESHOLD = 0.01  # Only keep results with score > 0.01 (Lowered for better recall)
    
        filtered_results = []
        for result in results:
            if result.get('retrieval_score', 0) > SCORE_THRESHOLD:
                filtered_results.append(result)
        
        logger.info(f"Filtered results from {len(results)} to {len(filtered_results)} (threshold={SCORE_THRESHOLD})")
        
        return filtered_results[:top_k]
    
    def _retrieve_bm25_primary(
        self,
        query: str,
        top_k: int = 5,
        bm25_k: int = 50
    ) -> List[Dict]:
        """
        BM25-primary retrieval with vector reranking.
        
        Strategy:
        1. BM25 retrieves top candidates (lexical matching)
        2. Vector similarity reranks BM25 candidates
        
        This is optimal for Sanskrit because:
        - General-purpose embedding models have limited Sanskrit support
        - BM25 with character n-grams handles sandhi/compound variations well
        - Vector search refines ranking within BM25 candidates
        """
        # Step 1: Preprocess query to SLP1
        preprocessed = self.preprocessor.process(query)
        query_slp1 = preprocessed.slp1
        logger.debug(f"Query preprocessed: '{query}' → '{query_slp1}'")
        
        # Step 2: BM25 search (primary retrieval)
        bm25_results = self.bm25_indexer.search(query_slp1, top_k=bm25_k)
        logger.debug(f"BM25 returned {len(bm25_results)} candidates")
        
        if not bm25_results:
            logger.warning("BM25 returned no results")
            return []
        
        # Step 3: Get query embedding for reranking
        query_embedding = self.embedding_generator.generate_query_embedding(query_slp1)
        
        # Step 4: Rerank BM25 candidates using vector similarity
        reranked = []
        for doc_id, bm25_score in bm25_results:
            chunk = self.metadata_store.get_chunk_by_index(int(doc_id))
            if chunk:
                # Get chunk embedding from vector index for similarity calculation
                # Use the chunk's SLP1 text to generate/retrieve embedding
                chunk_text = chunk.get('text_slp1', '')
                if chunk_text:
                    chunk_embedding = self.embedding_generator.generate_query_embedding(chunk_text)
                    # Cosine similarity (embeddings are normalized)
                    similarity = float(np.dot(query_embedding, chunk_embedding))
                else:
                    similarity = 0.0
                
                # Combined score: BM25 normalized + vector similarity
                # BM25 score varies widely, so we use rank-based weighting
                combined_score = bm25_score * 0.7 + similarity * 0.3
                reranked.append((doc_id, combined_score, chunk))
        
        # Sort by combined score (descending)
        reranked.sort(key=lambda x: x[1], reverse=True)
        
        # Step 5: Return top-k with metadata
        top_results = []
        for doc_id, score, chunk in reranked[:top_k]:
            chunk['retrieval_score'] = score
            chunk['rank'] = len(top_results) + 1
            chunk['retrieval_mode'] = 'bm25_primary'
            top_results.append(chunk)
        
        logger.info(f"Returning {len(top_results)} results (BM25-primary mode)")
        return top_results
    
    def _retrieve_hybrid_rrf(
        self,
        query: str,
        top_k: int = 5,
        bm25_k: int = 50,
        vector_k: int = 50,
        rrf_k: int = 60
    ) -> List[Dict]:
        """
        Hybrid retrieval using Reciprocal Rank Fusion.
        
        Pipeline:
        1. Preprocess query to SLP1
        2. BM25 search (keyword matching)
        3. Vector search (semantic similarity)
        4. Reciprocal Rank Fusion
        5. Return top-k with metadata
        """
        logger.info(f"Using hybrid RRF mode")
        
        # Step 1: Preprocess query to SLP1
        preprocessed = self.preprocessor.process(query)
        query_slp1 = preprocessed.slp1
        logger.debug(f"Query preprocessed: '{query}' → '{query_slp1}'")
        
        # Step 2: BM25 search
        bm25_results = self.bm25_indexer.search(query_slp1, top_k=bm25_k)
        logger.debug(f"BM25 returned {len(bm25_results)} results")
        
        # Step 3: Vector search
        query_embedding = self.embedding_generator.generate_query_embedding(query_slp1)
        vector_results = self.vector_indexer.search(query_embedding, top_k=vector_k)
        logger.debug(f"Vector search returned {len(vector_results)} results")
        
        # Step 4: Reciprocal Rank Fusion
        fused_results = reciprocal_rank_fusion(
            [bm25_results, vector_results],
            k=rrf_k
        )
        logger.debug(f"RRF combined to {len(fused_results)} unique results")
        
        # Debug: Show fused results
        logger.debug(f"Fused top-5: {fused_results[:5]}")
        
        # Step 5: Get top-k with metadata
        top_results = []
        for doc_id, rrf_score in fused_results[:top_k]:
            # Retrieve full chunk metadata
            logger.debug(f"Looking up doc_id={doc_id}")
            chunk = self.metadata_store.get_chunk_by_index(int(doc_id))
            logger.debug(f"Lookup result: {chunk is not None}")
            
            if chunk:
                # Add retrieval score
                chunk['retrieval_score'] = rrf_score
                chunk['rank'] = len(top_results) + 1
                chunk['retrieval_mode'] = 'hybrid_rrf'
                top_results.append(chunk)
        
        logger.info(f"Returning {len(top_results)} results (hybrid RRF mode)")
        return top_results
    
    def retrieve_with_explanation(
        self,
        query: str,
        top_k: int = 5
    ) -> Dict:
        """
        Retrieve with detailed explanation of ranking.
        
        Returns:
            {
                'query': original query,
                'query_slp1': preprocessed query,
                'results': list of chunks,
                'explanation': {
                    'bm25_top_3': [...],
                    'vector_top_3': [...],
                    'fusion_method': 'Reciprocal Rank Fusion'
                }
            }
        """
        # Preprocess
        preprocessed = self.preprocessor.process(query)
        query_slp1 = preprocessed.slp1
        
        # Search both indices
        bm25_results = self.bm25_indexer.search(query_slp1, top_k=50)
        query_embedding = self.embedding_generator.generate_query_embedding(query_slp1)
        vector_results = self.vector_indexer.search(query_embedding, top_k=50)
        
        # Fuse
        fused_results = reciprocal_rank_fusion([bm25_results, vector_results])
        
        # Get top-k chunks
        final_results = []
        for doc_id, score in fused_results[:top_k]:
            chunk = self.metadata_store.get_chunk_by_index(doc_id)
            if chunk:
                chunk['retrieval_score'] = score
                chunk['rank'] = len(final_results) + 1
                final_results.append(chunk)
        
        return {
            'query': query,
            'query_slp1': query_slp1,
            'script_detected': preprocessed.script,
            'results': final_results,
            'explanation': {
                'bm25_top_3': [
                    (idx, score) for idx, score in bm25_results[:3]
                ],
                'vector_top_3': [
                    (idx, dist) for idx, dist in vector_results[:3]
                ],
                'fusion_method': 'Reciprocal Rank Fusion (k=60)',
                'num_candidates': len(fused_results)
            }
        }
