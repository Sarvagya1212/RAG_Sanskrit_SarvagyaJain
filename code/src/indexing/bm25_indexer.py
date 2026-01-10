from rank_bm25 import BM25Okapi
import pickle
from typing import List, Dict, Tuple
from code.src.utils.logger import setup_logger

logger = setup_logger(__name__)

class BM25Indexer:
    """BM25 index using character n-grams."""
    
    def __init__(self, ngram_size: int = 4):
        """
        Args:
            ngram_size: Size of character n-grams (4 recommended)
        """
        self.ngram_size = ngram_size
        self.bm25 = None
        self.chunks = []
    
    def _create_ngrams(self, text: str) -> List[str]:
        """
        Create character n-grams from text.
        
        Example:
            Input: "धर्मः"
            Output: ["धर्म", "र्मः"] (if ngram_size=3)
        """
        # Remove spaces for continuous n-grams
        text = text.replace(' ', '')
        
        # Create n-grams
        ngrams = []
        for i in range(len(text) - self.ngram_size + 1):
            ngram = text[i:i + self.ngram_size]
            ngrams.append(ngram)
        
        return ngrams
    
    def build_index(self, chunks: List[Dict]):
        """
        Build BM25 index from chunks.
        
        Args:
            chunks: List of chunks with 'text_slp1' or 'slp1_text' field
        """
        self.chunks = chunks
        
        # Create n-gram corpus
        corpus = []
        for chunk in chunks:
            # Use SLP1 text for indexing (handle both field names)
            text_slp1 = chunk.get('text_slp1') or chunk.get('slp1_text', '')
            ngrams = self._create_ngrams(text_slp1)
            corpus.append(ngrams)
        
        # Build BM25 index
        self.bm25 = BM25Okapi(corpus)
        
        logger.info(f"BM25 index built with {len(chunks)} chunks")
    
    def search(self, query: str, top_k: int = 50) -> List[Tuple[int, float]]:
        """
        Search BM25 index.
        
        Args:
            query: Query string (will be converted to n-grams)
            top_k: Number of results to return
        
        Returns:
            List of (chunk_index, score) tuples
        """
        # Create query n-grams
        query_ngrams = self._create_ngrams(query)
        
        # Get BM25 scores
        scores = self.bm25.get_scores(query_ngrams)
        
        # Get top-k indices
        top_indices = scores.argsort()[-top_k:][::-1]
        
        # Return (index, score) pairs
        results = [(idx, scores[idx]) for idx in top_indices]
        
        return results
    
    def save(self, path: str):
        """Save BM25 index to disk."""
        with open(path, 'wb') as f:
            pickle.dump({
                'bm25': self.bm25,
                'chunks': self.chunks,
                'ngram_size': self.ngram_size
            }, f)
    
    def load(self, path: str):
        """Load BM25 index from disk."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.bm25 = data['bm25']
            self.chunks = data['chunks']
            self.ngram_size = data['ngram_size']