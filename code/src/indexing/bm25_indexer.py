from rank_bm25 import BM25Okapi
import pickle
from typing import List, Tuple, Dict
import re
from code.src.utils.logger import setup_logger

logger = setup_logger(__name__)


class BM25Indexer:
    """Fixed BM25 index using character n-grams AND word tokens."""
    
    def __init__(self, ngram_size: int = 3, use_hybrid: bool = True):
        """
        Initialize BM25 indexer with hybrid tokenization.
        
        Args:
            ngram_size: Size of character n-grams (3 recommended for Sanskrit)
            use_hybrid: Use both n-grams and word tokens (recommended)
        """
        self.ngram_size = ngram_size
        self.use_hybrid = use_hybrid
        self.bm25 = None
        self.chunks = []
        self.tokenized_corpus = []
        
        logger.info(f"BM25Indexer initialized (ngram_size={ngram_size}, hybrid={use_hybrid})")
    
    def _tokenize_text(self, text: str) -> List[str]:
        """
        Tokenize text using hybrid approach.
        
        Strategy:
        1. Word-level tokens (split by spaces)
        2. Character n-grams for each word
        3. Combine both for better matching
        
        Args:
            text: Input text (SLP1 encoded)
            
        Returns:
            List of tokens
        """
        if not text:
            return []
        
        tokens = []
        
        if self.use_hybrid:
            # Method 1: Word tokens (for exact matches)
            # Split by whitespace and strip punctuation
            raw_words = text.split()
            words = [w.strip(".,;:!?\"'()[]{}।॥-") for w in raw_words]
            words = [w for w in words if w] # Filter empty
            tokens.extend(words)
            
            # Method 2: Character n-grams (for fuzzy matches)
            # Remove spaces for continuous n-grams
            continuous_text = text.replace(' ', '')
            ngrams = self._create_ngrams(continuous_text)
            tokens.extend(ngrams)
            
            # Method 3: Vowel-neutralized tokens (for case/length-insensitive matches)
            # Neutralize: A->a, I->i, U->u, F->f, X->x
            neutralized_text = text.translate(str.maketrans("AIUFX", "aiufx"))
            
            # Add neutralized words (stripped)
            raw_neut_words = neutralized_text.split()
            neut_words = [w.strip(".,;:!?\"'()[]{}।॥-") for w in raw_neut_words]
            neut_words = [w for w in neut_words if w]
            tokens.extend(neut_words)
            
            # Add neutralized n-grams
            neutralized_continuous = neutralized_text.replace(' ', '')
            tokens.extend(self._create_ngrams(neutralized_continuous))
            
        else:
            # Pure n-gram mode
            continuous_text = text.replace(' ', '')
            tokens = self._create_ngrams(continuous_text)
            
            # Add neutralized n-grams too
            neutralized_continuous = continuous_text.translate(str.maketrans("AIUFX", "aiufx"))
            tokens.extend(self._create_ngrams(neutralized_continuous))
        
        # Remove duplicates while preserving order
        seen = set()
        unique_tokens = []
        for token in tokens:
            if token not in seen:
                seen.add(token)
                unique_tokens.append(token)
        
        return unique_tokens
    
    def _create_ngrams(self, text: str) -> List[str]:
        """
        Create character n-grams from text.
        
        Args:
            text: Input text
            
        Returns:
            List of n-grams
        """
        if len(text) < self.ngram_size:
            return [text]
        
        ngrams = []
        for i in range(len(text) - self.ngram_size + 1):
            ngram = text[i:i + self.ngram_size]
            ngrams.append(ngram)
        
        return ngrams
    
    def build_index(self, chunks: List[Dict]):
        """
        Build BM25 index from chunks.
        
        Args:
            chunks: List of chunks with 'text_slp1' field
        """
        self.chunks = chunks
        
        logger.info(f"Building BM25 index for {len(chunks)} chunks...")
        
        # Tokenize all chunks
        self.tokenized_corpus = []
        
        for i, chunk in enumerate(chunks):
            # Use SLP1 text for indexing (try multiple keys)
            text_slp1 = (chunk.get('preprocessed_text') or 
                        chunk.get('text_slp1') or 
                        chunk.get('slp1_text', ''))
            
            if not text_slp1:
                # If still empty, use original text as last resort
                text_slp1 = chunk.get('text_original', '')
                if not text_slp1 and chunk.get('text'):
                    text_slp1 = chunk.get('text', '')

            if not text_slp1:
                logger.warning(f"Chunk {i} has no text content!")
            
            # Tokenize
            tokens = self._tokenize_text(text_slp1)
            # Ensure at least one token to avoid ZeroDivisionError in rank_bm25
            if not tokens:
                tokens = ["<EMPTY>"]
                
            self.tokenized_corpus.append(tokens)
            
            # Log sample for debugging
            if i < 3:
                logger.debug(f"Chunk {i} tokens (first 10): {tokens[:10]}")
        
        # Build BM25 index
        if not self.tokenized_corpus:
             logger.error("No tokens generated from chunks. Cannot build index.")
             return
             
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        
        logger.info(f"BM25 index built successfully")
        
        # Validate index
        self._validate_index()
    
    def _validate_index(self):
        """Validate that index works correctly."""
        if not self.chunks or not self.tokenized_corpus:
            return
        
        # Test: Search for first chunk's tokens
        test_tokens = self.tokenized_corpus[0][:5]  # First 5 tokens
        test_query = ' '.join(test_tokens)
        
        scores = self.bm25.get_scores(test_tokens)
        top_idx = scores.argmax()
        
        if top_idx == 0:
            logger.info("✓ BM25 index validation passed")
        else:
            logger.warning(f"⚠ BM25 validation issue: Query from chunk 0 returned chunk {top_idx}")
    
    def search(self, query: str, top_k: int = 50) -> List[Tuple[int, float]]:
        """
        Search BM25 index.
        
        Args:
            query: Query string (should be preprocessed to SLP1)
            top_k: Number of results to return
        
        Returns:
            List of (chunk_index, score) tuples sorted by score
        """
        if self.bm25 is None:
            raise ValueError("BM25 index not built. Call build_index() first.")
        
        # Tokenize query (same way as documents)
        query_tokens = self._tokenize_text(query)
        
        logger.debug(f"Query tokens: {query_tokens[:10]}")
        
        # Get BM25 scores
        scores = self.bm25.get_scores(query_tokens)
        
        # Get top-k indices (sorted by score, descending)
        top_indices = scores.argsort()[-top_k:][::-1]
        
        # Return (index, score) pairs
        results = [(int(idx), float(scores[idx])) for idx in top_indices]
        
        # Filter out zero scores
        results = [(idx, score) for idx, score in results if score > 0]
        
        logger.debug(f"BM25 returned {len(results)} results with non-zero scores")
        
        return results
    
    def save(self, path: str):
        """Save BM25 index to disk."""
        data = {
            'bm25': self.bm25,
            'chunks': self.chunks,
            'tokenized_corpus': self.tokenized_corpus,
            'ngram_size': self.ngram_size,
            'use_hybrid': self.use_hybrid
        }
        
        with open(path, 'wb') as f:
            pickle.dump(data, f)
        
        logger.info(f"BM25 index saved to {path}")
    
    def load(self, path: str):
        """Load BM25 index from disk."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        self.bm25 = data['bm25']
        self.chunks = data['chunks']
        self.tokenized_corpus = data.get('tokenized_corpus', [])
        self.ngram_size = data.get('ngram_size', 3)
        self.use_hybrid = data.get('use_hybrid', True)
        
        logger.info(f"BM25 index loaded from {path}")
        logger.info(f"Loaded {len(self.chunks)} chunks, hybrid={self.use_hybrid}")


# Quick rebuild script
def rebuild_bm25_index():
    """Rebuild BM25 index with fixed implementation."""
    import json
    from pathlib import Path
    
    print("\n" + "="*60)
    print("REBUILDING BM25 INDEX")
    print("="*60)
    
    # Load chunks
    chunks_path = "data/processed/chunks.json"
    
    if not Path(chunks_path).exists():
        print(f"ERROR: {chunks_path} not found!")
        print("Please run indexing first.")
        return
    
    print(f"\n1. Loading chunks from {chunks_path}...")
    with open(chunks_path, 'r', encoding='utf-8') as f:
        chunks = json.load(f)
    
    print(f"   ✓ Loaded {len(chunks)} chunks")
    
    # Build new index
    print("\n2. Building BM25 index with hybrid tokenization...")
    indexer = BM25Indexer(ngram_size=3, use_hybrid=True)
    indexer.build_index(chunks)
    
    # Save
    output_path = "data/processed/bm25_index.pkl"
    print(f"\n3. Saving index to {output_path}...")
    indexer.save(output_path)
    
    print("\n" + "="*60)
    print("BM25 INDEX REBUILT SUCCESSFULLY")
    print("="*60)
    
    # Test it
    print("\n4. Testing new index...")
    
    test_queries = [
        ("kAlidAsa", "Should find Kalidasa story"),
        ("mUrkaBftya", "Should find foolish servant story"),
        ("vfdDA", "Should find old woman story"),
    ]
    
    for query_slp1, description in test_queries:
        results = indexer.search(query_slp1, top_k=3)
        
        print(f"\n   Query: {query_slp1} ({description})")
        
        if results:
            for rank, (idx, score) in enumerate(results[:3], 1):
                chunk = chunks[idx]
                story = chunk.get('story_title', 'Unknown')
                print(f"      {rank}. Score={score:.3f} | Story: {story}")
        else:
            print("      No results found")
    
    print("\n✓ Testing complete!\n")


if __name__ == "__main__":
    rebuild_bm25_index()