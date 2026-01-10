from sentence_transformers import SentenceTransformer
import numpy as np
from tqdm import tqdm
from typing import List, Dict
from code.src.utils.logger import setup_logger

logger = setup_logger(__name__)

class EmbeddingGenerator:
    """Generate semantic embeddings for chunks."""
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize embedding model.
        
        Args:
            model_name: HuggingFace model name
        """
        self.model_name = model_name
        self.model = None
        self.embedding_dim = 384  # Output dimension for MiniLM-L6-v2
    
    def load_model(self):
        """Load sentence transformer model."""
        logger.info(f"Loading embedding model: {self.model_name}")
        
        self.model = SentenceTransformer(self.model_name)
        
        # Set to CPU mode
        self.model = self.model.to('cpu')
        
        logger.info(f"Model loaded. Embedding dimension: {self.embedding_dim}")
    
    def generate_embeddings(
        self, 
        chunks: List[Dict], 
        batch_size: int = 32,
        normalize: bool = True
    ) -> np.ndarray:
        """
        Generate embeddings for all chunks.
        
        Args:
            chunks: List of chunks with 'text_slp1' or 'slp1_text' field
            batch_size: Batch size for processing (32 for CPU)
            normalize: Whether to L2-normalize embeddings
        
        Returns:
            numpy array of shape (n_chunks, embedding_dim)
        """
        if self.model is None:
            self.load_model()
        
        # Extract texts to embed (handle both field names)
        # Check if model requires prefixes (e.g. e5 models)
        is_e5 = "e5" in self.model_name.lower()
        prefix = "passage: " if is_e5 else ""
        
        texts = [
            f"{prefix}{chunk.get('text_slp1') or chunk.get('slp1_text', '')}" 
            for chunk in chunks
        ]
        
        logger.info(f"Generating embeddings for {len(texts)} chunks (model: {self.model_name}, prefix: '{prefix}')...")
        
        # Generate embeddings in batches with progress bar
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=normalize,
            convert_to_numpy=True
        )
        
        logger.info(f"Generated embeddings: {embeddings.shape}")
        
        return embeddings
    
    def generate_query_embedding(
        self, 
        query: str,
        normalize: bool = True
    ) -> np.ndarray:
        """
        Generate embedding for a single query.
        
        Args:
            query: Query text (should be preprocessed to SLP1)
            normalize: Whether to L2-normalize
        
        Returns:
            Embedding vector of shape (embedding_dim,)
        """
        if self.model is None:
            self.load_model()
        
        # Add prefix for e5 models
        is_e5 = "e5" in self.model_name.lower()
        prefix = "query: " if is_e5 else ""
        text_to_embed = f"{prefix}{query}"
        
        embedding = self.model.encode(
            [text_to_embed],
            normalize_embeddings=normalize,
            convert_to_numpy=True
        )[0]
        
        return embedding
    
    def save_embeddings(self, embeddings: np.ndarray, path: str):
        """Save embeddings to disk."""
        np.save(path, embeddings)
        logger.info(f"Embeddings saved to {path}")
    
    def load_embeddings(self, path: str) -> np.ndarray:
        """Load embeddings from disk."""
        embeddings = np.load(path)
        logger.info(f"Loaded embeddings: {embeddings.shape}")
        return embeddings