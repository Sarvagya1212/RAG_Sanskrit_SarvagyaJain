import faiss
import numpy as np
from typing import List, Dict, Tuple
from code.src.utils.logger import setup_logger

logger = setup_logger(__name__)

class VectorIndexer:
    """FAISS-based vector similarity search."""
    
    def __init__(self, embedding_dim: int = 384):
        """
        Initialize vector indexer.
        
        Args:
            embedding_dim: Dimension of embeddings (384 for MiniLM-L6-v2)
        """
        self.embedding_dim = embedding_dim
        self.index = None
        self.chunks = []
    
    def build_index(
        self, 
        embeddings: np.ndarray, 
        chunks: List[Dict],
        index_type: str = "FlatL2"
    ):
        """
        Build FAISS index from embeddings.
        
        Args:
            embeddings: numpy array of shape (n_chunks, embedding_dim)
            chunks: List of chunk dictionaries
            index_type: "FlatL2" (exact) or "IVFFlat" (approximate)
        """
        self.chunks = chunks
        
        # Ensure embeddings are float32 (FAISS requirement)
        embeddings = embeddings.astype('float32')
        
        if index_type == "FlatL2":
            # Exact search - perfect for small datasets
            self.index = faiss.IndexFlatL2(self.embedding_dim)
            
        elif index_type == "IVFFlat":
            # Approximate search for larger datasets
            # Number of clusters (sqrt of dataset size is good heuristic)
            n_clusters = min(100, int(np.sqrt(len(embeddings))))
            
            # Create quantizer
            quantizer = faiss.IndexFlatL2(self.embedding_dim)
            
            # Create IVF index
            self.index = faiss.IndexIVFFlat(
                quantizer, 
                self.embedding_dim, 
                n_clusters
            )
            
            # Train index (required for IVF)
            logger.info(f"Training IVF index with {n_clusters} clusters...")
            self.index.train(embeddings)
        
        else:
            raise ValueError(f"Unknown index type: {index_type}")
        
        # Add vectors to index
        self.index.add(embeddings)
        
        logger.info(
            f"FAISS index built: {index_type}, "
            f"{self.index.ntotal} vectors indexed"
        )
    
    def search(
        self, 
        query_embedding: np.ndarray, 
        top_k: int = 50
    ) -> List[Tuple[int, float]]:
        """
        Search for similar vectors.
        
        Args:
            query_embedding: Query vector of shape (embedding_dim,)
            top_k: Number of results to return
        
        Returns:
            List of (chunk_index, distance) tuples
            Note: Lower distance = more similar
        """
        # Ensure correct shape and type
        query_embedding = query_embedding.reshape(1, -1).astype('float32')
        
        # Search
        distances, indices = self.index.search(query_embedding, top_k)
        
        # Convert to list of tuples
        results = [
            (int(indices[0][i]), float(distances[0][i])) 
            for i in range(len(indices[0]))
        ]
        
        return results
    
    def save(self, path: str):
        """Save FAISS index to disk."""
        faiss.write_index(self.index, path)
        logger.info(f"FAISS index saved to {path}")
    
    def load(self, path: str):
        """Load FAISS index from disk."""
        self.index = faiss.read_index(path)
        logger.info(f"FAISS index loaded from {path}")