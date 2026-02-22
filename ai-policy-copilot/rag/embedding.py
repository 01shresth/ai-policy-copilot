"""
Embedding module using sentence-transformers
"""
from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer


class EmbeddingModel:
    """Generate embeddings using sentence-transformers"""
    
    _instance = None
    _model = None
    
    def __new__(cls, model_name: str = "all-MiniLM-L6-v2"):
        """Singleton pattern to avoid reloading the model"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._model_name = model_name
        return cls._instance
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """Initialize embedding model"""
        if self._model is None:
            self._model = SentenceTransformer(model_name)
    
    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for a list of texts
        
        Args:
            texts: List of text strings
            
        Returns:
            NumPy array of embeddings
        """
        if not texts:
            return np.array([])
        
        embeddings = self._model.encode(
            texts,
            show_progress_bar=False,
            convert_to_numpy=True
        )
        return embeddings
    
    def embed_query(self, query: str) -> np.ndarray:
        """
        Generate embedding for a single query
        
        Args:
            query: Query string
            
        Returns:
            NumPy array (1D) of the embedding
        """
        embedding = self._model.encode(
            query,
            show_progress_bar=False,
            convert_to_numpy=True
        )
        return embedding
    
    @property
    def dimension(self) -> int:
        """Get embedding dimension"""
        return self._model.get_sentence_embedding_dimension()
