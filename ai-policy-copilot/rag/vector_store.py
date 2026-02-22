"""
FAISS vector store module
"""
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import numpy as np
import faiss


class VectorStore:
    """FAISS-based vector store for document chunks"""
    
    def __init__(
        self, 
        index_path: Path = None, 
        metadata_path: Path = None,
        dimension: int = 384  # Default for all-MiniLM-L6-v2
    ):
        """
        Initialize vector store
        
        Args:
            index_path: Path to save/load FAISS index
            metadata_path: Path to save/load chunk metadata
            dimension: Embedding dimension
        """
        self.index_path = index_path
        self.metadata_path = metadata_path
        self.dimension = dimension
        self.index = None
        self.metadata: List[Dict] = []
    
    def create_index(
        self, 
        embeddings: np.ndarray, 
        metadata: List[Dict]
    ) -> None:
        """
        Create a new FAISS index from embeddings
        
        Args:
            embeddings: NumPy array of embeddings
            metadata: List of chunk metadata dictionaries
        """
        if len(embeddings) == 0:
            raise ValueError("Cannot create index with empty embeddings")
        
        # Use L2 distance (euclidean) with IndexFlatL2
        # For small to medium datasets, this is sufficient
        self.dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(self.dimension)
        
        # Add vectors to index
        self.index.add(embeddings.astype(np.float32))
        self.metadata = metadata
    
    def search(
        self, 
        query_embedding: np.ndarray, 
        top_k: int = 3
    ) -> List[Tuple[Dict, float]]:
        """
        Search for similar chunks
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return
            
        Returns:
            List of (metadata, distance) tuples
        """
        if self.index is None or self.index.ntotal == 0:
            return []
        
        # Reshape if needed
        if len(query_embedding.shape) == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        # Search
        k = min(top_k, self.index.ntotal)
        distances, indices = self.index.search(
            query_embedding.astype(np.float32), 
            k
        )
        
        results = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx >= 0 and idx < len(self.metadata):
                results.append((self.metadata[idx], float(dist)))
        
        return results
    
    def save(self) -> bool:
        """Save index and metadata to disk"""
        try:
            if self.index is None:
                return False
            
            # Save FAISS index
            if self.index_path:
                faiss.write_index(self.index, str(self.index_path))
            
            # Save metadata
            if self.metadata_path:
                with open(self.metadata_path, 'w') as f:
                    json.dump(self.metadata, f, indent=2)
            
            return True
        except Exception as e:
            print(f"Error saving index: {e}")
            return False
    
    def load(self) -> bool:
        """Load index and metadata from disk"""
        try:
            # Load FAISS index
            if self.index_path and self.index_path.exists():
                self.index = faiss.read_index(str(self.index_path))
            else:
                return False
            
            # Load metadata
            if self.metadata_path and self.metadata_path.exists():
                with open(self.metadata_path, 'r') as f:
                    self.metadata = json.load(f)
            else:
                return False
            
            return True
        except Exception as e:
            print(f"Error loading index: {e}")
            return False
    
    @property
    def is_empty(self) -> bool:
        """Check if index is empty or not initialized"""
        return self.index is None or self.index.ntotal == 0
    
    @property
    def total_chunks(self) -> int:
        """Get total number of indexed chunks"""
        return self.index.ntotal if self.index else 0
