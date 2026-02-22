"""
Retriever module for document search
"""
from typing import List, Dict
from .embedding import EmbeddingModel
from .vector_store import VectorStore


class Retriever:
    """Retrieve relevant document chunks for a query"""
    
    def __init__(
        self, 
        embedding_model: EmbeddingModel, 
        vector_store: VectorStore,
        top_k: int = 3
    ):
        """
        Initialize retriever
        
        Args:
            embedding_model: Embedding model instance
            vector_store: Vector store instance
            top_k: Number of results to retrieve
        """
        self.embedding_model = embedding_model
        self.vector_store = vector_store
        self.top_k = top_k
    
    def retrieve(self, query: str) -> List[Dict]:
        """
        Retrieve relevant chunks for a query
        
        Args:
            query: User's question
            
        Returns:
            List of chunk dictionaries with relevance scores
        """
        if self.vector_store.is_empty:
            return []
        
        # Embed the query
        query_embedding = self.embedding_model.embed_query(query)
        
        # Search vector store
        results = self.vector_store.search(query_embedding, self.top_k)
        
        # Format results
        retrieved_chunks = []
        for metadata, distance in results:
            chunk = {
                "text": metadata.get("text", ""),
                "doc_name": metadata.get("doc_name", "unknown"),
                "page_num": metadata.get("page_num"),
                "chunk_id": metadata.get("chunk_id", ""),
                "relevance_score": self._distance_to_score(distance)
            }
            retrieved_chunks.append(chunk)
        
        return retrieved_chunks
    
    def _distance_to_score(self, distance: float) -> float:
        """
        Convert L2 distance to a relevance score (0-1)
        Lower distance = higher relevance
        """
        # Using exponential decay for score calculation
        # This gives scores between 0 and 1
        return 1.0 / (1.0 + distance)
