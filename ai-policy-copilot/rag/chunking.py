"""
Text chunking module
"""
from typing import List, Dict
import re


class TextChunker:
    """Split text into overlapping chunks for RAG"""
    
    def __init__(self, chunk_size: int = 1000, overlap: int = 150):
        """
        Initialize chunker
        
        Args:
            chunk_size: Target size of each chunk in characters
            overlap: Number of overlapping characters between chunks
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def chunk_document(self, doc_data: Dict[str, any]) -> List[Dict[str, any]]:
        """
        Split a document into chunks with metadata
        
        Args:
            doc_data: Dictionary with doc_name, pages, full_text
            
        Returns:
            List of chunk dictionaries with text and metadata
        """
        chunks = []
        doc_name = doc_data.get("doc_name", "unknown")
        
        # If we have page-level data, chunk per page for better source tracking
        if doc_data.get("pages"):
            for page_data in doc_data["pages"]:
                page_chunks = self._chunk_text(
                    page_data["text"],
                    doc_name,
                    page_data["page_num"]
                )
                chunks.extend(page_chunks)
        else:
            # Fall back to full text chunking
            full_text = doc_data.get("full_text", "")
            if full_text:
                chunks = self._chunk_text(full_text, doc_name, None)
        
        # Assign unique chunk IDs
        for i, chunk in enumerate(chunks):
            chunk["chunk_id"] = f"{doc_name}_{i}"
        
        return chunks
    
    def _chunk_text(
        self, 
        text: str, 
        doc_name: str, 
        page_num: int = None
    ) -> List[Dict[str, any]]:
        """
        Split text into overlapping chunks
        
        Args:
            text: Text to chunk
            doc_name: Document name for metadata
            page_num: Page number (if available)
            
        Returns:
            List of chunk dictionaries
        """
        if not text or len(text.strip()) == 0:
            return []
        
        chunks = []
        
        # Try to split on sentence boundaries first
        sentences = self._split_into_sentences(text)
        
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence_len = len(sentence)
            
            if current_length + sentence_len <= self.chunk_size:
                current_chunk.append(sentence)
                current_length += sentence_len + 1  # +1 for space
            else:
                # Save current chunk
                if current_chunk:
                    chunk_text = " ".join(current_chunk)
                    chunks.append({
                        "text": chunk_text,
                        "doc_name": doc_name,
                        "page_num": page_num,
                        "char_count": len(chunk_text)
                    })
                
                # Start new chunk with overlap
                overlap_sentences = self._get_overlap_sentences(
                    current_chunk, 
                    self.overlap
                )
                current_chunk = overlap_sentences + [sentence]
                current_length = sum(len(s) + 1 for s in current_chunk)
        
        # Don't forget the last chunk
        if current_chunk:
            chunk_text = " ".join(current_chunk)
            chunks.append({
                "text": chunk_text,
                "doc_name": doc_name,
                "page_num": page_num,
                "char_count": len(chunk_text)
            })
        
        return chunks
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        # Simple sentence splitting
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _get_overlap_sentences(
        self, 
        sentences: List[str], 
        target_chars: int
    ) -> List[str]:
        """Get sentences for overlap from the end"""
        overlap = []
        char_count = 0
        
        for sentence in reversed(sentences):
            if char_count + len(sentence) <= target_chars:
                overlap.insert(0, sentence)
                char_count += len(sentence) + 1
            else:
                break
        
        return overlap
