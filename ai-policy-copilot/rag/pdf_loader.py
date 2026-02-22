"""
PDF text extraction module
"""
import re
from pathlib import Path
from typing import List, Dict, Optional
from pypdf import PdfReader


class PDFLoader:
    """Extract text from PDF files"""
    
    @staticmethod
    def extract_text(file_path: Path) -> Dict[str, any]:
        """
        Extract text from a PDF file
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            Dictionary with doc_name, pages, and full_text
        """
        try:
            reader = PdfReader(str(file_path))
            pages = []
            full_text = []
            
            for page_num, page in enumerate(reader.pages, 1):
                text = page.extract_text() or ""
                text = PDFLoader._clean_text(text)
                if text.strip():
                    pages.append({
                        "page_num": page_num,
                        "text": text
                    })
                    full_text.append(text)
            
            return {
                "doc_name": file_path.name,
                "pages": pages,
                "full_text": "\n\n".join(full_text),
                "total_pages": len(reader.pages)
            }
        except Exception as e:
            return {
                "doc_name": file_path.name if file_path else "unknown",
                "pages": [],
                "full_text": "",
                "total_pages": 0,
                "error": str(e)
            }
    
    @staticmethod
    def extract_from_bytes(file_bytes: bytes, filename: str) -> Dict[str, any]:
        """
        Extract text from PDF bytes (for uploaded files)
        
        Args:
            file_bytes: PDF file content as bytes
            filename: Name of the file
            
        Returns:
            Dictionary with doc_name, pages, and full_text
        """
        import io
        try:
            reader = PdfReader(io.BytesIO(file_bytes))
            pages = []
            full_text = []
            
            for page_num, page in enumerate(reader.pages, 1):
                text = page.extract_text() or ""
                text = PDFLoader._clean_text(text)
                if text.strip():
                    pages.append({
                        "page_num": page_num,
                        "text": text
                    })
                    full_text.append(text)
            
            return {
                "doc_name": filename,
                "pages": pages,
                "full_text": "\n\n".join(full_text),
                "total_pages": len(reader.pages)
            }
        except Exception as e:
            return {
                "doc_name": filename,
                "pages": [],
                "full_text": "",
                "total_pages": 0,
                "error": str(e)
            }
    
    @staticmethod
    def _clean_text(text: str) -> str:
        """Clean extracted text"""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters that might cause issues
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', text)
        return text.strip()
