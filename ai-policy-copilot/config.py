"""
Configuration settings for AI Policy Copilot
"""
import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
UPLOADS_DIR = DATA_DIR / "uploads"
INDEX_DIR = DATA_DIR / "index"
SAMPLE_DOCS_DIR = DATA_DIR / "sample_docs"

# Ensure directories exist
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
INDEX_DIR.mkdir(parents=True, exist_ok=True)
SAMPLE_DOCS_DIR.mkdir(parents=True, exist_ok=True)

# Chunking settings
CHUNK_SIZE = 1000  # characters (~500-800 tokens)
CHUNK_OVERLAP = 150  # ~15% overlap

# Embedding model
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Retrieval settings
TOP_K = 3

# Vector store paths
FAISS_INDEX_PATH = INDEX_DIR / "faiss_index.bin"
METADATA_PATH = INDEX_DIR / "metadata.json"

# LLM settings
LLM_PROVIDER = "openai"
LLM_MODEL = "gpt-4.1-mini"  # Cost-effective for RAG

# Environment variable for API key
EMERGENT_LLM_KEY = os.environ.get("EMERGENT_LLM_KEY", "")
