# AI Internal Policy Copilot

A portfolio-grade, production-style RAG (Retrieval-Augmented Generation) application for internal policy document Q&A.

## 🎯 Problem Statement

Organizations maintain numerous policy documents (HR policies, SOPs, compliance docs) that employees frequently need to reference. Finding specific information can be time-consuming. This AI Policy Copilot enables:

- **Instant Q&A**: Ask natural language questions about your policies
- **Source Citations**: Every answer includes references to source documents
- **Privacy-First**: Documents are processed locally, not sent to external services
- **Free to Run**: Works without API keys using extractive fallback mode

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        STREAMLIT UI                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
│  │  PDF Upload  │  │   Question   │  │  Answer + Citations  │  │
│  │   Sidebar    │  │    Input     │  │      Display         │  │
│  └──────────────┘  └──────────────┘  └──────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                        RAG PIPELINE                             │
│                                                                 │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────────┐ │
│  │  PDF     │──▶│  Text    │──▶│ Embedding│──▶│   FAISS      │ │
│  │  Loader  │   │  Chunker │   │  Model   │   │ Vector Store │ │
│  └──────────┘   └──────────┘   └──────────┘   └──────────────┘ │
│       │                                              │          │
│       │              INDEXING FLOW                   │          │
│       └──────────────────────────────────────────────┘          │
│                                                                 │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────────┐ │
│  │  Query   │──▶│ Embedding│──▶│ Retriever│──▶│   Answer     │ │
│  │  Input   │   │  Model   │   │  (Top-K) │   │  Generator   │ │
│  └──────────┘   └──────────┘   └──────────┘   └──────────────┘ │
│                                                     │           │
│                    QUERY FLOW                       ▼           │
│                                          ┌──────────────────┐   │
│                                          │  LLM (OpenAI)    │   │
│                                          │       or         │   │
│                                          │Extractive Fallback│  │
│                                          └──────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     LOCAL PERSISTENCE                           │
│  ┌──────────────────┐        ┌──────────────────────────────┐  │
│  │  FAISS Index     │        │  Chunk Metadata (JSON)       │  │
│  │  (faiss_index.bin)│       │  (metadata.json)             │  │
│  └──────────────────┘        └──────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| UI Framework | Streamlit |
| PDF Extraction | pypdf |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) |
| Vector Store | FAISS |
| LLM (optional) | OpenAI GPT-4.1-mini via Emergent LLM Key |
| Language | Python 3.10+ |

## 📁 Project Structure

```
ai-policy-copilot/
├── app.py                 # Main Streamlit application
├── config.py              # Configuration constants
├── rag/
│   ├── __init__.py        # RAG module exports
│   ├── pdf_loader.py      # PDF text extraction
│   ├── chunking.py        # Text chunking with overlap
│   ├── embedding.py       # Sentence-transformers embeddings
│   ├── vector_store.py    # FAISS vector store operations
│   ├── retriever.py       # Similarity search and retrieval
│   ├── generator.py       # LLM/extractive answer generation
│   └── utils.py           # Utility functions
├── data/
│   ├── uploads/           # Temporary upload storage
│   ├── index/             # Persisted FAISS index
│   └── sample_docs/       # Sample policy documents
├── requirements.txt       # Python dependencies
└── README.md              # This file
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
cd ai-policy-copilot
pip install -r requirements.txt
```

### 2. (Optional) Configure LLM

For AI-generated answers, set the Emergent LLM Key:

```bash
export EMERGENT_LLM_KEY=your-key-here
```

**Without an API key, the app works in extractive mode** - showing relevant document excerpts instead of AI-synthesized answers.

### 3. Run the Application

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`

## 📖 Usage

1. **Upload Documents**: Use the sidebar to upload PDF policy documents
2. **Index Documents**: Click "Index Documents" to process and vectorize
3. **Ask Questions**: Type your question in the input field
4. **View Answers**: Get answers with source citations

## 💬 Sample Questions

Once you've indexed your policy documents, try questions like:

- "What is the company's vacation policy?"
- "How do I submit an expense report?"
- "What are the guidelines for remote work?"
- "What is the approval process for purchase orders?"
- "How should I handle confidential information?"

## ⚙️ Configuration

Edit `config.py` to customize:

| Setting | Default | Description |
|---------|---------|-------------|
| `CHUNK_SIZE` | 1000 | Characters per chunk |
| `CHUNK_OVERLAP` | 150 | Overlap between chunks |
| `TOP_K` | 3 | Number of sources to retrieve |
| `EMBEDDING_MODEL` | all-MiniLM-L6-v2 | Sentence transformer model |
| `LLM_MODEL` | gpt-4.1-mini | OpenAI model for generation |

## 🎨 Features

### Premium UI
- Dark sidebar with light content area
- Gold accent colors for luxury feel
- Responsive chat interface
- Animated transitions

### RAG Pipeline
- Intelligent PDF text extraction
- Semantic chunking with overlap
- Efficient vector similarity search
- Context-aware answer generation

### Dual Mode Operation
- **LLM Mode**: Full AI-generated answers with strict context adherence
- **Extractive Mode**: Best-matching document excerpts (no API key needed)

## ⚠️ Limitations

1. **PDF Quality**: Text extraction depends on PDF quality (scanned documents may not work well)
2. **Language**: Optimized for English documents
3. **Context Length**: Very long documents may need multiple queries
4. **Memory**: Large document sets may require more RAM for FAISS

## 🔮 Future Improvements

1. **OCR Support**: Add OCR for scanned PDFs using pytesseract
2. **Multi-language**: Support for non-English documents
3. **Document Management**: Add/remove individual documents from index
4. **Conversation Memory**: Multi-turn conversations with context
5. **Export**: Export Q&A sessions as reports
6. **Authentication**: Add user authentication for enterprise use
7. **Analytics**: Track popular questions and gaps in documentation

## 📄 License

MIT License - Feel free to use for personal and commercial projects.

---

Built with ❤️ using RAG technology for intelligent document understanding.
