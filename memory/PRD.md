# AI Internal Policy Copilot - PRD

## Original Problem Statement
Build a portfolio-grade, production-style "AI Internal Policy Copilot" as a lightweight full-stack app using Retrieval-Augmented Generation (RAG). Create an internal assistant that lets a user upload policy/SOP PDFs and ask questions. The app must answer ONLY using retrieved document context and show the supporting source snippets (citations).

## User Personas
1. **HR Managers** - Need quick access to policy information for employee queries
2. **Compliance Officers** - Verify policy adherence and find specific regulations
3. **Employees** - Self-service access to company policies without HR contact

## Core Requirements (Static)
- PDF document upload and text extraction
- Intelligent text chunking with overlap for context preservation
- Vector embeddings using sentence-transformers (free, local)
- FAISS vector store for similarity search
- LLM-powered answer generation (OpenAI via Emergent LLM Key)
- Extractive fallback mode when LLM unavailable
- Source citations with relevance scores
- Premium dark/light mixed UI theme

## What's Been Implemented (Jan 2026)
- [x] Complete Streamlit UI with premium styling (dark sidebar, gold accents)
- [x] PDF upload with drag-drop interface
- [x] Document indexing with progress indicators
- [x] FAISS vector store with local persistence
- [x] sentence-transformers embeddings (all-MiniLM-L6-v2)
- [x] OpenAI GPT integration via Emergent LLM Key
- [x] Extractive fallback for non-LLM mode
- [x] Source citations with relevance percentages
- [x] Clear conversation functionality
- [x] Index status indicators
- [x] Sample policy PDF included
- [x] **User Authentication** (register/login with email/password)
- [x] **Audit Trail Logging** (tracks all queries, doc uploads, logins)
- [x] **Compliance Dashboard** (view all user activities for audit)

## Tech Stack
- **UI**: Streamlit
- **PDF**: pypdf
- **Embeddings**: sentence-transformers
- **Vector Store**: FAISS
- **LLM**: OpenAI GPT-4.1-mini (via Emergent)
- **Auth**: JSON-based user storage with SHA-256 password hashing

## Prioritized Backlog

### P0 (Critical)
- All core features implemented ✓
- Authentication & audit trail ✓

### P1 (Important)
- OCR support for scanned PDFs
- Multi-language document support
- Document management (add/remove individual docs)

### P2 (Nice to Have)
- Conversation memory for follow-up questions
- Export Q&A sessions as reports
- Admin role with elevated permissions
- Analytics charts/visualizations

## Next Tasks
1. Add OCR support using pytesseract
2. Implement conversation memory
3. Add document deletion from index
4. Create analytics charts for usage visualization
5. Add admin role with user management capabilities
