"""
AI Internal Policy Copilot - Main Streamlit Application
A premium RAG-based document Q&A system with user authentication and audit trails
"""
import streamlit as st
from pathlib import Path
import os
from dotenv import load_dotenv
from datetime import datetime

# Load environment variables
load_dotenv()

# Import RAG components
from rag.pdf_loader import PDFLoader
from rag.chunking import TextChunker
from rag.embedding import EmbeddingModel
from rag.vector_store import VectorStore
from rag.retriever import Retriever
from rag.generator import AnswerGenerator
from rag.utils import truncate_text, clean_filename
from rag.auth import (
    register_user, authenticate_user, log_query, 
    log_document_indexed, get_audit_log, get_user_stats, get_admin_stats
)
import config

# Page configuration
st.set_page_config(
    page_title="Policy Copilot",
    page_icon="📋",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Premium CSS styling
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&family=Playfair+Display:wght@400;600;700&display=swap');
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Global styles */
    .stApp {
        font-family: 'Inter', sans-serif;
    }
    
    /* Luxury Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0A0E17 0%, #111827 100%);
        border-right: 1px solid #1F2937;
    }
    
    [data-testid="stSidebar"] [data-testid="stMarkdown"] {
        color: #E5E7EB;
    }
    
    [data-testid="stSidebar"] .stButton button {
        background: linear-gradient(135deg, #D4AF37 0%, #B5952F 100%);
        color: #0A0E17;
        font-weight: 600;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 1.5rem;
        transition: all 0.3s ease;
        width: 100%;
    }
    
    [data-testid="stSidebar"] .stButton button:hover {
        background: linear-gradient(135deg, #F3E5AB 0%, #D4AF37 100%);
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(212, 175, 55, 0.3);
    }
    
    /* Main content area */
    .main .block-container {
        padding: 2rem 3rem;
        max-width: 1200px;
    }
    
    /* Title styling */
    .premium-title {
        font-family: 'Playfair Display', serif;
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #D4AF37 0%, #F3E5AB 50%, #B5952F 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.5rem;
    }
    
    .subtitle {
        color: #6B7280;
        font-size: 1rem;
        font-weight: 400;
        margin-bottom: 2rem;
    }
    
    /* Chat messages */
    .user-message {
        background: linear-gradient(135deg, #0A0E17 0%, #1F2937 100%);
        color: #FFFFFF;
        border-radius: 16px 16px 4px 16px;
        padding: 1.25rem 1.5rem;
        margin: 1rem 0;
        max-width: 85%;
        margin-left: auto;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
        font-size: 0.95rem;
        line-height: 1.6;
    }
    
    .ai-message {
        background: #FFFFFF;
        color: #111827;
        border: 1px solid #E5E7EB;
        border-left: 4px solid #D4AF37;
        border-radius: 16px 16px 16px 4px;
        padding: 1.25rem 1.5rem;
        margin: 1rem 0;
        max-width: 85%;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
        font-size: 0.95rem;
        line-height: 1.7;
    }
    
    /* Citation cards */
    .citation-card {
        background: #FFFFFF;
        border: 1px solid #E5E7EB;
        border-left: 3px solid #D4AF37;
        border-radius: 12px;
        padding: 1rem 1.25rem;
        margin: 0.75rem 0;
        transition: all 0.2s ease;
    }
    
    .citation-card:hover {
        border-color: #D4AF37;
        box-shadow: 0 4px 12px rgba(212, 175, 55, 0.1);
        transform: translateX(4px);
    }
    
    .citation-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 0.5rem;
    }
    
    .citation-source {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.75rem;
        color: #D4AF37;
        font-weight: 500;
    }
    
    .citation-score {
        background: linear-gradient(135deg, #D4AF37 0%, #B5952F 100%);
        color: #0A0E17;
        font-size: 0.7rem;
        font-weight: 600;
        padding: 0.25rem 0.5rem;
        border-radius: 20px;
    }
    
    .citation-text {
        color: #374151;
        font-size: 0.85rem;
        line-height: 1.6;
    }
    
    /* Status indicators */
    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 500;
    }
    
    .status-success {
        background: rgba(16, 185, 129, 0.1);
        color: #10B981;
        border: 1px solid rgba(16, 185, 129, 0.2);
    }
    
    .status-warning {
        background: rgba(245, 158, 11, 0.1);
        color: #F59E0B;
        border: 1px solid rgba(245, 158, 11, 0.2);
    }
    
    .status-info {
        background: rgba(212, 175, 55, 0.1);
        color: #D4AF37;
        border: 1px solid rgba(212, 175, 55, 0.2);
    }
    
    /* Document list */
    .doc-item {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid #1F2937;
        border-radius: 8px;
        padding: 0.75rem 1rem;
        margin: 0.5rem 0;
        display: flex;
        align-items: center;
        gap: 0.75rem;
    }
    
    .doc-icon {
        color: #D4AF37;
        font-size: 1.25rem;
    }
    
    .doc-name {
        color: #E5E7EB;
        font-size: 0.85rem;
        font-weight: 500;
    }
    
    /* Empty state */
    .empty-state {
        text-align: center;
        padding: 4rem 2rem;
        color: #6B7280;
    }
    
    .empty-state-icon {
        font-size: 4rem;
        margin-bottom: 1rem;
        opacity: 0.5;
    }
    
    .empty-state-title {
        font-family: 'Playfair Display', serif;
        font-size: 1.5rem;
        color: #374151;
        margin-bottom: 0.5rem;
    }
    
    /* Input styling */
    .stTextInput input {
        border: 2px solid #E5E7EB;
        border-radius: 12px;
        padding: 1rem;
        font-size: 1rem;
        transition: all 0.2s ease;
    }
    
    .stTextInput input:focus {
        border-color: #D4AF37;
        box-shadow: 0 0 0 3px rgba(212, 175, 55, 0.1);
    }
    
    /* Spinner */
    .stSpinner > div {
        border-color: #D4AF37 transparent transparent transparent;
    }
    
    /* Divider */
    .gold-divider {
        height: 2px;
        background: linear-gradient(90deg, transparent 0%, #D4AF37 50%, transparent 100%);
        margin: 2rem 0;
        border: none;
    }
    
    /* Mode indicator */
    .mode-badge {
        display: inline-block;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.7rem;
        padding: 0.25rem 0.75rem;
        border-radius: 4px;
        margin-top: 0.5rem;
    }
    
    .mode-llm {
        background: rgba(16, 185, 129, 0.1);
        color: #10B981;
    }
    
    .mode-extractive {
        background: rgba(245, 158, 11, 0.1);
        color: #F59E0B;
    }
    
    /* Auth forms */
    .auth-container {
        max-width: 400px;
        margin: 4rem auto;
        padding: 2rem;
        background: white;
        border-radius: 16px;
        box-shadow: 0 4px 24px rgba(0, 0, 0, 0.1);
    }
    
    .auth-title {
        font-family: 'Playfair Display', serif;
        font-size: 1.75rem;
        text-align: center;
        margin-bottom: 1.5rem;
        color: #111827;
    }
    
    /* User info card */
    .user-card {
        background: rgba(212, 175, 55, 0.1);
        border: 1px solid rgba(212, 175, 55, 0.3);
        border-radius: 12px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    .user-name {
        font-weight: 600;
        color: #D4AF37;
        font-size: 0.95rem;
    }
    
    .user-email {
        color: #9CA3AF;
        font-size: 0.8rem;
    }
    
    /* Audit log styles */
    .audit-entry {
        background: #FFFFFF;
        border: 1px solid #E5E7EB;
        border-radius: 8px;
        padding: 0.75rem 1rem;
        margin: 0.5rem 0;
    }
    
    .audit-action {
        font-weight: 600;
        color: #111827;
    }
    
    .audit-time {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.7rem;
        color: #6B7280;
    }
    
    .audit-user {
        color: #D4AF37;
        font-size: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)


def initialize_session_state():
    """Initialize session state variables"""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "indexed_docs" not in st.session_state:
        st.session_state.indexed_docs = []
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None
    if "embedding_model" not in st.session_state:
        st.session_state.embedding_model = None
    if "is_indexing" not in st.session_state:
        st.session_state.is_indexing = False
    if "uploaded_files_data" not in st.session_state:
        st.session_state.uploaded_files_data = []
    # Auth state
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    if "user" not in st.session_state:
        st.session_state.user = None
    if "show_register" not in st.session_state:
        st.session_state.show_register = False
    if "show_audit" not in st.session_state:
        st.session_state.show_audit = False


def get_or_create_embedding_model():
    """Get or create the embedding model"""
    if st.session_state.embedding_model is None:
        with st.spinner("Loading embedding model..."):
            st.session_state.embedding_model = EmbeddingModel(config.EMBEDDING_MODEL)
    return st.session_state.embedding_model


def get_or_load_vector_store():
    """Get or load the vector store"""
    if st.session_state.vector_store is None:
        vector_store = VectorStore(
            index_path=config.FAISS_INDEX_PATH,
            metadata_path=config.METADATA_PATH
        )
        if vector_store.load():
            st.session_state.vector_store = vector_store
        else:
            st.session_state.vector_store = vector_store
    return st.session_state.vector_store


def process_and_index_documents(uploaded_files):
    """Process uploaded PDFs and create vector index"""
    st.session_state.is_indexing = True
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        embedding_model = get_or_create_embedding_model()
        chunker = TextChunker(
            chunk_size=config.CHUNK_SIZE,
            overlap=config.CHUNK_OVERLAP
        )
        
        all_chunks = []
        indexed_docs = []
        
        for i, uploaded_file in enumerate(uploaded_files):
            status_text.markdown(f"📄 Processing: **{uploaded_file.name}**")
            progress_bar.progress((i) / len(uploaded_files) * 0.5)
            
            file_bytes = uploaded_file.read()
            doc_data = PDFLoader.extract_from_bytes(file_bytes, uploaded_file.name)
            
            if doc_data.get("error"):
                st.warning(f"⚠️ Could not process {uploaded_file.name}: {doc_data['error']}")
                continue
            
            if not doc_data.get("full_text"):
                st.warning(f"⚠️ No text extracted from {uploaded_file.name}")
                continue
            
            chunks = chunker.chunk_document(doc_data)
            all_chunks.extend(chunks)
            indexed_docs.append({
                "name": uploaded_file.name,
                "pages": doc_data.get("total_pages", 0),
                "chunks": len(chunks)
            })
            
            # Log document indexing
            if st.session_state.user:
                log_document_indexed(
                    user_id=st.session_state.user["id"],
                    user_email=st.session_state.user["email"],
                    user_name=st.session_state.user["name"],
                    doc_name=uploaded_file.name,
                    chunks_count=len(chunks)
                )
        
        if not all_chunks:
            st.error("❌ No text could be extracted from the uploaded documents.")
            return False
        
        status_text.markdown("🧠 Generating embeddings...")
        progress_bar.progress(0.6)
        
        chunk_texts = [chunk["text"] for chunk in all_chunks]
        embeddings = embedding_model.embed_texts(chunk_texts)
        
        status_text.markdown("📊 Building vector index...")
        progress_bar.progress(0.8)
        
        vector_store = VectorStore(
            index_path=config.FAISS_INDEX_PATH,
            metadata_path=config.METADATA_PATH,
            dimension=embedding_model.dimension
        )
        vector_store.create_index(embeddings, all_chunks)
        vector_store.save()
        
        st.session_state.vector_store = vector_store
        st.session_state.indexed_docs = indexed_docs
        
        progress_bar.progress(1.0)
        status_text.markdown("✅ **Indexing complete!**")
        
        return True
        
    except Exception as e:
        st.error(f"❌ Error during indexing: {str(e)}")
        return False
    finally:
        st.session_state.is_indexing = False


def ask_question(query: str):
    """Process a user question and generate an answer"""
    embedding_model = get_or_create_embedding_model()
    vector_store = get_or_load_vector_store()
    
    if vector_store.is_empty:
        return {
            "answer": "I don't have any documents indexed yet. Please upload and index some PDF documents first.",
            "mode": "no_index",
            "sources": []
        }
    
    retriever = Retriever(
        embedding_model=embedding_model,
        vector_store=vector_store,
        top_k=config.TOP_K
    )
    
    context_chunks = retriever.retrieve(query)
    
    generator = AnswerGenerator(api_key=config.EMERGENT_LLM_KEY)
    result = generator.generate(query, context_chunks)
    
    # Log query for audit trail
    if st.session_state.user:
        log_query(
            user_id=st.session_state.user["id"],
            user_email=st.session_state.user["email"],
            user_name=st.session_state.user["name"],
            query=query,
            answer_mode=result["mode"],
            sources_count=len(result["sources"])
        )
    
    return result


def render_auth_page():
    """Render the authentication page"""
    st.markdown("""
    <div style="text-align: center; padding: 2rem 0;">
        <div style="font-size: 4rem; margin-bottom: 1rem;">📋</div>
        <h1 class="premium-title">Policy Copilot</h1>
        <p class="subtitle">AI-Powered Document Assistant with Compliance Tracking</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        if st.session_state.show_register:
            render_register_form()
        else:
            render_login_form()


def render_login_form():
    """Render login form"""
    st.markdown("### Sign In")
    
    with st.form("login_form"):
        email = st.text_input("Email", placeholder="your@email.com")
        password = st.text_input("Password", type="password", placeholder="••••••••")
        
        col1, col2 = st.columns(2)
        with col1:
            submitted = st.form_submit_button("Sign In", use_container_width=True)
        with col2:
            if st.form_submit_button("Create Account", use_container_width=True):
                st.session_state.show_register = True
                st.rerun()
        
        if submitted:
            if email and password:
                result = authenticate_user(email, password)
                if result["success"]:
                    st.session_state.authenticated = True
                    st.session_state.user = result["user"]
                    st.success("Welcome back!")
                    st.rerun()
                else:
                    st.error(result["message"])
            else:
                st.warning("Please enter email and password")
    
    # Demo account hint
    st.markdown("""
    <div style="text-align: center; margin-top: 2rem; padding: 1rem; background: #F9FAFB; border-radius: 8px;">
        <p style="color: #6B7280; font-size: 0.85rem; margin: 0;">
            New user? Create an account to track your policy queries.
        </p>
    </div>
    """, unsafe_allow_html=True)


def render_register_form():
    """Render registration form"""
    st.markdown("### Create Account")
    
    with st.form("register_form"):
        name = st.text_input("Full Name", placeholder="John Doe")
        email = st.text_input("Email", placeholder="your@email.com")
        department = st.text_input("Department", placeholder="Engineering (optional)")
        password = st.text_input("Password", type="password", placeholder="Min 6 characters")
        confirm_password = st.text_input("Confirm Password", type="password")
        
        col1, col2 = st.columns(2)
        with col1:
            submitted = st.form_submit_button("Create Account", use_container_width=True)
        with col2:
            if st.form_submit_button("Back to Login", use_container_width=True):
                st.session_state.show_register = False
                st.rerun()
        
        if submitted:
            if not name or not email or not password:
                st.warning("Please fill in all required fields")
            elif password != confirm_password:
                st.error("Passwords do not match")
            elif len(password) < 6:
                st.error("Password must be at least 6 characters")
            else:
                result = register_user(email, password, name, department)
                if result["success"]:
                    st.success("Account created! Please sign in.")
                    st.session_state.show_register = False
                    st.rerun()
                else:
                    st.error(result["message"])


def render_sidebar():
    """Render the sidebar with upload and controls"""
    with st.sidebar:
        # User info
        if st.session_state.user:
            st.markdown(f"""
            <div class="user-card">
                <div class="user-name">👤 {st.session_state.user['name']}</div>
                <div class="user-email">{st.session_state.user['email']}</div>
                {f"<div style='font-size: 0.75rem; color: #6B7280; margin-top: 0.25rem;'>{st.session_state.user.get('department', '')}</div>" if st.session_state.user.get('department') else ""}
            </div>
            """, unsafe_allow_html=True)
        
        # Logo/Title
        st.markdown("""
        <div style="text-align: center; padding: 1rem 0;">
            <div style="font-size: 2rem; margin-bottom: 0.25rem;">📋</div>
            <div style="font-family: 'Playfair Display', serif; font-size: 1.25rem; font-weight: 700; color: #D4AF37;">
                Policy Copilot
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<hr style='border-color: #1F2937; margin: 0.5rem 0;'>", unsafe_allow_html=True)
        
        # File uploader
        st.markdown("""
        <div style="color: #E5E7EB; font-weight: 500; margin-bottom: 0.5rem; font-size: 0.9rem;">
            📁 Upload Documents
        </div>
        """, unsafe_allow_html=True)
        
        uploaded_files = st.file_uploader(
            "Upload PDF files",
            type=["pdf"],
            accept_multiple_files=True,
            label_visibility="collapsed",
            key="pdf_uploader"
        )
        
        if uploaded_files:
            st.markdown("""
            <div style="color: #9CA3AF; font-size: 0.75rem; margin: 0.5rem 0;">
                UPLOADED FILES
            </div>
            """, unsafe_allow_html=True)
            
            for file in uploaded_files:
                st.markdown(f"""
                <div class="doc-item">
                    <span class="doc-icon">📄</span>
                    <span class="doc-name">{file.name}</span>
                </div>
                """, unsafe_allow_html=True)
        
        if uploaded_files:
            if st.button("🔄 Index Documents", key="index_btn", use_container_width=True):
                with st.spinner(""):
                    success = process_and_index_documents(uploaded_files)
                    if success:
                        st.success("✅ Documents indexed!")
                        st.rerun()
        
        st.markdown("<hr style='border-color: #1F2937; margin: 1rem 0;'>", unsafe_allow_html=True)
        
        # Index status
        vector_store = get_or_load_vector_store()
        
        st.markdown("""
        <div style="color: #E5E7EB; font-weight: 500; margin-bottom: 0.5rem; font-size: 0.85rem;">
            📊 Index Status
        </div>
        """, unsafe_allow_html=True)
        
        if not vector_store.is_empty:
            st.markdown(f"""
            <div class="status-badge status-success">
                <span>●</span> {vector_store.total_chunks} chunks indexed
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="status-badge status-warning">
                <span>○</span> No documents indexed
            </div>
            """, unsafe_allow_html=True)
        
        # LLM status
        generator = AnswerGenerator()
        if generator.has_llm:
            st.markdown("""
            <div class="status-badge status-success">
                <span>⚡</span> LLM Active
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="status-badge status-info">
                <span>📝</span> Extractive Mode
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("<hr style='border-color: #1F2937; margin: 1rem 0;'>", unsafe_allow_html=True)
        
        # Audit log toggle
        if st.button("📜 View Audit Trail", key="audit_btn", use_container_width=True):
            st.session_state.show_audit = not st.session_state.show_audit
            st.rerun()
        
        # Clear conversation
        if st.button("🗑️ Clear Conversation", key="clear_btn", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
        
        # Logout
        st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)
        if st.button("🚪 Sign Out", key="logout_btn", use_container_width=True):
            st.session_state.authenticated = False
            st.session_state.user = None
            st.session_state.messages = []
            st.rerun()


def render_audit_trail():
    """Render audit trail view"""
    st.markdown("""
    <h1 class="premium-title">Audit Trail</h1>
    <p class="subtitle">Track all policy queries and document activities for compliance.</p>
    """, unsafe_allow_html=True)
    
    # User stats
    if st.session_state.user:
        stats = get_user_stats(st.session_state.user["id"])
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Your Queries", stats["total_queries"])
        with col2:
            st.metric("Docs Indexed", stats["documents_indexed"])
        with col3:
            if stats["last_activity"]:
                last_time = datetime.fromisoformat(stats["last_activity"].replace('Z', '+00:00'))
                st.metric("Last Activity", last_time.strftime("%b %d, %H:%M"))
            else:
                st.metric("Last Activity", "N/A")
    
    st.markdown("<div class='gold-divider'></div>", unsafe_allow_html=True)
    
    # Filter options
    col1, col2 = st.columns([2, 1])
    with col1:
        filter_action = st.selectbox(
            "Filter by action",
            ["All Actions", "POLICY_QUERY", "DOCUMENT_INDEXED", "USER_LOGIN", "USER_REGISTERED"],
            label_visibility="collapsed"
        )
    with col2:
        limit = st.selectbox("Show", [25, 50, 100], label_visibility="collapsed")
    
    # Get audit log
    action_filter = None if filter_action == "All Actions" else filter_action
    audit_entries = get_audit_log(limit=limit, action_type=action_filter)
    
    if not audit_entries:
        st.info("No audit entries found.")
        return
    
    # Display entries
    for entry in audit_entries:
        timestamp = datetime.fromisoformat(entry["timestamp"].replace('Z', '+00:00'))
        
        action_icons = {
            "POLICY_QUERY": "🔍",
            "DOCUMENT_INDEXED": "📄",
            "USER_LOGIN": "🔐",
            "USER_REGISTERED": "👤"
        }
        icon = action_icons.get(entry["action"], "📝")
        
        with st.container():
            st.markdown(f"""
            <div class="audit-entry">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <span style="font-size: 1.1rem;">{icon}</span>
                        <span class="audit-action">{entry["action"].replace("_", " ").title()}</span>
                        <span class="audit-user">by {entry["user_name"]}</span>
                    </div>
                    <span class="audit-time">{timestamp.strftime("%Y-%m-%d %H:%M:%S")}</span>
                </div>
                {f'<div style="margin-top: 0.5rem; padding-left: 1.5rem; color: #6B7280; font-size: 0.85rem;">{entry["details"].get("query", entry["details"].get("document_name", ""))}</div>' if entry["details"] else ''}
            </div>
            """, unsafe_allow_html=True)
    
    # Back button
    st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)
    if st.button("← Back to Chat", key="back_btn"):
        st.session_state.show_audit = False
        st.rerun()


def render_main_content():
    """Render the main chat interface"""
    st.markdown("""
    <h1 class="premium-title">Ask Your Policy Questions</h1>
    <p class="subtitle">Upload your policy documents and get instant, accurate answers with source citations.</p>
    """, unsafe_allow_html=True)
    
    chat_container = st.container()
    
    with chat_container:
        if not st.session_state.messages:
            st.markdown("""
            <div class="empty-state">
                <div class="empty-state-icon">📚</div>
                <div class="empty-state-title">Ready to Assist</div>
                <p>Upload your policy documents in the sidebar, click "Index Documents", then ask any question.</p>
                <div style="margin-top: 2rem; display: flex; justify-content: center; gap: 1rem; flex-wrap: wrap;">
                    <div style="background: #F9FAFB; border: 1px solid #E5E7EB; border-radius: 8px; padding: 0.75rem 1rem; font-size: 0.85rem; color: #6B7280;">
                        💡 "What is our vacation policy?"
                    </div>
                    <div style="background: #F9FAFB; border: 1px solid #E5E7EB; border-radius: 8px; padding: 0.75rem 1rem; font-size: 0.85rem; color: #6B7280;">
                        💡 "How do I submit an expense report?"
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            for message in st.session_state.messages:
                if message["role"] == "user":
                    st.markdown(f"""
                    <div class="user-message">{message["content"]}</div>
                    """, unsafe_allow_html=True)
                else:
                    answer = message.get("answer", message.get("content", ""))
                    mode = message.get("mode", "unknown")
                    sources = message.get("sources", [])
                    
                    mode_badge = ""
                    if mode == "llm":
                        mode_badge = '<span class="mode-badge mode-llm">AI Generated</span>'
                    elif mode == "extractive":
                        mode_badge = '<span class="mode-badge mode-extractive">Extractive</span>'
                    
                    st.markdown(f"""
                    <div class="ai-message">
                        {answer}
                        {mode_badge}
                    </div>
                    """, unsafe_allow_html=True)
                    
                    if sources and mode == "llm":
                        with st.expander("📚 View Source Citations", expanded=False):
                            for i, source in enumerate(sources, 1):
                                doc_name = source.get("doc_name", "Unknown")
                                page_num = source.get("page_num")
                                page_info = f" • Page {page_num}" if page_num else ""
                                score = source.get("relevance_score", 0)
                                text_preview = truncate_text(source.get("text", ""), 300)
                                
                                st.markdown(f"""
                                <div class="citation-card">
                                    <div class="citation-header">
                                        <span class="citation-source">Source {i}: {doc_name}{page_info}</span>
                                        <span class="citation-score">{score:.0%} match</span>
                                    </div>
                                    <div class="citation-text">{text_preview}</div>
                                </div>
                                """, unsafe_allow_html=True)
    
    st.markdown("<div class='gold-divider'></div>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([6, 1])
    
    with col1:
        user_question = st.text_input(
            "Ask a question",
            placeholder="Type your question about the policy documents...",
            label_visibility="collapsed",
            key="question_input"
        )
    
    with col2:
        ask_clicked = st.button("Ask", key="ask_btn", use_container_width=True)
    
    if (ask_clicked or user_question) and user_question:
        if not st.session_state.messages or st.session_state.messages[-1].get("content") != user_question:
            st.session_state.messages.append({
                "role": "user",
                "content": user_question
            })
            
            with st.spinner("Thinking..."):
                result = ask_question(user_question)
            
            st.session_state.messages.append({
                "role": "assistant",
                "answer": result["answer"],
                "mode": result["mode"],
                "sources": result["sources"]
            })
            
            st.rerun()


def main():
    """Main application entry point"""
    initialize_session_state()
    
    # Check authentication
    if not st.session_state.authenticated:
        render_auth_page()
        return
    
    # Render authenticated app
    render_sidebar()
    
    if st.session_state.show_audit:
        render_audit_trail()
    else:
        render_main_content()


if __name__ == "__main__":
    main()
