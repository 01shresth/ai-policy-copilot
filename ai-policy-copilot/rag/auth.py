"""
Authentication and audit trail module for AI Policy Copilot
"""
import json
import hashlib
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional, List


# Storage paths
AUTH_DIR = Path(__file__).parent / "data" / "auth"
USERS_FILE = AUTH_DIR / "users.json"
AUDIT_LOG_FILE = AUTH_DIR / "audit_log.json"

# Ensure directory exists
AUTH_DIR.mkdir(parents=True, exist_ok=True)


def _hash_password(password: str) -> str:
    """Hash password using SHA-256"""
    return hashlib.sha256(password.encode()).hexdigest()


def _load_users() -> Dict:
    """Load users from file"""
    if USERS_FILE.exists():
        with open(USERS_FILE, 'r') as f:
            return json.load(f)
    return {}


def _save_users(users: Dict) -> None:
    """Save users to file"""
    with open(USERS_FILE, 'w') as f:
        json.dump(users, f, indent=2)


def _load_audit_log() -> List[Dict]:
    """Load audit log from file"""
    if AUDIT_LOG_FILE.exists():
        with open(AUDIT_LOG_FILE, 'r') as f:
            return json.load(f)
    return []


def _save_audit_log(log: List[Dict]) -> None:
    """Save audit log to file"""
    with open(AUDIT_LOG_FILE, 'w') as f:
        json.dump(log, f, indent=2)


def register_user(email: str, password: str, name: str, department: str = "") -> Dict:
    """
    Register a new user
    
    Returns:
        Dict with success status and message
    """
    users = _load_users()
    
    if email.lower() in users:
        return {"success": False, "message": "Email already registered"}
    
    if len(password) < 6:
        return {"success": False, "message": "Password must be at least 6 characters"}
    
    user_id = str(uuid.uuid4())[:8]
    users[email.lower()] = {
        "id": user_id,
        "email": email.lower(),
        "name": name,
        "department": department,
        "password_hash": _hash_password(password),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "role": "user"
    }
    
    _save_users(users)
    
    # Log registration
    log_audit_event(
        user_id=user_id,
        user_email=email.lower(),
        user_name=name,
        action="USER_REGISTERED",
        details={"department": department}
    )
    
    return {"success": True, "message": "Registration successful", "user_id": user_id}


def authenticate_user(email: str, password: str) -> Dict:
    """
    Authenticate a user
    
    Returns:
        Dict with success status, message, and user data if successful
    """
    users = _load_users()
    
    email_lower = email.lower()
    if email_lower not in users:
        return {"success": False, "message": "Invalid email or password"}
    
    user = users[email_lower]
    if user["password_hash"] != _hash_password(password):
        return {"success": False, "message": "Invalid email or password"}
    
    # Log login
    log_audit_event(
        user_id=user["id"],
        user_email=email_lower,
        user_name=user["name"],
        action="USER_LOGIN",
        details={}
    )
    
    return {
        "success": True,
        "message": "Login successful",
        "user": {
            "id": user["id"],
            "email": user["email"],
            "name": user["name"],
            "department": user.get("department", ""),
            "role": user.get("role", "user")
        }
    }


def log_audit_event(
    user_id: str,
    user_email: str,
    user_name: str,
    action: str,
    details: Dict
) -> None:
    """Log an audit event"""
    audit_log = _load_audit_log()
    
    event = {
        "id": str(uuid.uuid4())[:12],
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "user_id": user_id,
        "user_email": user_email,
        "user_name": user_name,
        "action": action,
        "details": details
    }
    
    audit_log.append(event)
    
    # Keep only last 1000 events
    if len(audit_log) > 1000:
        audit_log = audit_log[-1000:]
    
    _save_audit_log(audit_log)


def log_query(
    user_id: str,
    user_email: str,
    user_name: str,
    query: str,
    answer_mode: str,
    sources_count: int
) -> None:
    """Log a policy query for audit trail"""
    log_audit_event(
        user_id=user_id,
        user_email=user_email,
        user_name=user_name,
        action="POLICY_QUERY",
        details={
            "query": query,
            "answer_mode": answer_mode,
            "sources_count": sources_count
        }
    )


def log_document_indexed(
    user_id: str,
    user_email: str,
    user_name: str,
    doc_name: str,
    chunks_count: int
) -> None:
    """Log document indexing for audit trail"""
    log_audit_event(
        user_id=user_id,
        user_email=user_email,
        user_name=user_name,
        action="DOCUMENT_INDEXED",
        details={
            "document_name": doc_name,
            "chunks_count": chunks_count
        }
    )


def get_audit_log(
    limit: int = 50,
    user_id: str = None,
    action_type: str = None
) -> List[Dict]:
    """
    Get audit log entries
    
    Args:
        limit: Maximum number of entries to return
        user_id: Filter by user ID
        action_type: Filter by action type
        
    Returns:
        List of audit log entries (most recent first)
    """
    audit_log = _load_audit_log()
    
    # Filter by user if specified
    if user_id:
        audit_log = [e for e in audit_log if e["user_id"] == user_id]
    
    # Filter by action type if specified
    if action_type:
        audit_log = [e for e in audit_log if e["action"] == action_type]
    
    # Return most recent first
    return list(reversed(audit_log[-limit:]))


def get_user_stats(user_id: str) -> Dict:
    """Get statistics for a user"""
    audit_log = _load_audit_log()
    user_events = [e for e in audit_log if e["user_id"] == user_id]
    
    queries = [e for e in user_events if e["action"] == "POLICY_QUERY"]
    docs_indexed = [e for e in user_events if e["action"] == "DOCUMENT_INDEXED"]
    
    return {
        "total_queries": len(queries),
        "documents_indexed": len(docs_indexed),
        "last_activity": user_events[-1]["timestamp"] if user_events else None
    }


def get_admin_stats() -> Dict:
    """Get admin statistics (for admin users)"""
    users = _load_users()
    audit_log = _load_audit_log()
    
    queries = [e for e in audit_log if e["action"] == "POLICY_QUERY"]
    
    # Get unique active users (last 30 days)
    from datetime import timedelta
    cutoff = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()
    recent_queries = [q for q in queries if q["timestamp"] > cutoff]
    active_users = len(set(q["user_id"] for q in recent_queries))
    
    return {
        "total_users": len(users),
        "total_queries": len(queries),
        "active_users_30d": active_users,
        "queries_today": len([q for q in queries if q["timestamp"][:10] == datetime.now(timezone.utc).isoformat()[:10]])
    }
