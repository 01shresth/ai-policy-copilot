"""
Utility functions for AI Policy Copilot
"""
from typing import List, Dict
import re


def truncate_text(text: str, max_length: int = 200) -> str:
    """Truncate text to max length with ellipsis"""
    if len(text) <= max_length:
        return text
    return text[:max_length].rsplit(' ', 1)[0] + "..."


def format_file_size(size_bytes: int) -> str:
    """Format file size in human readable format"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"


def highlight_query_terms(text: str, query: str) -> str:
    """Highlight query terms in text (for display)"""
    words = query.lower().split()
    for word in words:
        if len(word) > 2:  # Skip short words
            pattern = re.compile(f'({re.escape(word)})', re.IGNORECASE)
            text = pattern.sub(r'**\1**', text)
    return text


def clean_filename(filename: str) -> str:
    """Clean filename for display"""
    # Remove file extension
    name = re.sub(r'\.[^.]+$', '', filename)
    # Replace underscores and hyphens with spaces
    name = re.sub(r'[_-]+', ' ', name)
    # Title case
    return name.title()


def get_page_reference(page_num: int = None) -> str:
    """Format page reference"""
    if page_num:
        return f"Page {page_num}"
    return "Unknown page"
