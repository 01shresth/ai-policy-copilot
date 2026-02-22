"""
Answer generator module with LLM and extractive fallback
"""
import os
import asyncio
from typing import List, Dict, Optional


class AnswerGenerator:
    """Generate answers using LLM or extractive fallback"""
    
    def __init__(self, api_key: str = None):
        """
        Initialize generator
        
        Args:
            api_key: OpenAI API key (uses EMERGENT_LLM_KEY env var if not provided)
        """
        self.api_key = api_key or os.environ.get("EMERGENT_LLM_KEY", "")
        self._llm_available = bool(self.api_key)
    
    @property
    def has_llm(self) -> bool:
        """Check if LLM is available"""
        return self._llm_available
    
    def generate(
        self, 
        query: str, 
        context_chunks: List[Dict]
    ) -> Dict:
        """
        Generate an answer based on retrieved context
        
        Args:
            query: User's question
            context_chunks: List of retrieved chunk dictionaries
            
        Returns:
            Dictionary with answer and metadata
        """
        if not context_chunks:
            return {
                "answer": "I don't have any documents indexed yet. Please upload and index some PDF documents first.",
                "mode": "no_context",
                "sources": []
            }
        
        # Build context string
        context_text = self._build_context(context_chunks)
        
        if self._llm_available:
            try:
                answer = self._generate_with_llm(query, context_text)
                return {
                    "answer": answer,
                    "mode": "llm",
                    "sources": context_chunks
                }
            except Exception as e:
                print(f"LLM generation failed: {e}")
                # Fall back to extractive
                return self._extractive_answer(query, context_chunks)
        else:
            return self._extractive_answer(query, context_chunks)
    
    def _build_context(self, chunks: List[Dict]) -> str:
        """Build context string from chunks"""
        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            doc_name = chunk.get("doc_name", "Unknown")
            page_info = f" (Page {chunk.get('page_num')})" if chunk.get("page_num") else ""
            text = chunk.get("text", "")
            context_parts.append(f"[Source {i}: {doc_name}{page_info}]\n{text}")
        
        return "\n\n".join(context_parts)
    
    def _generate_with_llm(self, query: str, context: str) -> str:
        """Generate answer using LLM via emergentintegrations"""
        from emergentintegrations.llm.chat import LlmChat, UserMessage
        
        system_message = """You are an AI Policy Copilot that answers questions based ONLY on the provided document context.

STRICT RULES:
1. ONLY use information from the provided context to answer
2. If the context doesn't contain enough information, say "I don't have enough information in the uploaded documents to answer this question."
3. Be concise but thorough
4. Reference the source documents when relevant (e.g., "According to [Source 1]...")
5. Do NOT make up information or use external knowledge
6. If asked about something not in the documents, clearly state that"""
        
        prompt = f"""Context from uploaded documents:

{context}

---

Question: {query}

Please answer based ONLY on the above context. If the context doesn't contain the answer, say so clearly."""
        
        # Run async function synchronously
        chat = LlmChat(
            api_key=self.api_key,
            session_id="policy-copilot-session",
            system_message=system_message
        ).with_model("openai", "gpt-4.1-mini")
        
        # Use asyncio to run the async function
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            user_message = UserMessage(text=prompt)
            response = loop.run_until_complete(chat.send_message(user_message))
            return response
        finally:
            loop.close()
    
    def _extractive_answer(
        self, 
        query: str, 
        chunks: List[Dict]
    ) -> Dict:
        """
        Generate extractive answer when LLM is not available
        Returns the most relevant chunks as the answer
        """
        if not chunks:
            return {
                "answer": "No relevant information found in the indexed documents.",
                "mode": "extractive",
                "sources": []
            }
        
        # Build a structured extractive response
        answer_parts = [
            "**Based on the indexed documents, here are the most relevant excerpts:**\n"
        ]
        
        for i, chunk in enumerate(chunks, 1):
            doc_name = chunk.get("doc_name", "Unknown")
            page_info = f" (Page {chunk.get('page_num')})" if chunk.get("page_num") else ""
            score = chunk.get("relevance_score", 0)
            text_preview = chunk.get("text", "")[:500]
            if len(chunk.get("text", "")) > 500:
                text_preview += "..."
            
            answer_parts.append(
                f"**Source {i}: {doc_name}{page_info}** (Relevance: {score:.0%})\n"
                f"> {text_preview}\n"
            )
        
        answer_parts.append(
            "\n*Note: This is an extractive answer showing relevant document excerpts. "
            "For AI-generated synthesis, please configure an LLM API key.*"
        )
        
        return {
            "answer": "\n".join(answer_parts),
            "mode": "extractive",
            "sources": chunks
        }
