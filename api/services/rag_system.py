import re
from typing import List, Dict, Any

class BlockchainRAGSystem:
    def __init__(self, vector_store=None, llm_client=None):
        self.vector_store = vector_store
        self.llm_client = llm_client
        self.memory = {}

    def extract_citations(self, text: str) -> List[str]:
        """
        Add citation extraction from text.
        Extracts document IDs enclosed in brackets e.g., [doc_123]
        """
        pattern = r'\[(doc_[a-zA-Z0-9_]+)\]'
        return list(set(re.findall(pattern, text)))

    def update_memory(self, session_id: str, query: str, response: str):
        """
        Add conversation memory
        """
        if session_id not in self.memory:
            self.memory[session_id] = []
        self.memory[session_id].append({"role": "user", "content": query})
        self.memory[session_id].append({"role": "assistant", "content": response})
        
        # Keep last 10 messages to avoid context overflow
        if len(self.memory[session_id]) > 10:
            self.memory[session_id] = self.memory[session_id][-10:]

    def retrieve_context(self, query: str) -> str:
        """
        Design retrieval pipeline to fetch context from vector store.
        """
        if not self.vector_store:
            return "Mock Context for: " + query + " [doc_001]"
        
        docs = self.vector_store.search(query, k=5)
        context = "\n".join([f"[{d.id}] {d.content}" for d in docs])
        return context

    def answer_question(self, query: str, session_id: str = "default") -> Dict[str, Any]:
        """
        RAG system answering blockchain questions with citations.
        """
        # 1. Retrieve
        context = self.retrieve_context(query)
        
        # 2. Get conversation history
        history = self.memory.get(session_id, [])
        history_str = "\n".join([f"{m['role']}: {m['content']}" for m in history])
        
        # 3. Generate response
        prompt = f"Context:\n{context}\n\nHistory:\n{history_str}\n\nQuestion: {query}\nAnswer and include citations like [doc_id]:"
        
        if self.llm_client:
            raw_response = self.llm_client.generate(prompt).get("text", "Default response")
        else:
            raw_response = f"Based on the context, this is a response to '{query}'. Sources: [doc_001]"
            
        # 4. Extract citations
        citations = self.extract_citations(raw_response)
        
        # 5. Update memory
        self.update_memory(session_id, query, raw_response)
        
        return {
            "answer": raw_response,
            "citations": citations,
            "session_id": session_id
        }
