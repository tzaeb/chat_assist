import faiss
import numpy as np
import json
from sentence_transformers import SentenceTransformer

class ContextSearch:
    def __init__(self, context_data=None):
        """Initialize vectorization by embedding file content dynamically."""
        self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        self.context_data = context_data if context_data else []
        self.index = self.embed_context()
    
    def embed_context(self):
        """Embed file content and return FAISS index."""
        if not self.context_data:
            return None
        
        issue_embeddings = self.model.encode(self.context_data, convert_to_tensor=False)
        issue_embedding_matrix = np.array(issue_embeddings)
        
        index = faiss.IndexFlatL2(issue_embedding_matrix.shape[1])
        index.add(issue_embedding_matrix)
        
        return index
    
    def find_relevant_context(self, query, top_k=3, similarity_threshold=0.35):
        """Find relevant segments from the file content based on the user's query."""
        if not self.index:
            return ""
        
        query_vector = self.model.encode([query])
        D, I = self.index.search(np.array(query_vector), k=top_k)
        
        results = []
        for i, idx in enumerate(I[0]):
            if idx == -1:
                continue
            
            similarity_score = float(1 / (1 + D[0][i]))
            if similarity_score >= similarity_threshold:
                results.append(self.context_data[idx])
        
        return "\n".join(results) if results else ""

if __name__ == "__main__":
    sample_data = [
        "Autonomous vehicles struggle in heavy rain due to sensor interference.",
        "Machine learning models require extensive datasets to improve predictions.",
        "Cybersecurity threats are increasing with the rise of AI-powered systems."
    ]
    
    vectorization = ContextSearch(context_data=sample_data)
    test_query = "Tell me about AI and rain issues."
    relevant_text = vectorization.find_relevant_context(test_query)
    print("Test Query:", test_query)
    print("Relevant Context:", relevant_text)
