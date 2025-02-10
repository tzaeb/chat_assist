# vectorization.py
import faiss
import numpy as np
import json
from sentence_transformers import SentenceTransformer

class Vectorization:
    def __init__(self, context_file="context.json"):
        """Initialize vectorization by loading context and creating FAISS index."""
        self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        self.context_data = self.load_context(context_file)
        self.index = self.embed_context()
    
    def load_context(self, file_path):
        """Load summaries and descriptions from a JSON file."""
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    
    def embed_context(self):
        """Embed descriptions and return FAISS index."""
        descriptions = [entry["description"] for entry in self.context_data]
        issue_embeddings = self.model.encode(descriptions, convert_to_tensor=False)
        
        # Convert to NumPy array
        issue_embedding_matrix = np.array(issue_embeddings)
        
        # Create FAISS index
        index = faiss.IndexFlatL2(issue_embedding_matrix.shape[1])
        index.add(issue_embedding_matrix)
        
        return index
    
    def find_matches(self, query, top_k=3, similarity_threshold=0.3):
        """Find the top-k most relevant issues, but return only if similarity is above threshold."""
        query_vector = self.model.encode([query])

        # Search for the top-k matches
        D, I = self.index.search(np.array(query_vector), k=top_k)

        results = []
        for i, idx in enumerate(I[0]):
            if idx == -1:  # Ignore invalid indexes
                continue
            
            similarity_score = float(1 / (1 + D[0][i]))  # Convert FAISS distance to similarity score

            if similarity_score >= similarity_threshold:
                matched_summary = self.context_data[idx]["summary"]
                matched_description = self.context_data[idx]["description"]
                matched_tags = self.context_data[idx].get("tags", [])
                results.append({
                    "summary": matched_summary,
                    "description": matched_description,
                    "tags": matched_tags,
                    "score": similarity_score
                })

        return results  # Returns empty list if no match exceeds threshold

if __name__ == "__main__":
    vectorization = Vectorization()
    test_query = "Hi, can you tell me some things about issues with autonomous vehicles in heavy rain?"
    results = vectorization.find_matches(test_query)
    print("Test Query:", test_query)
    print("Results:", json.dumps(results))
