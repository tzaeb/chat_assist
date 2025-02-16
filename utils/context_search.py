import os
import re
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import docx

class ContextSearch:
    def __init__(self, file_content):
        """
        Initialize the ContextSearch instance by loading the document content, 
        splitting the text into sentences/paragraphs, and creating the FAISS index.
        """
        self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        self.document = self._load_document(file_content)
        self.sentences = self._split_sentences(self.document)
        self.index = self._create_faiss_index() if self.sentences else None

    def _load_document(self, file_content):
        """
        Reads content from uploaded file (TXT or DOCX) and returns as plain text.
        """
        if isinstance(file_content, bytes):  # Handle binary file content
            try:
                return file_content.decode("utf-8")  # Try decoding as text
            except UnicodeDecodeError:
                raise ValueError("File content could not be decoded as UTF-8 text.")
        elif isinstance(file_content, str):
            return file_content
        else:
            raise ValueError("Invalid file content type.")

    def _split_sentences(self, text):
        """
        Splits text into sentences or paragraphs.
        """
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if len(lines) > 1:
            return lines
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [sentence.strip() for sentence in sentences if sentence.strip()]

    def _create_faiss_index(self):
        """
        Computes embeddings for each sentence and builds a FAISS index.
        """
        embeddings = self.model.encode(self.sentences, convert_to_numpy=True)
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)
        return index

    @staticmethod
    def _merge_intervals(intervals):
        """
        Merges overlapping or adjacent intervals.
        """
        if not intervals:
            return []
        intervals.sort(key=lambda x: x[0])
        merged = [intervals[0]]
        for current in intervals[1:]:
            last = merged[-1]
            if current[0] <= last[1] + 1:
                new_start = last[0]
                new_end = max(last[1], current[1])
                new_score = max(last[2], current[2])
                new_match_set = last[3] | current[3]
                merged[-1] = (new_start, new_end, new_score, new_match_set)
            else:
                merged.append(current)
        return merged

    def query(self, query, top_k=5, score_threshold=0.5, window=10):
        """
        Searches for the top_k most similar sentences to the query and retrieves surrounding context.
        """
        if not self.index:
            return []
        q_embedding = self.model.encode([query], convert_to_numpy=True)
        q_embedding = q_embedding / np.linalg.norm(q_embedding, axis=1, keepdims=True)
        scores, indices = self.index.search(q_embedding, top_k)

        intervals = []
        for score, idx in zip(scores[0], indices[0]):
            if score < score_threshold:
                continue
            start = max(0, idx - window)
            end = min(len(self.sentences) - 1, idx + window)
            intervals.append((start, end, float(score), {idx}))

        merged_intervals = self._merge_intervals(intervals)
        results = []
        for start, end, score, match_set in merged_intervals:
            block_sentences = []
            for i in range(start, end + 1):
                s = self.sentences[i]
                if i in match_set:
                    s = f"[[[{s}]]]"  # Highlight matched sentence.
                block_sentences.append(s)
            block = " ".join(block_sentences)
            if start > 0:
                block = "..." + block
            if end < len(self.sentences) - 1:
                block = block + "..."
            results.append((block, score))
        return results

def main():
    """
    Simple test function for ContextSearch class.
    """
    file_path = input("Enter text or DOCX file path: ")
    if not os.path.isfile(file_path):
        print("File not found.")
        return
    
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".txt":
        with open(file_path, "r", encoding="utf-8") as f:
            file_content = f.read()
    elif ext in (".doc", ".docx"):
        doc = docx.Document(file_path)
        file_content = "\n".join(para.text for para in doc.paragraphs)
    else:
        print("Unsupported file type.")
        return

    cs = ContextSearch(file_content)
    while True:
        query = input("Enter your search query (or type 'exit' to quit): ")
        if query.lower() == "exit":
            break
        results = cs.query(query)
        for context, score in results:
            print(f"Relevance Score: {score:.3f}\nContext:\n{context}\n")

if __name__ == "__main__":
    main()
