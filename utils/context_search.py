import logging
import re
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import pdfplumber


# Configure logging for debugging and error tracking
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ContextSearch:
    def __init__(self, file_content, top_k=5, score_threshold=0.5, window=10):
        """
        Initialize ContextSearch with document content and configurable parameters.

        Args:
            file_content (str or bytes): Text content or binary data from a file.
            top_k (int): Number of top results to return. Default: 5.
            score_threshold (float): Minimum similarity score (0-1). Default: 0.5.
            window (int): Context window size around matches. Default: 10.
        """
        self.top_k = top_k
        self.score_threshold = score_threshold
        self.window = window
        try:
            self.document = self._load_document(file_content)
            self.sentences = self._split_sentences(self.document)
            self.model = None  # Lazy-loaded
            self.index = None  # Lazy-loaded
            logger.info(f"Initialized with {len(self.sentences)} sentences.")
        except Exception as e:
            logger.error(f"Initialization failed: {e}")
            raise

    def _load_document(self, file_content):
        """
        Load and decode file content into plain text.

        Args:
            file_content (str or bytes): Raw content to process.

        Returns:
            str: Decoded text content.
        """
        if isinstance(file_content, bytes):
            try:
                return file_content.decode("utf-8")
            except UnicodeDecodeError:
                try:
                    # Assume PDF if UTF-8 fails and try to extract text
                    with pdfplumber.open(file_content) as pdf:
                        return "\n".join(page.extract_text() or "" for page in pdf.pages)
                except Exception as e:
                    logger.error(f"Failed to decode file content as UTF-8 or PDF: {e}")
                    raise ValueError("File content must be UTF-8 text or a readable PDF.")
        elif isinstance(file_content, str):
            return file_content
        else:
            logger.error(f"Invalid file content type: {type(file_content)}")
            raise ValueError("File content must be string or bytes.")

    def _split_sentences(self, text):
        """
        Split text into sentences or paragraphs.

        Args:
            text (str): Input text to split.

        Returns:
            list: List of sentence/paragraph strings.
        """
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if len(lines) > 1:
            return lines
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]

    def _create_faiss_index(self):
        """
        Create a FAISS index from sentence embeddings.

        Returns:
            faiss.IndexFlatIP or None: Index if successful, None if failed.
        """
        if not self.sentences:
            return None
        try:
            if self.model is None:
                self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
            batch_size = 32  # Adjust based on hardware
            embeddings = []
            for i in range(0, len(self.sentences), batch_size):
                batch = self.sentences[i:i + batch_size]
                batch_embeddings = self.model.encode(batch, convert_to_numpy=True)
                embeddings.append(batch_embeddings)
            embeddings = np.vstack(embeddings)
            embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
            dim = embeddings.shape[1]
            index = faiss.IndexFlatIP(dim)
            index.add(embeddings)
            logger.info("FAISS index created successfully.")
            return index
        except Exception as e:
            logger.error(f"Failed to create FAISS index: {e}")
            return None

    def _ensure_index(self):
        """Ensure the model and index are loaded."""
        if self.model is None or self.index is None:
            self.index = self._create_faiss_index()

    @staticmethod
    def _merge_intervals(intervals):
        """
        Merge overlapping or adjacent intervals.

        Args:
            intervals (list): List of (start, end, score, match_set) tuples.

        Returns:
            list: Merged intervals.
        """
        if not intervals:
            return []
        intervals.sort(key=lambda x: x[0])
        merged = [intervals[0]]
        for current in intervals[1:]:
            last = merged[-1]
            if current[0] <= last[1] + 1:
                merged[-1] = (
                    last[0],
                    max(last[1], current[1]),
                    max(last[2], current[2]),
                    last[3] | current[3]
                )
            else:
                merged.append(current)
        return merged

    def query(self, query, top_k=None, score_threshold=None, window=None):
        """
        Search for similar sentences and return context blocks.

        Args:
            query (str): Search query.
            top_k (int, optional): Override default top_k.
            score_threshold (float, optional): Override default threshold.
            window (int, optional): Override default window.

        Returns:
            list: List of dicts with 'text', 'score', 'start_idx', 'end_idx', 'matches'.
        """
        top_k = top_k if top_k is not None else self.top_k
        score_threshold = score_threshold if score_threshold is not None else self.score_threshold
        window = window if window is not None else self.window

        self._ensure_index()
        if not self.index:
            logger.warning("No index available for search.")
            return []

        try:
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
                        s = f"[[[{s}]]]"  # Highlight matches
                    block_sentences.append(s)
                text = " ".join(block_sentences)
                result = {
                    "text": "… " + text + " …" if start > 0 or end < len(self.sentences) - 1 else text,
                    "score": score,
                    "start_idx": start,
                    "end_idx": end,
                    "matches": list(match_set)
                }
                results.append(result)
            logger.info(f"Query returned {len(results)} results.")
            return results
        except Exception as e:
            logger.error(f"Query failed: {e}")
            return []

def main():
    """Simple test function for ContextSearch."""
    import os
    import docx
    file_path = input("Enter text, DOCX, or PDF file path: ")
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
    elif ext == ".pdf":
        with pdfplumber.open(file_path) as pdf:
            file_content = "\n".join(page.extract_text() or "" for page in pdf.pages)
    else:
        print("Unsupported file type.")
        return

    cs = ContextSearch(file_content)
    while True:
        query = input("Enter your search query (or 'exit' to quit): ")
        if query.lower() == "exit":
            break
        results = cs.query(query)
        for result in results:
            print(f"Score: {result['score']:.3f}")
            print(f"Context: {result['text']}")
            print(f"Indices: {result['start_idx']}–{result['end_idx']}\n")

if __name__ == "__main__":
    main()