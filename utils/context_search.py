import logging
import re
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import pdfplumber

# Configure logging for debugging and error tracking
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def chunk_text(text: str, chunk_size: int = 300, chunk_overlap: int = 50) -> list[str]:
    """
    Splits text into overlapping chunks of specified size and overlap (measured in words).
    """
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = words[start:end]
        chunks.append(" ".join(chunk))
        # Move start by chunk_size - overlap to create overlap
        start += (chunk_size - chunk_overlap)
        if start >= len(words):
            break
    return chunks

class ContextSearch:
    def __init__(self, file_content, top_k=5, score_threshold=0.5):
        """
        Initialize ContextSearch with document content and configurable parameters.

        Args:
            file_content (str or bytes): Text content or binary data from a file.
            top_k (int): Default number of top results to return. Default: 5.
            score_threshold (float): Minimum similarity score cutoff in [0..1]. Default: 0.5.
        """
        self.top_k = top_k
        self.score_threshold = score_threshold
        try:
            self.document = self._load_document(file_content)
            self.chunks = self._create_chunks(self.document)
            self.model = None  # Lazy-loaded
            self.index = None  # Lazy-loaded
            logger.info(f"Initialized with {len(self.chunks)} chunks.")
        except Exception as e:
            logger.error(f"Initialization failed: {e}")
            # If there's a problem decoding or chunking, store empty text
            self.document = ""
            self.chunks = []

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
                # Attempt PDF extraction
                try:
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

    def _create_chunks(self, document: str) -> list[str]:
        """
        Splits the document into overlapping chunks.
        """
        if not document.strip():
            return []
        return chunk_text(document, chunk_size=200, chunk_overlap=50)

    def _create_faiss_index(self):
        """
        Create a FAISS index from chunk embeddings.
        """
        if not self.chunks:
            logger.warning("No chunks to index.")
            return None
        try:
            # Lazy load the model if needed
            if self.model is None:
                self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

            batch_size = 32
            all_embeddings = []
            for i in range(0, len(self.chunks), batch_size):
                batch = self.chunks[i : i + batch_size]
                batch_embeddings = self.model.encode(batch, convert_to_numpy=True)
                all_embeddings.append(batch_embeddings)

            embeddings = np.vstack(all_embeddings)
            # Normalize embeddings for dot-product use
            embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

            dim = embeddings.shape[1]
            index = faiss.IndexFlatIP(dim)
            index.add(embeddings)
            logger.info("FAISS index created successfully for chunked embeddings.")
            return index
        except Exception as e:
            logger.error(f"Failed to create FAISS index: {e}")
            return None

    def _ensure_index(self):
        """
        Ensure the model and FAISS index are loaded.
        """
        if self.index is None:
            self.index = self._create_faiss_index()

    def query(self, query: str, top_k: int = None) -> list[dict]:
        """
        Retrieve the top_k most relevant chunks to the query.
        Returns a list of dicts with 'score' and 'chunk'.
        """
        if not query.strip():
            return []

        self._ensure_index()
        if not self.index or not self.chunks:
            logger.warning("Index or chunks are not available.")
            return []

        if top_k is None:
            top_k = self.top_k

        try:
            # Embed the query
            if self.model is None:
                self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

            q_embedding = self.model.encode([query], convert_to_numpy=True)
            q_embedding = q_embedding / np.linalg.norm(q_embedding, axis=1, keepdims=True)

            scores, indices = self.index.search(q_embedding, top_k)
            # scores, indices are arrays of shape [1, top_k]

            results = []
            for score, idx in zip(scores[0], indices[0]):
                # Dot product of normalized vectors is effectively a similarity in [0..1]
                if score >= self.score_threshold:
                    results.append({
                        "score": float(score),
                        "chunk": self.chunks[idx],
                        "index": idx
                    })

            # Sort by highest similarity score first
            results.sort(key=lambda x: x["score"], reverse=True)
            logger.info(f"Query '{query}' returned {len(results)} relevant chunks.")
            return results
        except Exception as e:
            logger.error(f"Query failed: {e}")
            return []

def main():
    """
    Simple test to verify chunk creation and querying using ContextSearch.
    """
    # Sample text for demonstration
    sample_text = (
        "This is a short text talking about Python. "
        "Python is a popular programming language. "
        "We can also mention FAISS, which is a library for similarity search. "
        "FAISS is developed by Facebook AI Research. "
        "Sentence Transformers can produce embeddings for semantic search."
    )

    # Initialize the ContextSearch object
    search_tool = ContextSearch(sample_text)

    # Print out the chunks that got created
    print("Created Chunks:")
    for idx, chunk in enumerate(search_tool.chunks):
        print(f"Chunk {idx}: {chunk}")

    # Perform a query
    test_query = "What is FAISS?"
    print(f"\nRunning query: {test_query}\n")
    results = search_tool.query(test_query, top_k=3)

    # Display the results
    for i, res in enumerate(results):
        print(f"Result {i+1}:")
        print(f"  Score: {res['score']}")
        print(f"  Chunk: {res['chunk']}\n")

# Standard Python entry point
if __name__ == "__main__":
    main()
