import logging
import re
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import pdfplumber

# Configure logging for debugging and error tracking
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def split_into_paragraphs(text: str) -> list[str]:
    """
    Split the text into paragraphs using blank lines as delimiters.
    You can adjust this to suit your input formatting.
    """
    text = text.strip()
    if not text:
        return []
    # Splits on one or more blank lines
    paragraphs = re.split(r"\n\s*\n", text)
    # Remove paragraphs that are just whitespace
    paragraphs = [p.strip() for p in paragraphs if p.strip()]
    return paragraphs

def split_paragraph_into_sentences(paragraph: str) -> list[str]:
    """
    Very naive sentence splitting based on punctuation.
    This can be replaced with more robust methods (e.g., spaCy, NLTK) if desired.
    """
    paragraph = paragraph.strip()
    if not paragraph:
        return []
    # Regex that splits on a period, exclamation point, or question mark,
    # followed by whitespace or the paragraph end.
    # Keep the punctuation by using a lookbehind.
    sentences = re.split(r'(?<=[.!?])\s+', paragraph)
    # Clean up trailing whitespace
    sentences = [s.strip() for s in sentences if s.strip()]
    return sentences

def create_contextual_chunks(
    text: str,
    max_words_per_chunk: int = 80,
    min_words_per_chunk: int = 30,
    overlap: int = 10
) -> list[str]:
    """
    Naive approach to chunk text at paragraph & sentence boundaries:
      1) Split into paragraphs.
      2) For each paragraph, split into sentences.
      3) Accumulate sentences until hitting max_words_per_chunk.
      4) Optionally add overlap for better context continuity.

    Args:
        text (str): Original text to chunk.
        max_words_per_chunk (int): Target upper bound on chunk size in words.
        min_words_per_chunk (int): Minimum words before allowing a new chunk.
        overlap (int): Overlap in words between consecutive chunks for continuity.

    Returns:
        list[str]: List of chunked text segments.
    """
    paragraphs = split_into_paragraphs(text)
    all_chunks = []

    for paragraph in paragraphs:
        sentences = split_paragraph_into_sentences(paragraph)
        current_chunk = []
        current_length = 0

        for sent in sentences:
            words_in_sent = len(sent.split())
            # If adding this sentence goes beyond max_words, finalize current chunk
            if current_chunk and (current_length + words_in_sent) > max_words_per_chunk:
                chunk_text = " ".join(current_chunk)
                all_chunks.append(chunk_text)

                # Overlap: take the last X words from the finished chunk as a “prefix”
                # to preserve context for the next chunk
                if overlap > 0:
                    overlap_words = chunk_text.split()[-overlap:]
                    current_chunk = overlap_words.copy()
                    current_length = len(current_chunk)
                else:
                    current_chunk = []
                    current_length = 0

            # Add new sentence to the chunk
            current_chunk.append(sent)
            current_length += words_in_sent

        # End of paragraph: push whatever remains in current_chunk if not too small
        if current_chunk and current_length >= min_words_per_chunk:
            all_chunks.append(" ".join(current_chunk))

    return all_chunks


class ContextSearch:
    """
    ContextSearch with more “contextual” chunking instead of fixed word count.
    """
    def __init__(self, file_content, top_k=5, score_threshold=0.5):
        self.top_k = top_k
        self.score_threshold = score_threshold
        try:
            self.document = self._load_document(file_content)
            self.chunks = self._create_chunks(self.document)
            self.model = None  # Lazy-loaded
            self.index = None  # Lazy-loaded
            logger.info(f"Initialized with {len(self.chunks)} chunks.")
            logger.debug(f"Chunks {self.chunks}")
        except Exception as e:
            logger.error(f"Initialization failed: {e}")
            self.document = ""
            self.chunks = []

    def _load_document(self, file_content):
        if file_content is None:
            return ""
        elif isinstance(file_content, bytes):
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
        Create more context-aware chunks using paragraphs and sentences.
        Adjust the parameters (max_words_per_chunk, min_words_per_chunk, overlap)
        to tweak chunk size and overlap.
        """
        if not document.strip():
            return []
        return create_contextual_chunks(
            text=document,
            max_words_per_chunk=80,  # Tweak as needed
            min_words_per_chunk=10,  # Ensures we don't create tiny fragments
            overlap=10               # Overlap in words for continuity
        )

    def _create_faiss_index(self):
        if not self.chunks:
            logger.warning("No chunks to index.")
            return None
        try:
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
        if self.index is None:
            self.index = self._create_faiss_index()

    def query(self, query: str, top_k: int = None) -> list[dict]:
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
    sample_text = (
        "Paragraph one: Python is a popular programming language. It’s often used for "
        "data science, automation, and backend web development.\n\n"
        "Paragraph two: FAISS is a library for efficient similarity search and clustering "
        "of dense vectors. It is created by Facebook AI Research.\n\n"
        "Paragraph three: This chunk-based approach tries to keep ideas together, "
        "rather than simply slicing after a fixed number of words."
    )

    search_tool = ContextSearch(sample_text)

    print("Created Chunks:")
    for idx, chunk in enumerate(search_tool.chunks):
        print(f"Chunk {idx}:", chunk)
        print("-" * 70)

    test_query = "Tell me about FAISS."
    print(f"\nRunning query: {test_query}\n")
    results = search_tool.query(test_query, top_k=3)

    for i, res in enumerate(results):
        print(f"Result {i+1}:")
        print(f"  Score: {res['score']}")
        print(f"  Chunk: {res['chunk']}\n")

if __name__ == "__main__":
    main()
