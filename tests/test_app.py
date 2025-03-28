import pytest
import streamlit as st
from chat_assist import get_history
from utils.context_search import ContextSearch
from utils.text_handler import extract_image_urls

@pytest.fixture
def mock_session_state():
    st.session_state.messages = [
        {"role": "user", "content": "Hello, how are you?"},
        {"role": "assistant", "content": "I am fine, thank you!"}
    ]

def test_extract_image_urls():
    text = (
        "Check this image: https://example.com/image.jpg "
        "and this one: https://example.com/image2.png"
    )
    expected_urls = ["https://example.com/image.jpg", "https://example.com/image2.png"]
    assert extract_image_urls(text) == expected_urls

def test_get_history(mock_session_state):
    history = get_history()
    assert "user: Hello, how are you?" in history
    assert "assistant: I am fine, thank you!" in history

def test_get_history_max(mock_session_state):
    history = get_history(1)
    assert "user: Hello, how are you?" not in history
    assert "assistant: I am fine, thank you!" in history

def test_load_document_text():
    text = "This is a test document. It contains multiple sentences."
    cs = ContextSearch(text)
    assert cs.document == text

def test_load_document_binary():
    binary_content = b"This is a binary test document."
    cs = ContextSearch(binary_content)
    assert cs.document == "This is a binary test document."

def test_create_faiss_index():
    """
    Ensure FAISS index creation. Provide paragraphs to produce multiple chunks,
    then call .query() or cs._ensure_index() to force the index to be built.
    """
    text = (
        "Paragraph one: This is a sentence about testing. Here is another sentence.\n\n"
        "Paragraph two: We add more lines to ensure at least one or two chunks are created. "
        "This text references FAISS to match queries.\n\n"
        "Paragraph three: Enough content to test chunk creation, indexing, and searching."
    )
    cs = ContextSearch(text)

    cs.query("FAISS")  # triggers index creation
    assert cs.chunks is not None, "Chunks should be created."
    assert cs.index is not None, "FAISS index should be created after the query."
    assert cs.index.ntotal == len(cs.chunks), "Index size should match the number of chunks."

def test_query_results():
    """
    Provide text that actually contains 'test document' so the query can match.
    Make sure the chunk threshold isn’t so high that partial matches get filtered out.
    """
    text = (
        "This is a test document with enough text to produce at least one chunk. "
        "It references test document specifically, so the query can find it."
    )
    cs = ContextSearch(text, score_threshold=0.2)  # Lower threshold if needed
    results = cs.query("test document", top_k=1)

    assert len(results) > 0, "Expected at least one matching chunk."
    top_result = results[0]["chunk"].lower()
    assert "test document" in top_result, "Matching chunk should include 'test document'."

def test_query_no_results():
    text = "This text is about flowers and gardening, not quantum physics."
    cs = ContextSearch(text, score_threshold=0.8)  # High threshold
    results = cs.query("quantum entanglement", top_k=1)
    assert len(results) == 0, "Expected no results for a completely unrelated query."
