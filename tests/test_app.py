import pytest
import streamlit as st
import faiss
from chat_assist import get_history
from utils.context_search import ContextSearch
from utils.text_handler import extract_image_urls


# Mock session state for get_prompts_from_history
@pytest.fixture
def mock_session_state():
    st.session_state.messages = [
        {"role": "user", "content": "Hello, how are you?"},
        {"role": "assistant", "content": "I am fine, thank you!"}
    ]

def test_extract_image_urls():
    """Ensure that image URLs are correctly extracted from text."""
    text = "Check this image: https://example.com/image.jpg and this one: https://example.com/image2.png"
    expected_urls = ["https://example.com/image.jpg", "https://example.com/image2.png"]
    assert extract_image_urls(text) == expected_urls, "Extracted URLs should match expected list."


def test_get_history(mock_session_state):
    """Ensure chat history is correctly formatted."""
    history = get_history()
    assert "user: Hello, how are you?" in history, "User message should be in chat history."
    assert "assistant: I am fine, thank you!" in history, "Assistant response should be in chat history."

def test_get_history_max(mock_session_state):
    """Ensure chat history respects max message limit."""
    history = get_history(1)
    assert "user: Hello, how are you?" not in history, "User message should not be in truncated history."
    assert "assistant: I am fine, thank you!" in history, "Assistant response should be retained in history."

def test_load_document_text():
    text = "This is a test document. It contains multiple sentences."
    cs = ContextSearch(text)
    assert cs.document == text

def test_load_document_binary():
    binary_content = b"This is a binary test document."
    cs = ContextSearch(binary_content.decode("utf-8"))
    assert cs.document == "This is a binary test document."

def test_invalid_file_content():
    with pytest.raises(ValueError):
        ContextSearch(1234)  # Invalid input type

def test_create_faiss_index():
    text = "This is a sentence. Here is another."
    cs = ContextSearch(text)
    assert cs.index is not None
    assert cs.index.ntotal == len(cs.sentences)

def test_query_results():
    text = "This is a test document. It contains multiple sentences about different topics."
    cs = ContextSearch(text)
    results = cs.query("test document", top_k=1)
    assert len(results) > 0
    assert "test document" in results[0][0]

def test_query_no_results():
    text = "This is a completely different text with no relation to the query."
    cs = ContextSearch(text)
    results = cs.query("unrelated topic", top_k=1)
    assert len(results) == 0
