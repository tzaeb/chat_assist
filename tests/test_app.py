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

# Provide sample data for ContextSearch
@pytest.fixture
def sample_data():
    return [
        "Autonomous vehicles struggle in heavy rain due to sensor interference.",
        "Machine learning models require extensive datasets to improve predictions.",
        "Cybersecurity threats are increasing with the rise of AI-powered systems."
    ]

# Instantiate ContextSearch with sample data
@pytest.fixture
def context_search_instance(sample_data):
    return ContextSearch(context_data=sample_data)


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


def test_embedding_creation(context_search_instance):
    """Ensure that FAISS index is correctly created."""
    assert context_search_instance.index is not None, "FAISS index should be initialized."
    assert isinstance(context_search_instance.index, faiss.IndexFlatL2), "FAISS index should be of type IndexFlatL2."


def test_find_relevant_context(context_search_instance):
    """Ensure relevant context is retrieved for a query."""
    query = "Why do self-driving cars have issues in rain?"
    result = context_search_instance.find_relevant_context(query)
    assert "Autonomous vehicles struggle in heavy rain due to sensor interference." in result, "Relevant text should be found in results."


def test_empty_query(context_search_instance):
    """Ensure that an empty query returns an empty string."""
    query = ""
    result = context_search_instance.find_relevant_context(query)
    assert result == "", "Empty query should return an empty string."


def test_no_relevant_match(context_search_instance):
    """Ensure that an irrelevant query returns an empty string."""
    query = "How do fish swim?"
    result = context_search_instance.find_relevant_context(query)
    assert result == "", "Irrelevant query should return an empty string."


def test_threshold_behavior(context_search_instance):
    """Ensure that results respect the similarity threshold."""
    query = "Tell me about AI predictions."
    result = context_search_instance.find_relevant_context(query, similarity_threshold=0.9)
    assert result == "", "High threshold should filter out less relevant results."


def test_empty_context():
    """Ensure that an empty context list does not cause errors."""
    vectorization = ContextSearch(context_data=[])
    query = "What is AI?"
    result = vectorization.find_relevant_context(query)
    assert result == "", "Query on empty context should return an empty string."
