import pytest
import json
from chat_assist import extract_image_urls, get_history
import utils.vectorization
import streamlit as st
import copy

# Mocking session state for get_prompts_from_history
@pytest.fixture
def mock_session_state():
    st.session_state.messages = [
        {"role": "user", "content": "Hello, how are you?"},
        {"role": "assistant", "content": "I am fine, thank you!"}
    ]

def test_extract_image_urls():
    text = "Check this image: https://example.com/image.jpg and this one: https://example.com/image2.png"
    expected_urls = ["https://example.com/image.jpg", "https://example.com/image2.png"]
    assert extract_image_urls(text) == expected_urls, "Should contain URLs."

def test_load_context():
    vectorization = utils.vectorization.Vectorization()
    data =copy.deepcopy(vectorization.context_data)
    
    # Ensure the data is a list
    assert isinstance(data, list), "JSON data should be a list of dictionaries."
    
    for entry in data:
        # Ensure each entry is a dictionary
        assert isinstance(entry, dict), "Each item in the list should be a dictionary."
        
        # Ensure required keys exist
        assert "summary" in entry, "Each dictionary should have a 'summary' key."
        assert "description" in entry, "Each dictionary should have an 'description' key."
        assert "tags" in entry, "Each dictionary should have an 'tags' key."
        
        # Ensure the values are strings
        assert isinstance(entry["summary"], str), "'summary' should be a string."
        assert isinstance(entry["description"], str), "'description' should be a string."
        assert isinstance(entry["tags"], list), "'tags' should be a list."

def test_get_history(mock_session_state):
    history = get_history()
    assert "user: Hello, how are you?" in history, "String should be in history."
    assert "assistant: I am fine, thank you!" in history, "String should be in history."

def test_get_history_max(mock_session_state):
    history = get_history(1)
    print(history)
    assert not "user: Hello, how are you?" in history, "String should not be in history."
    assert "assistant: I am fine, thank you!" in history, "String should be in history."
