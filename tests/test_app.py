import pytest
import json
from chat_assist import extract_image_urls, get_history
import streamlit as st

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


def test_get_history(mock_session_state):
    history = get_history()
    assert "user: Hello, how are you?" in history, "String should be in history."
    assert "assistant: I am fine, thank you!" in history, "String should be in history."

def test_get_history_max(mock_session_state):
    history = get_history(1)
    print(history)
    assert not "user: Hello, how are you?" in history, "String should not be in history."
    assert "assistant: I am fine, thank you!" in history, "String should be in history."
