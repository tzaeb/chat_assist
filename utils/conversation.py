import streamlit as st
from typing import List, Dict, Any, Optional, Union

class ConversationManager:
    """
    Manages conversation history, formatting, and state in the Streamlit session.
    """
    
    @staticmethod
    def initialize_session():
        """Initialize the conversation memory in the session state if it doesn't exist."""
        if "messages" not in st.session_state:
            st.session_state.messages = []
    
    @staticmethod
    def get_history(max_messages: int = 6) -> str:
        """
        Collects the last 'max_messages' messages in session state and returns them as
        user/assistant lines for context.
        
        Args:
            max_messages: Maximum number of messages to include in the history.
            
        Returns:
            A string representation of the conversation history.
        """
        if "messages" not in st.session_state or not st.session_state.messages:
            return ""
            
        messages = st.session_state.messages[-max_messages:] if max_messages > 0 else st.session_state.messages
        return "\n".join(f"{msg['role']}: {msg['content']}" for msg in messages)
    
    @staticmethod
    def add_message(role: str, content: str):
        """
        Add a message to the conversation history.
        
        Args:
            role: The role of the message sender (e.g., "user", "assistant").
            content: The content of the message.
        """
        ConversationManager.initialize_session()
        st.session_state.messages.append({"role": role, "content": content})
    
    @staticmethod
    def get_messages(max_messages: Optional[int] = None) -> List[Dict[str, str]]:
        """
        Get the conversation history messages.
        
        Args:
            max_messages: Maximum number of most recent messages to return, or None for all.
            
        Returns:
            A list of message dictionaries.
        """
        ConversationManager.initialize_session()
        
        if max_messages is not None and max_messages > 0:
            return st.session_state.messages[-max_messages:]
        
        return st.session_state.messages
    
    @staticmethod
    def clear_history():
        """Clear the conversation history."""
        st.session_state.messages = [] 