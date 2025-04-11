import streamlit as st

class StreamingResponseHandler:
    """
    Handles the streaming response from an LLM, processing thought/think tags
    and updating the UI.
    """
    def __init__(self):
        """Initialize the streaming response handler."""
        self.stream_placeholder = st.empty()
        self.main_text = ""
        self.reasoning_text = ""
        self.in_reasoning = False
        self.current_tag = None
        self.buffer = ""  # Buffer to handle partial tags across chunk boundaries

    def process_chunk(self, chunk):
        """
        Process a chunk of text from the streaming response.
        """
        new_text = chunk["response"]
        # Append new text to the buffer
        self.buffer += new_text
        
        # Process complete tags from buffer
        while self.buffer:
            if not self.in_reasoning:
                # Look for opening tags
                think_pos = self.buffer.find("<think>")
                thought_pos = self.buffer.find("<thought>")
                
                # No opening tags found, add everything up to the last 10 chars to main_text
                # (preserving a possible partial opening tag)
                if think_pos == -1 and thought_pos == -1:
                    if len(self.buffer) > 10:
                        self.main_text += self.buffer[:-10]
                        self.buffer = self.buffer[-10:]
                    break
                
                # Found at least one opening tag
                if think_pos != -1 and (thought_pos == -1 or think_pos < thought_pos):
                    # Add text before the tag to main text
                    self.main_text += self.buffer[:think_pos]
                    # Set state to in_reasoning
                    self.in_reasoning = True
                    self.current_tag = "think"
                    # Remove processed text and the tag from buffer
                    self.buffer = self.buffer[think_pos + len("<think>"):]
                elif thought_pos != -1:
                    # Add text before the tag to main text
                    self.main_text += self.buffer[:thought_pos]
                    # Set state to in_reasoning
                    self.in_reasoning = True
                    self.current_tag = "thought"
                    # Remove processed text and the tag from buffer
                    self.buffer = self.buffer[thought_pos + len("<thought>"):]
            else:
                # In reasoning mode, look for closing tag
                closing_tag = f"</{self.current_tag}>"
                closing_pos = self.buffer.find(closing_tag)
                
                # No closing tag found, add everything up to the last 10 chars to reasoning_text
                # (preserving a possible partial closing tag)
                if closing_pos == -1:
                    if len(self.buffer) > 10:
                        self.reasoning_text += self.buffer[:-10]
                        self.buffer = self.buffer[-10:]
                    break
                
                # Found closing tag
                self.reasoning_text += self.buffer[:closing_pos]
                self.in_reasoning = False
                self.buffer = self.buffer[closing_pos + len(closing_tag):]
        
        # Update the placeholder with the current streamed content
        self._update_placeholder()
    
    def _update_placeholder(self, final=False):
        """Update the UI with the current state of the response."""
        with self.stream_placeholder.container():
            with st.chat_message("assistant"):
                # Display reasoning FIRST in an expander (if it exists)
                if self.reasoning_text.strip():
                    with st.expander("Bot's Internal Reasoning", expanded=not final):  # Only expand during streaming
                        st.markdown(self.reasoning_text)
                
                # Display main text AFTER reasoning
                #cursor = "" if final else "▌"  # Add cursor effect during streaming
                st.markdown(self.main_text)# + cursor)
    
    def finalize(self):
        """
        Finalize the streaming response, showing the final state without the cursor
        and with the reasoning expander collapsed.
        """
        # Process any remaining text in the buffer
        if len(self.buffer) > 0:
            if self.in_reasoning:
                self.reasoning_text += self.buffer
            else:
                self.main_text += self.buffer
            self.buffer = ""
            
        # Update the UI with the final state
        self._update_placeholder(final=True)
        
        # Return the main text to add to the session state
        return self.main_text 