# Chat Assist

A Streamlit-based chatbot application that leverages AI models to provide responses. The bot includes features like conversation history, image display, and internal reasoning.

## Features

- **AI-Powered Responses**: Uses the DeepSeek-R1-1.5B model via Ollama for intelligent interactions.
- **Conversation History**: Maintains a chat history with up to 6 previous messages to provide context.
- **Image Display**: Extracts and displays image URLs from chat messages using regular expressions.
- **Internal Reasoning**: Shows detailed reasoning steps when available, stored in an expander section.

## Dependencies

To run this application, ensure the following libraries are installed:

streamlit json re ollama

## Setup Instructions

1. **Install Dependencies**:
   Run `pip install streamlit json re ollama`.

2. **Configure Ollama**:
   Ensure you have Ollama installed and running locally on port `11434`.

3. **Prepare Additional Context**:
   Create a JSON file named `additional_context.json` containing any additional context to be used by the AI.

4. **Run the Application**:
   Execute with `streamlit run main.py`.

## Usage

- Enter your messages in the chat input.
- The bot will respond using its AI model.
- Images included in messages will be displayed inline.
- Detailed reasoning steps (if any) will appear as an expander section.

## Notes

- **Context**: Additional context is loaded from `additional_context.json` and provided to the AI as part of the prompt.
- **Images**: URLs for images are extracted using regular expressions and displayed within the chat interface.