# Chat Assist

Chat Assist is an AI-powered chatbot built using `streamlit` and `ollama`, with a contextual search system leveraging `FAISS` and `sentence-transformers` for relevant response retrieval.

## Features

- **Multiple AI Model Support**: Supports models such as DeepSeek-R1 (1.5B, 8B, 14B) and LLaMA 3.1 (8B).
- **Context-Aware Responses**: Uses vectorized context matching to enhance response accuracy.
- **Chat History Retention**: Maintains conversation history to provide coherent interactions.
- **Embedded Context Search**: Uses FAISS to retrieve relevant stored context dynamically.

## Installation

### Prerequisites

- Python 3.8+
- Install required dependencies:
  ```sh
  pip install -r requirements.txt
  ```

## Usage

1. Start the chatbot:
   ```sh
   streamlit run chat_assist.py
   ```
2. Select the AI model from the dropdown menu.
3. Enter your message in the chat input.
4. The bot retrieves relevant context and generates a response using an embedded context search system.

## How It Works

Chat Assist leverages FAISS-based vectorization to dynamically retrieve relevant context from a predefined dataset. The chatbot interface is built with `streamlit`, while the AI model is hosted via `ollama`.

## Example Query

```
User: "Tell me about autonomous vehicles in heavy rain."
Bot: "Autonomous vehicles face challenges in heavy rain due to sensor interference and reduced visibility. Relevant studies suggest..."
```

## Future Improvements

- Enhance model selection.
- Improve context matching and retrieval.
- Support for additional AI models.

## License

This project is licensed under the MIT License.
