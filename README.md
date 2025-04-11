# Chat Assist

Chat Assist is an AI-powered chatbot built using `streamlit` and `ollama`, with a contextual search system leveraging `FAISS` and `sentence-transformers` for relevant response retrieval.

![Chat Assist UI](chat_assist_ui.png)

## Features

- **Multiple AI Model Support**: Supports models such as DeepSeek-R1 (1.5B, 8B), LLaMA 3.1 (8B), LLaMA 3.2 (3B), Mistral 7B, Gemma3 4B, and more.
- **Context-Aware Responses**: Uses vectorized context matching to enhance response accuracy.
- **Chat History Retention**: Maintains conversation history to provide coherent interactions.
- **Embedded Context Search**: Uses FAISS to retrieve relevant stored context dynamically.
- **File-Based Context Expansion**: Allows users to upload files for additional context.

## Installation

### Prerequisites

- Python 3.8+
- Install required dependencies:
  ```sh
  pip install -r requirements.txt
  ```

### Install Ollama
Ollama is required to run AI models locally. Follow the steps below to install it:

- **macOS**: Download the installer from the [Ollama website](https://ollama.com/download) and follow the instructions.
- **Linux**: Open a terminal and run:
  ```sh
  curl -fsSL https://ollama.com/install.sh | sh
  ```
  Refer to the [Ollama Linux guide](https://github.com/ollama/ollama/blob/main/docs/linux.md) for more details.
- **Windows**: Download and install Ollama from the [official website](https://ollama.com/download).

### Install AI Models
After installing Ollama, download and run the required AI models, e.g.:

```sh
ollama run llama3.1
```

This command will automatically download and install the LLaMA 3.1 model. For more models, visit the [Ollama model library](https://ollama.com/library).

### Configure AI Models
AI models are configured in the `config.yml` file, which lists available models and their configurations. This list can be extended to include additional models. The file structure looks like this:
```yaml
model_options:
  "DeepSeek-R1 1.5B": "deepseek-r1:1.5b"
  "DeepSeek-R1 8B": "deepseek-r1:8b"
  "llama 3.1 8B": "llama3.1:8b"
  "llama 3.2 3B": "llama3.2"
  "Mistral 7B": "mistral"
  "Gemma3 4B": "gemma3:4b"
```

## Usage

1. Start the chatbot:

   To guarantee that your app is only accessible from your machine, run Streamlit with the `--server.address` option set to localhost. It's also recommended to disable the file watcher to avoid RuntimeError from torch.classes:
   ```sh
   streamlit run chat_assist.py --server.address localhost --server.fileWatcherType none
   ```

2. Select the AI model from the dropdown menu.
3. Upload an optional file to provide additional context.
4. Choose whether to include the full document or use smart context search.
5. Enter your message in the chat input.
6. The bot retrieves relevant context and generates a response.

## How It Works

Chat Assist uses smart document indexing to find information relevant to your questions. It creates vectors from document chunks, searches for matches to your queries, and combines the results with chat history to generate informed responses. The interface is built with `streamlit`, while AI processing happens via `ollama`.

## Future Improvements

- Enhance model selection.
- Improve context matching and retrieval.
- Support for additional AI models.
- Optimize performance for large document uploads.

## License

This project is licensed under the MIT License.
