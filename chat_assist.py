import streamlit as st
import yaml
from ollama import Client
from utils.context_search import ContextSearch
import utils.text_handler as th
from utils.streaming import StreamingResponseHandler
from utils.prompt_builder import PromptBuilder
from utils.file_handler import FileHandler
from utils.conversation import ConversationManager

# Note: Run with --server.fileWatcherType=none to avoid RuntimeError from torch.classes in Streamlit 1.42.0 watcher
max_history_in_prompt = 6
context_prompt = """You are an AI assistant. Use the following extracted information (if relevant) 
to provide a helpful and accurate answer to the user's question. 
If you do not see relevant information in the context, respond accordingly.
"""

client = Client(host="http://localhost:11434")

def load_config():
    try:
        with open("config.yml", "r") as file:
            return yaml.safe_load(file)
    except FileNotFoundError:
        st.error("config.yml not found. Please create it with the required fields.")
        return {}

# Load configuration values
config = load_config()
model_options = config.get("model_options")

st.title("Chat Assist")

# Sidebar for model selection and file upload
with st.sidebar:
    st.header("Settings")
    selected_model = st.selectbox(
        "Select the AI model:",
        options=list(model_options.keys()),
        index=len(model_options) - 1
    )

    # File uploader with support for multiple file types
    uploaded_file = FileHandler.get_file_uploader("Upload a file for additional context")
    file_content = ""
    if uploaded_file is not None:
        file_content = FileHandler.extract_content(uploaded_file)

    # Button to include full document
    include_full_doc = st.checkbox("Include full document in prompt", value=False)

# Initialize ContextSearch with uploaded context (if available)
context_search = ContextSearch(file_content=file_content)

# Initialize the prompt builder
prompt_builder = PromptBuilder(context_prompt, context_search)

# Initialize the conversation memory
ConversationManager.initialize_session()

# Render the conversation history
for message in ConversationManager.get_messages():
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        for url in th.extract_image_urls(message["content"]):
            st.image(url, use_container_width=True)

# Capture user input
if prompt := st.chat_input("Enter your message"):
    # Add user message to conversation history
    ConversationManager.add_message("user", prompt)
    
    with st.chat_message("user"):
        st.markdown(prompt)

    # Build the prompt using the PromptBuilder
    final_prompt = prompt_builder.build_prompt(
        prompt, 
        ConversationManager.get_history(max_history_in_prompt), 
        uploaded_file, 
        file_content, 
        include_full_doc
    )

    # Display the relevant context (only if not using full doc)
    if uploaded_file and not include_full_doc:
        retrieved_contexts = prompt_builder.get_relevant_context(prompt, top_k=3)
        if retrieved_contexts:
            with st.expander("Relevant Context Matches"):
                for i, res in enumerate(retrieved_contexts):
                    st.markdown(f"**Chunk #{i+1} (Score {res['score']:.2f}):**\n{res['chunk']}")

    # Debug printing
    print("-------------------------------------------------------------------")
    print(final_prompt)
    print("-------------------------------------------------------------------")

    # Initialize the streaming response handler
    handler = StreamingResponseHandler()

    # Generate response stream
    stream = client.generate(
        model=model_options[selected_model],
        prompt=final_prompt,
        options={"temperature": 0.6},
        stream=True
    )

    # Process streaming chunks using the handler
    for chunk in stream:
        handler.process_chunk(chunk)

    # Finalize the response and get the main text
    main_text = handler.finalize()

    # Add the assistant response to the conversation history
    ConversationManager.add_message("assistant", main_text)
