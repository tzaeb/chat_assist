import streamlit as st
import yaml
import re
from ollama import Client
from utils.context_search import ContextSearch
from utils.streaming import StreamingResponseHandler
from utils.prompt_builder import PromptBuilder
from utils.file_handler import FileHandler
from utils.conversation import ConversationManager

# Note: Run with --server.fileWatcherType=none to avoid RuntimeError from torch.classes in Streamlit 1.42.0 watcher
max_history_in_prompt = 6
context_prompt = "Use the following context (if relevant) to answer the question. If no relevant info is found, say so."

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
    
    if uploaded_file is not None:
        with st.spinner('Processing uploaded file...'):
            file_content = FileHandler.get_buffered_content(uploaded_file)
    else:
        file_content = None

    # Button to include full document
    include_full_doc = st.checkbox("Include full document", value=False)
    
    # Add a horizontal separator
    st.sidebar.divider()
    
    # Add a clear chat history section
    st.sidebar.markdown("### Chat Management")
    
    # Initialize confirmation state if it doesn't exist
    if "confirm_clear" not in st.session_state:
        st.session_state.confirm_clear = False
    
    # Primary clear history button
    if not st.session_state.confirm_clear:
        if st.sidebar.button("🗑️ Clear Chat History", use_container_width=True):
            st.session_state.confirm_clear = True
            st.rerun()
    
    # Show confirmation UI when confirm_clear is True
    if st.session_state.confirm_clear:
        st.sidebar.warning("Are you sure? This cannot be undone.")
        col1, col2 = st.sidebar.columns(2)
        
        with col1:
            if st.button("Yes, Clear", type="primary", use_container_width=True):
                # Clear the conversation history
                ConversationManager.clear_history()
                st.session_state.confirm_clear = False
                st.sidebar.success("Chat history cleared!")
                st.rerun()
        
        with col2:
            if st.button("Cancel", use_container_width=True):
                st.session_state.confirm_clear = False
                st.rerun()

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
        image_urls = re.findall(r"(https?://\S+\.(?:png|jpg|jpeg|gif))", message["content"])
        for url in image_urls:
            st.image(url, use_container_width=True)

# Capture user input
if prompt := st.chat_input("Enter your message"):
    # Add user message to conversation history
    ConversationManager.add_message("user", prompt)
    
    with st.chat_message("user"):
        st.markdown(prompt)

    # Build the prompt using the PromptBuilder
    with st.spinner('Indexing and searching for relevant context...'):
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
    with st.spinner('Generating response...'):
        stream = client.generate(
            model=model_options[selected_model],
            prompt=final_prompt,
            options={"temperature": 0.6},
            stream=True
        )
        
        # Process first chunk to stop the spinner
        first_chunk = next(stream)
        handler.process_chunk(first_chunk)

    # Process remaining chunks without the spinner
    for chunk in stream:
        handler.process_chunk(chunk)

    # Finalize the response and get the main text
    main_text = handler.finalize()

    # Add the assistant response to the conversation history
    ConversationManager.add_message("assistant", main_text)
