import streamlit as st
import json
import docx
import pdfplumber
import yaml
from ollama import Client
from utils.context_search import ContextSearch
import utils.text_handler as th

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

    # File uploader with PDF support
    uploaded_file = st.file_uploader(
        "Upload a text, DOCX, or PDF file for additional context",
        type=["txt", "json", "docx", "pdf"]
    )
    file_content = ""
    if uploaded_file is not None:
        if uploaded_file.type == "text/plain":
            file_content = uploaded_file.read().decode("cp1252", errors="replace")
        elif uploaded_file.type == "application/json":
            file_content = json.dumps(json.load(uploaded_file))
        elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            doc = docx.Document(uploaded_file)
            file_content = "\n".join(para.text for para in doc.paragraphs)
        elif uploaded_file.type == "application/pdf":
            with pdfplumber.open(uploaded_file) as pdf:
                file_content = "\n".join(page.extract_text() or "" for page in pdf.pages)

    # Button to include full document
    include_full_doc = st.checkbox("Include full document in prompt", value=False)

# Initialize ContextSearch with uploaded context (if available)
context_search = ContextSearch(file_content=file_content)

def get_history(max=max_history_in_prompt):
    """
    Collects the last 'max' messages in session state and returns them as
    user/assistant lines for context.
    """
    return "\n".join(f"{msg['role']}: {msg['content']}" for msg in st.session_state.messages[-max:])

# Initialize the conversation memory
if "messages" not in st.session_state:
    st.session_state.messages = []

# Render the conversation history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        for url in th.extract_image_urls(message["content"]):
            st.image(url, use_container_width=True)

# Capture user input
if prompt := st.chat_input("Enter your message"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Construct the final prompt based on context retrieval or full document
    if uploaded_file:
        # Option 1: Use full document in the prompt
        if include_full_doc:
            final_prompt = f"{context_prompt}\n" \
                           f"{th.format_file_context(uploaded_file.name, file_content)}\n" \
                           f"{get_history()}\n"
        # Option 2: Retrieve top-k chunks
        else:
            retrieved_contexts = context_search.query(prompt, top_k=3)  # adjust top_k as needed
            if retrieved_contexts:
                # Combine relevant chunks
                context_text = "\n".join(
                    f"Chunk #{i+1} (Score: {res['score']:.2f}):\n{res['chunk']}"
                    for i, res in enumerate(retrieved_contexts)
                )
                final_prompt = f"{context_prompt}\n" \
                           f"{th.format_file_context(uploaded_file.name, context_text)}\n" \
                           f"{get_history()}\n"
            else:
                final_prompt = f"{context_prompt}\n\n{get_history()}\n"
    else:
        # No file uploaded, just use conversation history
        final_prompt = f"{context_prompt}\n\n{get_history()}\n"

    # Display the relevant context (only if not using full doc)
    if uploaded_file and not include_full_doc:
        retrieved_contexts = context_search.query(prompt, top_k=3)
        if retrieved_contexts:
            with st.expander("Relevant Context Matches"):
                for i, res in enumerate(retrieved_contexts):
                    st.markdown(f"**Chunk #{i+1} (Score {res['score']:.2f}):**\n{res['chunk']}")

    # Stream the response
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        main_text = ""
        thinking_text = ""
        in_think = False

        # Debug printing
        print("-------------------------------------------------------------------")
        print(final_prompt)
        print("-------------------------------------------------------------------")

        stream = client.generate(
            model=model_options[selected_model],
            prompt=final_prompt,
            options={"temperature": 0.6},
            stream=True
        )

        # Process streaming chunks
        for chunk in stream:
            text = chunk["response"]
            if not in_think:
                if "<think>" in text:
                    parts = text.split("<think>", 1)
                    main_text += "<think>\n"
                    in_think = True
                    if "</think>" in parts[1]:
                        think_part, after_think = parts[1].split("</think>", 1)
                        thinking_text += think_part
                        in_think = False
                        main_text += after_think
                    else:
                        thinking_text += parts[1]
                else:
                    main_text += text
            else:
                if "</think>" in text:
                    think_part, after_think = text.split("</think>", 1)
                    thinking_text += think_part
                    in_think = False
                    main_text += after_think
                else:
                    thinking_text += text

            response_placeholder.markdown(main_text)

    # Optionally display the bot's internal reasoning
    if thinking_text.strip():
        with st.expander("Bot's Internal Reasoning"):
            st.markdown(thinking_text)

    # Add assistant response to session messages
    st.session_state.messages.append({"role": "assistant", "content": main_text.replace("<think>\n", "")})
