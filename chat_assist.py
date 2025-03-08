import streamlit as st
import json
import docx
import pdfplumber
from ollama import Client
from utils.context_search import ContextSearch
import utils.text_handler as th

# Note: Run with --server.fileWatcherType=none to avoid RuntimeError from torch.classes in Streamlit 1.42.0 watcher
max_history_in_prompt = 6
context_prompt = "You are an AI assistant, answering user questions accurately."

client = Client(host="http://localhost:11434")

st.title("Chat Assist")

# Sidebar for model selection and file upload
with st.sidebar:
    st.header("Settings")
    model_options = {
        "DeepSeek-R1 1.5B": "deepseek-r1:1.5b",
        "DeepSeek-R1 8B": "deepseek-r1:8b",
        "DeepSeek-R1 14B": "deepseek-r1:14b",
        "llama 3.1 8B": "llama3.1:8b",
        "llama 3.2 3B": "llama3.2",
        "Mistral 7B": "mistral",
        "Phi 4": "phi4"
    }
    selected_model = st.selectbox(
        "Select the AI model:",
        options=list(model_options.keys()),
        index=1
    )
    
    # File uploader with PDF support
    uploaded_file = st.file_uploader(
        "Upload a text, DOCX, or PDF file for additional context",
        type=["txt", "json", "docx", "pdf"]
    )
    file_content = ""
    if uploaded_file is not None:
        if uploaded_file.type == "text/plain":
            file_content = uploaded_file.read().decode("utf-8")
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
    return "\n".join(f"{msg['role']}: {msg['content']}" for msg in st.session_state.messages[-max:])

if "messages" not in st.session_state:
    st.session_state.messages = []

# Render the conversation history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        for url in th.extract_image_urls(message["content"]):
            st.image(url, use_container_width=True)

# Get user input
if prompt := st.chat_input("Enter your message"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Retrieve relevant context
    relevant_file_context = context_search.query(prompt)

    if uploaded_file:
        if include_full_doc:
            final_prompt = f"{context_prompt}\nFull document content:\n{file_content}\n\n{get_history()}\n"
        elif relevant_file_context:
            final_prompt = f"{context_prompt}\n{th.format_file_context(uploaded_file.name, relevant_file_context)}\n{get_history()}\n"
        else:
            final_prompt = f"{context_prompt}\n\n{get_history()}\n"
    else:
        final_prompt = f"{context_prompt}\n\n{get_history()}\n"

    if uploaded_file and relevant_file_context and not include_full_doc:
        with st.expander("Relevant Context Matches"):
            st.markdown(f"{th.format_file_context(uploaded_file.name, relevant_file_context)}")

    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        main_text = ""
        thinking_text = ""
        in_think = False

        print("-------------------------------------------------------------------")
        print(final_prompt)
        print("-------------------------------------------------------------------")

        stream = client.generate(
            model=model_options[selected_model],
            prompt=final_prompt,
            options={"temperature": 0.6},
            stream=True
        )

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

    if thinking_text.strip():
        with st.expander("Bot's Internal Reasoning"):
            st.markdown(thinking_text)

    st.session_state.messages.append({"role": "assistant", "content": main_text.replace("<think>\n", "")})