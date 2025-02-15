import streamlit as st
import json
from ollama import Client
from utils.context_search import ContextSearch
import utils.text_handler as th

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
        "llama 3.1 8B": "llama3.1:8b"
    }
    selected_model = st.selectbox(
        "Select the AI model:",
        options=list(model_options.keys()),
        index=1
    )
    
    # File uploader
    uploaded_file = st.file_uploader("Upload a text file for additional context", type=["txt", "json"])
    if uploaded_file is not None:
        file_content = uploaded_file.read().decode("utf-8")
        if uploaded_file.name.endswith(".json"):
            try:
                parsed_json = json.loads(file_content)
                file_content = json.dumps(parsed_json, indent=2)
            except json.JSONDecodeError:
                st.error("Invalid JSON file")
        
        st.session_state.file_context = th.chunk_text_by_sections(file_content)  # Store as list for FAISS indexing

# Initialize ContextSearch with uploaded context (if available)
context_search = ContextSearch(context_data=st.session_state.get("file_context", []))


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
    relevant_file_context = context_search.find_relevant_context(prompt)

    if uploaded_file and relevant_file_context.strip():
        final_prompt = f"{context_prompt}\n{th.format_file_context(uploaded_file.name, relevant_file_context)}\n{get_history()}\n\n"
    else:
        final_prompt = f"{context_prompt}\n\n{get_history()}\n\n"

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
    
    if uploaded_file and relevant_file_context.strip():
        with st.expander("Relevant Context Matches"):
            st.markdown(f"```{th.format_file_context(uploaded_file.name, relevant_file_context)}```")

    st.session_state.messages.append({"role": "assistant", "content": main_text.replace("<think>\n", "")})
