import streamlit as st
import json
import re
from ollama import Client

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
                st.session_state.file_context = json.dumps(parsed_json, indent=2)
            except json.JSONDecodeError:
                st.error("Invalid JSON file")
        else:
            st.session_state.file_context = file_content

max_history_in_prompt = 6
context_prompt = "You are an AI assistant, answering user questions accurately. Use the following additional \
context only when relevant to the question or when it aligns with the provided topics"

client = Client(host="http://localhost:11434")

def extract_image_urls(text):
    return re.findall(r"(https?://\S+\.(?:png|jpg|jpeg|gif))", text)

def get_history(max=max_history_in_prompt):
    lines = []
    messages = st.session_state.messages
    for message in messages[-max:]:
        lines.append(f"{message['role']}: {message['content']}")
    return "\n".join(lines)

if "messages" not in st.session_state:
    st.session_state.messages = []

# Render the conversation history.
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        for url in extract_image_urls(message["content"]):
            st.image(url, use_container_width=True)

# Get user input using st.chat_input.
if prompt := st.chat_input("Enter your message"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    file_context = st.session_state.get("file_context", "")
    if uploaded_file is not None:
        file_template = """[file name]: {file_name}\n[file content begin]\n{file_content}\n[file content end]\n{question}"""
        formatted_file_context = file_template.format(file_name=uploaded_file.name, file_content=file_context, question=prompt)
    else:
        formatted_file_context = file_context
    
    final_prompt = f"{context_prompt} {formatted_file_context}\n{get_history()}"
    
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        main_text = ""
        thinking_text = ""
        in_think = False
        print("-------------------------------------------------------------------------")
        print(final_prompt)
        print("-------------------------------------------------------------------------")
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

    st.session_state.messages.append({"role": "assistant", "content": main_text})
