import streamlit as st
import json
import re
from ollama import Client

st.title("Chat Assist")

# Configure the AI model client.
ai_model = "deepseek-r1:1.5b"
max_history_in_prompt = 6
additional_context_prompt = "You are an AI assistant, answering user questions accurately. Use the following additional \
                    context when relevant to the question or when it aligns with the provided topics:"
additional_context_file = "additional_context.json"
client = Client(host="http://localhost:11434")

@st.cache_data
def load_additional_context():
    with open(additional_context_file, "r") as f:
        data = json.load(f)
    return json.dumps(data,indent=2)

def extract_image_urls(text):
    return re.findall(r"(https?://\S+\.(?:png|jpg|jpeg|gif))", text)

def get_prompts_from_history():
    lines = []
    messages = st.session_state.messages
    #print(messages)
    for message in messages[-max_history_in_prompt:]:
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
    # Append and display the user's message.
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    additional_context = load_additional_context()
    final_prompt = f"{additional_context_prompt} {additional_context}\n{get_prompts_from_history()}"
    # Use a single assistant chat message block to stream the response.
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        main_text = ""
        thinking_text = ""
        in_think = False  # True while inside a <think> block.
        stream = client.generate(model=ai_model, prompt=final_prompt, stream=True)
        for chunk in stream:
            text = chunk["response"]
            if not in_think:
                if "<think>" in text:
                    parts = text.split("<think>", 1)
                    main_text += "<think>\n" #parts[0]
                    in_think = True
                    # If the closing tag is in the same chunk:
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
                # Already in a thinking block.
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

    # Append the final assistant message only once to session state.
    st.session_state.messages.append({"role": "assistant", "content": main_text})
 