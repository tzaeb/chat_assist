import streamlit as st
import json
import re
from ollama import Client
import torch
import utils.vectorization

# avoid RuntimeError: Tried to instantiate class '__path__._path'...
torch.classes.__path__ = []

st.title("Chat Assist")

# Model selection
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

max_history_in_prompt = 6
context_prompt = "You are an AI assistant, answering user questions accurately. Use the following additional \
                    context only when relevant to the question or when it aligns with the provided topics:"
context_file = "context.json"
client = Client(host="http://localhost:11434")

# vectorize context
vectorization = utils.vectorization.Vectorization(context_file)

def extract_image_urls(text):
    return re.findall(r"(https?://\S+\.(?:png|jpg|jpeg|gif))", text)

def get_history(max=max_history_in_prompt):
    lines = []
    messages = st.session_state.messages
    #print(messages)
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
    # Append and display the user's message.
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    results = vectorization.find_matches(prompt)  # Fetch top 3 matches
    matches = json.dumps(results)
    final_prompt = f"system:{context_prompt} {str(matches)}\n{get_history()}"
    # Use a single assistant chat message block to stream the response.
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        main_text = ""
        thinking_text = ""
        in_think = False  # True while inside a <think> block.
        stream = client.generate(model=model_options[selected_model], prompt=final_prompt, stream=True)
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
    if len(results)>0:
        with st.expander("Relevant Context Matches"):
            st.markdown("```json\n" + json.dumps(results, indent=2) + "\n```")

    # Append the final assistant message only once to session state.
    st.session_state.messages.append({"role": "assistant", "content": main_text})
 