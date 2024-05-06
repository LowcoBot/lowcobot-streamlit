import os

if not os.path.exists("tools"):
    raise Exception("The 'tools/' folder does not exist!")

import re

import streamlit as st
from agent import LLMAgent
from tools.prompt import getTitle

st.set_page_config(
    page_title=getTitle(),
)
st.title(getTitle())

if "agent" not in st.session_state:
    st.session_state.agent = LLMAgent()

if "messages" not in st.session_state:
    st.session_state.messages = []

if "current_idx" not in st.session_state:
    st.session_state.current_idx = 0


def form_callback(form_data):
    new_index = st.session_state.current_idx + 1
    st.session_state.current_idx = new_index
    with st.spinner("Loading an answer..."):
        prompt = f"Go ahead with {form_data}"
        response_output = st.session_state.agent.run_question(prompt)
        st.session_state.messages.append({"type": "assistant", "text": response_output})


def print_message(message: str, key: str):
    """
    Given something like: "Please, introduce your @@name@@, @@surname@@ and @@age@@."
    It will print: "Please, introduce your name, surname and age", and detects that name, surname and age are placeholders
    """

    # Detect placeholders
    placeholders = re.findall(r"@@(.*?)@@", message)
    message = message.replace("@@", "")

    # Print message
    st.markdown(message)

    # Print textareas per placeholder
    for placeholder in placeholders:
        st.text_input(placeholder, key=f"fields|{key}_{placeholder}")

    if placeholders:
        start = f"fields|{key}_"
        keys = st.session_state.keys()
        filtered_keys = filter(lambda x: x.startswith(start), keys)
        form_data = {k.replace(start, ""): st.session_state[k] for k in filtered_keys}
        st.button(
            "Send ➡️", key=f"button|{key}", on_click=form_callback, args=(form_data,)
        )


for idx, message in enumerate(st.session_state.messages):
    type = message["type"]
    with st.chat_message(type):
        print_message(message["text"], key=idx)

if prompt := st.chat_input("Talk with " + getTitle() + "..."):
    new_index = st.session_state.current_idx + 1
    st.session_state.current_idx = new_index

    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"type": "user", "text": prompt})

    with st.spinner("Loading an answer..."):
        response_output = st.session_state.agent.run_question(prompt)
        with st.chat_message("assistant"):
            print_message(response_output, key=str(new_index))
        st.session_state.messages.append({"type": "assistant", "text": response_output})
