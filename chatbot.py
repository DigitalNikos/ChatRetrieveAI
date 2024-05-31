import os
import tempfile
import streamlit as st
import time
from streamlit_chat import message
from rag import ChatPDF
from langchain_core.documents import Document




st.set_page_config(page_title="ChatPDF")


def display_messages():
    # st.subheader("Chat")
    # for i, (msg, is_user) in enumerate(st.session_state["messages"]):
    #     message(msg, is_user=is_user, key=str(i))
    # st.session_state["thinking_spinner"] = st.empty()

    st.subheader("Chat")
    for i, (msg, is_user) in enumerate(st.session_state["messages"]):
        # Ensure the message is JSON serializable
        if isinstance(msg, str):
            display_msg = msg
        elif isinstance(msg, Document):  
            display_msg = msg.page_content  
        else:
            display_msg = str(msg)  # Convert to string or handle as needed
        
        message(display_msg, is_user=is_user, key=str(i))
    st.session_state["thinking_spinner"] = st.empty()


def process_input():
    if st.session_state["user_input"] and len(st.session_state["user_input"].strip()) > 0:
        user_text = st.session_state["user_input"].strip()
        with st.session_state["thinking_spinner"], st.spinner(f"Thinking"):
            agent_text = st.session_state["assistant"].ask(user_text)

        st.session_state["messages"].append((user_text, True))
        st.session_state["messages"].append((agent_text, False))


def read_and_save_file(domain):
    print("===read_and_save_file===")  
    # st.session_state["assistant"].clear()
    # st.session_state["messages"] = []
    st.session_state["user_input"] = ""

    for file in st.session_state["file_uploader"]:
        _, file_extension = os.path.splitext(file.name)
        file_extension = file_extension.lower()

        with tempfile.NamedTemporaryFile(delete=False) as tf:
            tf.write(file.getbuffer())
            file_path = tf.name

        with st.session_state["ingestion_spinner"], st.spinner(f"Ingesting {file.name}"):
            # domain = st.session_state["domain"]
            answer = st.session_state["assistant"].ingest(file_path, domain, file.name)
            if answer == "no":
                st.session_state["messages"].append(("Document does not fall within the specified domain", False))
        os.remove(file_path)

def initialize_session_state():
    if len(st.session_state) == 0:
        clear_session_state()

def clear_session_state():
    st.session_state["messages"] = []
    st.session_state["assistant"] = ChatPDF()
    st.session_state["domain"] = ""


def page():
    print("===page()===")
    initialize_session_state()

    # Update the header with the domain if set
    if st.session_state["domain"].strip() == "":
        st.header("🤖 Chatbot")
    else:
        
        col1, col2 = st.columns([4, 1])
        with col1:
            st.header(f"🤖 Chatbot {st.session_state['domain']} domain")
        with col2:
            if st.button("Reset Domain", type="primary"):
                clear_session_state()
                st.rerun()

    if st.session_state["domain"].strip() == "":
        with st.form('chat_input_form'):
              col1, col2 = st.columns([1,1])     
        with col1:          
            domain = st.text_input( 
                "Enter a domain to start the chat.",
                placeholder='Enter you specific domain to start the chat.',
                label_visibility='collapsed'
            ) 
        with col2:
            st.form_submit_button('Add Domain')     
            st.session_state["domain"] = domain
            st.session_state["assistant"].set_domain(domain)
        
        if st.session_state["domain"].strip() != "":
            st.rerun() 
    else:        
        domain = st.session_state["domain"]

        st.file_uploader(
            "Upload document",
            type=["pdf"],
            key="file_uploader",
            on_change=read_and_save_file,
            args = (st.session_state["domain"], ),
            label_visibility="collapsed",
            accept_multiple_files=True,
        )

        st.session_state["ingestion_spinner"] = st.empty()
        display_messages()
        st.text_input("Message", key="user_input", on_change=process_input)


if __name__ == "__main__":
    print("===__main__===")
    page()