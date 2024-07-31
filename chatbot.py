import streamlit as st
from config import Config as cfg

from langchain_core.documents import Document
from streamlit_chat import message

from extract_source import upload_pdf, upload_url
from rag.rag import ChatPDF


st.set_page_config(page_title="ChatPDF", page_icon="🤖")

def display_messages():
    """Display chat messages in the Streamlit app."""
    print("\nCalling =>chatbot.py - display_messages()")

    for i, (msg, is_user) in enumerate(st.session_state["messages"]):
        display_msg = msg.page_content if isinstance(msg, Document) else str(msg)
        message(display_msg, is_user=is_user, key=str(i), avatar_style="bottts", seed=2)
    st.session_state["thinking_spinner"] = st.empty()

def process_input():
    """Process user input and get a response from the assistant."""
    print("\nCalling =>chatbot.py - process_input()")
    
    if st.session_state["user_input"] and len(st.session_state["user_input"].strip()) > 0:
        user_text = st.session_state["user_input"].strip()
        with st.session_state["thinking_spinner"], st.spinner(f"Thinking"):
            agent_text = st.session_state["assistant"].ask(user_text)

        st.session_state["messages"].append((user_text, True))
        st.session_state["messages"].append((agent_text, False))


def read_and_save_file(domain: str, source_type: str):
    """Handle file or URL upload and initiate document ingestion."""
    print("\nCalling =>chatbot.py - read_and_save_file()")

    action = cfg.UPLOAD_ACTIONS.get(source_type)
    if action:
        action(domain, st)
    else:
        raise ValueError(f"Invalid source type: {source_type}. Expected '.pdf', '.docx', '.txt' or 'url'.")

def initialize_session_state():
    if len(st.session_state) == 0:
        clear_session_state()

def clear_session_state():
    st.session_state["messages"] = []
    st.session_state["assistant"] = ChatPDF(cfg)
    st.session_state["domain"] = ""


def page():
    """Main function to handle the Streamlit page layout and interactions."""
    print("\nCalling =>chatbot.py - page()")
    initialize_session_state()

    # Update the header with the domain if set
    if st.session_state["domain"].strip() == "":
        st.header("Chatbot 🤖 💬")
        st.caption("🚀 ChatRetrieveAI: Using RAG for Document and Wikipedia Interaction.")
    else: 
        col1, col2 = st.columns([4, 1])
        with col1:
            st.header(f"Chatbot {st.session_state['domain']} domain 🤖 💬")
            st.caption("🚀 ChatRetrieveAI: Using RAG for Document and Web search Interaction.")
            if not st.session_state["messages"]:
                domain_message = f"Welcome to our  {st.session_state['domain']} domain support chat! How can I assist you today?"
                st.session_state["messages"] = [(domain_message, False)]
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
        with st.sidebar:          
            file_upload = st.file_uploader(
                "Upload document",
                type=["pdf", "docx", "txt"],
                key="file_uploader",
                on_change=read_and_save_file,
                args = (st.session_state["domain"], "document" ),
                label_visibility="collapsed",
                accept_multiple_files=True,
            )
            print("File Uploader", st.session_state["file_uploader"])
            
            try:
                st.text_input(
                    "Give Url", 
                    help = "Enter the URL of the webpage you want to ingest. The URL should be publicly accessible and not require any authentication. The URL should start with 'http' or 'https'.",
                    key="url_upload",
                    on_change=read_and_save_file,
                    args = (st.session_state["domain"], "url" ),)
            except:
                #TODO need to handle this error???
                # st.error("Invalid URL")
                st.write("Invalidd URL")

        st.session_state["ingestion_spinner"] = st.empty()
        display_messages()
        st.chat_input("Send message to Chatbot", key="user_input", on_submit=process_input)


if __name__ == "__main__":
    print("Calling =>chatbot.py - __main__")
    page()