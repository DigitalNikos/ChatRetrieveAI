import time
import requests
import streamlit as st

from rag.rag import ChatPDF
from config import Config 
from utils.upload_source import UploadStatus, upload_document, upload_url
from utils.text_doc_processing import trim_url_to_domain
from utils.ui_helpers import add_divider, add_heading, add_padding, display_table, hide_file_names, add_centered_heading_with_description  


st.set_page_config(page_title="ChatPDF", page_icon="🤖")

def display_messages():
    """
    Display chat messages in the Streamlit app.

    If no messages exist, initialize with a default assistant welcome message.
    Iterate through session messages and display them according to their role 
    (e.g., 'user' or 'assistant') in a chat format.
    """
    print("chatbot.py - display_messages()")

    # Initialize default welcome message if no messages are present
    if not st.session_state["messages"]:
        domain_message = f"Welcome to our  {st.session_state['domain']} domain support chat! How can I assist you today?"
        st.session_state["messages"] = [{"role": "assistant", "content": domain_message}]
        
    # Display all messages from the session state
    for msg in st.session_state["messages"]:
        role, content = (msg["role"], msg["content"]) if isinstance(msg, dict) else msg if isinstance(msg, tuple) else ("system", "Unexpected message format.") 
        st.chat_message(role).write(content)
    
    # Initialize the thinking spinner
    st.session_state["thinking_spinner"] = st.empty()
    

def process_input():
    """
    Process user input and generate a response from the assistant.

    Validates and retrieves user input, sends it to the assistant for a response, 
    and appends both the user input and assistant response to the chat history.
    """
    print("chatbot.py - process_input()")
    
    # Check if the user input is valid
    if st.session_state["user_input"] and len(st.session_state["user_input"].strip()) > 0:
        user_text = st.session_state["user_input"].strip()
        
        # Get response from the assistant while displaying a "thinking" spinner
        with st.session_state["thinking_spinner"], st.spinner(f"Thinking"):
            agent_text = st.session_state["assistant"].ask(user_text)
            
        # Append user input and assistant response to the message history
        st.session_state.messages.append({"role": "user", "content": user_text})
        st.session_state.messages.append({"role": "assistant", "content": agent_text})


def read_and_save_file(domain: str, source_type: str):
    """
    Process file or URL uploads and initiate document ingestion.

    Depending on the `source_type` ('document' or 'url'), this function handles the 
    upload, executes the relevant action (uploading a .pdf, .text, .docx or processing a URL), 
    and updates the session state with the result.

    Raises a ValueError if the `source_type` is invalid.
    """
    print("chatbot.py - read_and_save_file()")
    
    if source_type == "document":
        status = upload_document(domain, st)  
    elif source_type == "url":
        status = upload_url(domain, st)  
    else:
        raise ValueError(f"Invalid source type: {source_type}. Expected 'document' or 'url'.")
    
    if source_type == "url":
        st.session_state["url_status"] = status['status']
    elif source_type == "document":
        st.session_state['document_status'] = {'status': status['status'], 'file_name': status.get('file_name')}
    

def initialize_session_state():
    """
    Initialize the session state if it is currently empty.
    """
    print("chatbot.py - initialize_session_state()")
    
    if len(st.session_state) == 0:
        clear_session_state()
        

def clear_session_state():
    """
        Reset the session state to default values.
    """
    st.session_state["messages"] = []
    st.session_state["assistant"] = ChatPDF()
    st.session_state["domain"] = ""
    st.session_state['urls_store'] = []
    st.session_state['url_upload'] = "" 
    st.session_state['file_names'] = []
    st.session_state["document_status"] = None
    st.session_state["file_hashes"] = []


@st.dialog("Show Uploaded Files")
def show_file_names(names):
    """
    Display a list of uploaded file names in a table format.
    
    If no files are uploaded, a message is displayed.
    """
    if not names:
        st.write("No files uploaded.")
        return
    
    headers = ["#", "File name"]
    rows = [[i + 1, name] for i, name in enumerate(names)]
    display_table(headers, rows, max_visible_items=15)
    

@st.dialog("Upload URL")
def ingest_url():
    """
    Validate and upload a user-provided URL.

    Ensures the URL format is valid, checks for connectivity, and attempts 
    to upload the URL for document ingestion. Displays appropriate messages 
    based on success or failure of the upload process.
    """
    print("chatbot.py - ingest_url()")
    
    st.write(f"Add URL in the input box below and click 'Submit'.")
    url = st.text_input("Enter URL...")
    
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
    
    if st.button("Submit"):
        if url.strip() == "":
            st.error("URL cannot be empty.", icon="🚨")
            return
        
        # Basic validation for URL format
        if not url.startswith("http://") and not url.startswith("https://"):
            st.error("Invalid URL: Please includeeee 'http://' or 'https://'.", icon="🚨")
            time.sleep(4)
            st.rerun()
        
        try:
            response = requests.head(url, headers=headers, timeout=5)  
            print("Status code:", response.status_code)

            if response.status_code != 200:
                st.error("URL is not reachable or returned an error.", icon="🚨")
                return
        except requests.RequestException as e:
            st.error(f"Error reaching the URL: {str(e)}", icon="🚨")
            return
            
        st.session_state.url_upload = url
        read_and_save_file(st.session_state["domain"], "url")
        
        # Check if the URL was successfully processed
        if st.session_state.get("url_status") == UploadStatus.SUCCESS:
            url_cut = trim_url_to_domain(url)
            st.session_state['urls_store'].append(url_cut)
            st.success("URL uploaded successfully.", icon="✅")
            time.sleep(4)
            st.rerun()
        elif st.session_state.get("url_status") == UploadStatus.INVALID_DOMAIN:
            st.warning("URL does not fall within the specified domain.", icon="⚠️")
        else:
            st.error("Failed to upload the URL.", icon="🚨")


@st.dialog("Show Uploaded URLs")
def show_uploaded_urls(names):
    """
    Display a table of uploaded URLs.

    If no URLs are uploaded, a message is shown.
    """
    if not names:
        st.write("No files uploaded.")
        return
    
    headers = ["#", "Base URL"]
    rows = [[i + 1, f"<a href='{url['original_url']}' target='_blank'>{url['base_url']}</a>"] for i, url in enumerate(names)]
    display_table(headers, rows, max_visible_items=15)



def page():
    """
    Main function to render the Streamlit app layout and manage interactions.

    Handles session state initialization, domain input, file/URL uploads, 
    and user interactions. Dynamically updates the interface based on whether 
    a domain is set. Displays chat messages, manages file and URL ingestion, 
    and allows users to reset the domain and session state.
    """
    print("chatbot.py - page()")
    
    initialize_session_state()

    if st.session_state["domain"].strip() == "":
        add_padding(padding_top=200)
        st.header("Chatbot for specific Domain 🤖 💬")
        st.caption("ChatRetrieveAI: Using RAG for Document📚 and Web search Interaction🔎.")
        
        with st.form('chat_input_form'):
            col1, col2 = st.columns([2,1])   
        with col1:          
            domain = st.text_input( 
                "Enter a domain to start the chat.",
                placeholder='Enter you specific domain to start the chat.',
                label_visibility='collapsed'
            ) 
        with col2:
            st.form_submit_button('Add Domain',  use_container_width=True)     
            st.session_state["domain"] = domain
            st.session_state["assistant"].set_domain(domain)
            
        if st.session_state["domain"].strip() != "":
            st.rerun() 
    else: 
        col1, col2 = st.columns([4, 1])
        with col1:
            st.header(f"🤖 💬 Chatbot {st.session_state['domain']} domain.")
            st.caption("ChatRetrieveAI: Using RAG for Document📚 and Web search Interaction🔎.")
            
            if st.button("Reset Domain", type="primary", use_container_width=True):
                clear_session_state()
                st.rerun() 
                
            col3, col4 = st.columns([2, 2])
            with col3:
                if st.button("Show Uploaded URLs", use_container_width=True):
                    show_uploaded_urls(st.session_state['urls_store'])
            with col4:
                if st.button("Show Uploaded Files", use_container_width=True):
                    show_file_names(st.session_state['file_names'])
                    
            add_divider(padding_top=0)
            
            if not st.session_state["messages"]:
                domain_message = f"Welcome to our  {st.session_state['domain']} domain support chat! How can I assist you today?"
                st.session_state["messages"] = [{"role": "assistant", "content": domain_message}]
             
        with st.sidebar: 
            add_heading("📝 Upload Files", level=2, padding_bottom=0)         
            st.markdown(hide_file_names(), unsafe_allow_html=True)

            file_uploader = st.file_uploader(
                "Upload document",
                type=["pdf", "docx", "txt"],
                key="file_uploader",
                on_change=read_and_save_file,
                args = (st.session_state["domain"], "document" ),
                label_visibility="collapsed",
                accept_multiple_files=False,
            )
            
            if st.session_state.get("document_status") and st.session_state["document_status"].get('status') == UploadStatus.DUPLICATE_FILE:
                st.warning(f"The document '{st.session_state['document_status']['file_name']}' has already been uploaded.")
            elif st.session_state.get("document_status") and st.session_state["document_status"].get('status') == UploadStatus.INVALID_DOMAIN:
                st.error(f"The document '{st.session_state['document_status']['file_name']}' does not fall within the specified domain.")
            elif st.session_state.get("document_status") and st.session_state["document_status"].get('status') == UploadStatus.SUCCESS:
                st.success(f"Document {st.session_state['document_status']['file_name']} uploaded successfully.")   
                
            add_divider(padding_top=0)    
            
            try:
                add_padding(padding_top=0)
                add_heading("🔗 Upload URL", level=2, padding_bottom=30)
                
                if st.button("Upload URL", use_container_width=True):
                    ingest_url()
                    
            except Exception as e:
                st.error(f"An unexpected error occurred: {str(e)}")
            
            add_divider(padding_top=0)             
            
        st.session_state["ingestion_spinner"] = st.empty()
        display_messages()
        st.chat_input("Send message to Chatbot", key="user_input", on_submit=process_input)


if __name__ == "__main__":
    print("chatbot.py - __main__")
    page()