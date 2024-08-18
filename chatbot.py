import time
import requests
import streamlit as st

from rag.rag import ChatPDF
from config import Config 
from utils.extract_source import UploadStatus
from utils.text_doc_processing import trim_url_to_domain
from utils.ui_helpers import add_divider, add_heading, add_padding, display_table, hide_file_names, add_centered_heading_with_description  


st.set_page_config(page_title="ChatPDF", page_icon="🤖")
cfg = Config()


def display_messages():
    """
    Display chat messages in the Streamlit app, ensuring consistent handling of message formats.
    
    This function checks if there are existing messages in the session state. If none exist, 
    it initializes a default welcome message. It then iterates through the list of messages 
    and displays each one according to its role (user, assistant, etc.).
    """
    print("\nCalling =>chatbot.py - display_messages()")

    # Initialize default message if no messages exist
    if not st.session_state["messages"]:
        domain_message = f"Welcome to our  {st.session_state['domain']} domain support chat! How can I assist you today?"
        st.session_state["messages"] = [{"role": "assistant", "content": domain_message}]
        
    # Display each message in the chat
    for msg in st.session_state["messages"]:
        role, content = (msg["role"], msg["content"]) if isinstance(msg, dict) else msg if isinstance(msg, tuple) else ("system", "Unexpected message format.") 
        st.chat_message(role).write(content)
    
    # Initialize the thinking spinner
    st.session_state["thinking_spinner"] = st.empty()
    

def process_input():
    """
    Process user input and get a response from the assistant.

    This function checks the user input from the session state. If the input is valid (i.e., 
    not empty after stripping whitespace), it sends the input to the assistant and retrieves 
    a response. The user input and the assistant's response are then appended to the 
    session state's message list, maintaining the conversation history.
    
    The function also manages the "thinking" spinner to provide visual feedback during the 
    response generation process.
    """
    print("\nCalling =>chatbot.py - process_input()")
    
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
    Handle file or URL upload and initiate document ingestion.
    
    This function processes the upload of a file or URL based on the provided source type 
    ('.pdf', '.docx', '.txt', or 'url'). It retrieves the appropriate action 
    from the configuration's `UPLOAD_ACTIONS` dictionary and executes it, passing the 
    domain and Streamlit context (`st`) as arguments.
    
    If the `source_type` is invalid or not recognized, the function raises a ValueError, 
    providing feedback on the expected file types or URL.
    
    Raises:
    - ValueError: If the `source_type` is not one of the expected values ('.pdf', '.docx', 
      '.txt', or 'url').
    """
    print("\nCalling =>chatbot.py - read_and_save_file()")

    # Retrieve the appropriate action based on the source type
    action = cfg.UPLOAD_ACTIONS.get(source_type)
    
    # Raise an error if the source type is invalid
    if not action:
            raise ValueError(f"Invalid source type: {source_type}. Expected '.pdf', '.docx', '.txt' or 'url'.")
    
    # Execute the action if valid
    status =action(domain, st)
    print('Status Read_and_save_files:  ', status)
    
    # Handle URL status updates
    if source_type == "url":
        st.session_state["error_status"] = status['status']
    
    # Handle document status updates
    elif source_type == "document":
        st.session_state['error_document'] = {'status': status['status'], 'file_name': status.get('file_name')}
    

def initialize_session_state():
    """
    Initialize the session state if it is currently empty.

    This function checks if the `st.session_state` dictionary is empty. If it is,
    it calls `clear_session_state` to set up the necessary initial values for the session.
    
    This ensures that the session state has the required keys and values before 
    the application proceeds, preventing potential errors due to missing state data.
    """
    print("\nCalling =>chatbot.py - initialize_session_state()")
    
    if len(st.session_state) == 0:
        clear_session_state()
        

def clear_session_state():
    """
    Clear and initialize the session state with default values.

    This function resets the `st.session_state` dictionary to ensure a clean state for the application.

    This function is typically called when the session is first initialized or needs to be reset.
    """
    st.session_state["messages"] = []
    st.session_state["assistant"] = ChatPDF(cfg)
    st.session_state["domain"] = ""
    st.session_state['urls_store'] = []
    st.session_state['url_upload'] = "" 
    st.session_state['file_names'] = []
    st.session_state["error_document"] = None


@st.dialog("Show Uploaded Files")
def show_file_names(names):
    if not names:
        st.write("No files uploaded.")
        return
    
    headers = ["#", "File name"]
    rows = [[i + 1, name] for i, name in enumerate(names)]
    display_table(headers, rows, max_visible_items=15)


@st.dialog("Upload URL")
def upload_url():
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
            response = requests.head(url, headers=headers, timeout=5)  # Attempt a HEAD request to the URL
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
        if st.session_state.get("error_status") == UploadStatus.SUCCESS:
            url_cut = trim_url_to_domain(url)
            st.session_state['urls_store'].append(url_cut)
            st.success("URL uploaded successfully.", icon="✅")
            time.sleep(4)
            st.rerun()
        elif st.session_state.get("error_status") == UploadStatus.INVALID_DOMAIN:
            st.warning("URL does not fall within the specified domain.", icon="⚠️")
        else:
            st.error("Failed to upload the URL.", icon="🚨")


@st.dialog("Show Uploaded URLs")
def show_uploaded_urls(names):
    if not names:
        st.write("No files uploaded.")
        return
    
    headers = ["#", "Base URL"]
    rows = [[i + 1, f"<a href='{url['original_url']}' target='_blank'>{url['base_url']}</a>"] for i, url in enumerate(names)]
    display_table(headers, rows, max_visible_items=15)


def page():
    """
    Main function to handle the Streamlit page layout and interactions.

    This function sets up the user interface and manages interactions within the Streamlit app. 
    It initializes the session state, displays headers and instructions, and handles user input 
    and file uploads. The function is structured to dynamically update the interface based on 
    whether a domain is set or not.

    Key functionalities include:
    - Initializing session state and clearing it when necessary.
    - Displaying appropriate headers and captions based on the current domain.
    - Managing user input for setting a domain and submitting messages.
    - Handling file uploads and URLs for document ingestion.
    - Displaying chat messages and managing the interaction flow.

    The function also includes error handling for invalid URLs and allows the user to reset the 
    domain context, clearing the session state and restarting the interaction.

    Raises:
    - ValueError: If the file or URL upload encounters an unsupported format or type.
    """

    print("\nCalling =>chatbot.py - page()")
    initialize_session_state()

    
    if st.session_state["domain"].strip() == "":
        st.header("Chatbot 🤖 💬")
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
        
        add_divider(padding_top=0)
        add_centered_heading_with_description("Configure Retrieval Settings", "Customize the retrieval process by adjusting the parameters below.")
        add_padding(padding_top=50)
               
        col1, col2, col3 = st.columns([1,4,1])
        with col2:
            model_temperature = st.slider("Model Temperature", min_value=0.0, max_value=1.0, step=0.1, value=0.0)
            chunk_size = st.slider("Chunk Size", min_value=20, max_value=1000, step=20, value=512)
            chunk_overlap = st.slider("Chunk Overlap", min_value=0, max_value=1000, step=5, value=51)
            n_documents_to_retrieve = st.slider("Nr. of retrieve Documents", min_value=0, max_value=20, step=1, value=4)
            retrieve_score_threshold = st.slider("Retrieve score threshold", min_value=0.0, max_value=1.0, step=0.1, value=0.7)
            st.session_state["assistant"].cfg.update_splitter_params(model_temperature, chunk_size, chunk_overlap, n_documents_to_retrieve, retrieve_score_threshold)
            
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

            file_upload = st.file_uploader(
                "Upload document",
                type=["pdf", "docx", "txt"],
                key="file_uploader",
                on_change=read_and_save_file,
                args = (st.session_state["domain"], "document" ),
                label_visibility="collapsed",
                accept_multiple_files=True,
            )
            
            if st.session_state.get("error_document") and st.session_state["error_document"].get('status') == UploadStatus.INVALID_URL:
                st.warning(f"The document '{st.session_state['error_document']['file_name']}' has already been uploaded.")
            elif st.session_state.get("error_document") and st.session_state["error_document"].get('status') == UploadStatus.INVALID_DOMAIN:
                st.error(f"The document '{st.session_state['error_document']['file_name']}' does not fall within the specified domain.")
            elif st.session_state.get("error_document") and st.session_state["error_document"].get('status') == UploadStatus.SUCCESS:
                st.success(f"Document {st.session_state['error_document']['file_name']} uploaded successfully.")   
                
            add_divider(padding_top=0)    
            
            try:
                add_padding(padding_top=0)
                add_heading("🔗 Upload URL", level=2, padding_bottom=30)
                
                if st.button("Upload URL", use_container_width=True):
                    upload_url()
                    
            except Exception as e:
                st.error(f"An unexpected error occurred: {str(e)}")
            
            add_divider(padding_top=0)             
            
        st.session_state["ingestion_spinner"] = st.empty()
        display_messages()
        st.chat_input("Send message to Chatbot", key="user_input", on_submit=process_input)


if __name__ == "__main__":
    print("Calling =>chatbot.py - __main__")
    page()