import tempfile
import os
import hashlib
import requests
from enum import Enum, auto

class UploadStatus(Enum):
    SUCCESS = auto()
    ERROR = auto()
    INVALID_URL = auto()
    DUPLICATE_FILE = auto()
    INVALID_DOMAIN = auto()

def upload_document(domain: str, st):
    print("upload_source.py - upload_pdf()")
    
    if st.session_state["file_uploader"]:
        file = st.session_state["file_uploader"] 
        
        # Calculate the hash of the current file to prevent duplicates
        file_hash = hashlib.md5(file.getbuffer()).hexdigest()
        
        # Check if the file has already been uploaded (by comparing hashes)
        if "file_hashes" not in st.session_state:
            st.session_state["file_hashes"] = []
        
        if file_hash in st.session_state["file_hashes"]:
            return {'status': UploadStatus.DUPLICATE_FILE, 'file_name': file.name}
        
        st.session_state["file_hashes"].append(file_hash)
        
        _, file_extension = os.path.splitext(file.name)
        file_extension = file_extension.lower()

        # Save the file to a temporary location
        with tempfile.NamedTemporaryFile(delete=False) as tf:
            tf.write(file.getbuffer())
            file_path = tf.name

        # Process the file with the assistant
        with st.session_state["ingestion_spinner"], st.spinner(f"Ingesting {file.name}"):
            file_details = {
                'file_path': file_path, 
                'source_extension': file_extension, 
                'file_name': file.name, 
                'domain': domain,
            }
            answer = st.session_state["assistant"].ingest(file_details)
            
            if answer == "no":
                return {'status': UploadStatus.INVALID_DOMAIN, 'file_name': file.name}
        
        # Add the file name to the session state and remove the temporary file
        st.session_state['file_names'].append(file.name)
        os.remove(file_path)
        return {'status': UploadStatus.SUCCESS, 'file_name': file.name}
        

def upload_url(domain: str, st):
    print("upload_source.py - upload_url()")
    file_datails = {
        'url': st.session_state["url_upload"], 
        'source_extension': "url", 
        'domain': domain,
        'file_name': st.session_state["url_upload"]
    }
    
    try:
        answer = st.session_state["assistant"].ingest(file_datails)
        if answer == "no":
            return {'status': UploadStatus.INVALID_DOMAIN}
    except requests.exceptions.MissingSchema:
        return {'status': UploadStatus.INVALID_URL}
    except requests.exceptions.RequestException as e:
        st.error(f"An error occurred while processing the URL: {str(e)}")
        return {'status': UploadStatus.ERROR}
    
    return {'status': UploadStatus.SUCCESS, 'url': st.session_state["url_upload"]}