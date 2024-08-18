import tempfile
import os
import hashlib
import requests
from enum import Enum, auto

class UploadStatus(Enum):
    SUCCESS = auto()
    INVALID_URL = auto()
    ERROR = auto()
    INVALID_DOMAIN = auto()

def upload_pdf(domain: str, st):
    print("Calling =>extract_source.py - upload_pdf()")
    
    if st.session_state["file_uploader"]:
        file = st.session_state["file_uploader"][-1]
        
        # Calculate the hash of the current file
        file_hash = hashlib.md5(file.getbuffer()).hexdigest()
        
        # Check if the file has already been uploaded
        for uploaded_file in st.session_state["file_uploader"][:-1]:
            existing_file_hash = hashlib.md5(uploaded_file.getbuffer()).hexdigest()
            if file_hash == existing_file_hash:
                # st.warning(f"The document '{file.name}' has already been uploaded.")
                return  {'status': UploadStatus.INVALID_URL, 'file_name': file.name}
        
        _, file_extension = os.path.splitext(file.name)
        file_extension = file_extension.lower()

        with tempfile.NamedTemporaryFile(delete=False) as tf:
            tf.write(file.getbuffer())
            file_path = tf.name

        with st.session_state["ingestion_spinner"], st.spinner(f"Ingesting {file.name}"):
            file_datails = {'file_path': file_path, 'source_extension': file_extension, 'file_name': file.name, 'domain': domain,}
            answer = st.session_state["assistant"].ingest(file_datails)
            
            print('Upload txt:', answer)
            
            if answer == "no":
                return {'status': UploadStatus.INVALID_DOMAIN, 'file_name': file.name}
            
        st.session_state['file_names'].append(file.name)
        os.remove(file_path)
        return  {'status': UploadStatus.SUCCESS, 'file_name': file.name}


def upload_url(domain: str, st):
    print("Calling =>extract_source.py - upload_url()")
    file_datails = {
        'url': st.session_state["url_upload"], 
        'source_extension': "url", 
        'domain': domain,
        'file_name': st.session_state["url_upload"]
    }
    
    try:
        answer = st.session_state["assistant"].ingest(file_datails)
        print('AnswerURL:', answer)
        if answer == "no":
            return {'status': UploadStatus.INVALID_DOMAIN}
    except requests.exceptions.MissingSchema:
        return {'status': UploadStatus.INVALID_URL}
    except requests.exceptions.RequestException as e:
        st.error(f"An error occurred while processing the URL: {str(e)}")
        return {'status': UploadStatus.ERROR}
    
    return {'status': UploadStatus.SUCCESS, 'url': st.session_state["url_upload"]}