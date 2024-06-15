import tempfile
import os
import requests

def upload_pdf(domain: str, st):
    print("Calling =>extract_source.py - upload_pdf()")
    for file in st.session_state["file_uploader"]:
        _, file_extension = os.path.splitext(file.name)
        file_extension = file_extension.lower()

        with tempfile.NamedTemporaryFile(delete=False) as tf:
            tf.write(file.getbuffer())
            file_path = tf.name

        with st.session_state["ingestion_spinner"], st.spinner(f"Ingesting {file.name}"):
            file_datails = {'file_path': file_path, 'source_extension': file_extension, 'file_name': file.name, 'domain': domain,}
            answer = st.session_state["assistant"].ingest(file_datails)
            if answer == "no":
                st.session_state["messages"].append(("Document does not fall within the specified domain", False))
            # elif answer == "already_ingested":
            #     st.session_state["messages"].append((f"The document '{file.name}' has already been ingested.", False))
        os.remove(file_path)

def upload_url(domain: str, st):
    print("Calling =>extract_source.py - upload_url()")
    file_datails = {'url': st.session_state["url_upload"], 'source_extension': "url", 'domain': domain,'file_name': st.session_state["url_upload"]}
    try:
        answer = st.session_state["assistant"].ingest(file_datails)
        if answer == "no":
            st.session_state["messages"].append(("Document does not fall within the specified domain", False))
        # elif answer == "already_ingested":
        #         st.session_state["messages"].append((f"The URL '{st.session_state['url_upload']}' has already been ingested.", False))
    except requests.exceptions.MissingSchema:
        st.error("Invalid URL: No scheme supplied. Please include 'http://' or 'https://'.")
    except requests.exceptions.RequestException as e:
        st.error(f"An error occurred while processing the URL: {str(e)}")
    
    