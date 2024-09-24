import re
from urllib.parse import urlparse

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage


def print_documents(documents):
    """
    Prints the documents with formatted metadata and content for better readability.
    
    Args:
        documents (list): List of Document objects with metadata and page content.
    """
    for i, doc in enumerate(documents):
        print(f"\n--- Document {i+1} ---")
        
        if doc.metadata:
            print("Source:", doc.metadata.get('source', 'Unknown Source'))
        
        if doc.page_content:
            print("\nContent Preview:\n")
            print(doc.page_content[:100] + ('...' if len(doc.page_content) > 100 else ''))
        
        print("\n" + "-"*40 + "\n")


def clean_text(chunks, file_name: str):
    """
    Cleans the text in the documents by removing newline characters and extra spaces.

    Returns:
        list: List of Document objects with cleaned text.
    """
    print("text_doc_processing.py - clean_text()")
    
    for chunk in chunks:
        chunk.page_content = chunk.page_content.replace('\n', ' ')
        chunk.page_content = re.sub(r'\s+', ' ', chunk.page_content).strip()
        chunk.metadata['source'] = file_name
        
    return chunks


# def convert_str_to_document(input: str):
#     """
#     Converts the string input to a list of Document objects

#     Returns:
#         list: List of Document objects
#     """
#     print("text_doc_processing.py - convert_str_to_document()")
    
#     # Clean the string from the outer square brackets
#     cleaned_string = input.strip("[]")

#     # Use regex to split on individual document snippets, titles, and links
#     items = re.findall(r'snippet:\s*(.*?),\s*title:\s*(.*?),\s*link:\s*(https[^\]]+)', cleaned_string, re.DOTALL)

#     documents = []

#     for snippet, title, link in items:
#         # Clean up the snippet to remove potential trailing commas or whitespace
#         snippet = snippet.strip().rstrip(",")
#         title = title.strip()
#         link = link.strip()

#         doc = Document(
#             page_content=snippet,
#             metadata={'source': link}
#         )
#         documents.append(doc)

#     return documents

def convert_str_to_document(input: str):
    """
    Converts the string input to a list of Document objects.

    Returns:
        list: List of Document objects
    """
    print("text_doc_processing.py - convert_str_to_document()")
    
    # Clean the string from potential outer square brackets (if any)
    cleaned_string = input.strip("[]")
    
    # Use regex to split on individual document snippets and links
    items = re.findall(r'snippet:\s*(.*?),\s*link:\s*(https[^\s,]+)', cleaned_string, re.DOTALL)

    documents = []

    for snippet, link in items:
        # Clean up the snippet to remove potential trailing commas or whitespace
        snippet = snippet.strip().rstrip(",")
        link = link.strip()

        doc = Document(
            page_content=snippet,
            metadata={'source': link}
        )
        documents.append(doc)

    return documents


def normalize_documents(documents):
    """
    Normalizes the documents metadata. If PDFs file, extract the page number from the metadata and incrementing it by one.

    Returns:
        list: List of Document objects with normalized metadata
    """
    print("text_doc_processing.py - normalize_documents()")
    normalized_documents = []

    for doc in documents:
        metadata = doc.metadata if hasattr(doc, 'metadata') else doc['metadata']
        page_content = doc.page_content if hasattr(doc, 'page_content') else doc['page_content']
        
        # Handle PDF documents and increment the page number by one
        if metadata['source'].endswith('.pdf'):
            # Extract the page number from metadata, if it exists
            page_number = int(metadata.get('page', 0)) + 1
            normalized_source = f"{metadata['source']} - page: {page_number}"
        else:
            normalized_source = metadata['source']
        
        normalized_doc = Document(
            metadata={'source': normalized_source},
            page_content=page_content
        )
        
        normalized_documents.append(normalized_doc)
    
    return normalized_documents


def extract_limited_chat_history(chat_rephrased_history, max_length=3500):
    """
    Extracts a limited chat history based on the maximum length of the chat messages.

    Returns:
        list: List of chat messages with a total length less than the specified maximum length.
    """
    print("text_doc_processing.py - extract_limited_chat_history()")

    current_length = 0
    chat_messages = []

    for message in chat_rephrased_history:
        if isinstance(message, HumanMessage) or isinstance(message, AIMessage):
            message_content = message.content
            message_length = len(message_content)
    
            if current_length + message_length > max_length:
                break
            chat_messages.append(message)
            current_length += message_length
    
    return chat_messages


def trim_url_to_domain(original_url):
    """
    Trims the original URL to the base domain URL.

    Returns:
        dict: Dictionary containing the base URL and the original URL.
    """
    print("text_doc_processing.py - trim_url_to_domain()")

    parsed_url = urlparse(original_url)
    base_url = f"{parsed_url.scheme}://{parsed_url.netloc}"
    
    return {'base_url': base_url, 'original_url': original_url}


