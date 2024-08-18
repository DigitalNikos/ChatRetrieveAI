from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage
import re
from urllib.parse import urlparse

def clean_text(chunks, file_name: str):
    print("Calling =>clean_text.py - clean_text()")
    
    for chunk in chunks:
        # Remove newline characters
        chunk.page_content = chunk.page_content.replace('\n', ' ')
        
        # Replace multiple spaces with a single space 
        chunk.page_content = re.sub(r'\s+', ' ', chunk.page_content).strip()
        
        chunk.metadata['source'] = file_name

    
    return chunks

def format_rag_output(rag_output):
    answer = rag_output.get('answer', 'No answer found.')
    metadata = rag_output.get('metadata', {})
    sources = set()

    if isinstance(metadata, dict):
        if 'sources' in metadata:
            sources.update(metadata['sources'])
            
        if 'source' in metadata:
            source = metadata['source']
            page = metadata.get('page')
            if page:
                source = f"{source}, page {page}"
            sources.add(source)
            
        if 'link' in metadata:
            sources.update(metadata['link'])
            
    elif isinstance(metadata, list):
        for item in metadata:
            sources.add(item['link'])

    sources_list = ', '.join(sources)
    
    formatted_output = f"Answer: {answer}\nSources: {sources_list}"
    return formatted_output


def convert_str_to_document(input:str):
        cleaned_string = input.strip("[]")

        # Split into individual items
        items = re.split(r'\], \[', cleaned_string)

        
        documents = []

        for item in items:
            parsed_dict = {}
            
            # Find all key-value pairs
            key_value_pairs = re.findall(r'(\w+):\s([^,]+)', item)
            
            for key, value in key_value_pairs:
                parsed_dict[key] = value.strip()
            
            doc = Document(
                page_content=parsed_dict['snippet'],
                metadata={
                    'link': parsed_dict['link']
                }
            )
            
            documents.append(doc)
            
        return documents

def normalize_documents(documents):
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
        
        # Construct the normalized document
        normalized_doc = Document(
            metadata={'source': normalized_source},
            page_content=page_content
        )
        
        normalized_documents.append(normalized_doc)
    
    return normalized_documents

        
def format_final_answer(answer):
    if 'answer' in answer and isinstance(answer['answer'], dict):
        response = answer['answer'].get('answer', 'No answer provided.')
        metadata = answer['answer'].get('metadata', 'No metadata available.')
        source = "No metadata available"
        # Check if metadata is a string
        if isinstance(metadata, str):
            # Attempt to parse the source and page if possible
            match = re.search(r"source='([^']+)'(?: - page: (\d+))?", metadata)
            if match:
                source = match.group(1)
                
        
        return f"{response}\n\nMetadata: {source}"
    else:
        return answer.get('answer', 'No answer provided.')
        


def extract_limited_chat_history(chat_rephrased_history, max_length=3500):

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

    parsed_url = urlparse(original_url)
    base_url = f"{parsed_url.scheme}://{parsed_url.netloc}"
    print(" trim_url_to_domain: ", base_url)
    return {'base_url': base_url, 'original_url': original_url}