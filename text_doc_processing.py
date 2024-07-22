from langchain_core.documents import Document
import re

def clean_text(chunks, file_name:str):
    print("Calling =>clean_text.py - clean_text()")
    # Remove HTML tags
    # text = BeautifulSoup(text, "lxml").get_text()
    
    for chunk in chunks:
        
        # Remove unwanted characters but keep alphanumeric, spaces, commas, and dots
        chunk.page_content = re.sub(r'[^A-Za-z0-9\s,.]', '', chunk.page_content)
        
        # Replace multiple spaces with a single space
        chunk.page_content = re.sub(r'\s+', ' ', chunk.page_content).strip()
        
        # Convert text to lowercase
        chunk.page_content = chunk.page_content.lower()
    
        chunk.metadata['source'] = file_name
    
    print("chunks in clear_text: ", chunks)
    
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

        # Initialize list to hold the parsed dictionaries
        documents = []

        # Parse each item
        for item in items:
            # Initialize dictionary
            parsed_dict = {}
            
            # Find all key-value pairs
            key_value_pairs = re.findall(r'(\w+):\s([^,]+)', item)
            
            # Fill the dictionary
            for key, value in key_value_pairs:
                parsed_dict[key] = value.strip()
            
            # Create Document instance
            doc = Document(
                page_content=parsed_dict['snippet'],
                metadata={
                    # 'source': parsed_dict['title'],
                    'link': parsed_dict['link']
                }
            )
            
            # Append to the list
            documents.append(doc)
        return documents