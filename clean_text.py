import re
# from bs4 import BeautifulSoup

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
    
    return chunks