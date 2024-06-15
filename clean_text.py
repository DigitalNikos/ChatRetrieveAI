import re
# from bs4 import BeautifulSoup

# def clean_text(text):
#     print("Calling =>clean_text.py - clean_text()")
#     # Remove HTML tags
#     # text = BeautifulSoup(text, "lxml").get_text()
    
#     text = re.sub(r'[^A-Za-z\s]', '', text)
    
#     text = re.sub(r'\s+', ' ', text).strip()
    
#     text = text.lower()
    
#     return text

def clean_text(chunks, file_name:str):
    print("Calling =>clean_text.py - clean_text()")
    # Remove HTML tags
    # text = BeautifulSoup(text, "lxml").get_text()
    
    for chunk in chunks:
        chunk.page_content = re.sub(r'[^A-Za-z\s]', '', chunk.page_content)
        chunk.page_content = re.sub(r'\s+', ' ', chunk.page_content).strip()
        chunk.page_content = chunk.page_content.lower()
    
        chunk.metadata['source'] = file_name
    
    return chunks