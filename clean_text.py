import re
# from bs4 import BeautifulSoup

def clean_text(text):
    # Remove HTML tags
    # text = BeautifulSoup(text, "lxml").get_text()
    
    text = re.sub(r'[^A-Za-z\s]', '', text)
    
    text = re.sub(r'\s+', ' ', text).strip()
    
    text = text.lower()
    
    return text