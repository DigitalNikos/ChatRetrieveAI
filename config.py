from dataclasses import dataclass
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader, WebBaseLoader

@dataclass
class Config:
    MODEL = "llama3"
    SPLITTER_CHUNK_SIZE = 256
    SPLITTER_CHUNK_OVERLAP = 100
    N_DOCUMENTS_TO_RETRIEVE = 3
    RETRIEVER_SCORE_THRESHOLD = 0.5

    WIKIPEDIA_TOP_K_RESULTS=1
    WIKIPEDIA_DOC_CONTENT_CHARS_MAX=100

    AGENT_MAX_ITERATIONS = 10

    LOADERS_TYPES = {
        ".pdf": PyPDFLoader,
        ".txt": TextLoader,
        ".docx": Docx2txtLoader,
        "url": WebBaseLoader,
    }    
