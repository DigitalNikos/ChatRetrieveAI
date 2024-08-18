from enum import Enum, auto
from dataclasses import dataclass
from utils.extract_source import upload_pdf, upload_url
from langchain_community.document_loaders import PyMuPDFLoader, TextLoader, Docx2txtLoader, WebBaseLoader


@dataclass
class Config:
    MODEL: str = "llama3.1"
    MATH_MODEL: str = "mathstral"
    MODEL_TEMPERATURE: float = 0.0
    
    SPLITTER_CHUNK_SIZE: int = 512
    SPLITTER_CHUNK_OVERLAP: int = 51
    N_DOCUMENTS_TO_RETRIEVE: int = 4
    RETRIEVER_SCORE_THRESHOLD: float = 0.7
    
    COLLECTION_NAME: str = "rag-chroma"

    LOADERS_TYPES = {
        ".pdf": PyMuPDFLoader,
        ".txt": TextLoader,
        ".docx": Docx2txtLoader,
        "url": WebBaseLoader,
    }    

    UPLOAD_ACTIONS = {
        "document": upload_pdf,
        "url": upload_url
    }

    def update_splitter_params(self, model_temperature: float ,chunk_size: int, chunk_overlap: int, n_documents_to_retrieve: int, retriever_score_threshold: float):
        self.MODEL_TEMPERATURE = model_temperature
        self.SPLITTER_CHUNK_SIZE = chunk_size
        self.SPLITTER_CHUNK_OVERLAP = chunk_overlap
        self.N_DOCUMENTS_TO_RETRIEVE = n_documents_to_retrieve
        self.RETRIEVER_SCORE_THRESHOLD = retriever_score_threshold
        

