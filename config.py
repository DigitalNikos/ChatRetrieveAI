from dataclasses import dataclass

@dataclass
class Config:
    print("Calling =>config.py - Config")
    # Model parameters
    MODEL: str = "llama3.1"
    MATH_MODEL: str = "mathstral"
    MODEL_TEMPERATURE: float = 0.0
    MODEL_FORMAT: str = "json"
    KEEP_IN_MEMORY: int = -1

    # Splitter parameters
    SPLITTER_CHUNK_SIZE: int = 512
    SPLITTER_CHUNK_OVERLAP: int = 51

    #Database and retriever parameters
    N_DOCUMENTS_TO_RETRIEVE: int = 4
    RETRIEVER_SCORE_THRESHOLD: float = 0.5    
    COLLECTION_NAME: str = "rag-chroma"
    SEARCH_TYPE: str = "similarity_score_threshold"
    N_DDG_TO_RETRIEVE: int = 4
    

