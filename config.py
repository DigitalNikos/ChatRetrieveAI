from dataclasses import dataclass


@dataclass
class Config:
    # Model parameters
    MODEL: str = "llama3.1"
    MODEL_TEMPERATURE: float = 0.0
    KEEP_IN_MEMORY: int = -1

    # Splitter parameters
    SPLITTER_CHUNK_SIZE: int = 512
    SPLITTER_CHUNK_OVERLAP: int = 51

    #Database and retriever parameters
    COLLECTION_NAME: str = "rag_chroma"
    URI: str = "./vector.db"
    
    N_DDG_TO_RETRIEVE: int = 4
    
