from dataclasses import dataclass

@dataclass
class Config:
    MODEL = "mistral"
    SPLITTER_CHUNK_SIZE = 256
    SPLITTER_CHUNK_OVERLAP = 100
    N_DOCUMENTS_TO_RETRIEVE = 3
    RETRIEVER_SCORE_THRESHOLD = 0.5
