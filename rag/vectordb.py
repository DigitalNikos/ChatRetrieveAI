
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_community.vectorstores.utils import filter_complex_metadata
from typing import List
from langchain_core.documents import Document
from config import Config as cfg

class VectorDB:
    def __init__(self):
        self.vector_store = None

    def initialize(self, chunks: List[Document]):
        self.vector_store = Chroma.from_documents(documents=chunks, collection_name="rag-chroma", embedding=FastEmbedEmbeddings())

    def add_documents(self, chunks: List[Document]):
        if not self.vector_store:
            self.initialize(chunks)
        else:
            self.vector_store.add_documents(chunks)

    def as_retriever(self):
        if not self.vector_store:
            raise ValueError("Vector store is not initialized with documents yet.")
        return self.vector_store.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={
                "k": cfg.N_DOCUMENTS_TO_RETRIEVE,
                "score_threshold": cfg.RETRIEVER_SCORE_THRESHOLD,
            },
        )