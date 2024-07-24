
from langchain_community.vectorstores import Chroma
# from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores.utils import filter_complex_metadata
from typing import List
from langchain_core.documents import Document
from config import Config as cfg

class VectorDB:
    def __init__(self):
        self.vector_store = None

    def initialize(self, chunks: List[Document]):
        model_name = "BAAI/bge-large-en"
        model_kwargs = {'device': 'cpu'}
        encode_kwargs = {'normalize_embeddings': True}
        hf = HuggingFaceBgeEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )
        self.vector_store = Chroma.from_documents(documents=chunks, collection_name="rag-chroma", embedding=hf)

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