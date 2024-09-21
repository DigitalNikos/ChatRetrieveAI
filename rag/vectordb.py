from uuid import uuid4
from typing import List
from config import Config as cfg
from langchain_core.documents import Document
# from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
# from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_community.vectorstores import Chroma


class VectorDB:
    def __init__(self):
        print("vectordb.py - __init__()")
        
        self.embedding =  OllamaEmbeddings(model="nomic-embed-text")
        self.vector_store = None

    def initialize(self, chunks: List[Document]):
        print("vectordb.py - initialize()")
        
        self.vector_store = Chroma.from_documents(documents=chunks, collection_name="rag-chroma", embedding= self.embedding)
      
    def add_documents(self, chunks: List[Document]):
        print("vectordb.py - add_documents()")
        
        if not self.vector_store:
            self.initialize(chunks)
        else:
            self.vector_store.add_documents(chunks)
        

    def as_retriever(self):
        print("vectordb.py - as_retriever()")
        
        if not self.vector_store:
            raise ValueError("Vector store is not initialized with documents yet.")
        
        return self.vector_store.as_retriever(
            search_type= cfg.SEARCH_TYPE,
            search_kwargs={
                "k": cfg.N_DOCUMENTS_TO_RETRIEVE,
                "score_threshold": cfg.RETRIEVER_SCORE_THRESHOLD,
            },
        )