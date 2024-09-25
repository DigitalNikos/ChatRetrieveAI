from typing import List
from uuid import uuid4

from config import Config as cfg
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_core.documents import Document
from langchain_milvus import Milvus



class VectorDB:
    def __init__(self):
        print("vectordb.py - __init__()")

        model_name = "BAAI/bge-large-en"
        model_kwargs = {'device': 'cpu'}
        encode_kwargs = {'normalize_embeddings': True}
        self.hf = HuggingFaceBgeEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )
        
        self.vector_store =  Milvus(
            collection_name = cfg.COLLECTION_NAME,
            embedding_function= self.hf,
            connection_args={"uri": cfg.URI},
            drop_old = True
        )
        
        self.retriever = self.vector_store.as_retriever()
        
        
    def add_documents(self, chunks: List[Document]):
        print("vectordb.py - add_documents()")
        
        uuids = [str(uuid4()) for _ in range(len(chunks))]
        self.vector_store.add_documents(documents=chunks, ids=uuids)
    

        

