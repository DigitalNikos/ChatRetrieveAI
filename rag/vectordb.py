
from langchain_community.vectorstores import Chroma
from langchain_milvus import Milvus
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores.utils import filter_complex_metadata
from typing import List
from langchain_core.documents import Document
from config import Config as cfg


import os
from uuid import uuid4

os.environ["TOKENIZERS_PARALLELISM"] = "false"



# class VectorDB:
#     def __init__(self):
#         self.vector_store = None

#     def initialize(self, chunks: List[Document]):
#         model_name = "BAAI/bge-large-en"
#         model_kwargs = {'device': 'cpu'}
#         encode_kwargs = {'normalize_embeddings': True}
#         hf = HuggingFaceBgeEmbeddings(
#             model_name=model_name,
#             model_kwargs=model_kwargs,
#             encode_kwargs=encode_kwargs
#         )

#         self.vector_store = Chroma.from_documents(documents=chunks, collection_name=cfg.COLLECTION_NAME, embedding=hf)

#     def add_documents(self, chunks: List[Document]):
#         if not self.vector_store:
#             self.initialize(chunks)
#         else:
#             self.vector_store.add_documents(chunks)

#     def as_retriever(self):
#         if not self.vector_store:
#             raise ValueError("Vector store is not initialized with documents yet.")
#         return self.vector_store.as_retriever(
#             search_type="similarity_score_threshold",
#             search_kwargs={
#                 "k": cfg.N_DOCUMENTS_TO_RETRIEVE,
#                 "score_threshold": cfg.RETRIEVER_SCORE_THRESHOLD,
#             },
#         )


class VectorDB:
    def __init__(self):
        model_name = "BAAI/bge-large-en"
        model_kwargs = {'device': 'cpu'}
        encode_kwargs = {'normalize_embeddings': True}
        hf = HuggingFaceBgeEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )
        self.vector_store = Milvus(embedding_function=hf, connection_args={"uri": "./chatbotDB.db"}, drop_old = True)
      

    def add_documents(self, chunks: List[Document]):
        uuids = [str(uuid4()) for _ in range(len(chunks))]
        self.vector_store.add_documents(documents=chunks, ids=uuids)

    def as_retriever(self):
        if not self.vector_store:
            raise ValueError("Vector store is not initialized with documents yet.")
        return self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={
                "k": cfg.N_DOCUMENTS_TO_RETRIEVE,
                "score_threshold": cfg.RETRIEVER_SCORE_THRESHOLD,
            },
        )