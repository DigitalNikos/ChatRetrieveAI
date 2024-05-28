from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# from langchain.vectorstores.utils import filter_complex_metadata
from langchain_community.vectorstores.utils import filter_complex_metadata
from config import Config as cfg
from typing import List

from langchain_core.documents import Document
from knowledge_base_system import KnowledgeBaseSystem

class ChatPDF:
    vector_store = None
    retriever = None
    chain = None

    def __init__(self):
        print("---Calling ChatPDF init function---")
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=cfg.SPLITTER_CHUNK_SIZE, chunk_overlap=cfg.SPLITTER_CHUNK_OVERLAP)
        self.retriever = None
        self.knowledge_base_system = None
    
    def initialize(self, chunks: List[Document]):
        print("---Calling initialize function---")
        vector_store = Chroma.from_documents(documents=chunks, collection_name="rag-chroma", embedding=FastEmbedEmbeddings())
        self.retriever = vector_store.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={
                "k": cfg.N_DOCUMENTS_TO_RETRIEVE,
                "score_threshold": cfg.RETRIEVER_SCORE_THRESHOLD,
            },
        )
        self.knowledge_base_system = KnowledgeBaseSystem(self.retriever, cfg.MODEL)

    def ingest(self, pdf_file_path: str):
        print("---Calling ingest function---")
        docs = PyPDFLoader(file_path=pdf_file_path).load()

        self.text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=250, chunk_overlap=0
        )

        chunks = self.text_splitter.split_documents(docs)
        chunks = filter_complex_metadata(chunks)

        if not self.retriever:
            self.initialize(chunks)
        else:
            self.retriever.add_documents(chunks)


    def ask(self, query: str):
        if not self.retriever:
            return "Please, add a PDF document first."
        result = self.knowledge_base_system.invoke({"question": query}) # self.agent_executor.invoke({"input": query, "chat_history": self.chat_history})
        # answer = result["output"]
        # self.chat_history += "Human:" + query + "\n" + "Assistant (YOU)" + answer + "\n"
        # print(answer)
        return result
