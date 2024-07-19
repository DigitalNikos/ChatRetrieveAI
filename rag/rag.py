from config import Config as cfg
from knowledge_base_system import KnowledgeBaseSystem

from langchain_community.chat_models import ChatOllama
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.utils import filter_complex_metadata
from text_doc_processing import clean_text
from prompts import domain_detection, domain_check
from langchain_core.output_parsers import JsonOutputParser
from rag.vectordb import VectorDB

class ChatPDF:
    print("Calling =>rag.py - ChatPDF")

    def __init__(self,):
        print("Calling =>rag.py - ChatPDF - __init__()")
        self.json_llm = ChatOllama(model=cfg.MODEL, format="json", temperature=0) 
        self.domain = None
        self.vector_db = VectorDB()
        self.retriever = None
        self.knowledge_base_system = KnowledgeBaseSystem(cfg.MODEL)
        
        self.summary_domain_chain = domain_detection | self.json_llm | JsonOutputParser()
        self.domain_checking = domain_check | self.json_llm | JsonOutputParser()


    def ingest(self,sources: dict):
        print('\nCalling => rag.py - ingest()')

        source_extension = sources['source_extension']

        if source_extension not in cfg.LOADERS_TYPES:
            raise Exception("Not valid upload source!!")

        if source_extension == "url":
            docs = cfg.LOADERS_TYPES[source_extension](sources["url"]).load()
        else:
            docs = cfg.LOADERS_TYPES[source_extension](sources["file_path"]).load()

        self.text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=cfg.SPLITTER_CHUNK_SIZE, chunk_overlap=cfg.SPLITTER_CHUNK_OVERLAP
        )

        chunks = self.text_splitter.split_documents(docs)
        chunks = filter_complex_metadata(chunks)
        chunks = clean_text(chunks, sources['file_name'])

        print("C\nhunks URL: ", chunks)

        result = self.summary_domain_chain.invoke({"documents": chunks})
        result = self.domain_checking.invoke({"domain": sources['domain'], "summary": result["summary"], "doc_domain": result["domain"]})  

        if result["score"] == "no":
            return result["score"]
        
        self.vector_db.add_documents(chunks)
        if not self.retriever:
            self.retriever = self.vector_db.as_retriever()
            self.knowledge_base_system.set_retriever(self.retriever)
            

    def ask(self, query: str):
        print("Calling =>rag.py - ask()")
        if self.domain is None:
            return "Please set the domain before asking questions."
        result = self.knowledge_base_system.invoke({"question": query, "domain": self.domain}) 
        return result

    
    def set_domain(self, domain: str):
        print("Calling =>rag.py - set_domain()")
        self.domain = domain
        
        