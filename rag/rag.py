from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import JsonOutputParser
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.utils import filter_complex_metadata

from rag.vectordb import VectorDB
from utils import clean_text, normalize_documents, format_final_answer

from qa_system.qa_manager import KnowledgeBaseSystem
from rag.rag_prompts import domain_detection, domain_check

from langchain.output_parsers import ResponseSchema, StructuredOutputParser

class ChatPDF:
    print("Calling =>rag.py - ChatPDF")

    def __init__(self, cfg):
        print("Calling =>rag.py - ChatPDF - __init__()")
        self.json_llm = ChatOllama(model=cfg.MODEL, format="json", temperature=0) 
        self.domain = None
        self.vector_db = VectorDB()
        self.retriever = None
        self.cfg = cfg
        self.knowledge_base_system = KnowledgeBaseSystem(
            general_llm_model_name=self.cfg.MODEL,
            math_llm_model_name=self.cfg.MATH_MODEL,
            temperature=self.cfg.MODEL_TEMPERATURE,
        )
        
        self.domain_checking = domain_check | self.json_llm | JsonOutputParser()
        self.summary_domain_chain = domain_detection | self.json_llm | JsonOutputParser()
        

    def ingest(self, sources: dict):
        print("\n--- INGEST DATA ---")

        source_extension = sources['source_extension']

        if source_extension not in self.cfg.LOADERS_TYPES:
            raise Exception("Not valid upload source!!")

        if source_extension == "url":
            docs = self.cfg.LOADERS_TYPES[source_extension](sources["url"]).load()
        else:
            docs = self.cfg.LOADERS_TYPES[source_extension](sources["file_path"]).load()
            

        self.text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=self.cfg.SPLITTER_CHUNK_SIZE, chunk_overlap=self.cfg.SPLITTER_CHUNK_OVERLAP
        )
        
        chunks = self.text_splitter.split_documents(docs)
        chunks = filter_complex_metadata(chunks)
        chunks = clean_text(chunks, sources['file_name'])
        chunks = normalize_documents(chunks)

        print("Summart Domain Chain: ", self.summary_domain_chain)
        print("\nChunks Before update:     ", chunks)
        
        response_schemas = [
            ResponseSchema(
                name="summary", 
                description="Summary of the documents."),
            ResponseSchema(
                name="domain",
                description="List of possible domains the documents could belong to.",
                type="list",
            ),
        ]
        
        output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
        format_instructions = output_parser.get_format_instructions()
        

        result = self.summary_domain_chain.invoke({"documents": chunks, "format_instructions" : format_instructions})
        print("\nResult data domain detection: ")
        print("Result:     ", result)
        print("Summary:    {}".format(result["summary"]))
        print("Domain:     {}".format(result["domain"]))
        
        result = self.domain_checking.invoke({"domain": sources['domain'], "summary": result["summary"], "doc_domain": result["domain"]})  
        print("Result for summary: ", result)
        
        if result["score"] == "no":
            return result["score"]
        
        self.vector_db.add_documents(chunks)
        if not self.retriever:
            self.retriever = self.vector_db.as_retriever()
            print("Retriever set")
            self.knowledge_base_system.set_retriever(self.retriever)
            

    def ask(self, query: str):
        print("Calling =>rag.py - ask()")
        if self.domain is None:
            return "Please set the domain before asking questions."
        
        state = self.invoke({"question": query, "domain": self.domain})
        print("State: ", state)
        result = format_final_answer(state)
        return result
    
    def invoke(self, state):
        print("Calling =>rag.py - invoke()")
        return self.knowledge_base_system.invoke(state)

    
    def set_domain(self, domain: str):
        print("Calling =>rag.py - set_domain()")
        self.domain = domain
        
        