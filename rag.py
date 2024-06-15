from config import Config as cfg
from knowledge_base_system import KnowledgeBaseSystem

class ChatPDF:
    print("Calling =>rag.py - ChatPDF")

    def __init__(self,):
        print("Calling =>rag.py - ChatPDF - __init__()")
        self.domain = None
        self.knowledge_base_system = KnowledgeBaseSystem(cfg.MODEL)

    def ingest(self, source_details: dict):
        print("Calling =>rag.py - ingest()")
        answer = self.knowledge_base_system.ingest(source_details)
        return answer


    def ask(self, query: str):
        print("Calling =>rag.py - ask()")
        result = self.knowledge_base_system.invoke({"question": query, "domain": self.domain}) 
        return result
    
    def set_domain(self, domain: str):
        print("Calling =>rag.py - set_domain()")
        self.domain = domain
        
        