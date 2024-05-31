from config import Config as cfg
from knowledge_base_system import KnowledgeBaseSystem

class ChatPDF:
    def __init__(self,):
        print("---init ChatPDF---")
        self.domain = None
        self.knowledge_base_system = KnowledgeBaseSystem(cfg.MODEL)

    def ingest(self, pdf_file_path: str, domain: str, file_name: str):
        answer = self.knowledge_base_system.ingest(pdf_file_path, domain, file_name)
        return answer


    def ask(self, query: str):
        print("---Ask ChatPDF---")
        result = self.knowledge_base_system.invoke({"question": query, "domain": self.domain}) 
        return result
    
    def set_domain(self, domain: str):
        self.domain = domain
        
        