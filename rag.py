from config import Config as cfg
from knowledge_base_system import KnowledgeBaseSystem

class ChatPDF:
    def __init__(self):
        print("---init ChatPDF---")
        self.knowledge_base_system = KnowledgeBaseSystem(cfg.MODEL)

    def ingest(self, pdf_file_path: str):
        self.knowledge_base_system.ingest(pdf_file_path)


    def ask(self, query: str):
        print("---Ask ChatPDF---")
        result = self.knowledge_base_system.invoke({"question": query}) 
        return result