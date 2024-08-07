import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import unittest
from rag.rag import ChatPDF
from config import Config as cfg
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.documents import Document



class TestExistRetrieverDocs(unittest.TestCase):
    def setUp(self):
        cfg.MODEL_TEMPERATURE = 0.0
        self.domain = "Sport"
        test_dir = os.path.dirname(__file__)
        file_path = os.path.join(test_dir, 'data', 'Application of Artificial_Intelligence_in_Basketball_Sport.pdf')
        source_extension = ".pdf"
        file_name = "Application of Artificial_Intelligence_in_Basketball_Sport.pdf"
        self.chat_pdf = ChatPDF(cfg)
        self.chat_pdf.ingest({'file_path': file_path, 'source_extension': source_extension, 'file_name': file_name, 'domain': self.domain})
        self.kbs = self.chat_pdf.knowledge_base_system
    
    def test_positive_retrive_positive_grade_docs(self):
        question = "How can AI improve basketball training?"
        inputs = {"question": question, "execution_path": []}
        retrieve_documents_classification = 'yes'
        grade_documents_classification = 'yes'
        
        state = self.kbs._retrieve(inputs)
        provide_classification = 'yes' if len(state['documents']) > 0 else 'no'
        self.assertEqual(provide_classification, retrieve_documents_classification)
        
        inputs = {"question": state['question'], 'documents': state['documents'], "execution_path": []}
        state = self.kbs._grade_documents(inputs)
        
        print(f"State: {state}")
        provide_classification = 'yes' if len(state['grade_documents']) > 0 else 'no'
        self.assertEqual(provide_classification, grade_documents_classification)
    
    def  test_positive_retrive_negative_grade_docs(self):
        question = "What are the recent advancements in AI for detecting emotions through speech in basketball games?"
        inputs = {"question": question, "execution_path": []}
        retrieve_documents_classification = 'yes'
        grade_documents_classification = 'no'
        
        state = self.kbs._retrieve(inputs)
        provide_classification = 'yes' if len(state['documents']) > 0 else 'no'
        self.assertEqual(provide_classification, retrieve_documents_classification)
        
        inputs = {"question": state['question'], 'documents': state['documents'], "execution_path": []}
        state = self.kbs._grade_documents(inputs)
        
        print(f"State: {state}")
        provide_classification = 'yes' if len(state['grade_documents']) > 0 else 'no'
        self.assertEqual(provide_classification, grade_documents_classification)
    
    def  test_negative_retrive_negative_grade_docs(self):
        question = "What are the most effective defensive techniques used in boxing?"
        inputs = {"question": question, "execution_path": []}
        retrieve_documents_classification = 'no'
        
        state = self.kbs._retrieve(inputs)
        provide_classification = 'yes' if len(state['documents']) > 0 else 'no'
        self.assertEqual(provide_classification, retrieve_documents_classification)
        
        
    def tearDown(self) -> None:
        return super().tearDown()

if __name__ == '__main__':
    unittest.main()