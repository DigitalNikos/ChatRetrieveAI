import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import unittest
from rag.rag import ChatPDF
from config import Config as cfg
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.documents import Document



class TestRetrieverExistingDocuments(unittest.TestCase):
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
    
    def test_positive_exist_docs(self):
        question = "How can AI improve basketball training?"
        expected_classification = 'yes'
        
        inputs = {"question": question, "execution_path": []}
        state = self.kbs._retrieve(inputs)
        
        self.assertTrue(len(state['documents']) > 0, "No documents retrieved.")
        
        inputs = {"question": question, "documents": state['documents'], "execution_path": []}
        state = self.kbs._grade_documents(inputs)
        
        provided_classification = 'yes' if len(state['documents']) > 0 else 'no'
        
        self.assertEqual(provided_classification, expected_classification)
    
    def test_negative_exist_docs(self):
        question = "How does AI contribute to enhancing athletic performance in swimming?"
        expected_classification = 'no'
        
        inputs = {"question": question, "execution_path": []}
        state = self.kbs._retrieve(inputs)
        
        self.assertTrue(len(state['documents']) > 0, "No documents retrieved.")
        
        inputs = {"question": question, "documents": state['documents'], "execution_path": []}
        state = self.kbs._grade_documents(inputs)
        
        provided_classification = 'no' if len(state['documents']) == 0 else 'yes'
        
        self.assertEqual(provided_classification, expected_classification)
        
        
    def tearDown(self) -> None:
        return super().tearDown()

if __name__ == '__main__':
    unittest.main()