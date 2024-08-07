import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import unittest
from rag.rag import ChatPDF
from config import Config as cfg
from langchain_core.messages import HumanMessage, AIMessage

class TestRetrieverPathChatPdf(unittest.TestCase):
    def setUp(self):
        print(cfg)
        cfg.MODEL_TEMPERATURE = 0.0
        self.domain = "Sport"
        test_dir = os.path.dirname(__file__)
        file_path = os.path.join(test_dir, 'data', 'Application of Artificial_Intelligence_in_Basketball_Sport.pdf')
        source_extension = ".pdf"
        file_name = "Application of Artificial_Intelligence_in_Basketball_Sport.pdf"
        
        self.chat_pdf = ChatPDF(cfg)
        self.chat_pdf.ingest({'file_path': file_path, 'source_extension': source_extension, 'file_name': file_name, 'domain': self.domain})
        self.kbs =self.chat_pdf.knowledge_base_system
        self.kbs.chat_history.extend([HumanMessage(content="What is the difference between a winning and a losing team?"), AIMessage(content="According to Ene et al.'s research (2018) published in the International Journal of Performance Analysis in Sport, the difference between a winning and a losing team lies in insights from Euroleague basketball.")])
    
    def test_retriever_rephrase_question(self):
        question = "What are the key factors?"
        expected_execution_path = ['check_query_domain', 'rephrase_based_history', 'check_query_domain', 'retrieve']    
        inputs = {"question": question, "domain": self.domain}
        
        state = self.chat_pdf.invoke(inputs)
        # check if the execution path is as expected until the last expected step
        self.assertEqual(state['execution_path'][:len(expected_execution_path)], expected_execution_path)
    
    def test_retriever_question(self):
        question = "How can AI improve basketball training?"
        expected_execution_path = ['check_query_domain','retrieve']    
        inputs = {"question": question, "domain": self.domain}
        
        state = self.chat_pdf.invoke(inputs)
        # check if the execution path is as expected until the last expected step
        self.assertEqual(state['execution_path'][:len(expected_execution_path)], expected_execution_path)
    
        
    def tearDown(self) -> None:
        return super().tearDown()

if __name__ == '__main__':
    unittest.main()