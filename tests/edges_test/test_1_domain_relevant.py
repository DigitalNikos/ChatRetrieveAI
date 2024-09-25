import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import unittest
import inspect

from rag.rag import ChatPDF
from config import Config as cfg


class TestDomainRelevance(unittest.TestCase):
    def setUp(self):
        cfg.MODEL_TEMPERATURE = 0.0
        self.domain = "Sport"
        self.chat_pdf = ChatPDF()
    
    
    def test_positive_domain_relavance(self):
        question = "How can AI improve basketball training?"
        inputs = {"question": question, "domain": self.domain}
        expected_classification = 'yes'
        
        state = self.chat_pdf.knowledge_base_system._check_query_domain(inputs)
        
        print(f"\n{self.__class__.__name__}:       {inspect.currentframe().f_code.co_name}")
        print(f"\nCheck_query_domain -> State:      {state}")
        
        self.assertEqual(state['question'], question)
        self.assertEqual(state['q_domain_relevance'], expected_classification)
        
        
    def test_negative__domain_relavance(self):
        question = "How is the weather today?"
        inputs = {"question": question, "domain": self.domain}
        expected_classification = 'no'
        
        state = self.chat_pdf.knowledge_base_system._check_query_domain(inputs)
        
        print(f"\n{self.__class__.__name__}:       {inspect.currentframe().f_code.co_name}")
        print(f"\nCheck_query_domain -> State:      {state}")
        
        self.assertEqual(state['question'], question)
        self.assertEqual(state['q_domain_relevance'], expected_classification)
        
   
    def tearDown(self) -> None:
        self.chat_pdf = None    
        self.domain = None
        return super().tearDown()

if __name__ == '__main__':
    unittest.main()