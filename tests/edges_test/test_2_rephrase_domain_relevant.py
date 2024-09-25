import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import unittest
import inspect

from rag.rag import ChatPDF
from config import Config as cfg
from langchain_core.messages import HumanMessage, AIMessage


class TestRephraseQuestion(unittest.TestCase):
    def setUp(self):
        cfg.MODEL_TEMPERATURE = 0.0
        self.domain = "Sport"
        self.chat_pdf = ChatPDF()
        self.chat_pdf.knowledge_base_system.chat_rephrased_history.extend([HumanMessage(
                                        content="What is AI's role in predicting basketball game outcomes?"
                                    ), 
                                      AIMessage(
                                        content="AI can predict the outcome of a basketball game by analyzing large amounts of internet information, providing marketing advice to stakeholders, and helping teams make strategic and tactical decisions."
                                    )])
    
    
    def test_positive_rephrased_quesiton(self):
        question = "What are some key metrics analyzed?"
        inputs = {"question": question, 'chat_history': self.chat_pdf.knowledge_base_system.chat_rephrased_history}
        expected_classification = 'yes'
        
        state = self.chat_pdf.knowledge_base_system._rephrase_query(inputs)
        
        print(f"\n{self.__class__.__name__}:       {inspect.currentframe().f_code.co_name}")
        print(f"\nRephrase_query -> State:      {state}")
        
        inputs = {"question": state['question'], "domain": self.domain}
        state = self.chat_pdf.knowledge_base_system._check_query_domain(inputs)
        
        print(f"\nCheck_query_domain -> State:      {state}")
 
        self.assertEqual(state['q_domain_relevance'], expected_classification)
    
        
    def test_negative_rephrased_quesiton(self):
        question = "How is the weather today?"
        inputs = {"question": question, 'chat_history': self.chat_pdf.knowledge_base_system.chat_rephrased_history}
        expected_classification = 'no'
        
        state = self.chat_pdf.knowledge_base_system._rephrase_query(inputs)
        
        print(f"\n{self.__class__.__name__}:       {inspect.currentframe().f_code.co_name}")
        print(f"\nRephrase_query -> State:      {state}")
        
        inputs = {"question":  state['question'], "domain": self.domain}
        state = self.chat_pdf.knowledge_base_system._check_query_domain(inputs)
        
        print(f"\nRephrase_query -> State:      {state}")
        
        self.assertEqual(state['q_domain_relevance'], expected_classification)
            
        
    def tearDown(self) -> None:
        self.chat_pdf.knowledge_base_system.chat_rephrased_history.clear()
        self.domain = None
        self.chat_pdf = None
        return super().tearDown()

if __name__ == '__main__':
    unittest.main()