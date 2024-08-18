import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import unittest
import inspect

from rag.rag import ChatPDF
from config import Config as cfg
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.documents import Document


class TestUsefulAnswer(unittest.TestCase):
    def setUp(self):
        cfg.MODEL_TEMPERATURE = 0.0
        self.domain = "Sport"
        self.chat_pdf = ChatPDF(cfg)
        self.kbs = self.chat_pdf.knowledge_base_system

    
    def test_positive_useful_answer(self):
        question = "What are the main applications of AI in improving player performance in basketball?"
        generation = "AI has several applications in improving player performance in basketball. It is used for performance analysis through wearable devices and video analysis, allowing coaches to develop personalized training programs. AI also helps in injury prevention by analyzing physiological data and predicting potential injuries. Additionally, AI assists in strategic planning by evaluating game data and suggesting optimal strategies."
        usuful_answer_classification = 'useful'
        
        inputs = {"question": question, "answer": generation}
        state = self.kbs._answer_check(inputs)
        
        print(f"\n{self.__class__.__name__}:       {inspect.currentframe().f_code.co_name}")
        print(f"\nAnswer_check -> State:      {state}")
        
        provide_classification = state['answer_useful']
        self.assertEqual(provide_classification, usuful_answer_classification)

    
    def  test_negative_useful_answer(self):
        question = "How does AI guarantee that basketball referees make perfect decisions during games?"
        generation = "Traditional methods used by basketball referees to ensure fair play include consulting a set of ancient basketball scrolls that outline the original rules of the game. Referees are also trained to use divination techniques such as reading tea leaves or using crystal balls to make close-call decisions."
        usuful_answer_classification = 'not useful'
        
        inputs = {"question": question, "answer": generation}
        state = self.kbs._answer_check(inputs)
        
        print(f"\n{self.__class__.__name__}:       {inspect.currentframe().f_code.co_name}")
        print(f"\nAnswer_check -> State:      {state}")
        
        provide_classification = state['answer_useful']
        self.assertEqual(provide_classification, usuful_answer_classification)

        
    def tearDown(self) -> None:
        self.chat_pdf = None
        self.kbs = None
        self.domain = None
        return super().tearDown()

if __name__ == '__main__':
    unittest.main()