import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import unittest
import inspect

from rag.rag import ChatPDF
from config import Config as cfg


class TestComputationMethod(unittest.TestCase):
    def setUp(self):
        cfg.MODEL_TEMPERATURE = 0.0
        self.domain = "Sport"
        chat_pdf = ChatPDF(cfg)
        self.kbs = chat_pdf.knowledge_base_system
    
    
    def test_positive_question_classifier(self):
        question = "Using a predictive model with an accuracy rate of 84%, how many games would you expect to correctly predict out of a total of 500 NBA games?"
        inputs = {"question": question, 'grade_documents': []}
        expected_classification = 'yes'
        
        state = self.kbs._math_generate(inputs)
        
        print(f"\n{self.__class__.__name__}:       {inspect.currentframe().f_code.co_name}")
        print(f"\nMath_gen -> State:      {state}")
        
        self.assertEqual(state['question'], question)
        self.assertEqual(state['generation_score'], expected_classification)
        
        
    def test_negative__question_classifier(self):
        question = "Find the derivative of f(x) = 3x^2 + 2x + 1."
        inputs = {"question": question,'grade_documents': []}
        expected_classification = 'no'
        
        state = self.kbs._math_generate(inputs)
        
        print(f"\n{self.__class__.__name__}:       {inspect.currentframe().f_code.co_name}")
        print(f"\nMath_gen -> State:      {state}")
        
        self.assertEqual(state['question'], question)
        self.assertEqual(state['generation_score'], expected_classification)
        
        
    def tearDown(self) -> None:
        self.kbs = None
        self.domain = None
        self.chat_pdf = None
        return super().tearDown()

if __name__ == '__main__':
    unittest.main()