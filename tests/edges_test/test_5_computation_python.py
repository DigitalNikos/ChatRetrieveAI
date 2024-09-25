import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import unittest
import inspect

from rag.rag import ChatPDF
from config import Config as cfg


class TestComputationMethod(unittest.TestCase):
    def setUp(self):
        cfg.MODEL_TEMPERATURE = 0.0
        self.domain = "Sport"
        self.chat_pdf = ChatPDF()
    
    
    def test_positive_question_classifier(self):
        question = "How much change does Sandy receive from a twenty-dollar bill after ordering three cappuccinos at $2 each, two iced teas at $3 each, two cafe lattes at $1 each, and two espressos at $1 each?"
        inputs = {"question": question, 'grade_documents': []}
        expected_classification = 'yes'
        
        state = self.chat_pdf.knowledge_base_system._math_generate(inputs)
        
        print(f"\n{self.__class__.__name__}:       {inspect.currentframe().f_code.co_name}")
        print(f"\nMath_gen -> State:      {state}")
        
        self.assertEqual(state['question'], question)
        self.assertEqual(state['math_score'], expected_classification)
        
        
    def test_negative__question_classifier(self):
        question = "Find the derivative of f(x) = 3x^2 + 2x + 1."
        inputs = {"question": question,'grade_documents': []}
        expected_classification = 'no'
        
        state = self.chat_pdf.knowledge_base_system._math_generate(inputs)
        
        print(f"\n{self.__class__.__name__}:       {inspect.currentframe().f_code.co_name}")
        print(f"\nMath_gen -> State:      {state}")
        
        self.assertEqual(state['question'], question)
        self.assertEqual(state['math_score'], expected_classification)
        
        
    def tearDown(self) -> None:
        self.domain = None
        self.chat_pdf = None
        return super().tearDown()

if __name__ == '__main__':
    unittest.main()