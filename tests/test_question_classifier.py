import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import unittest
from rag.rag import ChatPDF
from config import Config as cfg


class TestQuestionClassifier(unittest.TestCase):
    
    def setUp(self):
        cfg.MODEL_TEMPERATURE = 0.0
        self.domain = "Sport"
        chat_pdf = ChatPDF(cfg)
        self.kbs = chat_pdf.knowledge_base_system
    
    def test_positive_question_classifier(self):
        
        question = "Using a predictive model with an accuracy rate of 84%, how many games would you expect to correctly predict out of a total of 500 NBA games?"
        inputs = {"question": question, "execution_path": []}
        expected_classification = 'yes'
        
        state = self.kbs._question_classifier(inputs)
        print(f"State: {state}")
        self.assertEqual(state['question'], question)
        self.assertEqual(state['question_type'], expected_classification)
        
        
    def test_negative__question_classifier(self):
        question = "How has AI technology impacted the strategies employed by basketball coaches?"
        inputs = {"question": question, "execution_path": []}
        expected_classification = 'no'
        
        state = self.kbs._question_classifier(inputs)
        print(f"State: {state}")
        self.assertEqual(state['question'], question)
        self.assertEqual(state['question_type'], expected_classification)
        
        
    def tearDown(self) -> None:
        return super().tearDown()

if __name__ == '__main__':
    unittest.main()