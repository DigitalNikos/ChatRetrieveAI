import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import unittest
from rag.rag import ChatPDF
from config import Config as cfg

class TestMathClassifier(unittest.TestCase):
    def setUp(self):
        cfg.MODEL_TEMPERATURE = 0.0
        self.domain = "Technology"
        test_dir = os.path.dirname(__file__)
        file_path = os.path.join(test_dir, 'data', 'Quantum_Computing_Article_ChatGPT4o.pdf')
        source_extension = ".pdf"
        file_name = "Quantum_Computing_Article_ChatGPT4o.pdf"
        
        chat_pdf = ChatPDF(cfg)
        chat_pdf.ingest({'file_path': file_path, 'source_extension': source_extension, 'file_name': file_name, 'domain': self.domain})
        self.kbs = chat_pdf.knowledge_base_system
    
    def test_positive_classification(self):
        question = "If the demand for quantum computing specialists increases from 1,200 in 2021 to 3,000 in 2024, what is the total increase in specialist demand over these years?"
        inputs = {"question": question, "domain": self.domain, "execution_path": []}
        expected_classification = 'yes'
        
        state = self.kbs._question_classifier(inputs)
        print(f"State: {state}")
        self.assertEqual(state['question'], question)
        self.assertEqual(state['question_type'], expected_classification)
        
    def test_negative_classification(self):
        question = "How much did global investment in quantum computing reach in 2023?"
        inputs = {"question": question, "domain": self.domain, "execution_path": []}
        expected_classification = 'no'
        
        state = self.kbs._question_classifier(inputs)
        print(f"State: {state}")
        self.assertEqual(state['question'], question)
        self.assertEqual(state['question_type'], expected_classification)
        
        
        
    def tearDown(self) -> None:
        return super().tearDown()

if __name__ == '__main__':
    unittest.main()