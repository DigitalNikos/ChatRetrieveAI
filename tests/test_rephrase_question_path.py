import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import unittest
from rag.rag import ChatPDF
from config import Config as cfg

class TestRephraseQuestionPathChatPdf(unittest.TestCase):
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
    
    def test_rephrase_question(self):
        question = "How can AI improve basketball training?"
        # expected_answer ={'answer': 'Solution:\n\nStep 1: Calculate the difference between the demand for quantum computing specialists in 2024 and 2021\nStep 2: The result is the total increase in specialist demand over these years\n\n Final answer: (3000 - 1200) = 1800', 'metadata': 'Computed with python'}
        expected_execution_path = ['check_query_domain', 'retrieve']    
        inputs = {"question": question, "domain": self.domain}
    
        state = self.chat_pdf.invoke(inputs)
        print(f"State: {state}")
        
        self.assertEqual(state['question'], question)
        # check if the execution path is as expected until the last expected step
        self.assertEqual(state['execution_path'][:len(expected_execution_path)], expected_execution_path)
        
        
    def tearDown(self) -> None:
        return super().tearDown()

if __name__ == '__main__':
    unittest.main()