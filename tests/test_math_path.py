import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import unittest
from rag.rag import ChatPDF
from config import Config as cfg

class TestMathAnswerChatPdf(unittest.TestCase):
    def setUp(self):
        print(cfg)
        cfg.MODEL_TEMPERATURE = 0.0
        self.domain = "Technology"
        test_dir = os.path.dirname(__file__)
        file_path = os.path.join(test_dir, 'data', 'Quantum_Computing_Article_ChatGPT4o.pdf')
        source_extension = ".pdf"
        file_name = "Quantum_Computing_Article_ChatGPT4o.pdf"
        
        self.chat_pdf = ChatPDF(cfg)
        self.chat_pdf.ingest({'file_path': file_path, 'source_extension': source_extension, 'file_name': file_name, 'domain': self.domain})
    
    def test_math_question(self):
        question = "If the demand for quantum computing specialists increases from 1,200 in 2021 to 3,000 in 2024, what is the total increase in specialist demand over these years?"
        # expected_answer ={'answer': 'Solution:\n\nStep 1: Calculate the difference between the demand for quantum computing specialists in 2024 and 2021\nStep 2: The result is the total increase in specialist demand over these years\n\n Final answer: (3000 - 1200) = 1800', 'metadata': 'Computed with python'}
        expected_execution_path = [
            'check_query_domain', 'retrieve', 'grade_docs', 'question_classification', 'math_generate'
        ]    
        inputs = {"question": question, "domain": self.domain}
    
        state = self.chat_pdf.invoke(inputs)
        print(f"State: {state}")
        
        self.assertEqual(state['question'], question)
        # self.assertEqual(state['answer'], expected_answer)
        # check if the execution path is as expected until the last expected step
        self.assertEqual(state['execution_path'][:len(expected_execution_path)], expected_execution_path)
        
        
    def tearDown(self) -> None:
        return super().tearDown()

if __name__ == '__main__':
    unittest.main()