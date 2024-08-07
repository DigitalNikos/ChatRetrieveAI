import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import unittest
from rag.rag import ChatPDF
from config import Config as cfg

class TestPath4(unittest.TestCase):
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
    
    def test_path_4(self):
        question = "What is my Name?"
        expected_execution_path = [
            'check_query_domain', 
            'retrieve', 
            'grade_docs', 
            'ddg_search', 
            'grade_docs', 
            'question_classification', 
            'generate', 
            'hallucination_check', 
            'answer_check'
        ]
        
        inputs = {"question": question, "domain": self.domain}
        state = self.chat_pdf.invoke(inputs)
        
        # Check if the execution path is as expected until the last expected step
        self.assertEqual(state['execution_path'], expected_execution_path)
        
        # Check if the question is correctly classified as non-math
        self.assertEqual(state['question_type'], 'no')
        
        # Check if initial documents are not retrieved from PDF
        # self.assertTrue(len(state['documents']) == 0, "Initial documents should not be retrieved from the PDF.")
        
        # Check if documents are retrieved from DDG and graded
        # self.assertTrue(len(state['documents']) > 0, "Documents should be retrieved from DDG search.")
        
        # Check if the answer is generated and does not contain hallucinations
        self.assertEqual(state['hallucination'], 'no')
        
        # Check if the answer is not useful
        self.assertEqual(state['answer_useful'], 'not useful')
        
        # Check if the final answer is "I don't know the answer to that question."
        self.assertEqual(state['answer']['answer'], "I don't know the answer to that question.")
        
        # Print the final answer
        print(f"Final Answer: {state['answer']['answer']}")

    def tearDown(self) -> None:
        return super().tearDown()

if __name__ == '__main__':
    unittest.main()
