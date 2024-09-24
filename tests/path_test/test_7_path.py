import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import unittest
from rag.rag import ChatPDF
from config import Config as cfg
import random
import numpy as np

class TestPath7(unittest.TestCase):
    def setUp(self):
        random.seed(42)
        np.random.seed(42)
        cfg.MODEL_TEMPERATURE = 0.0
        cfg.COLLECTION_NAME = "test_7_path"
        self.domain = "Earth"
        test_dir = os.path.dirname(__file__)
        file_path = os.path.join(test_dir, '..', 'data', 'amazon.pdf')
        source_extension = ".pdf"
        file_name = "amazon.pdf"
        self.chat_pdf = ChatPDF()
        self.chat_pdf.ingest({'file_path': file_path, 'source_extension': source_extension, 'file_name': file_name, 'domain': self.domain})

    
    def test_path_7(self):
        question = "What are the specific strategies used by ancient civilizations in South America to prevent deforestation and their impact on modern conservation efforts?"
        expected_execution_path = [
            'check_query_domain', 
            'retrieve',
            'grade_docs',
            'question_classification',
            'generate',
            'hallucination_check',
            'answer_check',
        ]
        
        inputs = {"question": question, "domain": self.domain}
        state = self.chat_pdf.invoke(inputs)
        
        # Check if the question is in the spacified domain
        self.assertEqual(state['q_domain_relevance'], 'yes', msg="Question is not in the specified domain")
        
        # Check if the exist positive grade documents
        self.assertTrue(len(state['grade_documents']) > 0, msg="No positive grade documents")
        
        # Check if the question is correctly classified as math
        self.assertEqual(state['question_type'], 'no', msg="Question is classified as math")
        
        # Check if the answer is generated and contains hallucinations
        self.assertEqual(state['hallucination'], 'no', msg="Generation score is not as expected")
        
        # Check if the answer is not useful
        self.assertEqual(state['answer_useful'], 'not useful', msg="Answer is not useful")
        
        # Check the answer is the expected one
        self.assertEqual(state['answer']['answer'], "I don't know the answer to that question.", msg="Answer is not as expected")
        
        # Check if the execution path is as expected until the last expected step
        self.assertEqual(state['execution_path'], expected_execution_path, msg="Execution path is not as expected")


    def tearDown(self) -> None:
        self.chat_pdf = None
        return super().tearDown()

if __name__ == '__main__':
    unittest.main()
