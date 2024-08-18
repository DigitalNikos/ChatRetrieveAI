import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import unittest
from rag.rag import ChatPDF
from config import Config as cfg
from langchain_core.messages import HumanMessage, AIMessage

class TestPath1(unittest.TestCase):
    def setUp(self):
        print(cfg)
        cfg.MODEL_TEMPERATURE = 0.0
        self.domain = "Sport"
        test_dir = os.path.dirname(__file__)
        file_path = os.path.join(test_dir, '..', 'data', 'Application of Artificial_Intelligence_in_Basketball_Sport.pdf')
        source_extension = ".pdf"
        file_name = "Application of Artificial_Intelligence_in_Basketball_Sport.pdf"
        self.chat_pdf = ChatPDF(cfg)
        self.chat_pdf.ingest({'file_path': file_path, 'source_extension': source_extension, 'file_name': file_name, 'domain': self.domain})
    
    def test_path_1(self):
        question = "What are the benefits of using wearable devices in basketball training and performance analysis?"
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
        self.assertEqual(state['q_domain_relevance'], 'yes')
        
        # Check if the exist positive grade documents
        self.assertTrue(len(state['grade_documents']) > 0)
        
        # Check if the question is correctly classified as non-math
        self.assertEqual(state['question_type'], 'no')
        
        # Check if the answer is generated and does not contain hallucinations
        self.assertEqual(state['hallucination'], 'no')
        
        # Check if the answer is not useful
        self.assertEqual(state['answer_useful'], 'useful')
        
        # Check if the execution path is as expected until the last expected step
        self.assertEqual(state['execution_path'], expected_execution_path)


    def tearDown(self) -> None:
        return super().tearDown()

if __name__ == '__main__':
    unittest.main()
