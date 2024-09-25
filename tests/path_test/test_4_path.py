import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import unittest

from config import Config as cfg
from langchain_core.messages import AIMessage, HumanMessage
from rag.rag import ChatPDF


class TestPath4(unittest.TestCase):
    def setUp(self):
        cfg.MODEL_TEMPERATURE = 0.0
        self.domain = "Sport"
        test_dir = os.path.dirname(__file__)
        file_path = os.path.join(test_dir, '..', 'data', 'Application of Artificial_Intelligence_in_Basketball_Sport.pdf')
        source_extension = ".pdf"
        file_name = "Application of Artificial_Intelligence_in_Basketball_Sport.pdf"
        self.chat_pdf = ChatPDF()
        self.chat_pdf.ingest({'file_path': file_path, 'source_extension': source_extension, 'file_name': file_name, 'domain': self.domain})

    
    def test_path_4(self):
        question = "Domain: Sport - What are the key findings of the study on the impact of AI on player nutrition in Sport basketball?"
        expected_execution_path = [
            'check_query_domain', 
            'retrieve',
            'grade_docs',
            'ddg_search',
            'grade_docs',
        ]
        
        inputs = {"question": question, "domain": self.domain}
        state = self.chat_pdf.invoke(inputs)
        
        # Check if the question is in the spacified domain
        self.assertEqual(state['q_domain_relevance'], 'yes', msg="Question is not in the specified domain")
        
        # Check if the exist positive grade documents
        self.assertTrue(len(state['grade_documents']) == 0, msg="No positive grade documents")
        
        # Check if the answer is not useful
        self.assertEqual(state['answer']['answer'], "I don't know the answer to that question.", msg="Answer is not as expected")
        
        # Check if the execution path is as expected until the last expected step
        self.assertEqual(state['execution_path'], expected_execution_path, msg="Execution path is not as expected")


    def tearDown(self) -> None:
        self.chat_pdf = None
        return super().tearDown()

if __name__ == '__main__':
    unittest.main()
