import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))


import unittest
from rag.rag import ChatPDF
from config import Config as cfg
from langchain_core.messages import HumanMessage, AIMessage

class TestPath3(unittest.TestCase):
    def setUp(self):
        print(cfg)
        cfg.MODEL_TEMPERATURE = 0.0
        self.domain = "Technology"
        test_dir = os.path.dirname(__file__)
        file_path = os.path.join(test_dir, '..', 'data', 'Assignment1.pdf')
        source_extension = ".pdf"
        file_name = "Assignment1.pdf"
        self.chat_pdf = ChatPDF()
        self.kbs = self.chat_pdf.knowledge_base_system
        self.chat_pdf.ingest({'file_path': file_path, 'source_extension': source_extension, 'file_name': file_name, 'domain': self.domain})
        self.kbs.chat_rephrased_history.extend([HumanMessage(
                                        content="Is the Camera Service already implemented?"
                                    ), 
                                      AIMessage(
                                        content="Yes, the Camera Service is already implemented."
                                    )])
    
    def test_path_3(self):
        question = "How is the weather today?"
        expected_execution_path = [
            'check_query_domain', 
            'rephrase_based_history',
            'check_query_domain', 
        ]
        
        inputs = {"question": question, "domain": self.domain}
        state = self.chat_pdf.invoke(inputs)
        
        # Check if the question is in the spacified domain
        self.assertEqual(state['q_domain_relevance'], 'no')
        
        # Check if the answer is not useful
        self.assertEqual(state['answer']['answer'], "I don't know the answer to that question.")
        
        # Check if the execution path is as expected until the last expected step
        self.assertEqual(state['execution_path'], expected_execution_path)


    def tearDown(self) -> None:
        return super().tearDown()

if __name__ == '__main__':
    unittest.main()
