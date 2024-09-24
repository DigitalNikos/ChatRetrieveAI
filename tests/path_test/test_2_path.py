import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import unittest
from rag.rag import ChatPDF
from config import Config as cfg
from langchain_core.messages import HumanMessage, AIMessage


class TestPath2(unittest.TestCase):
    def setUp(self):
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
    
    def test_path_2(self):
        question = "What is the use it?"
        expected_execution_path = [
            'check_query_domain', 
            'rephrase_based_history',
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
