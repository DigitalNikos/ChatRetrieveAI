import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import unittest
from rag.rag import ChatPDF
from config import Config as cfg


class TestPath5(unittest.TestCase):
    def setUp(self):
        cfg.MODEL_TEMPERATURE = 0.0
        cfg.COLLECTION_NAME = "test_5_path"
        self.domain = "Sport"
        test_dir = os.path.dirname(__file__)
        file_path = os.path.join(test_dir, '..', 'data', 'Application of Artificial_Intelligence_in_Basketball_Sport.pdf')
        source_extension = ".pdf"
        file_name = "Application of Artificial_Intelligence_in_Basketball_Sport.pdf"
        self.chat_pdf = ChatPDF()
        self.chat_pdf.ingest({'file_path': file_path, 'source_extension': source_extension, 'file_name': file_name, 'domain': self.domain})

    
    def test_path_5(self):
        question = "AI analysis of basketball games revealed that in the 2009/2010 NBA season, the Naive Bayes classifier was able to correctly predict 67% of 778 games. How many games did the classifier fail to predict correctly?"
        expected_execution_path = [
            'check_query_domain', 
            'retrieve',
            'grade_docs',
            'question_classification',
            'math_generate',
            'answer_check'
        ]
        
        inputs = {"question": question, "domain": self.domain}
        state = self.chat_pdf.invoke(inputs)
        
        # Check if the question is in the spacified domain
        self.assertEqual(state['q_domain_relevance'], 'yes', msg="Question is not in the specified domain")
        
        # Check if the exist positive grade documents
        self.assertTrue(len(state['grade_documents']) > 0, msg="No positive grade documents")
        
        # Check if the question is correctly classified as math
        self.assertEqual(state['question_type'], 'yes', msg="Question is not classified as math")
        
        self.assertEqual(state['answer_useful'], 'useful', msg="Answer is not useful")
        
        # Check if the execution path is as expected until the last expected step
        self.assertEqual(state['execution_path'], expected_execution_path, msg="Execution path is not as expected")


    def tearDown(self) -> None:
        self.chat_pdf = None
        self.kbs = None
        self.domain = None
        return super().tearDown()

if __name__ == '__main__':
    unittest.main()
