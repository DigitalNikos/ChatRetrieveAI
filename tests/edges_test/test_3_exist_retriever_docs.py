import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import unittest
import inspect

from rag.rag import ChatPDF
from config import Config as cfg


class TestExistRetrieverDocs(unittest.TestCase):
    def setUp(self):
        cfg.MODEL_TEMPERATURE = 0.0
        
        self.domain = "Sport"
        test_dir = os.path.dirname(__file__)
        file_path = os.path.join(test_dir, '..' ,'data', 'Application of Artificial_Intelligence_in_Basketball_Sport.pdf')
        source_extension = ".pdf"
        file_name = "Application of Artificial_Intelligence_in_Basketball_Sport.pdf"
        self.chat_pdf = ChatPDF()
        self.chat_pdf.ingest({'file_path': file_path, 'source_extension': source_extension, 'file_name': file_name, 'domain': self.domain})
    
    
    
    def test_positive_retrive_positive_grade_docs(self):      
        question = "How can AI improve basketball training?"
        inputs = {"question": question}
        retrieve_documents_classification = 'yes'
        grade_documents_classification = 'yes'
        
        state = self.chat_pdf.knowledge_base_system._retrieve(inputs)
        
        print(f"\n{self.__class__.__name__}:       {inspect.currentframe().f_code.co_name}")
        print(f"\nRetriever -> State:      {state}")
        
        provide_classification = 'yes' if len(state['documents']) > 0 else 'no'
        self.assertEqual(provide_classification, retrieve_documents_classification)
        
        inputs = {"question": state['question'], 'documents': state['documents']}
        state = self.chat_pdf.knowledge_base_system._grade_documents(inputs)
        
        print(f"\nGrade_documents -> State:      {state}")
        
        provide_classification = 'yes' if len(state['grade_documents']) > 0 else 'no'
        self.assertEqual(provide_classification, grade_documents_classification)
    
    
    def test_positive_retrive_negative_grade_docs(self):
        question = "What are the recent advancements in AI for detecting emotions through speech in basketball games?"
        inputs = {"question": question}
        retrieve_documents_classification = 'yes'
        grade_documents_classification = 'no'
        
        state = self.chat_pdf.knowledge_base_system._retrieve(inputs)
        
        print(f"\n{self.__class__.__name__}:       {inspect.currentframe().f_code.co_name}")
        print(f"\nRetriever -> State:      {state}")
        
        provide_classification = 'yes' if len(state['documents']) > 0 else 'no'
        self.assertEqual(provide_classification, retrieve_documents_classification)
        
        inputs = {"question": state['question'], 'documents': state['documents']}
        state = self.chat_pdf.knowledge_base_system._grade_documents(inputs)
        
        print(f"\nGrade_documents -> State:      {state}")
        
        provide_classification = 'yes' if len(state['grade_documents']) > 0 else 'no'
        self.assertEqual(provide_classification, grade_documents_classification)
        
    def tearDown(self) -> None:
        self.chat_pdf = None
        self.domain = None
        return super().tearDown()

if __name__ == '__main__':
    unittest.main()