import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import unittest
import inspect

from rag.rag import ChatPDF
from config import Config as cfg
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.documents import Document


class TestAnswerHallucination(unittest.TestCase):
    def setUp(self):
        cfg.MODEL_TEMPERATURE = 0.0
        cfg.RETRIEVER_SCORE_THRESHOLD = 0.5
        self.domain = "Sport"
        test_dir = os.path.dirname(__file__)
        file_path = os.path.join(test_dir, '..', 'data', 'Application of Artificial_Intelligence_in_Basketball_Sport.pdf')
        source_extension = ".pdf"
        file_name = "Application of Artificial_Intelligence_in_Basketball_Sport.pdf"
        self.chat_pdf = ChatPDF(cfg)
        self.chat_pdf.ingest({'file_path': file_path, 'source_extension': source_extension, 'file_name': file_name, 'domain': self.domain})
        self.kbs = self.chat_pdf.knowledge_base_system
        
    
    def test_positive_answer_hallucination(self):
        question = "How does AI help in preventing injuries in basketball players?"
        retrieve_documents_classification = 'yes'
        
        inputs = {"question": question,}
        state = self.kbs._retrieve(inputs)
        
        print(f"\n{self.__class__.__name__}:       {inspect.currentframe().f_code.co_name}")
        print(f"\nRetrieve -> State:      {state}")
        
        provide_classification = 'yes' if len(state['documents']) > 0 else 'no'
        self.assertEqual(provide_classification, retrieve_documents_classification, msg="Retrieve classification is not as expected")
        
        inputs = {"documents": state["documents"], "question": state["question"]}
        state = self.kbs._grade_documents(inputs)
        
        print(f"\nGrade_documents -> State:      {state}")
        
        inputs = {"grade_documents": state["grade_documents"], "question": question}
        state = self.kbs._generate(inputs)

        print(f"\nGenerate -> State:      {state}")
        
        hallucination_classification = 'no'
        print(f"\nHallucination_check -> State:      {state['answer']}")
        inputs = {'grade_documents': state['grade_documents'], "answer": state["answer"]}
        state = self.kbs._hallucination_check(inputs)
        
        provide_classification = state['hallucination']
        self.assertEqual(provide_classification, hallucination_classification, msg="Hallucination classification is not as expected")
        
    
    def  test_negative_answer_hallucination(self):
        question = "How do boxers effectively use feints to set up their punches?"
        retrieve_documents_classification = 'yes'
        hallucination_classification = 'yes'
        
        inputs = {"question": question}
        state = self.kbs._retrieve(inputs)
        
        print(f"\n{self.__class__.__name__}:       {inspect.currentframe().f_code.co_name}")
        print(f"\nRetrieve -> State:      {state}")
        
        provide_classification = 'yes' if len(state['documents']) > 0 else 'no'
        self.assertEqual(provide_classification, retrieve_documents_classification, msg="Retrieve classification is not as expected")
        
        inputs = {"grade_documents": state["documents"], "question": state["question"]}
        state = self.kbs._generate(inputs)
        
        print(f"\nGenerate -> State:      {state}")
        
        inputs = {'grade_documents': state['grade_documents'], "answer": state["answer"]}
        state = self.kbs._hallucination_check(inputs)
        
        print(f"\nHallucination_check -> State:      {state}")
        
        provide_classification = state['hallucination']
        self.assertEqual(provide_classification, hallucination_classification, msg="Hallucination classification is not as expected")

        
    def tearDown(self) -> None:
        self.chat_pdf = None
        self.kbs = None
        self.domain = None
        return super().tearDown()

if __name__ == '__main__':
    unittest.main()