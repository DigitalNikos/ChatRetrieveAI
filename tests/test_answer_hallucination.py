import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import unittest
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
        file_path = os.path.join(test_dir, 'data', 'Application of Artificial_Intelligence_in_Basketball_Sport.pdf')
        source_extension = ".pdf"
        file_name = "Application of Artificial_Intelligence_in_Basketball_Sport.pdf"
        self.chat_pdf = ChatPDF(cfg)
        self.chat_pdf.ingest({'file_path': file_path, 'source_extension': source_extension, 'file_name': file_name, 'domain': self.domain})
        self.kbs = self.chat_pdf.knowledge_base_system
    
    def test_positive_answer_hallucination(self):
        question = "How does AI help in preventing injuries in basketball players?"
        inputs = {"question": question, "execution_path": []}
        retrieve_documents_classification = 'yes'
        
        state = self.kbs._retrieve(inputs)
        provide_classification = 'yes' if len(state['documents']) > 0 else 'no'
        self.assertEqual(provide_classification, retrieve_documents_classification)
        
        inputs = {"documents": state["documents"], "question": state["question"], "execution_path": []}
        state = self.kbs._grade_documents(inputs)
        
        inputs = {"grade_documents": state["grade_documents"], "question": question, "execution_path": []}
        state = self.kbs._generate(inputs)
        
        hallucination_classification = state['answer']
        inputs = {'grade_documents': state['grade_documents'], "answer": state["answer"], "execution_path": []}
        state = self.kbs._hallucination_check(inputs)
        
        print(f"State: {state}")
        provide_classification = state['answer']
        self.assertEqual(provide_classification, hallucination_classification)
    
    def  test_negative_answer_hallucination(self):
        question = "How do boxers effectively use feints to set up their punches?"
        inputs = {"question": question, "execution_path": []}
        retrieve_documents_classification = 'yes'
        hallucination_classification = 'yes'
        
        state = self.kbs._retrieve(inputs)
        provide_classification = 'yes' if len(state['documents']) > 0 else 'no'
        self.assertEqual(provide_classification, retrieve_documents_classification)
        
        inputs = {"grade_documents": state["documents"], "question": state["question"], "execution_path": []}
        state = self.kbs._generate(inputs)
        
        inputs = {'grade_documents': state['grade_documents'], "answer": state["answer"], "execution_path": []}
        state = self.kbs._hallucination_check(inputs)
        
        print(f"State: {state}")
        provide_classification = state['hallucination']
        self.assertEqual(provide_classification, hallucination_classification)

        
    def tearDown(self) -> None:
        return super().tearDown()

if __name__ == '__main__':
    unittest.main()