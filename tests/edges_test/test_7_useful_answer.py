import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import unittest
import inspect

from rag.rag import ChatPDF
from config import Config as cfg

class TestUsefulAnswer(unittest.TestCase):
    def setUp(self):
        cfg.MODEL_TEMPERATURE = 0.0
        self.chat_pdf = ChatPDF()
        self.kbs = self.chat_pdf.knowledge_base_system

    
    def test_positive_useful_answer(self):
        question = "What are the main applications of AI in improving player performance in basketball?"
        generation = {
            "answer": {
                "answer": "AI has several applications in improving player performance in basketball. It is used for performance analysis through wearable devices and video analysis, allowing coaches to develop personalized training programs. AI also helps in injury prevention by analyzing physiological data and predicting potential injuries. Additionally, AI assists in strategic planning by evaluating game data and suggesting optimal strategies."
            }
        }
        usuful_answer_classification = 'useful'
        
        inputs = {"question": question, "answer": generation}
        state = self.kbs._answer_check(inputs)
        
        print(f"\n{self.__class__.__name__}:       {inspect.currentframe().f_code.co_name}")
        print(f"\nAnswer_check -> State:      {state}")
        
        provide_classification = state['answer_useful']
        self.assertEqual(provide_classification, usuful_answer_classification)

    
    def  test_negative_useful_answer(self):
        question = "What are the benefits of space tourism?"
        generation ={
            "answer": {
                "answer": "The process of photosynthesis allows plants to convert sunlight into energy, which is essential for their growth and oxygen production."
            }
        }

        usuful_answer_classification = 'not useful'
        
        inputs = {"question": question, "answer": generation}
        state = self.kbs._answer_check(inputs)
        
        print(f"\n{self.__class__.__name__}:       {inspect.currentframe().f_code.co_name}")
        print(f"\nAnswer_check -> State:      {state}")
        
        provide_classification = state['answer_useful']
        self.assertEqual(provide_classification, usuful_answer_classification)

        
    def tearDown(self) -> None:
        self.chat_pdf = None
        self.kbs = None
        return super().tearDown()

if __name__ == '__main__':
    unittest.main()