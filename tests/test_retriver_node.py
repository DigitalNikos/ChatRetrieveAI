import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import unittest
from rag.rag import ChatPDF
from config import Config as cfg
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.documents import Document



class TestRetrieverNode(unittest.TestCase):
    def setUp(self):
        cfg.MODEL_TEMPERATURE = 0.0
        self.domain = "Sport"
        test_dir = os.path.dirname(__file__)
        file_path = os.path.join(test_dir, 'data', 'Application of Artificial_Intelligence_in_Basketball_Sport.pdf')
        source_extension = ".pdf"
        file_name = "Application of Artificial_Intelligence_in_Basketball_Sport.pdf"
        
        self.chat_pdf = ChatPDF(cfg)
        self.chat_pdf.ingest({'file_path': file_path, 'source_extension': source_extension, 'file_name': file_name, 'domain': self.domain})
        self.kbs = self.chat_pdf.knowledge_base_system
    
    def test_positive_retrive_docs(self):
        question = "How can AI improve basketball training?"
        inputs = {"question": question, "execution_path": []}
        expected_classification = [Document(metadata={'source': 'Application of Artificial_Intelligence_in_Basketball_Sport.pdf - page: 8'}, page_content='61multipleteacherstohelpstudentsinindividualizedandautonomouslearning.zhaoandxieused aquestionnairesurveymethodtoevaluatetheeffectoficaiinbasketballtraining.thesurvey resultsshowedthatstudentsandteachersweresatisfiedwiththeeffectsoficai,buttherewere deficienciesintheuseprocessthatmustbefurtherimproved39.moreover,yangintroduced anaibasketballcoachingsystembasedonthebaumwelchalgorithm.thesystemcould formulateandadjustthetrainingplanbasedonindividualplayersphysicalconditions,athletic ability,andchangesinsportskillsmeasuredduringtraining.theauthorconductedan experimentalstudyon20juniorbasketballplayerstocomparethetrainingeffectsoftheai basketballcoachingsystemandtraditionalcoachingmethod.theresultsshowedthattheai technologysignificantlyimprovedthetrainingefficiencyofbasketballplayers40. aibasketballtrainingmachine intelligentmachinerycanimprovethetrainingefficiencyofbasketballplayers.liuand'), Document(metadata={'source': 'Application of Artificial_Intelligence_in_Basketball_Sport.pdf - page: 8'}, page_content='practice,theauthorsconfirmedthattheequipmentcouldeffectivelyimprovetheathletes shootingpercentage41.basketballtrainingrobotsareincreasinglyusedinbasketballtraining. inordertoavoidinjurycausedbythecollisionbetweenthemobiletrainingrobotandtheathlete, xuandtangusedultrasonicsignalsofobstaclescollectedaroundtherobotasinputandapplied animprovedqlearningalgorithmbasedonmachinelearning,sothattherobotcaneffectively avoidcollisionswithathletesduringthetrainingcourseandallowathletestoreceivescientific andefficienttraining42. intelligentarena liuetal.proposedadeeplearningbasedbasketballvideoanalysissolutionforuseinan intelligentbasketballarena.thissolutioncouldautomaticallybroadcastbasketballgames,detect scores,andgeneratehighlightvideosinrealbasketballgames.theprogramwasimplemented intoabusinessintelligencebasketballarenaapplication,standzbasketball.accordingtoa'), Document(metadata={'source': 'Application of Artificial_Intelligence_in_Basketball_Sport.pdf - page: 8'}, page_content='ability,andchangesinsportskillsmeasuredduringtraining.theauthorconductedan experimentalstudyon20juniorbasketballplayerstocomparethetrainingeffectsoftheai basketballcoachingsystemandtraditionalcoachingmethod.theresultsshowedthattheai technologysignificantlyimprovedthetrainingefficiencyofbasketballplayers40. aibasketballtrainingmachine intelligentmachinerycanimprovethetrainingefficiencyofbasketballplayers.liuand liintroducedanintelligentbasketballshootingtrainingvehicle,whichcouldimproveathletes basketballshootingtargetcaptureabilityandresponsespeed.basedontheshootingtraining practice,theauthorsconfirmedthattheequipmentcouldeffectivelyimprovetheathletes shootingpercentage41.basketballtrainingrobotsareincreasinglyusedinbasketballtraining. inordertoavoidinjurycausedbythecollisionbetweenthemobiletrainingrobotandtheathlete, xuandtangusedultrasonicsignalsofobstaclescollectedaroundtherobotasinputandapplied')]
        
        state = self.kbs._retrieve(inputs)
        print(f"State: {state}")
        self.assertEqual(state['documents'], expected_classification)
        
        
    def tearDown(self) -> None:
        return super().tearDown()

if __name__ == '__main__':
    unittest.main()