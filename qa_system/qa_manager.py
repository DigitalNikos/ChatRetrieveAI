import numexpr as ne
from tqdm import tqdm
from typing import List, Dict
from config import Config as cfg
from typing_extensions import TypedDict
from qa_system.lang_graph import WorkflowInitializer
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from langchain.chains import create_history_aware_retriever
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_experimental.llms.ollama_functions import OllamaFunctions
from utils import convert_str_to_document, extract_limited_chat_history, print_documents
from qa_system.structure_answer import AnswerWithSources, AnswerWithSourcesMath, AnswerWithWebSourcesMath
from qa_system.prompts import (generate_answer,rephrase_prompt, grader_document_prompt, hallucination_grader_prompt, 
                               answers_grader_prompt, query_domain_check, question_classifier_prompt,math_solver, math_solver_web)


class KnowledgeBaseSystem:
    print('\n--- KNOWLEDGE BASE SYSTEM ---')
    class GraphState(TypedDict):
        """
        Represents the state of our graph.

        """
        question: str
        rephrase_question: str
        q_domain_relevance: str
        documents: List[str]
        grade_documents: List[str]
        domain : str
        hallucination: str
        math_score: str
        question_type: str
        execution_path: List[str] = []
        answer_useful: str
        answer: Dict


    def __init__(self):  
        print('knowledge_base_system.py - __init__()')    
        self.retriever = None
        self.chat_history = []
        self.chat_rephrased_history = []
        
        # LLMs
        self.json_llm = ChatOllama(model=cfg.MODEL, format=cfg.MODEL_FORMAT, temperature=cfg.MODEL_TEMPERATURE)  
        self.llm = OllamaFunctions(model=cfg.MODEL, keep_alive=-1, format=cfg.MODEL_FORMAT, temperature=cfg.MODEL_TEMPERATURE) 
        self.llm_mathematic_resoning = OllamaFunctions(model=cfg.MODEL, keep_alive=cfg.KEEP_IN_MEMORY, format=cfg.MODEL_FORMAT, temperature=cfg.MODEL_TEMPERATURE) 
        self.llm_mathematic_web_based_resoning = OllamaFunctions(model=cfg.MODEL, keep_alive=cfg.KEEP_IN_MEMORY, format=cfg.MODEL_FORMAT, temperature=cfg.MODEL_TEMPERATURE) 
        
        # STRUCTURED LLMs
        self.structured_llm = self.llm.with_structured_output(AnswerWithSources)
        self.structured_llm_numexpr = self.llm_mathematic_resoning.with_structured_output(AnswerWithSourcesMath)
        self.structured_llm_not_numexpr = self.llm_mathematic_web_based_resoning.with_structured_output(AnswerWithWebSourcesMath)
        
        # CHAINS
        self.generate_answer = generate_answer | self.structured_llm
        self.query_domain_check = query_domain_check | self.json_llm | JsonOutputParser()
        self.rephrase_query_chain = rephrase_prompt | self.json_llm | JsonOutputParser()
        self.retrieval_grader_document_chain = grader_document_prompt | self.json_llm | JsonOutputParser()
        self.hallucination_grader_chain = hallucination_grader_prompt | self.json_llm | JsonOutputParser()
        self.answer_grader_chain = answers_grader_prompt | self.json_llm | JsonOutputParser()
        self.question_classifier = question_classifier_prompt | self.json_llm | JsonOutputParser()
        self.search_ddg_search_results = DuckDuckGoSearchResults(num_results = cfg.N_DDG_TO_RETRIEVE, verbose = True)
        self.chain_math_numexpr = math_solver |self.structured_llm_numexpr 
        self.chain_math_not_numexpr = math_solver_web | self.structured_llm_not_numexpr
        
        # GRAPH APP
        self.app = WorkflowInitializer(self).initialize()
            
    
    def _check_query_domain(self, state: GraphState):
        """
        Check if the query belongs to the specified domain using an LLM call.
        
        Returns:
            state (dict): Updated state with the domain relevance score and a default answer.
        """
        print("\n--- CHECK QUERY DOMAIN ---")
        
        if "execution_path" in state:
            state['execution_path'].extend(["check_query_domain"])
        
        print("\nQuestion:  {}".format(state["question"]))
        print("\nDomain:    {}".format(state["domain"]))

        try:
            answer = self.query_domain_check.invoke({"question": state["question"], "domain": state["domain"]})
            state['q_domain_relevance'] = answer['score']
            state['answer'] = {'answer': "I don't know the answer to that question.", 'metadata': "No metadata"}
            print("\nResult:    {}".format(answer['score']))
        except Exception as e:
            print(f"KeyError: {e}. _check_query_domain() - Response may not contain expected fields.")
            state['q_domain_relevance'] = 'no'
            state['answer'] = {'answer': "I don't know the answer to that question due to an internal error .", 'metadata': "No metadata"}
        
        return state
    
    
    def _rephrase_query(self, state: GraphState):
        """
        Transform the query based on the existing "chat_rephrased_history" if it is relevant.
            
        Returns:
            state (dict): Updates 'question' key with a re-phrased question value
        """
        print("\n--- REPHRASE QUERY ---")
        
        if "execution_path" in state:
            state['execution_path'].extend(["rephrase_based_history"])
        
        print("\nQuestion:         {}".format(state["question"]))
        
        try:
            chat_history_content = extract_limited_chat_history(self.chat_rephrased_history, max_length=3500)
            rephrased_query = self.rephrase_query_chain.invoke({"input": state["question"], "chat_history": chat_history_content})
            print("\nRephrased query:  {}".format(rephrased_query))
            state["question"] = rephrased_query['question']
        except Exception as e:
            print(f"KeyError: {e}. _rephrase_query() - Response may not contain expected fields.")
            state["question"] = state["question"]
            
        return state
   
    
    def _retrieve(self, state: GraphState):
        """
        Retrieve documents from the retriever.
            
        Returns:
            state (dict): Updated state with 'documents' containing retrieved documents or an empty list if no retriever is available.
        """
        print("\n--- RETRIEVE DOCUMENTS ---")
        
        if "execution_path" in state:
            state['execution_path'].extend(["retrieve"])
        
        print("\nQuestion to retrive:    {}".format(state["question"]))
        
        if self.retriever is None: 
            print("\nNo files or URLs detected. Returning an empty document list.")
            state["documents"] = []
            return state

        chat_retriever_chain = create_history_aware_retriever(self.json_llm, self.retriever, rephrase_prompt)
        documents = chat_retriever_chain.invoke({"input": state["question"], "chat_history": self.chat_history})
        
        print("\nRetrieved Documents:    ")
        print_documents(documents)
        
        state['documents'] = documents
        return state
    
    
    def _grade_documents(self, state: GraphState):
        """
        Determines whether the retrieved documents are relevant to the question.
            
        Returns:
            state (dict): Updated state with 'grade_documents' key containing only relevant documents.
        """
        print("\n--- GRADE RETRIEVED DOCUMENTS---")
        if "execution_path" in state:
            state['execution_path'].extend(["grade_docs"])
            
        documents = state["documents"]
        num_documents = len(documents)
        
        print("\nNumber of documents:  {}".format(num_documents))
        
        try: 
            filtered_docs = []
            with tqdm(total=num_documents, desc="Grading Documents", ncols=100) as pbar:
                for d in documents:
                    score = self.retrieval_grader_document_chain.invoke({"question": state["question"], "document": d.page_content})
                    grade = score["score"]
                    if grade == "yes":
                        filtered_docs.append(d)
                    else:
                        continue
                    pbar.update(1)
        except Exception as e:
            print(f"KeyError: {e}. _grade_documents() - Response may not contain expected fields.")
            filtered_docs = []
                

        print("\nRelevant document filter:   {}/{}".format(len(filtered_docs), num_documents))
        print("\nFiltered Documents:     ")
        print_documents(filtered_docs)
                
        state["grade_documents"] = filtered_docs
        return state
    

    def _generate(self, state: GraphState):
        """
        Generate an answer using the provided context and question.
            
        Returns:
            state (dict): Updated state key 'answer" with the new answer.
        """
        print("\n--- GENERATE ANSWER ---")
        
        if "execution_path" in state:
            state['execution_path'].extend(["generate"])
        
        print("\nQuestion:               {}".format(state["question"]))
        
        try:
            generation = self.generate_answer.invoke({"context": state['grade_documents'], "question": state["question"]})
            metadata = ' '.join(generation.sources)
            
            print("\nAnswer:                 {}".format(generation.answer))
            print("\nMetadata:               {}".format(metadata))
            
            state ["answer"] = {'answer': generation.answer, 'metadata': metadata}
        except Exception as e:
            print(f"KeyError: {e}. _generate() - Response may not contain expected fields.")
            state["answer"] = {'answer': "I don't know the answer to that question.", 'metadata': "No metadata"}
            
        return state

    
    def _question_classifier(self, state: GraphState):
        """
        Classify the question needs math solution or not .
            
        Returns:
            state (dict): Updated state with 'question_type' key containing the classification.
        """
        print("\n--- QUESTION CLASSIFIER ---")
        
        if "execution_path" in state:
            state['execution_path'].extend(["question_classification"])
        
        try:
            question_type = self.question_classifier.invoke({"question": state["question"]})
            print("\nQuestion Type: {}".format("math" if question_type['score'] == "yes" else "text"))
            
            state["question_type"] = question_type['score']
        except Exception as e:
            print(f"KeyError: {e}. _question_classifier() - Response may not contain expected fields.")
            state["question_type"] = 'error'
            
        return state
    

    def _math_generate(self, state: GraphState):
        """
        Generate an arithmetic resoning answer using the provided context and question.
        
        Returns:
            state (dict): Updated state key 'answer' with the new answer.
        """
        print("\n--- GENERATE MATH ANSWER ---")
        
        if "execution_path" in state:
            state['execution_path'].extend(["math_generate"])
                    
        try:
            print("\nQuestion:                {}".format(state["question"]))
            
            generation = self.chain_math_numexpr.invoke({"question": state["question"], "documents": state["grade_documents"]})
            stepwise_str = generation.step_wise_reasoning
            expr_str = generation.expr
            sources = ','.join(generation.sources)
            
            print("\nStepwise Reasoning:      {}".format(stepwise_str))
            print("\nExpression:              {}".format(expr_str))  
            print("\nSources:                 {}".format(sources))
            
            answer_to_neEvaluate = ne.evaluate(expr_str)
            
            print("\nAnswer to ne.evaluate:   {}".format(answer_to_neEvaluate))
            
            steps_str = [f"Step {i+1}: {step}\n" for i, step in enumerate(stepwise_str)]
            stepwise_str = "💡 Solution:\n\n" + "\n".join(steps_str)
            
            state['answer'] = {"answer": f"{stepwise_str}\n\n Final answer: {answer_to_neEvaluate} ", "metadata": sources, "calculation": 'Computed with python'}
            state['math_score'] = "yes"
        except Exception as e:
            try:
                answer_web = self.chain_math_not_numexpr.invoke({"question": state["question"], "documents": state["grade_documents"]})
                print("\nAnswer Web:              {}".format(answer_web))
                stepwise_str = answer_web.step_wise_reasoning
                solution = answer_web.solution
                sources = ','.join(answer_web.sources)
                print("\nStepwise Reasoning:      {}".format(stepwise_str))
                print("\nSolution:              {}".format(solution))  
                print("\nSources:                 {}".format(sources))
                
                steps_str = [f"Step {i+1}: {step}\n" for i, step in enumerate(stepwise_str)]
                stepwise_str = "🤔 Solution:\n\n" + "\n".join(steps_str)
                
                state['math_score'] = "no"
                state['answer'] = {"answer": f"{stepwise_str}\n\n Final answer: {solution} ", "metadata": sources, "calculation": 'Not python computed or Web based solution, \n maybe not be accurate. (Check the sources)🚨'}
            except Exception as e:
                state['answer'] = {
                    "answer": "Unexpected error occurred",
                    "metadata": "No metadata",
                    "calculation": "Error occurred"
                }
                state['math_score'] = "no"
                
        return state


    def _ddg_search(self, state: GraphState):
        """
        Perform a DuckDuckGo search and retrieve documents.
            
        Returns:
            state (dict): Updated state with 'documents' containing retrieved documents from DDG.
        """
        print("\n--- DDG SEARCH ---")
        
        state['execution_path'].extend(["ddg_search"])
        
        try:
            documents = self.search_ddg_search_results.invoke({"query": state["question"]})
            documents = convert_str_to_document(documents)
            
            print("\nDocuments DDG:")
            print_documents(documents)
            
            state["documents"] = documents
        except Exception as e:
            print(f"KeyError: {e}. _ddg_search() - Response may not contain expected fields.")
            state["documents"] = []
        
        return state

    
    def _hallucination_check(self, state: GraphState):
        """
        Determines whether the generation is grounded in the document and answers question.

        Returns:
            state (dict): Updated state with hallucination check result
        """
        print("\n--- HALLUCINATIONS CHECK ---")
        
        if "execution_path" in state:
            state['execution_path'].extend(["hallucination_check"])
        
        try:
            score = self.hallucination_grader_chain.invoke({"documents":  state["grade_documents"], "generation": state["answer"]}) 
            state["hallucination"] = "no" if score["score"] == "yes" else "yes"
            
            if score["score"] == "no":
                state["answer"] = {'answer': "I don't know the answer to that question.", 'metadata': "No metadata"}
            
            print(f"\nDECISION: generation is {'grounded in documents' if score['score'] == 'yes' else 'not grounded in documents, re-try'}")
        except Exception as e:
            print(f"KeyError: {e}. _hallucination_check() - Response may not contain expected fields.")
            state["hallucination"] = "yes"
            
        return state


    def _answer_check(self, state: GraphState):
        """
        Determines whether the generation answers the question.

        Returns:
            state (dict): Updated state 'answer_useful' key with the result of the answer check.
        """
        print("\n--- FINAL ANSWER CHECK ---")
        
        if "execution_path" in state:
            state['execution_path'].extend(["answer_check"])

        try:
            print("\nQuestion:      {}".format(state["question"]))
            print("\nAnswer:      {}".format(state["answer"]['answer']))
            
            score = self.answer_grader_chain.invoke({"question": state["question"], "generation": state["answer"]})
            print("\nScore:      {}".format(score))
            state["answer_useful"] = "useful" if score["score"] == "yes" else "not useful"
            print(f"\nDECISION: generation {'addresses' if score['score'] == 'yes' else 'does not address'} question")
            
            if score["score"] == "no":
                state["answer"] = {'answer': "I don't know the answer to that question.", 'metadata': "No metadata"} 
                
        except Exception as e:
            print(f"KeyError: {e}. _answer_check() - Response may not contain expected fields.")
            state["answer_useful"] = "not useful"
            
        return state
    

    def invoke(self, inputs):
        """
        Invoke the Knowledge Base System with the provided inputs.

        Returns:
            answer (dict): The answer to the question.
        """
        print("\n--- INOVKE START ---")
        print('\nDomain:       {}'.format(inputs['domain']))
        print("\nQuestion:     {}".format(inputs['question']))
        
        # Initialize the inputs with 'execution_path' key for the Unit Test
        inputs['execution_path'] = []
                
        try:
            answer = self.app.invoke(inputs)
        except Exception as e:
            print("\nException:   {}".format(e))            
            answer = {"answer": "I don't know the answer to that question", "metadata": "No metadata"}
        
        self.chat_history.extend([HumanMessage(content=inputs['question']), AIMessage(content=answer['answer']['answer'])])
        self.chat_rephrased_history.extend([HumanMessage(content=answer['question']), AIMessage(content=answer['answer']['answer'])])   
        return answer
    
    def set_retriever(self, retriever):
        self.retriever = retriever