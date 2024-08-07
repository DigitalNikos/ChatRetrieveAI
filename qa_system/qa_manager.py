from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from langchain.chains import create_history_aware_retriever
from langchain_community.tools import DuckDuckGoSearchResults

from tqdm import tqdm
from typing import List
from typing_extensions import TypedDict

from qa_system.lang_graph import WorkflowInitializer
from utils import convert_str_to_document, extract_limited_chat_history
from qa_system.prompts import (generate_answer_propmpt,rephrase_prompt, 
                     grader_document_prompt, hallucination_grader_prompt, 
                     answers_grader_prompt, query_domain_check, question_classifier_prompt,math_solver)

import numexpr as ne


class KnowledgeBaseSystem:
    print('\n--- KNOWLEDGE BASE SYSTEM ---')
    class GraphState(TypedDict):
        """
        Represents the state of our graph.

        """
        print('\n--- GRAPH STATE ---')
        question: str
        rephrase_question: str
        answer: str
        documents: List[str]
        domain : str
        hallucination: str
        generation_score: str
        question_type: str
        execution_path: List[str] = []

    def __init__(self, general_llm_model_name: str, math_llm_model_name: str, retriever = None, temperature=0):  
        print('\nknowledge_base_system.py - __init__()')    
        self.retriever = retriever
        self.chat_history = []
        self.chat_rephrased_history = []
        
        # LLMs
        self.json_llm = ChatOllama(model=general_llm_model_name, format="json", temperature=temperature) 
        self.math_llm = ChatOllama(model=math_llm_model_name, format="json", temperature=temperature)  
        
        # CHAINS
        self.query_domain_check = query_domain_check | self.json_llm | JsonOutputParser()
        self.rephrase_query_chain = rephrase_prompt | self.json_llm | JsonOutputParser()
        self.retrieval_grader_document_chain = grader_document_prompt | self.json_llm | JsonOutputParser()
        self.generate_answer = generate_answer_propmpt | self.json_llm | JsonOutputParser()
        self.hallucination_grader_chain = hallucination_grader_prompt | self.json_llm | JsonOutputParser()
        self.answer_grader_chain = answers_grader_prompt | self.json_llm | JsonOutputParser()
        self.question_classifier = question_classifier_prompt | self.json_llm | JsonOutputParser()
        self.search_ddg_search_results = DuckDuckGoSearchResults(num_results = 4, verbose = True)
        self.chain_math = math_solver |self.math_llm | JsonOutputParser()
        
        # GRAPH APP
        self.app = WorkflowInitializer(self).initialize()
            
    
    def _check_query_domain(self, state: GraphState):
        """
        Check if the query belongs to the specified domain using the specific chain.
        
        Args:
            state (GraphState): Contains 'question' and 'domain' keys.

        Returns:
            str: 'yes' query within the domain, otherwise 'no'.
        """
        print("\n--- CHECK QUERY DOMAIN ---")
        state['execution_path'].extend(["check_query_domain"])
        
        question = state["question"]
        domain = state["domain"]
        print("\nQuestion:  {}".format(question))
        print("\nDomain:    {}".format(domain))

        answer = self.query_domain_check.invoke({"question": question, "domain": domain})
        print("\nAnswer:    {}".format(answer))
        
        state['generation_score'] = answer['score']
        state['generation_score'] = 'yes'
        state['answer'] = {"answer": "I don't know the answer to that question.", "metadata": "No metadata"}
        return state
    
    
    def _rephrase_query(self, state: GraphState):
        """
        Transform the query to produce a better question.
        
        Args:
            state (dict): The current graph state
            
        Returns:
            state (dict): Updates 'question' key with a re-phrased question
        """
        print("\n--- REPHRASE QUERY ---")
        state['execution_path'].extend(["rephrase_based_history"])
        print("\nQuestion:         {}".format(state["question"]))
        
        chat_history_content = extract_limited_chat_history(self.chat_rephrased_history, max_length=3500)
        rephrased_query = self.rephrase_query_chain.invoke({"input": state["question"], "chat_history": chat_history_content})
        
        print("\nRephrased query:  {}".format(rephrased_query))
        state["question"] = rephrased_query['question']
        return state
   
    
    def _retrieve(self, state: GraphState):
        """
        Retrieve documents from the retriever.
        
        Args:
            state (dict): Contains the current graph state, including the question.
            
        Returns:
             GraphState: Updated state with 'documents' containing retrieved documents or an empty list if no retriever is available.
        """
        print("\n--- RETRIEVE DOCUMENTS ---")
        state['execution_path'].extend(["retrieve"])
        
        if self.retriever is None: # or raise error
            print("\nNo files or URLs uploaded. Returning empty documents.")
            state["documents"] = []
            return state
        
        print("\nQuestion to retrive:    {}".format(state["question"]))

        chat_retriever_chain = create_history_aware_retriever(self.json_llm, self.retriever, rephrase_prompt)
        documents = chat_retriever_chain.invoke({"input": state["question"], "chat_history": self.chat_history})
        state['documents'] = documents 
        print("\nRetrieved Documents:    {}".format(documents))
        return state
    

    def _generate(self, state: GraphState):
        """
        Generate an answer using the provided context and question.
        
        Args:
            state (dict): The current graph state
            
        Returns:
            dict: Updated state with a new key 'generation' containing the LLM generation.
        """
        print("\n--- GENERATE ANSWER ---")
        print("\nQuestion:                {}".format(state["question"]))
        state['execution_path'].extend(["generate"])
        
        generation = self.generate_answer.invoke({"context": state["documents"], "question": state["question"], "chat_history": self.chat_history})
        
        print("\nAnswer:                 {}".format(generation))

        return {"documents": state["documents"], "question": state["question"], "answer": generation}


    def _grade_documents(self, state: GraphState):
        """
        Determines whether the retrieved documents are relevant to the question.
        
        Args:
            state (dict): The current graph state
            
        Returns:
            dict: Updated state with 'documents' key containing only relevant documents.
        """
        print("\n--- GRADE RETRIEVED DOCUMENTS---")
        state['execution_path'].extend(["grade_docs"])
        
        documents = state["documents"]
        num_documents = len(documents)
        
        print("\nNumber of documents:  {}".format(num_documents))
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
        
        print("\nRelevant documents:   {}/{}".format(len(filtered_docs), num_documents))
        state["documents"] = filtered_docs
        return state
    
    def _question_classifier(self, state: GraphState):
        """
        Classify the question needs math solution or not .
        
        Args:
            state (dict): The current graph state
            
        Returns:
            dict: Updated state with 'question_type' key containing the classification.
        """
        print("\n--- QUESTION CLASSIFIER ---")
        state['execution_path'].extend(["question_classification"])
        
        question = state["question"]
        question_type = self.question_classifier.invoke({"question": question})
        state["question_type"] = question_type['score']
        print("\nQuestion Type:  {}".format(state["question_type"]))
        return state
    

    def _math_generate(self, state: GraphState):
        """
        Generate an answer using the provided context and question.
        
        Args:
            state (dict): The current graph state
            
        Returns:
            dict: Updated state with a new key 'generation' containing the LLM generation.
        """
        print("\n--- GENERATE MATH ANSWER ---")
        print("\nQuestion:                {}".format(state["question"]))
        state['execution_path'].extend(["math_generate"])
        
        generation = self.chain_math.invoke({"question": state["question"], "documents": state["documents"]})
        steps_str = [f"Step {i+1}: {step}" for i, step in enumerate(generation['step-wise reasoning'])]
        stepwise_str = "Solution:\n\n" + "\n".join(steps_str).replace('*', '\\*')
        expr_str = generation['expr'].replace('*', '\\*')
        try:
            state['answer'] = {"answer": f"{stepwise_str}\n\n Final answer: {expr_str} = {ne.evaluate(generation['expr'])}", "metadata": "Computed with python"}
            state['generation_score'] = "yes"
        except Exception as e:
            state['answer'] = {"answer": f"{stepwise_str}\n\n Final answer: {expr_str}", "metadata": "No metadata"}
            state['generation_score'] = "no"

        print("\nAnswer:                 {}".format(state['answer']))

        return state

    def _ddg_search(self, state: GraphState):
        """
        Perform a DuckDuckGo search and retrieve documents.
        
        Args:
            state (GraphState): The current graph state
            
        Returns:
            state (dict): Updates state with retrieved documents
        """
        print("\n--- DDG SEARCH ---")
        state['execution_path'].extend(["ddg_search"])
        
        documents = self.search_ddg_search_results.invoke({"query": state["question"]})
        documents = convert_str_to_document(documents)
        print("Documents DDG: ", documents)
        
        state["documents"] = documents
        return state

    
    def _hallucination_check(self, state: GraphState):
        """
        Determines whether the generation is grounded in the document and answers question.

        Args:
            state (dict): The current graph state

        Returns:
            str: Decision for next node to call
        """
        print("\n--- HALLUCINATIONS CHECK ---")
        state['execution_path'].extend(["hallucination_check"])
        
        documents = state["documents"]
        generation = state["answer"]
        score = self.hallucination_grader_chain.invoke({"documents": documents, "generation": generation}) 
        state["hallucination"] = "no" if score["score"] == "yes" else "yes"
        print(f"\nDECISION: generation is {'grounded in documents' if score['score'] == 'yes' else 'not grounded in documents, re-try'}")
        if score == "yes":
            print("SCORE: ", score)
            state["answer"] = {'answer': "I don't know the answer to that question.", 'metadata': "No metadata"} 
            
        return state


    def _answer_check(self, state: GraphState):
        """
        Determines whether the generation answers the question.

        Args:
            state (dict): The current graph state

        Returns:
            dict: Updated state with answer check result
        """
        print("\n--- FINAL ANSWER CHECK ---")
        state['execution_path'].extend(["answer_check"])
        
        question = state["question"]
        generation = state["answer"]

        score = self.answer_grader_chain.invoke({"question": question, "generation": generation})
        grade = score["score"]
        
        print("Anser Check Score:      {}".format(grade))

        state["score"] = "useful" if grade == "yes" else "not useful"
        print(f"\nDECISION: generation {'addresses' if grade == 'yes' else 'does not address'} question")
        if grade == "no":
            state["answer"] = {'answer': "I don't know the answer to that question.", 'metadata': "No metadata"} 
        return state
    

    def invoke(self, inputs):
        print(f"inputs: {inputs}")
        print('\nCalling => knowledge_base_system.py - invoke()')
        inputs['execution_path'] = []
                
        try:
            answer = self.app.invoke(inputs)
            print("\n--- INOVKE ANSWER ---")
            print("\nAnswer:  {}".format(answer['answer']))
        except Exception as e:
            answer = {"answer": "I don't know the answer to that question", "metadata": "No metadata"}
        
        self.chat_history.extend([HumanMessage(content=inputs['question']), AIMessage(content=answer['answer']['answer'])])
        self.chat_rephrased_history.extend([HumanMessage(content=answer['question']), AIMessage(content=answer['answer']['answer'])])
        for message in self.chat_history:
            print("\nChat History:  {}".format(message))
        return answer
    
    def set_retriever(self, retriever):
        self.retriever = retriever
