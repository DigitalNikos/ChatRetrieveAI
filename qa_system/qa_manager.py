import numexpr as ne
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




from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import FlashrankRerank

class KnowledgeBaseSystem:
    print('\n--- KNOWLEDGE BASE SYSTEM ---')
    class GraphState(TypedDict):
        """
        Represents the state of our graph.

        """
        print('\n--- GRAPH STATE ---')
        question: str
        rephrase_question: str
        q_domain_relevance: str
        documents: List[str]
        grade_documents: List[str]
        domain : str
        hallucination: str
        generation_score: str
        question_type: str
        execution_path: List[str] = []
        answer_useful: str
        answer: str

    def __init__(self, general_llm_model_name: str, math_llm_model_name: str, retriever = None, temperature=0):  
        print('\nknowledge_base_system.py - __init__()')    
        self.retriever = retriever
        self.chat_history = []
        self.chat_rephrased_history = []
        
        # LLMs
        self.json_llm = ChatOllama(model=general_llm_model_name, format="json", temperature=temperature) 
        self.math_llm = ChatOllama(model=math_llm_model_name, format="json", temperature=temperature)  
        self.llm = ChatOllama(model=general_llm_model_name, temperature=temperature) 
        
        # CHAINS
        self.query_domain_check = query_domain_check | self.json_llm | JsonOutputParser()
        self.rephrase_query_chain = rephrase_prompt | self.json_llm | JsonOutputParser()
        self.retrieval_grader_document_chain = grader_document_prompt | self.json_llm | JsonOutputParser()
        self.generate_answer = generate_answer_propmpt | self.llm | JsonOutputParser()
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
        
        if "execution_path" in state:
            state['execution_path'].extend(["check_query_domain"])
        
        print("\nQuestion:  {}".format(state["question"]))
        print("\nDomain:    {}".format(state["domain"]))

        answer = self.query_domain_check.invoke({"question": state["question"], "domain": state["domain"]})
        print("\nAnswer:    {}".format(answer))
        
        state['q_domain_relevance'] = answer['score']
        state['answer'] = {'answer': "I don't know the answer to that question.", 'metadata': "No metadata"}
        return state
    
    
    def _rephrase_query(self, state: GraphState):
        """
        Transform the query based on the existing "chat_rephrased_history".
        
        Args:
            state (dict): The current graph state
            
        Returns:
            state (dict): Updates 'question' key with a re-phrased question
        """
        print("\n--- REPHRASE QUERY ---")
        
        if "execution_path" in state:
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
        
        if "execution_path" in state:
            state['execution_path'].extend(["retrieve"])
        
        print("\nQuestion to retrive:    {}".format(state["question"]))
        
        if self.retriever is None: 
            print("\nNo files or URLs uploaded. Returning empty documents.")
            state["documents"] = []
            return state

        compressor = FlashrankRerank(model="ms-marco-MiniLM-L-12-v2")         
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor, base_retriever=self.retriever
        )
        
        compressed_docs = compression_retriever.invoke(state["question"] )
        print("\nRetrieved Documents:    {}".format(compressed_docs))
        
        state['documents'] = compressed_docs
        return state
    
    
    def _grade_documents(self, state: GraphState):
        """
        Determines whether the retrieved documents are relevant to the question.
        
        Args:
            state (dict): The current graph state
            
        Returns:
            dict: Updated state with 'documents' key containing only relevant documents.
        """
        print("\n--- GRADE RETRIEVED DOCUMENTS---")
        if "execution_path" in state:
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
        
        state["grade_documents"] = filtered_docs
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
        
        if "execution_path" in state:
            state['execution_path'].extend(["generate"])
        
        print("\nQuestion:                {}".format(state["question"]))
        print("\nContext:                 {}".format(state["grade_documents"]))
         
        generation = self.generate_answer.invoke({"context": state['grade_documents'], "question": state["question"], "chat_history": self.chat_history})
        print("\nAnswer:                 {}".format(generation))
        
        state ["answer"] = generation
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
        
        if "execution_path" in state:
            state['execution_path'].extend(["question_classification"])
        
        question_type = self.question_classifier.invoke({"question": state["question"]})
        print("\nQuestion Type:  {}".format(question_type))
        
        state["question_type"] = question_type['score']
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
        
        if "execution_path" in state:
            state['execution_path'].extend(["math_generate"])
            
        print("\nQuestion:                {}".format(state["question"]))
        
        generation = self.chain_math.invoke({"question": state["question"], "documents": state["grade_documents"]})
        
        stepwise_str = generation['step-wise reasoning'] if 'step-wise reasoning' in generation else "No step-wise reasoning available."
        expr_str = generation['expr'] if 'expr' in generation else "No expression available."
        
        print("\nGeneration:             {}".format(generation))
        print("\nGeneration:             {}".format(generation['expr']))
            
        if 'step-wise reasoning' in generation:
            steps_str = [f"Step {i+1}: {step}\n" for i, step in enumerate(generation['step-wise reasoning'])]
            stepwise_str = "Solution:\n\n" + "\n".join(steps_str).replace('*', '\\*')
        
        if 'expr' in generation:
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
        print("\nDocuments DDG:         {}".format(documents))
        
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
        
        if "execution_path" in state:
            state['execution_path'].extend(["hallucination_check"])
        
        score = self.hallucination_grader_chain.invoke({"documents":  state["grade_documents"], "generation": state["answer"]}) 
        state["hallucination"] = "no" if score["score"] == "yes" else "yes"
        print(f"\nDECISION: generation is {'grounded in documents' if score['score'] == 'yes' else 'not grounded in documents, re-try'}")
        
        if score == "yes":
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
        
        if "execution_path" in state:
            state['execution_path'].extend(["answer_check"])

        score = self.answer_grader_chain.invoke({"question": state["question"], "generation": state["answer"]})
        print("Anser Check Score:      {}".format(score["score"]))

        state["answer_useful"] = "useful" if score["score"] == "yes" else "not useful"
        print(f"\nDECISION: generation {'addresses' if score['score'] == 'yes' else 'does not address'} question")
        if score["score"] == "no":
            state["answer"] = {'answer': "I don't know the answer to that question.", 'metadata': "No metadata"} 
            
        return state
    

    def invoke(self, inputs):
        print("\n--- INOVKE START ---")
        print("\nInputs:      {}".format(inputs))
        
        inputs['execution_path'] = []
                
        try:
            answer = self.app.invoke(inputs)
            print("\n--- INOVKE ANSWER ---")
            print("\nAnswer:      {}".format(answer))
        except Exception as e:
            print("\nException:   {}".format(e))            
            answer = {"answer": "I don't know the answer to that question", "metadata": "No metadata"}
        
        self.chat_history.extend([HumanMessage(content=inputs['question']), AIMessage(content=answer['answer']['answer'])])
        self.chat_rephrased_history.extend([HumanMessage(content=answer['question']), AIMessage(content=answer['answer']['answer'])])   
        return answer
    
    
    def set_retriever(self, retriever):
        self.retriever = retriever
