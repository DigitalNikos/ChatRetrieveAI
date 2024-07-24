from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from langchain.chains import create_history_aware_retriever
from langchain_community.tools import DuckDuckGoSearchResults

from tqdm import tqdm
from typing import List
from typing_extensions import TypedDict

from qa_system.lang_graph import WorkflowInitializer
from text_doc_processing import convert_str_to_document
from qa_system.prompts import (generate_answer_propmpt,rephrase_prompt, 
                     grader_document_prompt, hallucination_grader_prompt, 
                     answers_grader_prompt, query_domain_check)


class KnowledgeBaseSystem:
    print('\n--- KNOWLEDGE BASE SYSTEM ---')
    class GraphState(TypedDict):
        """
        Represents the state of our graph.

        """
        print('\n--- GRAPH STATE ---')
        question: str
        answer: str
        documents: List[str]
        domain: str
        hallucination: str
        generation_score: str

    def __init__(self, llm_name: str, retirver = None):  
        print('\nknowledge_base_system.py - __init__()')    
        self.retriever = retirver
        self.chat_history = []
        
        # LLMs
        self.json_llm = ChatOllama(model=llm_name, format="json", temperature=0)  
        
        # CHAINS
        self.query_domain_check = query_domain_check | self.json_llm | JsonOutputParser()
        self.rephrase_query_chain = rephrase_prompt | self.json_llm | JsonOutputParser()
        self.retrieval_grader_document_chain = grader_document_prompt | self.json_llm | JsonOutputParser()
        self.generate_answe = generate_answer_propmpt | self.json_llm | JsonOutputParser()
        self.hallucination_grader_chain = hallucination_grader_prompt | self.json_llm | JsonOutputParser()
        self.answer_grader_chain = answers_grader_prompt | self.json_llm | JsonOutputParser()
        self.search_ddg_search_results = DuckDuckGoSearchResults(num_results = 2, verbose = True)
        
        # GRAPH APP
        self.app = None
        self.initialize_graph()
            
    
    def _check_query_domain(self, state: GraphState):
        """
        Check if the query belongs to the specified domain using the specific chain.
        
        Args:
            state (GraphState): Contains 'question' and 'domain' keys.

        Returns:
            str: 'yes' query within the domain, otherwise 'no'.
        """
        print("\n--- CHECK QUERY DOMAIN ---")
        
        question = state["question"]
        domain = state["domain"]
        print("\nQuestion:  {}".format(question))
        print("\nDomain:    {}".format(domain))

        answer = self.query_domain_check.invoke({"question": question, "domain": domain})
        print("\nAnswer:    {}".format(answer))
        
        state['generation_score'] = answer['score']  
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
        print("\nAnswer:           {}".format(state["answer"]))
        print("\nQuestion:         {}".format(state["question"]))
        
        rephrased_query = self.rephrase_query_chain.invoke({"input": state["question"], "chat_history": self.chat_history})
        
        print("\nRephrased query:  {}".format(rephrased_query))
        return {"question": rephrased_query['question']}
   
    
    def _retrieve(self, state: GraphState):
        """
        Retrieve documents from the retriever.
        
        Args:
            state (dict): Contains the current graph state, including the question.
            
        Returns:
             GraphState: Updated state with 'documents' containing retrieved documents or an empty list if no retriever is available.
        """
        print("\n--- RETRIEVE DOCUMENTS ---")
        
        if self.retriever is None: # or raise error
            print("\nNo files or URLs uploaded. Returning empty documents.")
            state["documents"] = []
            return state
        
        print("\nQuestion to retrive:    {}".format(state["question"]))

        chat_retriever_chain = create_history_aware_retriever(self.json_llm, self.retriever, rephrase_prompt)
        documents = chat_retriever_chain.invoke({"input": state["question"], "chat_history": self.chat_history})
        
        print("\nRetrieved Documents:    {}".format(documents))
        return {"documents": documents, "question": state["question"]}
    

    def _generate(self, state: GraphState):
        """
        Generate an answer using the provided context and question.
        
        Args:
            state (dict): The current graph state
            
        Returns:
            dict: Updated state with a new key 'generation' containing the LLM generation.
        """
        print("\n--- GENERATE ANSWER ---")
        
        question = state["question"]
        documents = state["documents"]
        
        generation = self.generate_answe.invoke({"context": documents, "question": question, "chat_history": self.chat_history})
        
        print("\nAnswer:                 {}".format(generation))

        return {"documents": documents, "question": question, "answer": generation}


    def _grade_documents(self, state: GraphState):
        """
        Determines whether the retrieved documents are relevant to the question.
        
        Args:
            state (dict): The current graph state
            
        Returns:
            dict: Updated state with 'documents' key containing only relevant documents.
        """
        print("\n--- GRADE RETRIEVED DOCUMENTS---")
        
        question = state["question"]
        documents = state["documents"]
        
        num_documents = len(documents)
        
        print("\nNumber of documents:  {}".format(num_documents))
        filtered_docs = []
        with tqdm(total=num_documents, desc="Grading Documents", ncols=100) as pbar:
            for d in documents:
                score = self.retrieval_grader_document_chain.invoke({"question": question, "document": d.page_content})
                grade = score["score"]
                if grade == "yes":
                    filtered_docs.append(d)
                else:
                    continue
                pbar.update(1)
        
        print("\nRelevant documents:   {}/{}".format(len(filtered_docs), num_documents))
        return {"documents": filtered_docs, "question": question}


    def _ddg_search(self, state: GraphState):
        """
        Perform a DuckDuckGo search and retrieve documents.
        
        Args:
            state (GraphState): The current graph state
            
        Returns:
            state (dict): Updates state with retrieved documents
        """
        print("\n--- DDG SEARCH ---")
        
        question = state["question"]
        
        documents = self.search_ddg_search_results.invoke({"query": state["question"]})
        documents = convert_str_to_document(documents)
        print("Documents DDG: ", documents)
        
        return {"documents": documents, "question": question}

    
    def _hallucination_check(self, state: GraphState):
        """
        Determines whether the generation is grounded in the document and answers question.

        Args:
            state (dict): The current graph state

        Returns:
            str: Decision for next node to call
        """

        print("\n--- HALLUCINATIONS CHECK ---")
        documents = state["documents"]
        generation = state["answer"]
        
        print("*"*40)
        print("generation: ", generation)
        print("*"*40)

        score = self.hallucination_grader_chain.invoke({"documents": documents, "generation": generation})
        grade = score["score"]
        
        if grade == "yes":
            print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
            state["hallucination"] = "no"
            print("State in Hallucination: ", state)
        else:
            print("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY---")
            state["hallucination"] = "yes"
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
        question = state["question"]
        generation = state["answer"]

        score = self.answer_grader_chain.invoke({"question": question, "generation": generation})
        grade = score["score"]
        
        print("Anser Check Score:      {}".format(grade))

        state["score"] = "useful" if grade == "yes" else "not useful"
        print(f"---DECISION: GENERATION {'ADDRESSES' if grade == 'yes' else 'DOES NOT ADDRESS'} QUESTION---")
        return state
    
    def _end_with_document_message(self, state: GraphState):
        """
        Ends the workflow with a message indicating no documents were found.

        Args:
            state (GraphState): The current graph state

        Returns:
            dict: Updated state with document message
        """
        print("\nCalling => knowledge_base_system.py - _end_with_document_message()")
        state["generation"] = "I don't have any documents to answer that question."
        return state
    
    
    def _end_with_hallucination_message(self, state: GraphState):
        """
        Ends the workflow with a hallucination message.

        Args:
            state (GraphState): The current graph state

        Returns:
            dict: Updated state with hallucination message
        """
        print("\nCalling => knowledge_base_system.py - _end_with_hallucination_message()")
        state["generation"] = "I don't know the answer to that question."
        return state
        
        
    def initialize_graph(self):    
        print('\n--- INITIALIZING GRAPH ---')
            
        workflow_initializer = WorkflowInitializer(self)
        self.app = workflow_initializer.initialize()
        return self.app
    

    def invoke(self, inputs):
        print('\nCalling => knowledge_base_system.py - invoke()')
        print("Inputs: ", inputs)
        print("Inputs type: ", type(inputs))
        
        try:
            answer = self.app.invoke(inputs)
            print("Answer in invoke: ", answer)
        except Exception as e:
            print("Error: ", e)
            answer = {"answer": "I don't know the answer to that question"}
            
        if 'answer' in answer and isinstance(answer['answer'], dict):
            response = answer['answer'].get('answer', 'No answer provided.')
            metadata = answer['answer'].get('metadata', 'No metadata available.')
            formatted_response = f"{response}\n\nMetadata: {metadata}"
        else:
            formatted_response = answer.get('answer', 'No answer provided.')

        
        # self.update_chat_history(inputs['question'], answer['generation']['answer'])
        return formatted_response
     
    def update_chat_history(self, question: str, answer: str):
        self.chat_history.extend([HumanMessage(content=question), AIMessage(content=answer)])
        return self.chat_history
    
    def set_retriever(self, retriever):
        self.retriever = retriever
