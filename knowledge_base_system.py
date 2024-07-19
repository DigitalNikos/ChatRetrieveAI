from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import JsonOutputParser
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_core.messages import HumanMessage, AIMessage
from langchain.chains import create_history_aware_retriever

from tqdm import tqdm
from typing import List
from typing_extensions import TypedDict

from lang_graph import WorkflowInitializer
from text_doc_processing import convert_str_to_document

from prompts import (generate_answer_propmpt,rephrase_prompt, 
                     grader_document_prompt, hallucination_grader_prompt, 
                     answers_grader_prompt, domain_check, query_domain_check, domain_detection)


class KnowledgeBaseSystem:
    print('\n--- KNOWLEDGE BASE SYSTEM ---')
    class GraphState(TypedDict):
        """
        Represents the state of our graph.

        """
        print('\n--- GRAPH STATE ---')
        question: str
        generation: str
        documents: List[str]
        domain: str

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
        self.summary_domain_chain = domain_detection | self.json_llm | JsonOutputParser()
        self.domain_checking = domain_check | self.json_llm | JsonOutputParser()
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
        
        state['generation'] = answer['score']  
        return state
    
    
    def _handle_domain_relevance_with_rephrase(self, state: GraphState):
        """
        Determine the next step based on domain relevance, with rephrasing option.
        
        Args:
            state (GraphState): Contains the 'generation' key.

        Returns:
            str: "retrieve" if relevant, otherwise "rephrase".
        """
        print("\n--- CHECKING DOMAIN RELEVANCE QUERY (WITH REPHRASE) ---")
        
        if state["generation"] == "yes":
            print("\nQuery is relevant to domain:       retrieve")
            return "retrieve"
        else:
            print("\nQuery is not relevant to domain:   rephrase")
            return "rephrase"
    
    
    def _handle_domain_relevance_with_end(self, state: GraphState):
        """
        Determine the next step based on domain relevance, with ending option.
        
        Args:
            state (GraphState): Contains the 'generation' key.

        Returns:
            str: "retrieve" if relevant, otherwise "end".
        """
        print("\n--- CHECKING DOMAIN RELEVANCE QUERY (WITH END) ---")
        
        if state["generation"] == "yes":
            print("\nQuery is relevant to domain:       retrieve")
            return "retrieve"
        else:
            print("\nQuery is not relevant to domain:   end")
            return "end"
   
        
    def _existing_docs_condition(self, state: GraphState):
        """
        Determine the next step based on the existence of documents, with "ddg_search" option.

        Args:
            state (GraphState): Contains the 'documents' key.
            
        Returns:
            str: "generate" if documents exist, otherwise "ddg_search".
        """
        print("\n--- CHECK EXISTING DOCUMENTS CONDITION ---")
        
        if state["documents"]:
            print("\nDECISION: documents exist, proceed to genarate.")
            return "grade_docs"
        else:
            print("\nDECISION: documents NOT exist, proceed to ddg search.")
            return "ddg_search"
    
    
    def _existing_ddg_docs_condition(self, state: GraphState):
        """
        Determine the next step based on the existence of ddg documents, with "end" option.
        
        Args:
            state (GraphState): Contains the 'documents' key.

        Returns:
            str: "grade_docs" if documents exist, otherwise "end".
        """
        print("\n--- CHECK EXISTING DOCUMENTS CONDITION ---")
        
        if state["documents"]:
            print("\nDECISION: documents exist, proceed to genarate.")
            return "grade_docs"
        else:
            print("\nDECISION: documents NOT exist, proceed to ddg search.")
            return "end"

    def hallucination_condition(self, state: GraphState):
        """
            Determines the next step based on the hallucination score of the generation.
            
            Args:
                state (GraphState): Contains the 'score' and 'generation' keys.
                
            Returns:
                str: The value of 'score' indicating the hallucination condition.
            """
        print("\n--- CHECK HALLUCINATION CONDITION ---")
        
        if state["score"] != "useful":
            print("\n---DECISION: GENERATION IS NOT USEFUL, CLEARING GENERATION---")
            state['generation'] = ""    
        else:
            print("\n---DECISION: GENERATION IS USEFUL---")        
        return state['score']
        
    
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
        
        question = state["question"]
        
        print("\nQuestion to retrive:    {}".format(question))

        chat_retriever_chain = create_history_aware_retriever(self.json_llm, self.retriever, rephrase_prompt)
        documents = chat_retriever_chain.invoke({"input": question, "chat_history": self.chat_history})
        print("\nRetrieved Documents:    {}".format(documents))
        
        return {"documents": documents, "question": question}
    

    def _generate(self, state: GraphState):
        """
        Generate an answer using the provided context and question.
        
        Args:
            state (dict): The current graph state
            
        Returns:
            dict: Updated state with a new key 'generation' containing the LLM generation.
        """
        print("\n---GENERATE ANSWER---")
        
        question = state["question"]
        documents = state["documents"]
        
        generation = self.generate_answe.invoke({"context": documents, "question": question, "chat_history": self.chat_history})
        print("\nAnswer from RAG:", generation)

        return {"documents": documents, "question": question, "generation": generation}


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


    def _rephrase_query(self, state: GraphState):
        """
        Transform the query to produce a better question.
        
        Args:
            state (dict): The current graph state
            
        Returns:
            state (dict): Updates 'question' key with a re-phrased question
        """
        print("\n--- REPHRASE QUERY ---")
        
        question = state["question"]
        documents = state["documents"]
        print("\nQuestion:         {}".format(question))
        
        rephrased_query = self.rephrase_query_chain.invoke({"input": question, "chat_history": self.chat_history})
        print("\nRephrased query:  {}".format(rephrased_query))
        
        return {"documents": documents, "question": rephrased_query['question']}


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
        

    def _decide_to_generate(self, state: GraphState):
        """
        Determines whether to generate an answer, or re-generate a question.
        
        Args:
            state (dict): The current graph state
            
        Returns:
            str: Binary decision for next node to call
        """
        print("\n--- DECIDE TO GENERATE ---")
        
        question = state["question"]
        filtered_documents = state["documents"]

        if not filtered_documents:
            print("DESICION:  No relevant documents found.")
            return "end"
        else:
            print("DECISION: generate.")
            return "generate"

    
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
        generation = state["generation"]

        score = self.hallucination_grader_chain.invoke({"documents": documents, "generation": generation})
        grade = score["score"]
        
        if grade == "yes":
            print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
            state["hallucination"] = "no"
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
        generation = state["generation"]

        score = self.answer_grader_chain.invoke({"question": question, "generation": generation})
        grade = score["score"]
        
        print("Score: ", score)

        if grade == "yes":
            print("---DECISION: GENERATION ADDRESSES QUESTION---")
            state["score"] = "useful"
        else:
            print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
            state["score"] = "not useful"
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
            answer = {"generation": "I don't know the answer to that question"}
            
        # self.update_chat_history(inputs['question'], answer['generation']['answer'])
        return answer['generation']
     
    def update_chat_history(self, question: str, answer: str):
        self.chat_history.extend([HumanMessage(content=question), AIMessage(content=answer)])
        return self.chat_history
    
    def set_retriever(self, retriever):
        self.retriever = retriever
