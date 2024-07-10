from langchain import hub
from langchain_community.chat_models import ChatOllama
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader, WebBaseLoader
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain.agents import create_react_agent, AgentExecutor
from langgraph.graph import END, StateGraph, START
from langchain_core.documents import Document
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain.tools import DuckDuckGoSearchRun
from langchain_community.tools import DuckDuckGoSearchResults

from langchain_core.messages import HumanMessage, AIMessage

from langchain.chains import create_history_aware_retriever

import re
import os

from prompts import rephrase_prompt, grader_prompt, rag_prompt, hallucination_grader_prompt, answers_grader_prompt, re_write_prompt, domain_check, query_domain_check, domain_detection
from clean_text import clean_text
from config import Config as cfg
from get_tools import create_tool

from typing import List
from typing_extensions import TypedDict


class KnowledgeBaseSystem:
    print('Calling => knowledge_base_system.py - KnowledgeBaseSystem()')
    class GraphState(TypedDict):
        """
        Represents the state of our graph.

        Attributes:
            question: question
            generation: LLM generation
            documents: list of documents
        """
        print('Calling => knowledge_base_system.py - GraphState()')
        question: str
        generation: str
        documents: List[str]
        domain: str

    def __init__(self, llm_name: str):  
        print('Calling => knowledge_base_system.py - __init__()')      
        self.retriever = None
        
        # LLMs
        self.json_llm = ChatOllama(model=llm_name, format="json", temperature=0)
        self.llm = ChatOllama(model=llm_name, temperature=0)

        # Prompts
        self.grader_prompt = grader_prompt
        self.hallucination_grader_prompt = hallucination_grader_prompt
        self.rag_prompt = rag_prompt
        self.answers_grader_prompt = answers_grader_prompt
        self.re_write_prompt = re_write_prompt
        self.rephrase_prompt = rephrase_prompt

        # Chains
        self.retrieval_grader_chain = self.grader_prompt | self.json_llm | JsonOutputParser()
        self.rag_chain = self.rag_prompt | self.llm | JsonOutputParser()
        self.hallucination_grader_chain = self.hallucination_grader_prompt | self.json_llm | JsonOutputParser()
        self.answer_grader_chain = self.answers_grader_prompt | self.json_llm | JsonOutputParser()
        # self.question_rewriter_chain = self.re_write_prompt | self.json_llm | JsonOutputParser()
        self.question_rewriter_chain = rephrase_prompt | self.json_llm | JsonOutputParser()
        # ===============

        self.summary_domain_chain = domain_detection | self.llm | JsonOutputParser()
        self.domain_checking = domain_check | self.llm | JsonOutputParser()
        self.query_check = query_domain_check | self.json_llm | JsonOutputParser()
        
        # self.chat_retriever_chain = create_history_aware_retriever(self.llm, self.retriever, rephrase_prompt)
        
        self.chat_history = []
        
        self.search_duckduckGo_search_results = DuckDuckGoSearchResults(num_results = 2, verbose = True)
        

        self.app = None
        self.initialize_graph()

    def initialize(self, chunks: List[Document]):
        print('Calling => knowledge_base_system.py - initialize()')
        vector_store = Chroma.from_documents(documents=chunks, collection_name="rag-chroma", embedding=FastEmbedEmbeddings())
        self.retriever = vector_store.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={
                "k": cfg.N_DOCUMENTS_TO_RETRIEVE,
                "score_threshold": cfg.RETRIEVER_SCORE_THRESHOLD,
            },
        )

    def ingest(self,sources: dict):
        print('Calling => knowledge_base_system.py - ingest()')

        source_extension = sources['source_extension']

        if source_extension not in cfg.LOADERS_TYPES:
            raise Exception("Not valid upload source!!")

        # Load document
        if source_extension == "url":
            docs = cfg.LOADERS_TYPES[source_extension](sources["url"]).load()
        else:
            print("Upload file again!")
            docs = cfg.LOADERS_TYPES[source_extension](sources["file_path"]).load()

        self.text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=cfg.SPLITTER_CHUNK_SIZE, chunk_overlap=cfg.SPLITTER_CHUNK_OVERLAP
        )

        chunks = self.text_splitter.split_documents(docs)
        chunks = filter_complex_metadata(chunks)
        chunks = clean_text(chunks, sources['file_name'])

        print("Chunks URL: ", chunks)

        result = self.summary_domain_chain.invoke({"documents": chunks})

        result = self.domain_checking.invoke({"domain": sources['domain'], "summary": result["summary"], "doc_domain": result["domain"]})  

        # if result["score"] == "no":
        #     return result["score"]
        
        if not self.retriever:
            self.initialize(chunks)
        else:
            self.retriever.add_documents(chunks)
    
        # print("DB DATA: ", self.vector_store.get())

    def format_docs(docs):
        print('Calling => knowledge_base_system.py - format_docs()')
        return "\n\n".join(doc.page_content for doc in docs)
    
    
    
    
    
    
    # ===============START NEW GRAPH================
    
    def _check_query_domain(self, state: GraphState):
        """
        Determine if the provided query falls within the specified domain.

        This method evaluates whether a given question belongs to a particular
        domain by invoking an external query check method.

        Args:
            state (GraphState): The current state of the graph containing the
                                'question' and 'domain' keys.

        Returns:
            str: "yes" if the query is within the specified domain, otherwise "no".
        """
        print('Calling => knowledge_base_system.py - _check_query_domain()')
        
        question = state["question"]
        domain = state["domain"]
        print("\nQuestion: ", question)
        print("\nDomain: ", state["domain"])
        print("\nState: ", state)
        answer = self.query_check.invoke({"question": question, "domain": domain})
        print("\nAnswer: ", answer)
        state['generation'] = answer['score']
        return state
    
    def domain_relevance_condition(self, state: GraphState):
        print('Calling => knowledge_base_system.py - domain_relevance_condition()')
        if state["generation"] == "yes":
            return "retrieve"
        else:
            return "rephrase"
        
    def existing_docs_condition(self, state: GraphState):
        print('Calling => knowledge_base_system.py - existing_docs_condition()')
        print("State: ", state)
        if state["documents"]:
            print("Documents exist")
            return "generate"
        else:
            print("Documents do not exist")
            return "duckDuckGo_search"

    def hallucination_condition(self, state: GraphState):
        print('Calling => knowledge_base_system.py - hallucination_condition()')
        if state["score"] != "useful":
            state['generation'] = ""            
        return state['score']
        
    
    def _retrieve(self, state: GraphState):
        """
        Retrieve documents

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): New key added to state, documents, that contains retrieved documents
        """
        print("Calling => knowledge_base_system.py - _retrieve()")
        if self.retriever is None: # or raise error
            state["documents"] = []
            return state
        
        question = state["question"]

        chat_retriever_chain = create_history_aware_retriever(self.llm, self.retriever, rephrase_prompt)
        documents = chat_retriever_chain.invoke({"input": question, "chat_history": self.chat_history})
        return {"documents": documents, "question": question}
    
    
    
    
    
    # ===============END NEW GRAPH================





    def _generate(self, state: GraphState):
        """
        Generate answer

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): New key added to state, generation, that contains LLM generation
        """
        print("Calling => knowledge_base_system.py - _generate()")
        question = state["question"]
        documents = state["documents"]

        # RAG generation
        print(" genarate documents:", documents)
        generation = self.rag_chain.invoke({"context": documents, "question": question})
        print("answer from RAG:", generation['answer'])
        print("answer from RAG:", generation['metadata'])

        return {"documents": documents, "question": question, "generation": generation['answer']}


    def _grade_documents(self, state: GraphState):
        """
        Determines whether the retrieved documents are relevant to the question.

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): Updates documents key with only filtered relevant documents
        """

        print("Calling => knowledge_base_system.py - _grade_documents()")
        question = state["question"]
        documents = state["documents"]

        # Score each doc
        filtered_docs = []
        for d in documents:
            score = self.retrieval_grader_chain.invoke(
                {"question": question, "document": d.page_content}
            )
            grade = score["score"]
            if grade == "yes":
                print("---GRADE: DOCUMENT RELEVANT---")
                filtered_docs.append(d)
            else:
                print("---GRADE: DOCUMENT NOT RELEVANT---")
                continue
        return {"documents": filtered_docs, "question": question}


    def _transform_query(self, state: GraphState):
        """
        Transform the query to produce a better question.

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): Updates question key with a re-phrased question
        """

        print("Calling => knowledge_base_system.py - _transform_query()")
        question = state["question"]
        documents = state["documents"]
        print(f"Chat history: {self.chat_history}")
        # Re-write question
        better_question = self.question_rewriter_chain.invoke({"input": question, "chat_history": self.chat_history})
        print("Better question: ", better_question)
        return {"documents": documents, "question": better_question['question']}

    def _duckDuckGo_search(self, state: GraphState):
        print("Calling => knowledge_base_system.py -duckGO()")
        
        question = state["question"]
        print("\nQuestion: ", question)
        
        documents = self.search_duckduckGo_search_results.invoke({"query": state["question"]})
        print("\nDocuments as string: ", documents)
        documents = self.convert_str_to_document(documents)
        print("\nDocuments: ", documents)
        # answer = self.rag_chain.invoke({"context": documents, "question": question})
        # print("Answer: ", answer)

        # return { "question": question, "generation": answer}
        return {"documents": documents, "question": question}
        
    ### Edges

    def _decide_to_generate(self, state: GraphState):
        """
        Determines whether to generate an answer, or re-generate a question.

        Args:
            state (dict): The current graph state

        Returns:
            str: Binary decision for next node to call
        """

        print("Calling => knowledge_base_system.py - _decide_to_generate()")
        question = state["question"]
        filtered_documents = state["documents"]

        if not filtered_documents:
            # All documents have been filtered check_relevance
            # We will re-generate a new query
            print(
                "---DECISION: ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, TRANSFORM QUERY---"
            )
            return "transform_query"
        else:
            # We have relevant documents, so generate answer
            print("---DECISION: GENERATE---")
            return "generate"


    def _grade_generation_v_documents_and_question(self, state: GraphState):
        """
        Determines whether the generation is grounded in the document and answers question.

        Args:
            state (dict): The current graph state

        Returns:
            str: Decision for next node to call
        """

        print("Calling => knowledge_base_system.py - _grade_generation_v_documents_and_question()")
        question = state["question"]
        documents = state["documents"]
        generation = state["generation"]

        score = self.hallucination_grader_chain.invoke(
            {"documents": documents, "generation": generation}
        )
        grade = score["score"]

        # Check hallucination
        if grade == "yes":
            print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
            # Check question-answering
            print("---GRADE GENERATION vs QUESTION---")
            score = self.answer_grader_chain.invoke({"question": question, "generation": generation})
            grade = score["score"]
            if grade == "yes":
                print("---DECISION: GENERATION ADDRESSES QUESTION---")
                state["score"] = "useful"
                return state
            else:
                print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
                state["score"] = "not useful"
                return state
        else:
            print("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY---")
            state["score"] = "not supported"
            return state
        
    def initialize_graph(self):    
        print('Calling => knowledge_base_system.py - initialize_graph()')    
        workflow = StateGraph(self.GraphState)

        workflow.add_edge(START, "check_query_domain")
        workflow.add_node("check_query_domain", self._check_query_domain)
        
        workflow.add_node("retrieve", self._retrieve)
        workflow.add_node("rephrase", self._transform_query)
        
        workflow.add_node("generate", self._generate)
        workflow.add_node("duckDuckGo_search", self._duckDuckGo_search)
        workflow.add_node("hallucination_grader", self._grade_generation_v_documents_and_question)
        
        
        workflow.add_conditional_edges(
            "check_query_domain",
            self.domain_relevance_condition,
            {
                "retrieve": "retrieve",
                "rephrase": "rephrase",
            },
        )
        
        workflow.add_edge("rephrase", "check_query_domain")
        
        workflow.add_conditional_edges(
            "retrieve",
            self.existing_docs_condition,
            {
                "generate": "generate",
                "duckDuckGo_search": "duckDuckGo_search",
            },
        )
        
        workflow.add_edge("generate", "hallucination_grader")
        workflow.add_edge("duckDuckGo_search", "generate")
        
        workflow.add_conditional_edges(
            "hallucination_grader",
            self.hallucination_condition,
            {
                "not supported": END,
                "useful": END,
                "not useful": END,
            },
        )
        

        

        # # Define the nodes
        # workflow.add_node("check_query_domain", self._check_query_domain)
        # workflow.add_node("retrieve", self._retrieve)
        # # workflow.add_node("retrieve", self._retrieve)  # retrieve
        # # workflow.add_node("grade_documents", self._grade_documents)  # grade documents
        # # workflow.add_node("generate", self._generate)  # generatae
        # # workflow.add_node("transform_query", self._transform_query)  # transform_query
        # # workflow.add_node("duckDuckGo_search", self._duckDuckGo_search)  # wikipedia_search

        # # # Build graph
        # workflow.set_entry_point("check_query_domain")
        # workflow.add_conditional_edges(
        #     ""
        #     self._check_query_domain,
        #     {
        #         "yes": "retrieve",
        #         "no": END,
        #     },
        # )
        # workflow.set_entry_point("retrieve")
        # workflow.add_edge("retrieve", "grade_documents")
        # workflow.add_conditional_edges(
        #     "grade_documents",
        #     self._decide_to_generate,
        #     {
        #         # "transform_query": "transform_query",
        #         "transform_query": "duckDuckGo_search",
        #         "generate": "generate",
        #     },
        # )
        # workflow.add_edge("duckDuckGo_search", END)
        # workflow.add_edge("transform_query", "retrieve")
        # workflow.add_conditional_edges(
        #     "generate",
        #     self._grade_generation_v_documents_and_question,
        #     {
        #         "not supported": "generate",
        #         "useful": END,
        #         "not useful": "transform_query",
        #     },
        # )

        # Compile
        self.app = workflow.compile()
    
    def stream(self, inputs):
        print('Calling => knowledge_base_system.py - stream()')
        return self.app.stream(inputs, {"recursion_limit": 10})

    def invoke(self, inputs):
        print('Calling => knowledge_base_system.py - invoke()')
        print("Inputs: ", inputs)
        print("Inputs type: ", type(inputs))
        # answer = self.query_check.invoke(inputs)
        # if answer["score"] == "no":
        #     print("Query does not fall within the specified domain")
        #     return "Query does not fall within the specified domain"
        # else:
        #     return "MALAKIA"
        # inputs["chat_history"] = self.chat_history
        try:
            answer = self.app.invoke(inputs, {"recursion_limit": 10})
        except Exception as e:
            print("Error: ", e)
            answer = {"generation": "I don't know the answer to that question"}
        self.update_chat_history(inputs['question'], answer['generation'])
        return answer['generation']
    
    def convert_str_to_document(self, input:str):
        print('Calling => knowledge_base_system.py - convert_str_to_document()')
        cleaned_string = input.strip("[]")

        # Split into individual items
        items = re.split(r'\], \[', cleaned_string)

        # Initialize list to hold the parsed dictionaries
        documents = []

        # Parse each item
        for item in items:
            # Initialize dictionary
            parsed_dict = {}
            
            # Find all key-value pairs
            key_value_pairs = re.findall(r'(\w+):\s([^,]+)', item)
            
            # Fill the dictionary
            for key, value in key_value_pairs:
                parsed_dict[key] = value.strip()
            
            # Create Document instance
            doc = Document(
                page_content=parsed_dict['snippet'],
                metadata={
                    'title': parsed_dict['title'],
                    'link': parsed_dict['link']
                }
            )
            
            # Append to the list
            documents.append(doc)
        return documents
    
    def update_chat_history(self, question: str, answer: str):
        self.chat_history.extend([HumanMessage(content=question), AIMessage(content=answer)])
        return self.chat_history