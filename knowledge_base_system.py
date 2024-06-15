from langchain import hub
from langchain_community.chat_models import ChatOllama
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader, WebBaseLoader
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain.agents import create_react_agent, AgentExecutor
from langgraph.graph import END, StateGraph
from langchain_core.documents import Document
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain.prompts import PromptTemplate
import os

from prompts import grader_prompt, rag_prompt, hallucination_grader_prompt, answers_grader_prompt, re_write_prompt, domain_check, query_domain_check, domain_detection
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

    def __init__(self, llm_name: str):  
        print('Calling => knowledge_base_system.py - __init__()')      
        self.retriever = None

        # Set to keep track of ingested documents and URLs
        self.ingested_documents = set()  
        self.ingested_urls = set()  

        # LLMs
        self.json_llm = ChatOllama(model=llm_name, format="json", temperature=0)
        self.llm = ChatOllama(model=llm_name, temperature=0)

        # Prompts
        self.grader_prompt = grader_prompt
        self.hallucination_grader_prompt = hallucination_grader_prompt
        self.rag_prompt = rag_prompt
        self.answers_grader_prompt = answers_grader_prompt
        self.re_write_prompt = re_write_prompt

        # Chains
        self.retrieval_grader_chain = self.grader_prompt | self.json_llm | JsonOutputParser()
        self.rag_chain = self.rag_prompt | self.llm | JsonOutputParser()
        self.hallucination_grader_chain = self.hallucination_grader_prompt | self.json_llm | JsonOutputParser()
        self.answer_grader_chain = self.answers_grader_prompt | self.json_llm | JsonOutputParser()
        self.question_rewriter_chain = self.re_write_prompt | self.llm | StrOutputParser()
        # ===============

        self.summary_domain_chain = domain_detection | self.llm | JsonOutputParser()
        self.domain_checking = domain_check | self.llm | JsonOutputParser()
        self.query_check = query_domain_check | self.llm | JsonOutputParser()
        
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

        result = self.summary_domain_chain.invoke({"documents": chunks})

        result = self.domain_checking.invoke({"domain": sources['domain'], "summary": result["summary"], "doc_domain": result["domain"]})  

        if result["score"] == "no":
            return result["score"]
        
        if not self.retriever:
            self.initialize(chunks)
        else:
            self.retriever.add_documents(chunks)
    

    def format_docs(docs):
        print('Calling => knowledge_base_system.py - format_docs()')
        return "\n\n".join(doc.page_content for doc in docs)


    def _retrieve(self, state: GraphState):
        """
        Retrieve documents

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): New key added to state, documents, that contains retrieved documents
        """
        if self.retriever is None: # or raise error
            return {"documents": [], "question": state["question"]}
        
        print("Calling => knowledge_base_system.py - _retrieve()")
        question = state["question"]

        # Retrieval
        documents = self.retriever.get_relevant_documents(question)
        return {"documents": documents, "question": question}


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

        return {"documents": documents, "question": question, "generation": generation}


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

        # Re-write question
        better_question = self.question_rewriter_chain.invoke({"question": question})
        return {"documents": documents, "question": better_question}

    def _wikipedia_search(self, state: GraphState):
        """
        Perform a search using Wikipedia to retrieve information.

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): Updates the state with the original question and the generated answer from Wikipedia
        """

        print("Calling => knowledge_base_system.py - _wikipedia_search()")

        prompt = hub.pull("hwchase17/react-chat")
        question = state["question"]

        tool = create_tool("wikipedia")
        tools = [tool]

        agent = create_react_agent(llm=self.llm, tools=tools, prompt=prompt, )
        agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, max_iterations=cfg.AGENT_MAX_ITERATIONS, return_intermediate_steps=True, handle_parsing_errors=True)

        input_data = {
            "input": question,
            "agent_scratchpad": "",
            "chat_history": []
        }

        result = agent_executor.invoke(input_data)
        answer = result["output"]
        print("ANSWER FROM ANGENT WIKI:", answer)

        return { "question": question, "generation": answer}
        
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
                return "useful"
            else:
                print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
                return "not useful"
        else:
            print("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY---")
            return "not supported"
        
    def initialize_graph(self):    
        print('Calling => knowledge_base_system.py - initialize_graph()')    
        workflow = StateGraph(self.GraphState)

        # Define the nodes
        workflow.add_node("retrieve", self._retrieve)  # retrieve
        workflow.add_node("grade_documents", self._grade_documents)  # grade documents
        workflow.add_node("generate", self._generate)  # generatae
        workflow.add_node("transform_query", self._transform_query)  # transform_query
        workflow.add_node("wikipedia_search", self._wikipedia_search)  # wikipedia_search

        # Build graph
        workflow.set_entry_point("retrieve")
        workflow.add_edge("retrieve", "grade_documents")
        workflow.add_conditional_edges(
            "grade_documents",
            self._decide_to_generate,
            {
                # "transform_query": "transform_query",
                "transform_query": "wikipedia_search",
                "generate": "generate",
            },
        )
        workflow.add_edge("wikipedia_search", END)
        workflow.add_edge("transform_query", "retrieve")
        workflow.add_conditional_edges(
            "generate",
            self._grade_generation_v_documents_and_question,
            {
                "not supported": "generate",
                "useful": END,
                "not useful": "transform_query",
            },
        )

        # Compile
        self.app = workflow.compile()
    
    def stream(self, inputs):
        print('Calling => knowledge_base_system.py - stream()')
        return self.app.stream(inputs)

    def invoke(self, inputs):
        print('Calling => knowledge_base_system.py - invoke()')
        answer = self.query_check.invoke(inputs)
        if answer["score"] == "no":
            print("Query does not fall within the specified domain")
            return "Query does not fall within the specified domain"
        answer = self.app.invoke(inputs)
        final = answer['generation']
        print("FINAL ANSWER:", final)
        return final