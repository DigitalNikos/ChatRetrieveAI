from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import JsonOutputParser
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from prompts import grader_prompt, rag_prompt, hallucination_grader_prompt, answers_grader_prompt, re_write_prompt
from typing_extensions import TypedDict
from typing import List
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.agents import create_react_agent
from langchain.agents import AgentExecutor
from langchain import hub
from langgraph.graph import END, StateGraph

class KnowledgeBaseSystem:

    class GraphState(TypedDict):
        """
        Represents the state of our graph.

        Attributes:
            question: question
            generation: LLM generation
            documents: list of documents
        """

        question: str
        generation: str
        documents: List[str]

    def __init__(self, retriever, llm_name: str):
        self.retriever = retriever
        
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
        self.rag_chain = self.rag_prompt | self.llm | StrOutputParser()
        self.hallucination_grader_chain = self.hallucination_grader_prompt | self.json_llm | JsonOutputParser()
        self.answer_grader_chain = self.answers_grader_prompt | self.json_llm | JsonOutputParser()
        self.question_rewriter_chain = self.re_write_prompt | self.llm | StrOutputParser()
        
        self.app = None
        self.initialize_graph()

    # Post-processing
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)


    def retrieve(self, state: GraphState):
        """
        Retrieve documents

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): New key added to state, documents, that contains retrieved documents
        """
        print("---RETRIEVE---")
        question = state["question"]

        # Retrieval
        documents = self.retriever.get_relevant_documents(question)
        return {"documents": documents, "question": question}


    def generate(self, state: GraphState):
        """
        Generate answer

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): New key added to state, generation, that contains LLM generation
        """
        print("---GENERATE---")
        question = state["question"]
        documents = state["documents"]

        # RAG generation
        generation = self.rag_chain.invoke({"context": documents, "question": question})
        return {"documents": documents, "question": question, "generation": generation}


    def grade_documents(self, state: GraphState):
        """
        Determines whether the retrieved documents are relevant to the question.

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): Updates documents key with only filtered relevant documents
        """

        print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
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


    def transform_query(self, state: GraphState):
        """
        Transform the query to produce a better question.

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): Updates question key with a re-phrased question
        """

        print("---TRANSFORM QUERY---")
        question = state["question"]
        documents = state["documents"]

        # Re-write question
        better_question = self.question_rewriter_chain.invoke({"question": question})
        return {"documents": documents, "question": better_question}

    def wikipedia_search(self, state: GraphState):

        print("---WIKIPEDIA SEARCH---")
        prompt = hub.pull("hwchase17/react-chat")
        print("---PULL the prompt---")
        question = state["question"]
        print("---Retrive question from state---")
        print("QUESTION:", question)
        documents = state["documents"]
        print("---Retrive documents from state---")
        print("DOCUMENTS:", documents)
        description = "Search for information in Wikipedia. Whenever you cannot answer the question based on the private knowledge base use wikipedia tool instead."
        api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=1000)
        print("---Create WikipediaAPIWrapper---")
        wiki_tool = WikipediaQueryRun(api_wrapper=api_wrapper, name="Wikipedia search", description=description)
        print("---Create WikipediaQueryRun---")

        tools = [wiki_tool]
        print("---Create tools---")
        agent = create_react_agent(llm=self.llm, tools=tools, prompt=prompt, )
        print("---Create agent---")
        agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, max_iterations=10, return_intermediate_steps=True, handle_parsing_errors=True)
        print("---Create agent_executor---")
        
        input_data = {
            "input": question,
            "agent_scratchpad": "",
            "chat_history": []
        }

        # result = agent_executor.invoke({"input": question})
        result = agent_executor.invoke(input_data)
        print("---Invoke agent_executor---")
        print("RESULT:", result)
        answer = result["output"]
        print("---Get answer from result---")
        print("ANSWER FROM ANGENT WIKI:", answer)

        return { "question": question, "generation": answer}
        

    ### Edges


    def decide_to_generate(self, state: GraphState):
        """
        Determines whether to generate an answer, or re-generate a question.

        Args:
            state (dict): The current graph state

        Returns:
            str: Binary decision for next node to call
        """

        print("---ASSESS GRADED DOCUMENTS---")
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


    def grade_generation_v_documents_and_question(self, state: GraphState):
        """
        Determines whether the generation is grounded in the document and answers question.

        Args:
            state (dict): The current graph state

        Returns:
            str: Decision for next node to call
        """

        print("---CHECK HALLUCINATIONS---")
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
        workflow = StateGraph(self.GraphState)

        # Define the nodes
        workflow.add_node("retrieve", self.retrieve)  # retrieve
        workflow.add_node("grade_documents", self.grade_documents)  # grade documents
        workflow.add_node("generate", self.generate)  # generatae
        workflow.add_node("transform_query", self.transform_query)  # transform_query
        workflow.add_node("wikipedia_search", self.wikipedia_search)  # wikipedia_search

        # Build graph
        workflow.set_entry_point("retrieve")
        workflow.add_edge("retrieve", "grade_documents")
        workflow.add_conditional_edges(
            "grade_documents",
            self.decide_to_generate,
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
            self.grade_generation_v_documents_and_question,
            {
                "not supported": "generate",
                "useful": END,
                "not useful": "transform_query",
            },
        )

        # Compile
        self.app = workflow.compile()
    
    def stream(self, inputs):
        return self.app.stream(inputs)

    def invoke(self, inputs):
        answer = self.app.invoke(inputs)
        final = answer['generation']
        print("FINAL ANSWER:", final)
        return final