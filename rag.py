from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain.schema.output_parser import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.runnable import RunnablePassthrough
from langchain.prompts import PromptTemplate
from langchain.vectorstores.utils import filter_complex_metadata
from config import Config as cfg
from langchain.tools.retriever import create_retriever_tool

from langchain import hub
from langchain.agents import create_react_agent
from langchain.agents import AgentExecutor
from typing import List

from langchain_core.documents import Document

class ChatPDF:
    vector_store = None
    retriever = None
    chain = None

    def __init__(self):
        print("Constructor ChatPDF called...")
        self.model = ChatOllama(model=cfg.MODEL)
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=cfg.SPLITTER_CHUNK_SIZE, chunk_overlap=cfg.SPLITTER_CHUNK_OVERLAP)

        self.prompt =  hub.pull("hwchase17/react-chat")

        self.agent = None
        self.agent_executor = None
        self.tools = []
        self.chat_history = ""
        self.retriever = None
    
    def initialize(self, chunks: List[Document]):
        print("Initializing")
        vector_store = Chroma.from_documents(documents=chunks, embedding=FastEmbedEmbeddings())
        self.retriever = vector_store.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={
                "k": cfg.N_DOCUMENTS_TO_RETRIEVE,
                "score_threshold": cfg.RETRIEVER_SCORE_THRESHOLD,
            },
        )

        retriever_tool = create_retriever_tool(self.retriever, name="Private knowledge base", description="YOU MUST ALWAYS USE THIS AS YOUR FIRST TOOL. Lookup information in a private knowledge base.")
        self.tools = [retriever_tool]

        self.agent = create_react_agent(llm=self.model, tools=self.tools, prompt=self.prompt, )
        self.agent_executor = AgentExecutor(agent=self.agent, tools=self.tools, verbose=True, max_iterations=4, return_intermediate_steps=True, handle_parsing_errors=True)

    def ingest(self, pdf_file_path: str):
        print("Ingesting...")
        docs = PyPDFLoader(file_path=pdf_file_path).load()

        chunks = self.text_splitter.split_documents(docs)
        chunks = filter_complex_metadata(chunks)

        if not self.retriever:
            self.initialize(chunks)
        else:
            self.retriever.add_documents(chunks)

        


        # self.chain = ({"context": self.retriever, "question": RunnablePassthrough()}
        #               | self.prompt
        #               | self.model
        #               | StrOutputParser())

    def ask(self, query: str):
        if not self.agent_executor:
            return "Please, add a PDF document first."
        result = self.agent_executor.invoke({"input": query, "chat_history": self.chat_history})
        answer = result["output"]
        self.chat_history += "Human:" + query + "\n" + "Assistant (YOU)" + answer + "\n"
        print(answer)
        return answer

    def clear(self):
        self.vector_store = None
        self.retriever = None
        self.chain = None
        self.agent = None
        self.agent_executor = None
        self.tools = []