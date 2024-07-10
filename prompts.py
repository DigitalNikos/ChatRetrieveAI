from langchain import hub
from langchain.prompts import PromptTemplate

# rephrase_prompt = hub.pull("langchain-ai/chat-langchain-rephrase")
rephrase_prompt = PromptTemplate(
    template=""""
    Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question.

    Chat History:

    {chat_history}

    Follow Up Input: {input}

    Standalone Question:
    answer Template:
    {{
        "question": "Standalone Question",
    }}
    """,
    input_variables=["chat_history", "input"],
)


grader_prompt = PromptTemplate(
    template="""You are a grader assessing relevance of a retrieved document to a user question. \n 
    Here is the retrieved document: \n\n {document} \n\n
    Here is the user question: {question} \n
    If the document contains keywords related to the user question, grade it as relevant. \n
    It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
    Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question. \n
    Provide the binary score as a JSON with a single key 'score' and no premable or explanation.""",
    input_variables=["question", "document"],
)

# rag_prompt = hub.pull("rlm/rag-prompt")
rag_prompt =PromptTemplate(
    template="""You are an assistant for question-answering tasks. 
    Use the following pieces of retrieved context to answer the question. 
    If you don't know the answer, just say that you don't know. 
    Use three sentences maximum and keep the answer concise.
    return a JSON with the key 'answer' and 'metadata', metadata should reference the metadata from used Document objects. 

    Question: {question} 

    Context: {context} 
    """,
    input_variables=["contexy", "question"],
)

hallucination_grader_prompt = PromptTemplate(
    template="""You are a grader assessing whether an answer is grounded in / supported by a set of facts. \n 
    Here are the facts:
    \n ------- \n
    {documents} 
    \n ------- \n
    Here is the answer: {generation}
    Give a binary score 'yes' or 'no' score to indicate whether the answer is grounded in / supported by a set of facts. \n
    Provide the binary score as a JSON with a single key 'score' and no preamble or explanation.""",
    input_variables=["generation", "documents"],
)

answers_grader_prompt = PromptTemplate(
    template="""You are a grader assessing whether an answer is useful to resolve a question. \n 
    Here is the answer:
    \n ------- \n
    {generation} 
    \n ------- \n
    Here is the question: {question}
    Give a binary score 'yes' or 'no' to indicate whether the answer is useful to resolve a question. \n
    Provide the binary score as a JSON with a single key 'score' and no preamble or explanation.""",
    input_variables=["generation", "question"],
)

re_write_prompt =PromptTemplate(
    template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question.
    if the chat history is empty, the follow up question should be returned exactly as it is.

    Chat History:

    {chat_history}

    Follow Up Input: {question}

    Answer as a JSON object with a single key 'standalone_question' and no preamble or explanation.
    """,
    input_variables=["chat_history", "question"],
)

domain_detection = PromptTemplate(
    template="""You are a grader assessing the summarization and domain of a set of documents.
    Here are the documents:
    \n ------- \n
    {documents}
    \n ------- \n
    First, provide a summary of the documents. Then, indicate the domain the documents belong to (e.g., sports, movies, tech).
    Provide the summary and domain as a JSON object with keys 'summary' and 'domain' respectively, and no preamble or explanation. Not a List but JSON object always""",
    input_variables=["documents"],
)

domain_check = PromptTemplate(
    template="""You are a grader assessing whether a set of documents falls within a specified domain.
    Here is the specified domain: {domain}
    Here is the summary of the document:
    \n ------- \n
    {summary}
    \n ------- \n
    Here is the domain you have identified from the document: {doc_domain}
    Give a binary 'yes' or 'no' score to indicate whether the documents fall within the specified domain. Not 0 or 1 only yes or no.
    Provide the binary score as a JSON with a single key 'score' and no preamble or explanation.""",
    input_variables=["domain", "summary", "doc_domain"],
)

query_domain_check = PromptTemplate(
    template="""You are a grader assessing whether a query falls within a specified domain.
    Here is the specified domain: {domain}
    Here is the query:
    \n ------- \n
    {question}
    \n ------- \n
    Give a binary 'yes' or 'no' score to indicate whether the query fall within the specified domain. Not 0 or 1 only yes or no.
    Provide the binary score as a JSON with a single key 'score' and no preamble or explanation.""",
    input_variables=["domain", "question"],
)