from langchain import hub
from langchain.prompts import PromptTemplate
from langchain.output_parsers import ResponseSchema, StructuredOutputParser

query_domain_check =PromptTemplate(
    template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
    You are a grader assessing whether a user question falls within the specified {domain} domain.\n 
    Your task is to determine if the question is directly related to {domain} by considering the content and context of the question.\n
    Give a binary score 'yes' or 'no' score to indicate whether the domain is relevant to the question. \n
    Provide the binary score as a JSON with a single key 'score' and no premable or explanation.
    <|eot_id|><|start_header_id|>user<|end_header_id|>
    Here is the user question: {question} \n 
    Answer: <|eot_id|><|start_header_id|>assistant<|end_header_id|>
    """,
    input_variables=["question", "domain"],
)

rephrase_prompt = PromptTemplate(
    template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
    If the follow-up question is already a standalone question or is not relevant to the last user question, return it exactly as it is.
    If the follow-up question needs context from the last user question to be understood, rephrase it to be a standalone question.
    
    Given the last user question and a follow-up question:
    
    answer Template:
    {{
        'question': "Standalone Question",
    }}
    
    <|eot_id|><|start_header_id|>user<|end_header_id|>
    Last User Question: {chat_history}
    Follow-Up Question: {input}
    <|eot_id|><|start_header_id|>assistant<|end_header_id|>
    """,
    input_variables=[ "intput", "chat_history"],
)

grader_document_prompt = PromptTemplate(
    template="""You are a grader assessing relevance of a retrieved document to a user question. \n 
    Here is the retrieved document: \n {document} \n
    If the document contains keywords related to the user question, grade it as relevant. \n
    It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
    Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question. \n
    Provide the binary score as a JSON with a single key 'score' and no premable or explanation.

    Here is the user question: {question} \n
    """,
    input_variables=["question", "document"],
)

response_schemas = [
            ResponseSchema(name="answer", description="answer to the user's question"),
            ResponseSchema(
                name="metadata",
                description="source used to answer the user's question.",
            ),
        ]
output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
format_instructions = output_parser.get_format_instructions()
generate_answer_propmpt = PromptTemplate(
        template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
        You are an assistant for question-answering tasks. 
        Use the following pieces of retrieved context to answer the question. 
        Also i will provide you the chat history. If it is empty, you can ignore it. if it is not empty, you can use it to answer the question.
        If you don't know the answer, just say that you don't know. 
        return a JSON with the key 'answer' and 'metadata', metadata should reference the metadata from used Document objects. 
        <|eot_id|><|start_header_id|>user<|end_header_id|>
        Context: {context} 
        
        Chat History: {chat_history}
        
        Question: {question} 
        
        format instructions: {format_instructions}
        <|eot_id|><|start_header_id|>assistant<|end_header_id|>
        """,
    input_variables=["context", "question", "chat_history"],
    partial_variables={"format_instructions": format_instructions},
)

hallucination_grader_prompt = PromptTemplate(
    template="""You are a grader assessing whether an answer is grounded in / supported by a set of facts. \n 
    Here are the facts:
    \n ------- \n
    {documents} 
    \n ------- \n
    Here is the answer: {generation}
    Give a binary score 'yes' or 'no' score to indicate whether the answer is grounded in / supported by a set of facts. \n
    Provide the binary score as a JSON with a single key 'score' and no preamble or explanation.
    """,
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
    Provide the binary score as a JSON with a single key 'score' and no preamble or explanation.
    """,
    input_variables=["generation", "question"],
)



