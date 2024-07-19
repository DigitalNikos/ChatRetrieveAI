from langchain import hub
from langchain.prompts import PromptTemplate
from langchain.output_parsers import ResponseSchema, StructuredOutputParser

query_domain_check =PromptTemplate(
    template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are a grader assessing for this 
    specific {domain}, relevance whether a uesr question falls within a specified domain. \n
    Give a binary score 'yes' or 'no' score to indicate whether the domain is relevant to the question. \n
    Provide the binary score as a JSON with a single key 'score' and no premable or explanation.
     <|eot_id|><|start_header_id|>user<|end_header_id|>
    Here is the user question: {question} \n <|eot_id|><|start_header_id|>assistant<|end_header_id|>
    """,
    input_variables=["question", "domain"],
)

rephrase_prompt = PromptTemplate(
    template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> 
    You are a helpful AI assistant specialized in rephrasing and improving questions.\n
    Given the following conversation and a follow up question,\n 
    rephrase the follow up question to be a standalone question.
    if the history is empty, the follow up question should be returned exactly as it is.
    if the follow up question is already a standalone question, return it as it is.
    if the follow up question is not relevant to the chat history, return it as it is.
    Chat History:  {chat_history}
    
    Standalone Question:
    answer Template:
    {{
        "question": "Standalone Question",
    }}
    <|eot_id|><|start_header_id|>user<|end_header_id|> 
    Here is the Follow Up question: {input} 
    <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
    input_variables=[ "intput", "chat_history"],
)

grader_document_prompt = PromptTemplate(
    template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> 
    You are a grader assessing relevance of a retrieved document to a user question. \n 
    Here is the retrieved document: \n {document} \n
    If the document contains keywords related to the user question, grade it as relevant. \n
    It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
    Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question. \n
    Provide the binary score as a JSON with a single key 'score' and no premable or explanation.
    <|eot_id|><|start_header_id|>user<|end_header_id|> 
    Here is the user question: {question} \n
    <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
    input_variables=["question", "document"],
)

response_schemas = [
            ResponseSchema(name="answer", description="answer to the user's question"),
            ResponseSchema(
                name="source",
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
        Use three sentences maximum and keep the answer concise.
        return a JSON with the key 'answer' and 'metadata', metadata should reference the metadata from used Document objects. 

        Context: {context} 
        
        Chat History: {chat_history}
        <|eot_id|><|start_header_id|>user<|end_header_id|> 
        Question: {question} 
        <|eot_id|><|start_header_id|>assistant<|end_header_id|>
        format instructions: {format_instructions}
        """,
    input_variables=["context", "question", "chat_history"],
    partial_variables={"format_instructions": format_instructions},
)

hallucination_grader_prompt = PromptTemplate(
    template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
    You are a grader assessing whether an answer is grounded in / supported by a set of facts. \n 
    Here are the facts:
    \n ------- \n
    {documents} 
    \n ------- \n
    Here is the answer: {generation}
    Give a binary score 'yes' or 'no' score to indicate whether the answer is grounded in / supported by a set of facts. \n
    Provide the binary score as a JSON with a single key 'score' and no preamble or explanation.
    <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
    input_variables=["generation", "documents"],
)

answers_grader_prompt = PromptTemplate(
    template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
    You are a grader assessing whether an answer is useful to resolve a question. \n 
    Here is the answer:
    \n ------- \n
    {generation} 
    \n ------- \n
    Here is the question: {question}
    Give a binary score 'yes' or 'no' to indicate whether the answer is useful to resolve a question. \n
    Provide the binary score as a JSON with a single key 'score' and no preamble or explanation.
    <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
    input_variables=["generation", "question"],
)



