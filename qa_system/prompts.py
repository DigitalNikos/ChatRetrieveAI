from langchain import hub
from langchain.prompts import PromptTemplate
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

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

# rephrase_prompt = PromptTemplate(
#     template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
#     If the follow-up question lacks context rephrase it from the last user question to be understood, rephrase it to be a standalone question.
#     If the follow-up question is already a standalone question or is not relevant to the last user question, return it exactly as it is.
    
#     Given the last user question and a follow-up question:
    
#     Answer Template:
#     {{
#         'question': "Rephrased Standalone Question or Original Follow-Up Question",
#     }}
    
#     <|eot_id|><|start_header_id|>user<|end_header_id|>
#     Last User Question: {chat_history}
#     Follow-Up Question: {input}
#     <|eot_id|><|start_header_id|>assistant<|end_header_id|>
#     """,
#     input_variables=["input", "chat_history"],
# )

contextualize_q_system_prompt = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
Given a chat history and the latest user question \
which might reference context in the chat history, formulate a standalone question \
which can be understood without the chat history. Do NOT answer the question, \
just reformulate it if needed and otherwise return it as is.

Answer Template:
     {{
        'question': "Rephrased Standalone Question or Original Follow-Up Question",
     }}
 <|eot_id|><|start_header_id|>assistant<|end_header_id|>"""
rephrase_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)


question_classifier_prompt = PromptTemplate(
    template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
    You are an intelligent assistant trained to answer questions with 'yes' or 'no'. 
    Your task is to determine if a mathematical calculation is required to answer the given query. 
    If the query involves performing any kind of mathematical calculation (such as addition, subtraction, multiplication, division, finding percentages, solving equations, etc.), you should respond with 'yes'. 
    If the query can be answered without performing any mathematical calculations, respond with 'no'. 

    Give a binary score 'yes' or 'no' score to indicate whether the domain is relevant to the question. \n
    Provide the binary score as a JSON with a single key 'score' and no premable or explanation.
    <|eot_id|><|start_header_id|>user<|end_header_id|>
    Here is the user question: {question} \n 
    Answer: <|eot_id|><|start_header_id|>assistant<|end_header_id|> 
    """,
    input_variables=[ "question"],
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
    <|eot_id|><|start_header_id|>assistant<|end_header_id|> 
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
        <|begin_of_text|><|start_header_id|>system<|end_header_id|>
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

math_solver = PromptTemplate(
    template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
    You are an expert AI specialized in generating expressions for ne.evaluate(.) to solve arithmetic tasks.

    Given specific numbers, write a NumExpr-compatible expression that directly calculates the result. The final expression must contain only numbers and operators, with no variables. Provide the output as a JSON object with 'step-wise reasoning' and 'expr'.

    Example Task: 
    Related Documents: [Cappuccinos cost $2, iced teas cost $3, cafe lattes cost $1.5 and espressos cost $1 each.]
    Task: Sandy orders some drinks for herself and some friends. She orders three cappuccinos, two iced teas, two cafe lattes, and two espressos. 
    How much change does she receive back for a twenty-dollar bill?
    Answer: {{
    'step-wise reasoning': [
    'Calculate the total cost of three cappuccinos: 3 * 2',
    'Calculate the total cost of two iced teas: 2 * 3',
    'Calculate the total cost of two cafe lattes: 2 * 1.5',
    'Calculate the total cost of two espressos: 2 * 1',
    'Sum these costs to get the total expenditure',
    'Subtract the total expenditure from 20 to find the change'
    ],
    'expr': '20 - (3 * 2 + 2 * 3 + 2 * 1.5 + 2 * 1)'
    }}
    
    <|begin_of_text|><|start_header_id|>system<|end_header_id|>
    Related Documents: {documents}\n
    Task: {question} \n 
     <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
    input_variables=["question", "documents"],
)


