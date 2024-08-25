from langchain.prompts import PromptTemplate
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


response_schemas_domain_check = [
    ResponseSchema(
        name="score", 
        description="Answer with 'yes' if the question is related to the domain, 'no' otherwise."),
]
output_parser_domain_check = StructuredOutputParser.from_response_schemas(response_schemas_domain_check)
format_instructions_domain_check = output_parser_domain_check.get_format_instructions(only_json =True)
query_domain_check =PromptTemplate(
    template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
    You are a grader assessing whether a user question falls within the specified {domain} domain.
    <|eot_id|> <|start_header_id|>user<|end_header_id|>
    User question: {question}
    format instructions: {format_instructions}
    <|eot_id|><|start_header_id|>assistant<|end_header_id|>
    """,
    input_variables=["question", "domain"],
    partial_variables={"format_instructions": format_instructions_domain_check}
)


contextualize_q_system_prompt = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
Given a chat history and the latest user question which might reference context in the chat history, 
formulate a standalone question which can be understood without the chat history. 
Do NOT answer the question, just reformulate it if needed and otherwise return it as is.

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


response_schemas_question_classifier = [
    ResponseSchema(
        name="score", 
        description="Answer with 'yes' if the question is a math question, 'no' otherwise."),
]
output_parser_question_classifier = StructuredOutputParser.from_response_schemas(response_schemas_question_classifier)
format_instructions_question_classifier = output_parser_question_classifier.get_format_instructions(only_json =True)
question_classifier_prompt = PromptTemplate(
    template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
    Classify the question as a math question or not.
    <|eot_id|><|start_header_id|>user<|end_header_id|>
    question: {question}
    <|eot_id|><|start_header_id|>assistant<|end_header_id|>
    format instructions: {format_instructions}
    """,
    input_variables=["question"],
    partial_variables={"format_instructions": format_instructions_question_classifier} 
)


response_schemas_grader_document = [
    ResponseSchema(
        name="score", 
        description="Answer with 'yes' if the question is relevant to the document, 'no' otherwise."),
]
output_parser_grader_document = StructuredOutputParser.from_response_schemas(response_schemas_grader_document)
format_instructions_grader_document = output_parser_grader_document.get_format_instructions(only_json =True)
grader_document_prompt = PromptTemplate(
    template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
    You are a grader assessing relevance of a document to a user's question. \n 
    Document: {document}
    <|eot_id|><|start_header_id|>user<|end_header_id|>
    User question: {question} 
    <|eot_id|><|start_header_id|>assistant<|end_header_id|> 
    format instructions: {format_instructions}
    """,
    input_variables=["question", "document"],
    partial_variables={"format_instructions": format_instructions_grader_document} 
)


generate_answer = PromptTemplate(
        template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
        You are an assistant for question-answering tasks. 
        Use only the context to answer the user's question.
        Context: {context} 
        <|eot_id|><|begin_of_text|><|start_header_id|>user<|end_header_id|>
        User Question: {question} 
        <|eot_id|><|start_header_id|>assistant<|end_header_id|>
        """,
    input_variables=["context", "question"],
)


response_schemas_hallucination = [
            ResponseSchema(
                name="score", 
                description="Answer with 'yes' if the answer is grounded in / supported by a set of facts, 'no' otherwise.")
        ]
output_parser_hallucination = StructuredOutputParser.from_response_schemas(response_schemas_hallucination)
format_instructions_hallucination = output_parser_hallucination.get_format_instructions()
hallucination_grader_prompt = PromptTemplate(
    template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
    You are a grader assessing whether an answer is grounded in / supported by a set of facts. 
    Here are the facts: {documents} 
    Here is the answer: {generation}.
    <|eot_id|><|begin_of_text|><|start_header_id|>user<|end_header_id|>
    format instructions: {format_instructions}
    <|eot_id|><|start_header_id|>assistant<|end_header_id|>
    """,
    input_variables=["generation", "documents"],
    partial_variables={"format_instructions": format_instructions_hallucination},
)


response_schemas_answer_grader = [
            ResponseSchema(
                name="score", 
                description="Answer with 'yes' if the answer is useful to resolve a question, 'no' otherwise.")
        ]
output_parser__answer_grader = StructuredOutputParser.from_response_schemas(response_schemas_answer_grader)
format_instructions_answer_grader = output_parser__answer_grader.get_format_instructions()
answers_grader_prompt = PromptTemplate(
    template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
    You are a grader assessing whether an answer is useful to resolve a question.
    Here is the answer: {generation}. 
    Here is the question: {question}
    <|eot_id|><|begin_of_text|><|start_header_id|>user<|end_header_id|>
    format instructions: {format_instructions}
    <|eot_id|><|start_header_id|>assistant<|end_header_id|>
    """,
    input_variables=["generation", "question"],
    partial_variables={"format_instructions": format_instructions_answer_grader},
)


math_solver = PromptTemplate(
    template="""
    <|begin_of_text|><|start_header_id|>system<|end_header_id|>
    You are an expert AI specialized in generating expressions for ne.evaluate(.) to solve arithmetic tasks.
    Given specific numbers, write a NumExpr-compatible expression that directly calculates the result.
    The final expression must contain only numbers and operators, with no variables.
    
    Related Documents: {documents}\n
    
    Example Task: 
    Related Documents: [Cappuccinos cost $2, iced teas cost $3, cafe lattes cost $1.5 and espressos cost $1 each.]
    Task: Sandy orders some drinks for herself and some friends. She orders three cappuccinos, 
    two iced teas, two cafe lattes, and two espressos. How much change does she receive back for a twenty-dollar bill?
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
    <|eot_id|><|begin_of_text|><|start_header_id|>user<|end_header_id|>
    Task: {question} 
    <|eot_id|><|start_header_id|>assistant<|end_header_id|>
     """,
    input_variables=["question", "documents"],
)


math_solver_web = PromptTemplate(
    template="""
    <|begin_of_text|><|start_header_id|>system<|end_header_id|>
    Solve the arithmetic task step by step using the related documents.
    Related Documents: {documents}\n
    <|eot_id|><|begin_of_text|><|start_header_id|>user<|end_header_id|>
    Arithmetic task: {question} 
    <|eot_id|><|start_header_id|>assistant<|end_header_id|>
     """,
    input_variables=["question", "documents"],
)