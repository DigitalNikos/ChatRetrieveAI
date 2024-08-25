from langchain.prompts import PromptTemplate
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
    
response_schemas_document_domain_check = [
            ResponseSchema(
                name="score", 
                description="'yes' if the summary and domain fall within the specified user domain, 'no' otherwise."),
        ]
output_parser_document_domain_check = StructuredOutputParser.from_response_schemas(response_schemas_document_domain_check)
format_output_parser_document_domain_check = output_parser_document_domain_check.get_format_instructions()
domain_check = PromptTemplate(
    template="""
    <|begin_of_text|><|start_header_id|>system<|end_header_id|>
    Here is the list from 3 domains you have identified from the document: {doc_domain}
    Here is the summary of the document: {summary}
    Here is the specified domain from the user: {domain}
    {format_instructions}
    <|eot_id|><|start_header_id|>assistant<|end_header_id|>
    """,
    input_variables=["domain", "summary", "doc_domain"],
    partial_variables={"format_instructions": format_output_parser_document_domain_check},
)

response_schemas_domain_detection= [
            ResponseSchema(
                name="summary", 
                description="Summary of the documents."),
            ResponseSchema(
                name="domain",
                description="List of possible domains the documents could belong to.",
                type="list",
            ),
        ]
output_parser_domain_detection = StructuredOutputParser.from_response_schemas(response_schemas_domain_detection)
format_domain_detection = output_parser_domain_detection.get_format_instructions()
domain_detection = PromptTemplate(
    template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
    You are a grader assessing the summarization and domain of a set of documents.
    Here are the documents:
    \n ------- \n
    {documents}
    \n ------- \n
    First, provide a summary of the documents. 
    Then, indicate three possible domains the documents could belong(e.g., sports, movies, technology).
    {format_instructions}
    <|eot_id|><|start_header_id|>assistant<|end_header_id|>
    """,
    input_variables=["documents"],
    partial_variables={"format_instructions": format_domain_detection},
)