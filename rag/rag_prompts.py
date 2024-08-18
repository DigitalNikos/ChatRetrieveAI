from langchain.prompts import PromptTemplate
from langchain.output_parsers import ResponseSchema, StructuredOutputParser

domain_check = PromptTemplate(
    template="""
    You are a grader assessing whether a set of documents falls within a specified domain.\n
    Here is the specified domain: {domain}
    Here is the summary of the document:
    \n ------- \n
    {summary}
    \n ------- \n
    Here is the domain you have identified from the document: {doc_domain}
    Give a binary 'yes' or 'no' score to indicate whether the documents fall within the specified domain. Not 0 or 1 only yes or no.
    Provide the binary score as a JSON with a single key 'score' and no preamble or explanation.
    """,
    input_variables=["domain", "summary", "doc_domain"],
)

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
    input_variables=["documents","format_instructions"],
)

#TODO try in one prompt