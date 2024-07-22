from langchain.prompts import PromptTemplate

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
    template="""You are a grader assessing the summarization and domain of a set of documents.\n
    Here are the documents:
    \n ------- \n
    {documents}
    \n ------- \n
    First, provide a summary of the documents. Then, indicate the domain the documents belong to (e.g., sports, movies, tech).\n
    Provide the summary and domain as a JSON object with keys 'summary' and 'domain' respectively, and no preamble or explanation. Not a List but JSON object always.
    """,
    input_variables=["documents"],
)