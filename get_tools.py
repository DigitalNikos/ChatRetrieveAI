from langchain.tools.retriever import create_retriever_tool
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper

from config import Config as cfg

def create_tool(tool_name, name: str = "", description: str = "", **kwargs):
    if tool_name == "wikipedia":
        if name == "":
            name = "Wikipedia search"
        if description == "":
            description = "Search for information in Wikipedia."
        api_wrapper = WikipediaAPIWrapper(top_k_results=cfg.WIKIPEDIA_TOP_K_RESULTS, doc_content_chars_max=cfg.WIKIPEDIA_DOC_CONTENT_CHARS_MAX)
        wiki_tool = WikipediaQueryRun(api_wrapper=api_wrapper, name=name, description=description, **kwargs)
        return wiki_tool
    else:
        raise ValueError(f"Tool {tool_name} not found.")