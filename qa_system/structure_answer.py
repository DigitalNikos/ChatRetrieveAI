from typing import List, Set

from pydantic import BaseModel
from typing_extensions import Annotated


class AnswerWithSources(BaseModel):
    """An answer to the question, with keys 'answer' and 'sources'.
        'answer' : The answer to user's question.
        'sources' : The key 'sources' of metadata of the Documents that are used to generate the answer."""
    answer: str
    sources: Annotated[
        Set[str],
        ...,
        "The key 'sources' of the Metadata",
    ]

class AnswerWithSourcesMath(BaseModel):
    """An Arithmetic Reasoning step wise reasoning to the question, with keys 'step_wise_reasoning', 'expr' and 'sources'.
        'step_wise_reasoning' : The step wise reasoning to the user's question.
        'expr' :Generate only parameter expressions (like 3x + 1) to be evaluated by ne.evaluate(.) for solving arithmetic tasks.
        'sources' : ONLY the key 'sources' of metadata of the Documents that are used to generate the answer."""
    step_wise_reasoning: List[str]
    expr: str
    sources: Annotated[
        Set[str],
        ...,
        "The key 'sources' of the Metadata from the provided documents",   
    ]

class AnswerWithWebSourcesMath(BaseModel):
    """An Arithmetic Reasoning step wise reasoning to the question, with keys 'step_wise_reasoning', 'solution' and 'sources'.
        'step_wise_reasoning': The step wise reasoning to the arithmetic task.
        'solution': Generated solution.
        'sources': The key 'sources' of metadata of the Documents that are used to generate the answer."""
    step_wise_reasoning: List[str]
    solution: str
    sources: Annotated[
        Set[str],
        ...,
        "The key 'sources' of the Metadata",   
    ]
    
class AnswerHallucination(BaseModel):
    """Answer with 'yes' if the answer is grounded in / supported by a set of facts, 'no' otherwise.
        'score': the key that it will have the value 'yes' or 'no'."""
    score: str
    