# Unit Tests for Path Execution ChatBot

This test ensures that the system correctly follows the designated execution paths when processing a user's question. It does not cover paths that involve retrieving documents from web sources.

## Test File: test_path_1.py

**Test Path:**

- 'check_query_domain'
- 'retrieve'
- 'grade_docs'
- 'question_classification'
- 'generate'
- 'hallucination_check'
- 'answer_check'

**The test ensures that:**

- The question is correctly identified as relevant to the domain.
- Positive grade documents are retrieved.
- The question is classified as non-mathematical.
- The generated answer is free of hallucinations.
- The final answer is useful.

## Test File: test_path_2.py

**Test Path:**

- 'check_query_domain'
- 'rephrase_based_history'
- 'check_query_domain'
- 'retrieve'
- 'grade_docs'
- 'question_classification'
- 'generate'
- 'hallucination_check'
- 'answer_check'

**The test ensures that:**

- The question is correctly identified as relevant to the domain after rephrasing based on chat history.
- Positive grade documents are retrieved.
- The question is classified as non-mathematical.
- The generated answer is free of hallucinations.
- The final answer is useful.

**Test File: test_path_3.py**

**Test Path:**

- 'check_query_domain'
- 'rephrase_based_history'
- 'check_query_domain'

**The test ensures that:**

- The question is correctly identified as irrelevant to the domain.
- The question it will be rephrased based on chta_history
- The system provides a fallback response: "I don't know the answer to that question."

**Test File: test_path_4.py**

**Test Path:**

- 'check_query_domain'
- 'retrieve'
- 'grade_docs'
- 'ddg_search'
- 'grade_docs'

**The test ensures that:**

- The question is correctly identified as relevant to the domain.
- No positive grade documents are retrieved from the Vector DB.
- The system performs a web search (ddg_search) and still fails to retrieve relevant documents.
- The system provides a fallback response: "I don't know the answer to that question."

**Test File: test_path_5.py**

**Test Path:**

- 'check_query_domain'
- 'retrieve'
- 'grade_docs'
- 'question_classification'
- 'math_generate'
- 'answer_check'

**The test ensures that:**

- The question is correctly identified as relevant to the domain.
- Positive grade documents are retrieved.
- The question is correctly classified as a mathematical question.
- The system generates a useful answer based on Python computation (ne.evaluate(.)).
- The execution path matches the expected path until the last specified step.

**Test File: test_path_6.py**

**Test Path:**

- 'check_query_domain'
- 'retrieve'
- 'grade_docs'
- 'question_classification'
- 'math_generate'

**The test ensures that:**

- The question is correctly identified as relevant to the domain.
- Positive grade documents are retrieved.
- The question is correctly classified as a mathematical question.
- The system generates a mathematical answer based in a web-based calculation/solution
- The execution path matches the expected path until the last specified step.

**Test File: test_path_7.py**

**Test Path:**

- 'check_query_domain'
- 'retrieve'
- 'grade_docs'
- 'question_classification'
- 'generate'
- 'hallucination_check'
- 'answer_check'

**The test ensures that:**

- The question is correctly identified as relevant to the domain.
- Positive grade documents are retrieved.
- The question is correctly classified as non-mathematical.
- The system generates an answer free of hallucinations.
- The final answer is classified as not useful, with a fallback response: "I don't know the answer to that question."
- The execution path matches the expected path until the last specified step.
