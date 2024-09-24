# Unit Tests for Graph Edges ChatBot

This project contains unit tests for verifying the behavior of different edges in the ChatPDF system. Each edge represents a different type of interaction, such as domain relevance checks, rephrased queries, and document retrieval. These tests ensure that the system correctly handles various scenarios across these edges.

## Test Files

### 1. 'test_1_domain_relevant.py'

**Description:** Tests the domain relevance of a question.

- **Positive Test:** Verifies that a question relevant to the domain ("Sport") is correctly classified as relevant.

- **Negative Test:** Verifies that a question unrelated to the domain is correctly classified as irrelevant.

### 2. 'test_2_rephrase_domain_relevant.py'

**Description:** Tests if a rephrased question remains relevant to the domain.

- **Positive Test:** Verifies that a rephrased question related to the domain is correctly classified as relevant.

- **Negative Test:** Verifies that an unrelated rephrased question is correctly classified as irrelevant.

### 3.'test_3_exist_retriever_docs.py'

**Description:** Tests the document retrieval and grading process for relevance.

- **Positive Test:** Ensures that relevant documents are retrieved and correctly graded as relevant.

### 4.'test_4_question_classifier.py'

**Description:** Tests the classification of a question's type based on 'math' or 'text'.

- **Positive Test:** Verifies that a specific predictive model question is classified as a question of type math 'yes'.

- **Negative Test:** Verifies that a specific predictive model question is classified as a question of type text 'no'.

### 5.'test_5_computation_method.py'

**Description:** Tests the computation method (math_generate) to determine whether the answer is computed using Python (ne.evaluate(.)) or if it's a fallback web-based solution.

- **Positive Test:** Verifies that the system computes the answer directly using Python (with ne.evaluate()). The classification for this case is yes, indicating that the computation was performed locally.

- **Negative Test:** Verifies that when the system is unable to compute the answer using Python, it falls back to a web-based solution. The classification for this case is no, indicating that the result was retrieved externally and may not be as reliable.

### 6.'test_6_answer_hallucination.py'

**Description:** Tests the system's ability to check for hallucinations in generated answers. This involves verifying whether the answer provided by the system is grounded in the retrieved documents or if it includes unsupported information (hallucinations).

- **Positive Test:** Verifies that the system correctly identifies when the generated answer is fully supported by the retrieved documents and classifies it as not hallucinated ('no').

- **Negative Test:** Verifies that the system correctly identifies when the generated answer includes information that is not supported by the retrieved documents and classifies it as hallucinated ('yes').

### 7.'test_7_useful_answer.py'

**Description:** Tests the system's ability to evaluate whether the generated answer is useful or not based on the context of the question.

- **Positive Test:** Verifies that the system correctly classifies an answer as useful when the answer is relevant and directly addresses the question asked.
- **Negative Test:**Verifies that the system correctly classifies an answer as not useful when the answer is irrelevant or unrelated to the question asked.
