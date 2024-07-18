from langgraph.graph import END, StateGraph, START

class WorkflowInitializer:
    def __init__(self, graph_state, check_query_domain, rephrase_query, retrieve, grade_documents, generate, ddg_search, hallucinations_answer_check, handle_domain_relevance_with_rephrase, handle_domain_relevance_with_end, existing_docs_condition, decide_to_generate, hallucination_condition):
        self.graph_state = graph_state
        self.check_query_domain = check_query_domain
        self.rephrase_query = rephrase_query
        self.retrieve = retrieve
        self.grade_documents = grade_documents
        self.generate = generate
        self.ddg_search = ddg_search
        self.hallucinations_answer_check = hallucinations_answer_check
        self.handle_domain_relevance_with_rephrase = handle_domain_relevance_with_rephrase
        self.handle_domain_relevance_with_end = handle_domain_relevance_with_end
        self.existing_docs_condition = existing_docs_condition
        self.decide_to_generate = decide_to_generate
        self.hallucination_condition = hallucination_condition

    def initialize(self):
        print('\nCalling => langgraph.py - WorkflowInitializer.initialize()')    
        workflow = StateGraph(self.graph_state)

        workflow.add_edge(START, "check_query_domain")
        workflow.add_node("check_query_domain", self.check_query_domain)
        workflow.add_node("rephrase", self.rephrase_query)
        workflow.add_node("retrieve", self.retrieve)   
        workflow.add_node("grade_docs", self.grade_documents)     
            
        workflow.add_node("generate", self.generate)
        workflow.add_node("ddg_search", self.ddg_search)
        workflow.add_node("hallucination_grader", self.hallucinations_answer_check)
            
        workflow.add_conditional_edges(
            "check_query_domain",
            self.handle_domain_relevance_with_rephrase,
            {
                "retrieve": "retrieve",
                "rephrase": "rephrase",
            },
        )
            
        workflow.add_edge("rephrase", "check_query_domain_end")

        # Node for the second check, only called after rephrase
        workflow.add_node("check_query_domain_end", self.check_query_domain)
        workflow.add_conditional_edges(
            "check_query_domain_end",
            self.handle_domain_relevance_with_end,
            {
                "retrieve": "retrieve",
                "end": END,
            },
        )
        workflow.add_conditional_edges(
            "retrieve",
            self.existing_docs_condition,
            {
                "grade_docs": "grade_docs",
                "ddg_search": "ddg_search",
            },
        )
            
        workflow.add_conditional_edges(
            "grade_docs",
            self.decide_to_generate,
            {
                "generate": "generate",
                "end": END,
            }
        )
            
        workflow.add_conditional_edges(
            "ddg_search",
            self.existing_docs_condition,
            {
                "grade_docs": "grade_docs",
                "end": END,
            }
        )
            
        workflow.add_edge("generate", "hallucination_grader")
        workflow.add_edge("ddg_search", "generate")
            
        workflow.add_conditional_edges(
            "hallucination_grader",
            self.hallucination_condition,
            {
                "not supported": END,
                "useful": END,
                "not useful": END,
            },
        )
            
        return workflow.compile()