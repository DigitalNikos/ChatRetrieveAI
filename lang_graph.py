from langgraph.graph import END, StateGraph, START

class WorkflowInitializer:
    # def __init__(self, graph_state, check_query_domain, rephrase_query, retrieve, grade_documents, generate, ddg_search, hallucination_check, answer_check, end_with_hallucination_message, handle_domain_relevance_with_rephrase, handle_domain_relevance_with_end, existing_docs_condition, decide_to_generate, hallucination_condition, existing_ddg_docs_condition):
    #     self.graph_state = graph_state
    #     self.check_query_domain = check_query_domain
    #     self.rephrase_query = rephrase_query
    #     self.retrieve = retrieve
    #     self.grade_documents = grade_documents
    #     self.generate = generate
    #     self.ddg_search = ddg_search
    #     self.hallucination_check = hallucination_check
    #     self.answer_check = answer_check
    #     self.end_with_hallucination_message = end_with_hallucination_message
    #     self.handle_domain_relevance_with_rephrase = handle_domain_relevance_with_rephrase
    #     self.handle_domain_relevance_with_end = handle_domain_relevance_with_end
    #     self.existing_docs_condition = existing_docs_condition
    #     self.decide_to_generate = decide_to_generate
    #     self.hallucination_condition = hallucination_condition
    #     self.existing_ddg_docs_condition = existing_ddg_docs_condition
    def __init__(self, system):
        self.system = system

    def initialize(self):
        print('\nCalling => langgraph.py - WorkflowInitializer.initialize()')    
        workflow = StateGraph(self.system.GraphState)

        workflow.add_edge(START, "check_query_domain")
        workflow.add_node("check_query_domain", self.system._check_query_domain)
        workflow.add_node("rephrase", self.system._rephrase_query)
        workflow.add_node("retrieve", self.system._retrieve)   
        workflow.add_node("grade_docs", self.system._grade_documents)     
        
        workflow.add_node("generate", self.system._generate)
        workflow.add_node("ddg_search", self.system._ddg_search)
        workflow.add_node("answer_check", self.system._answer_check)
        workflow.add_node("hallucination_check", self.system._hallucination_check)
        workflow.add_node("end_with_hallucination_message", self.system._end_with_hallucination_message)
        
        workflow.add_conditional_edges(
            "check_query_domain",
            self.system._handle_domain_relevance_with_rephrase,
            {
                "retrieve": "retrieve",
                "rephrase": "rephrase",
            },
        )
        
        workflow.add_edge("rephrase", "check_query_domain_end")

        # Node for the second check, only called after rephrase
        workflow.add_node("check_query_domain_end", self.system._check_query_domain)
        workflow.add_conditional_edges(
            "check_query_domain_end",
            self.system._handle_domain_relevance_with_end,
            {
                "retrieve": "retrieve",
                "end": END,
            },
        )
        workflow.add_conditional_edges(
            "retrieve",
            self.system._existing_docs_condition,
            {
                "grade_docs": "grade_docs",
                "ddg_search": "ddg_search",
            },
        )
        
        workflow.add_conditional_edges(
            "grade_docs",
            self.system._decide_to_generate,
            {
                "generate": "generate",
                "end": END,
            }
        )
        
        workflow.add_conditional_edges(
            "ddg_search",
            self.system._existing_ddg_docs_condition,
            {
                "grade_docs": "grade_docs",
                "end": END,
            }
        )
        
        workflow.add_edge("generate", "hallucination_check")
        
        workflow.add_conditional_edges(
            "hallucination_check",
            lambda state: state["hallucination"],
            {
                "yes": "end_with_hallucination_message",
                "no": "answer_check",
            },
         )
        
        workflow.add_conditional_edges(
            "answer_check",
            lambda state: state["score"],
            {
                "useful": END,
                "not useful": END,
            },
        )
        
        return workflow.compile()
