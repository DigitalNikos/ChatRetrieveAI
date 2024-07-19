from langgraph.graph import END, StateGraph, START

class WorkflowInitializer:

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
        workflow.add_node("check_query_domain_end", self.system._check_query_domain)
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
