from langgraph.graph import END, StateGraph, START

class WorkflowInitializer:

    def __init__(self, system):
        self.system = system

    def initialize(self):
        print('\nCalling => langgraph.py - WorkflowInitializer.initialize()')    
        workflow = StateGraph(self.system.GraphState)

        workflow.set_entry_point("check_query_domain")
        workflow.add_node("check_query_domain", self.system._check_query_domain)
        workflow.add_node("rephrase_based_history", self.system._rephrase_query)
        workflow.add_node("retrieve", self.system._retrieve)   
        workflow.add_node("grade_docs", self.system._grade_documents)     
        workflow.add_node("check_query_domain_end", self.system._check_query_domain)
        workflow.add_node("generate", self.system._generate)
        workflow.add_node("ddg_search", self.system._ddg_search)
        workflow.add_node("answer_check", self.system._answer_check)
        workflow.add_node("hallucination_check", self.system._hallucination_check)
        
        workflow.add_conditional_edges(
            "check_query_domain",
            lambda state: state["generation_score"],
            {
                "yes": "retrieve",
                "no": "rephrase_based_history",
            },
        )
        
        workflow.add_edge("rephrase_based_history", "check_query_domain_end")
        workflow.add_conditional_edges(
            "check_query_domain_end",
            lambda state: state["generation_score"],
            {
                "yes": "retrieve",
                "no": END,
            },
        )
        
        workflow.add_edge("retrieve", "grade_docs")
        
        workflow.add_conditional_edges(
            "grade_docs",
            lambda state: "yes" if state["documents"] else "no",
            {
                "yes": "generate",
                "no": "ddg_search",
            },
        )
        
        workflow.add_conditional_edges(
            "ddg_search",
            lambda state: "yes" if state["documents"] else "no",
            {
                "yes": "grade_docs",
                "no": END,
            }
        )
        
        workflow.add_edge("generate", "hallucination_check")
        workflow.add_conditional_edges(
            "hallucination_check",
            lambda state: state["hallucination"],
            {
                "yes": END,
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
        
        app = workflow.compile()
        
        try:
            image_data = app.get_graph(xray=True).draw_mermaid_png()
            with open("graph_img/output_image.png", "wb") as file:
                file.write(image_data)
            print("\nImage saved to output_image.png")
        except Exception:
            print("Could not save image")
            pass
        return app
