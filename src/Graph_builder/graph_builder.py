from langgraph.graph import StateGraph, END
from src.State.rag_state import RAGState
from src.Nodes.react_node import RAGNodes

class GraphBuilder:
    """Builds and manages the langgraph workflow
    """
    def __init__(self,retriever,llm):
        """Initialize garph builder

        Args:
            retriever (_type_): doc retriver instance
            llm (_type_): language model instance
        """
        self.nodes = RAGNodes(retriever,llm)
        self.graph = None
        
    def build(self):
        """Build the RAG workflow graph
        Returns:
            Compiled graph instance
        """
        #create state graph
        builder = StateGraph(RAGState)
        
        #Add nodes
        builder.add_node("retriever",self.nodes.retriever_docs) # self.nodes.retriever_docs is the definition of the node called retriever
        builder.add_node("responder",self.nodes.generate_answer)
        
        #set entry point
        builder.set_entry_point("retriever")
        
        #Add edges
        builder.add_edge("retriever","responder")
        builder.add_edge("responder",END)
        
        #Compile graph
        self.graph = builder.compile()
        return self.graph
    
    def run(self, question:str)-> dict:
        """Run the RAG workflow

        Args:
            question (str): User question

        Returns:
            dict: Final state with answer
        """
        if self.graph is None:
            self.build()
            
        initial_state = RAGState(question=question)
        return self.graph.invoke(initial_state)