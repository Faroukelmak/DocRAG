from src.State.rag_state import RAGState

class RAGNodes:
    """Contains node functions for RAG workflow
    """
    
    def __init__(self,retriever,llm):
        """Initialize RAG nodes

        Args:
            retriever (_type_): doc retriever instance
            llm (_type_): language model instance
        """
        self.retriever = retriever
        self.llm = llm 
    
    def retriever_docs(self,state:RAGState)->RAGState:
        """Retrieve relevant doc node

        Args:
            state (RAGState): Current RAG state

        Returns:
            RAGState: update RAG state with retrieved documents
        """
        docs = self.retriever.invoke(state.question)
        return RAGState(
            question=state.question,
            retrieved_docs=docs
            )
        
    def generate_answer(self,state:RAGState) -> RAGState:
        """Generate answer from retrived documents node

        Args:
            state (RAGState): Current RAG state with retrieved documents

        Returns:
            RAGState: Update RAG state with generated answer
        """
        context = "\n\n".join([doc.page_content for doc in state.retrieved_docs])
        
        prompt = f"""Answer the question based on the context
        
        Context:
        {context}
        
        Question: {state.question}
        
        """
        
        reponse = self.llm.invoke(prompt)
        
        return RAGState(
            question=state.question,
            retrieved_docs=state.retrieved_docs,
            answer=reponse.content
        )
        
        