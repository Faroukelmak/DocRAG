from typing import List
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document


class VectorStore:
    """Manages vector store application
    """
    def __init__(self):
        self.embedding = OpenAIEmbeddings()
        self.vectorstore = None
        self.retriever = None
        
    def create_retriever(self,documents:List[Document]):
        """create vecor store from documents

        Args:
            documents (List[Document]): list of documents to embed
        """
        self.vectorstore = FAISS.from_documents(documents,self.embedding)
        self.retriever = self.vectorstore.as_retriever()

    
    def get_retriever(self):
        """Get the retriver instance
        Returns:
            retriever instance
        """
        if self.retriever is None:
            raise ValueError("vector store not initialized. call create_retriever first.")
        return self.retriever
    
    def retrieve(self,query: str, k: int=4) -> List[Document]:
        """Retrieve relevant doc for a query

        Args:
            query (str): search query
            k (int, optional): number of doc to retrieve. Defaults to 4.

        Returns:
            List[Document]: list of releveant docs
        """
        if self.retriever is None:
            raise ValueError("vector store not initialized. call create_retriever first.")
        return self.retriever.invoke(query)