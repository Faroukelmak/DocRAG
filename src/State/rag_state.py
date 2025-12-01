"""Rag State definition for Langgraph
    """

from typing import List
from pydantic import BaseModel
from langchain_core.documents import Document

class RAGState(BaseModel):
    """State object for RAG workflow

    Args:
        BaseModel (_type_): 
    """
    question: str
    retrieved_docs: List[Document]= []
    answer:str =""