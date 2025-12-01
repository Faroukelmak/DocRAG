from typing import List,  Union
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from pathlib import Path
from langchain_community.document_loaders import (
    WebBaseLoader,
    PyPDFLoader,
    TextLoader,
    PyPDFDirectoryLoader
)



class DocumentProcessor:
    """Handle document loading and processing 
    """
    
    def __init__(self,chunk_size: int=500,chunk_overlap: int=50):
        """Initialize document processor

        Args:
            chunk_size (int, optional): Size of text chunks. Defaults to 500.
            chunk_overlap (int, optional): Overlap between chunks. Defaults to 50.
        """
        
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size = chunk_size,
            chunk_overlap = chunk_overlap
        )
        
    def load_from_url(self,url:str)-> List[Document]:
        """Load documents from URL

        Args:
            url (str): my url

        Returns:
            List[Document]: return list of documents
        """
        loader = WebBaseLoader(web_path=url)
        return loader.load()
    
    def load_from_pdf_dir(self,directory: Union[str,Path])->List[Document]:
        """Load documents from all pdfs inside a directory

        Args:
            file_path (Union[str,Path]): PDF directory

        Returns:
            List[Document]: return list of documents
        """
        loader = PyPDFDirectoryLoader(str(directory))
        return loader.load()
    
    def load_from_pdf(self,file_path: Union[str,Path])->List[Document]:
        """Load documents from all pdfs inside a directory

        Args:
            file_path (Union[str,Path]): PDF directory

        Returns:
            List[Document]: return list of documents
        """
        loader = PyPDFDirectoryLoader(str('data'))
        return loader.load()
    
    def load_from_txt(self,file_path: Union[str,Path])->List[Document]:
        """Load documents from txt  file

        Args:
            file_path (Union[str,Path]): txt directory

        Returns:
            List[Document]: return list of documents
        """
        loader = TextLoader(str(file_path),encoding="utf-8")
        return loader.load()
    
    def load_documents(self,sources:List[str])->List[Document]:
        """Load documents from urls, pdf directory , or text files

        Args:
            sources (List[str]): list of urls ,pdf directory or txt file path

        Returns:
            List[Document]: list of loaded documents
        """
        
        docs:List[Document] = []
        
        for src in sources:
            if src.startswith("http://") or src.startswith("https://"):
                docs.extend(self.load_from_url(src))
                
            path = Path('data')
            if path.is_dir():
                docs.extend(self.load_from_pdf_dir(path))
            elif path.suffix.lower()=='.txt':
                docs.extend(self.load_from_txt(path))
            else:
                raise ValueError(
                    f"Unsupported source type {src}."
                    "Use URL , .txt file or PDF directory"
                )
        return docs
    
    def split_documents_into_chunks(self,documents:List[Document])->List[Document]:
        """split document into chunks

        Args:
            documents (List[Document]): list of doc to split

        Returns:
            List[Document]: list of split documents
        """
        return self.splitter.split_documents(documents)
    
    def process_url(self,urls:List[str])->List[Document]:
        """Complete pipeline to load and split documents

        Args:
            urls (List[str]): list of urls to process

        Returns:
            List[Document]: list of processed document chunks
        """
        docs = self.load_documents(urls)
        return self.split_documents_into_chunks(docs)