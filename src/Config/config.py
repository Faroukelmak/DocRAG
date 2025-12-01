
import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

# Load environment variables
load_dotenv()

class Config:
    """Configuration class for RAG system"""
    
    # API Keys
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")

    LLM_MODEL = "openai:gpt-4o"
    
    # Document Processing
    CHUNK_SIZE = 500
    CHUNK_OVERLAP = 50
    
    # Default URLs
    DEFAULT_URLS = [
        "https://rwai.ch/en/blog/how-deepseek-trained-r1",
        "https://medium.com/@techsachin/s1-simple-test-time-scaling-approach-to-exceed-openais-o1-preview-performance-ec5a624c5d2f",
        
    ]
    
    @classmethod
    def get_llm(cls):
        """Initialize and return the LLM model"""
        os.environ["OPENAI_API_KEY"] = cls.OPENAI_API_KEY
        return init_chat_model(cls.LLM_MODEL)