"""
Configuration management for MEDRAG application.
"""

import os
from typing import Optional
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()


@dataclass
class Config:
    """Configuration class for MEDRAG application."""
    
    # LLM Configuration
    openai_api_key: Optional[str] = None
    openai_model: str = "gpt-3.5-turbo"
    openai_max_tokens: int = 1000
    openai_temperature: float = 0.7
    
    # Vector Store Configuration
    vector_store_path: str = "./vector_store"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    chunk_size: int = 1000
    chunk_overlap: int = 200
    
    # PDF Processing Configuration
    max_pdf_pages: int = 100
    min_chunk_length: int = 50
    
    # RAG Configuration
    top_k_retrieval: int = 5
    similarity_threshold: float = 0.7
    
    def __post_init__(self):
        """Load configuration from environment variables."""
        self.openai_api_key = os.getenv("OPENAI_API_KEY", self.openai_api_key)
        self.openai_model = os.getenv("OPENAI_MODEL", self.openai_model)
        self.openai_max_tokens = int(os.getenv("OPENAI_MAX_TOKENS", self.openai_max_tokens))
        self.openai_temperature = float(os.getenv("OPENAI_TEMPERATURE", self.openai_temperature))
        
        self.vector_store_path = os.getenv("VECTOR_STORE_PATH", self.vector_store_path)
        self.embedding_model = os.getenv("EMBEDDING_MODEL", self.embedding_model)
        self.chunk_size = int(os.getenv("CHUNK_SIZE", self.chunk_size))
        self.chunk_overlap = int(os.getenv("CHUNK_OVERLAP", self.chunk_overlap))
        
        self.max_pdf_pages = int(os.getenv("MAX_PDF_PAGES", self.max_pdf_pages))
        self.min_chunk_length = int(os.getenv("MIN_CHUNK_LENGTH", self.min_chunk_length))
        
        self.top_k_retrieval = int(os.getenv("TOP_K_RETRIEVAL", self.top_k_retrieval))
        self.similarity_threshold = float(os.getenv("SIMILARITY_THRESHOLD", self.similarity_threshold))
    
    def validate(self) -> bool:
        """Validate configuration settings."""
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY is required. Please set it in your environment variables.")
        
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        
        if self.chunk_overlap < 0:
            raise ValueError("chunk_overlap must be non-negative")
        
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")
        
        return True
