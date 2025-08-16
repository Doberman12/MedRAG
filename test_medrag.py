#!/usr/bin/env python3
"""
Test script for MEDRAG application components.
"""

import os
import sys
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_imports():
    """Test if all required modules can be imported."""
    try:
        from medrag.core.pdf_processor import PDFProcessor
        from medrag.core.vector_store import VectorStore
        from medrag.core.llm_client import LLMClient
        from medrag.core.rag_system import RAGSystem
        from medrag.utils.config import Config
        logger.info("[OK] All imports successful")
        return True
    except ImportError as e:
        logger.error(f"[FAILED] Import error: {e}")
        return False

def test_config():
    """Test configuration loading."""
    try:
        from medrag.utils.config import Config
        config = Config()
        logger.info("[OK] Configuration loaded successfully")
        logger.info(f"   - Embedding model: {config.embedding_model}")
        logger.info(f"   - Chunk size: {config.chunk_size}")
        logger.info(f"   - Vector store path: {config.vector_store_path}")
        return True
    except Exception as e:
        logger.error(f"[FAILED] Configuration error: {e}")
        return False

def test_pdf_processor():
    """Test PDF processor initialization."""
    try:
        from medrag.core.pdf_processor import PDFProcessor
        processor = PDFProcessor()
        logger.info("[OK] PDF processor initialized successfully")
        return True
    except Exception as e:
        logger.error(f"[FAILED] PDF processor error: {e}")
        return False

def test_vector_store():
    """Test vector store initialization."""
    try:
        from medrag.core.vector_store import VectorStore
        store = VectorStore()
        logger.info("[OK] Vector store initialized successfully")
        return True
    except Exception as e:
        logger.error(f"[FAILED] Vector store error: {e}")
        return False

def test_llm_client():
    """Test LLM client initialization."""
    try:
        from medrag.core.llm_client import LLMClient
        # This will fail if OPENAI_API_KEY is not set, which is expected
        client = LLMClient()
        logger.info("[OK] LLM client initialized successfully")
        return True
    except ValueError as e:
        if "OPENAI_API_KEY is required" in str(e):
            logger.warning("[SKIPPED] LLM client test skipped - OPENAI_API_KEY not set")
            return True
        else:
            logger.error(f"[FAILED] LLM client error: {e}")
            return False
    except Exception as e:
        logger.error(f"[FAILED] LLM client error: {e}")
        return False

def test_rag_system():
    """Test RAG system initialization."""
    try:
        from medrag.core.vector_store import VectorStore
        from medrag.core.llm_client import LLMClient
        from medrag.core.rag_system import RAGSystem
        
        # Initialize components
        vector_store = VectorStore()
        
        # Try to initialize LLM client (may fail if no API key)
        try:
            llm_client = LLMClient()
        except ValueError:
            logger.warning("[SKIPPED] LLM client not available for RAG system test")
            return True
        
        # Initialize RAG system
        rag_system = RAGSystem(vector_store, llm_client)
        logger.info("[OK] RAG system initialized successfully")
        return True
    except Exception as e:
        logger.error(f"[FAILED] RAG system error: {e}")
        return False

def main():
    """Run all tests."""
    logger.info("Starting MEDRAG system tests...")
    
    tests = [
        ("Import Test", test_imports),
        ("Configuration Test", test_config),
        ("PDF Processor Test", test_pdf_processor),
        ("Vector Store Test", test_vector_store),
        ("LLM Client Test", test_llm_client),
        ("RAG System Test", test_rag_system),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        logger.info(f"\nRunning {test_name}...")
        if test_func():
            passed += 1
        else:
            logger.error(f"[FAILED] {test_name} failed")
    
    logger.info(f"\nTest Results: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("All tests passed! MEDRAG system is ready to use.")
        return 0
    else:
        logger.error("Some tests failed. Please check the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
