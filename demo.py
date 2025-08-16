#!/usr/bin/env python3
"""
Demo script for MEDRAG application.
This script demonstrates how to use the MEDRAG system programmatically.
"""

import os
import sys
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def demo_basic_usage():
    """Demonstrate basic MEDRAG usage."""
    print("MEDRAG Demo - Basic Usage")
    print("=" * 50)
    
    try:
        from medrag.core.pdf_processor import PDFProcessor
        from medrag.core.vector_store import VectorStore
        from medrag.core.llm_client import LLMClient
        from medrag.core.rag_system import RAGSystem
        from medrag.utils.config import Config
        
        # Initialize components
        print("1. Initializing components...")
        config = Config()
        pdf_processor = PDFProcessor()
        vector_store = VectorStore()
        
        # Try to initialize LLM client
        try:
            llm_client = LLMClient()
            rag_system = RAGSystem(vector_store, llm_client)
            llm_available = True
            print("   [OK] LLM client initialized")
        except ValueError:
            print("   [WARNING] LLM client not available (OPENAI_API_KEY not set)")
            llm_available = False
            rag_system = None
        
        print("   [OK] PDF processor initialized")
        print("   [OK] Vector store initialized")
        
        # Show system stats
        print("\n2. System Statistics:")
        doc_count = vector_store.get_document_count()
        print(f"   - Documents in store: {doc_count}")
        print(f"   - Embedding model: {config.embedding_model}")
        print(f"   - Chunk size: {config.chunk_size}")
        print(f"   - Similarity threshold: {config.similarity_threshold}")
        
        # Demonstrate document processing (if we had a sample PDF)
        print("\n3. Document Processing:")
        print("   To process a PDF document, you would use:")
        print("   ```python")
        print("   pdf_path = Path('your_medical_document.pdf')")
        print("   chunks = pdf_processor.process_pdf(pdf_path)")
        print("   vector_store.add_documents(chunks, pdf_path.name)")
        print("   ```")
        
        # Demonstrate querying (if LLM is available)
        if llm_available:
            print("\n4. Querying Documents:")
            print("   To query documents, you would use:")
            print("   ```python")
            print("   response = rag_system.query('What are the symptoms of diabetes?')")
            print("   print(response['answer'])")
            print("   ```")
        else:
            print("\n4. Querying Documents:")
            print("   [WARNING] LLM not available - set OPENAI_API_KEY to enable querying")
            print("   Example query: 'What are the symptoms of diabetes?'")
        
        # Show available methods
        print("\n5. Available Methods:")
        print("   - pdf_processor.process_pdf(pdf_path)")
        print("   - vector_store.add_documents(chunks, source_name)")
        print("   - vector_store.search(query, top_k=5)")
        print("   - vector_store.get_document_count()")
        print("   - vector_store.get_document_list()")
        if llm_available:
            print("   - rag_system.query(question, top_k=5, temperature=0.7)")
            print("   - rag_system.get_relevant_documents(query)")
            print("   - rag_system.explain_retrieval(query)")
        
        print("\n[OK] Demo completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Demo failed: {e}")
        return False
    
    return True

def demo_cli_usage():
    """Demonstrate CLI usage."""
    print("\nMEDRAG Demo - CLI Usage")
    print("=" * 50)
    
    print("Available CLI commands:")
    print("1. Add documents:")
    print("   uv run python cli.py add path/to/medical_document.pdf")
    
    print("\n2. Query documents:")
    print("   uv run python cli.py query 'What are the symptoms of diabetes?'")
    
    print("\n3. List documents:")
    print("   uv run python cli.py list")
    
    print("\n4. System statistics:")
    print("   uv run python cli.py stats")
    
    print("\n5. Test system:")
    print("   uv run python cli.py test")
    
    print("\n6. Clear all documents:")
    print("   uv run python cli.py clear")

def demo_web_interface():
    """Demonstrate web interface usage."""
    print("\nMEDRAG Demo - Web Interface")
    print("=" * 50)
    
    print("To start the web interface:")
    print("1. Set your OpenAI API key:")
    print("   export OPENAI_API_KEY='your_api_key_here'")
    print("   # or create a .env file with OPENAI_API_KEY=your_api_key_here")
    
    print("\n2. Run the Streamlit app:")
    print("   uv run streamlit run main.py")
    
    print("\n3. Open your browser to http://localhost:8501")
    
    print("\n4. Features available in the web interface:")
    print("   - Upload multiple PDF documents")
    print("   - Interactive query interface")
    print("   - View source documents and relevance scores")
    print("   - Adjust retrieval parameters")
    print("   - System statistics and monitoring")

def main():
    """Run the complete demo."""
    print("Welcome to MEDRAG (Medical RAG) Demo!")
    print("This demo shows how to use the MEDRAG system for medical document querying.")
    print()
    
    # Run demos
    demo_basic_usage()
    demo_cli_usage()
    demo_web_interface()
    
    print("\n" + "=" * 60)
    print("Demo completed!")
    print("\nNext steps:")
    print("1. Set your OPENAI_API_KEY environment variable")
    print("2. Add some medical PDF documents")
    print("3. Start querying your documents!")
    print("\nFor more information, see the README.md file.")

if __name__ == "__main__":
    main()
