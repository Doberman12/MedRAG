#!/usr/bin/env python3
"""
Command-line interface for MEDRAG application.
"""

import argparse
import sys
import logging
from pathlib import Path
from typing import List, Optional

from medrag.core.pdf_processor import PDFProcessor
from medrag.core.vector_store import VectorStore
from medrag.core.llm_client import LLMClient
from medrag.core.rag_system import RAGSystem
from medrag.utils.config import Config

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MEDRAGCLI:
    """Command-line interface for MEDRAG application."""
    
    def __init__(self):
        """Initialize CLI application."""
        self.config = Config()
        self.pdf_processor = PDFProcessor()
        self.vector_store = VectorStore()
        
        # Initialize LLM client (may fail if no API key)
        try:
            self.llm_client = LLMClient()
            self.rag_system = RAGSystem(self.vector_store, self.llm_client)
            self.llm_available = True
        except ValueError as e:
            logger.warning(f"LLM client not available: {e}")
            self.llm_available = False
            self.rag_system = None
    
    def add_documents(self, pdf_paths: List[Path]) -> bool:
        """Add PDF documents to the vector store."""
        if not pdf_paths:
            logger.error("No PDF files provided")
            return False
        
        success_count = 0
        for pdf_path in pdf_paths:
            if not pdf_path.exists():
                logger.error(f"File not found: {pdf_path}")
                continue
            
            if not pdf_path.suffix.lower() == '.pdf':
                logger.warning(f"Skipping non-PDF file: {pdf_path}")
                continue
            
            try:
                logger.info(f"Processing {pdf_path.name}...")
                chunks = self.pdf_processor.process_pdf(pdf_path)
                
                if chunks:
                    self.vector_store.add_documents(chunks, pdf_path.name)
                    logger.info(f"✅ Successfully processed {pdf_path.name} ({len(chunks)} chunks)")
                    success_count += 1
                else:
                    logger.warning(f"No content extracted from {pdf_path.name}")
                    
            except Exception as e:
                logger.error(f"❌ Error processing {pdf_path.name}: {e}")
        
        logger.info(f"📊 Processed {success_count}/{len(pdf_paths)} documents successfully")
        return success_count > 0
    
    def query_documents(self, question: str, top_k: int = 5, temperature: float = 0.7) -> Optional[dict]:
        """Query documents with a question."""
        if not self.llm_available:
            logger.error("❌ LLM client not available. Please set OPENAI_API_KEY environment variable.")
            return None
        
        try:
            logger.info(f"🔍 Searching for: {question}")
            response = self.rag_system.query(
                query=question,
                top_k=top_k,
                temperature=temperature
            )
            
            return response
            
        except Exception as e:
            logger.error(f"❌ Error querying documents: {e}")
            return None
    
    def list_documents(self):
        """List all documents in the vector store."""
        doc_count = self.vector_store.get_document_count()
        documents = self.vector_store.get_document_list()
        
        print(f"\nDocument Store Status:")
        print(f"   Total documents: {doc_count}")
        print(f"   Unique sources: {len(documents)}")
        
        if documents:
            print(f"\nDocuments:")
            for i, doc in enumerate(documents, 1):
                print(f"   {i}. {doc}")
        else:
            print("   No documents found.")
    
    def clear_documents(self):
        """Clear all documents from the vector store."""
        try:
            self.vector_store.clear_all()
            logger.info("[OK] All documents cleared from vector store")
        except Exception as e:
            logger.error(f"[FAILED] Error clearing documents: {e}")
    
    def system_stats(self):
        """Display system statistics."""
        try:
            if not self.rag_system:
                print(f"\nSystem Statistics:")
                print(f"   Vector Store:")
                doc_count = self.vector_store.get_document_count()
                documents = self.vector_store.get_document_list()
                print(f"     - Total documents: {doc_count}")
                print(f"     - Unique sources: {len(documents)}")
                print(f"     - Embedding model: {self.config.embedding_model}")
                print(f"\n   Model Configuration:")
                print(f"     - LLM model: Not available (OPENAI_API_KEY not set)")
                print(f"     - Max tokens: {self.config.openai_max_tokens}")
                print(f"     - Temperature: {self.config.openai_temperature}")
                return
                
            stats = self.rag_system.get_system_stats()
            
            print(f"\nSystem Statistics:")
            print(f"   Vector Store:")
            print(f"     - Total documents: {stats.get('vector_store', {}).get('total_documents', 0)}")
            print(f"     - Unique sources: {stats.get('vector_store', {}).get('unique_sources', 0)}")
            print(f"     - Embedding model: {stats.get('vector_store', {}).get('embedding_model', 'N/A')}")
            
            print(f"\n   Model Configuration:")
            print(f"     - LLM model: {stats.get('model', {}).get('model', 'N/A')}")
            print(f"     - Max tokens: {stats.get('model', {}).get('max_tokens', 'N/A')}")
            print(f"     - Temperature: {stats.get('model', {}).get('temperature', 'N/A')}")
            
            print(f"\n   RAG Configuration:")
            config = stats.get('config', {})
            print(f"     - Similarity threshold: {config.get('similarity_threshold', 'N/A')}")
            print(f"     - Top-K retrieval: {config.get('top_k_retrieval', 'N/A')}")
            print(f"     - Chunk size: {config.get('chunk_size', 'N/A')}")
            print(f"     - Chunk overlap: {config.get('chunk_overlap', 'N/A')}")
            
        except Exception as e:
            logger.error(f"[FAILED] Error getting system stats: {e}")
    
    def test_system(self):
        """Test the complete system."""
        try:
            if not self.rag_system:
                print(f"\nSystem Test Results:")
                print(f"   Vector Store: [OK]")
                print(f"   LLM Client: [FAILED] (OPENAI_API_KEY not set)")
                print(f"   Overall System: [WARNING] (Limited functionality)")
                return
                
            test_results = self.rag_system.test_system()
            
            print(f"\nSystem Test Results:")
            print(f"   Vector Store: {'[OK]' if test_results.get('vector_store') else '[FAILED]'}")
            print(f"   LLM Client: {'[OK]' if test_results.get('llm_client') else '[FAILED]'}")
            print(f"   Overall System: {'[OK]' if test_results.get('overall') else '[FAILED]'}")
            
            if 'error' in test_results:
                print(f"   Error: {test_results['error']}")
                
        except Exception as e:
            logger.error(f"[FAILED] Error testing system: {e}")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="MEDRAG - Medical RAG (Retrieval-Augmented Generation) CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Add PDF documents
  python cli.py add docs/medical_report.pdf docs/patient_notes.pdf
  
  # Query documents
  python cli.py query "What are the symptoms of diabetes?"
  
  # List documents
  python cli.py list
  
  # Get system stats
  python cli.py stats
  
  # Test system
  python cli.py test
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Add documents command
    add_parser = subparsers.add_parser('add', help='Add PDF documents to the vector store')
    add_parser.add_argument('pdf_files', nargs='+', type=Path, help='PDF files to process')
    
    # Query command
    query_parser = subparsers.add_parser('query', help='Query documents with a question')
    query_parser.add_argument('question', type=str, help='Question to ask about the documents')
    query_parser.add_argument('--top-k', type=int, default=5, help='Number of documents to retrieve (default: 5)')
    query_parser.add_argument('--temperature', type=float, default=0.7, help='Response creativity (0-1, default: 0.7)')
    
    # List documents command
    subparsers.add_parser('list', help='List all documents in the vector store')
    
    # Clear documents command
    subparsers.add_parser('clear', help='Clear all documents from the vector store')
    
    # Stats command
    subparsers.add_parser('stats', help='Display system statistics')
    
    # Test command
    subparsers.add_parser('test', help='Test the complete system')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    # Initialize CLI
    cli = MEDRAGCLI()
    
    try:
        if args.command == 'add':
            success = cli.add_documents(args.pdf_files)
            return 0 if success else 1
            
        elif args.command == 'query':
            response = cli.query_documents(args.question, args.top_k, args.temperature)
            if response:
                print(f"\nAI Response:")
                print(f"{response['answer']}")
                
                if response.get('sources'):
                    print(f"\nSources:")
                    for i, source in enumerate(response['sources'], 1):
                        print(f"   {i}. {source['document']} (Score: {source['score']:.3f})")
                        print(f"      {source['content'][:200]}...")
                
                print(f"\nProcessing time: {response['processing_time']:.2f}s")
                print(f"Documents retrieved: {response['documents_retrieved']}")
                print(f"Documents used: {response['documents_used']}")
                return 0
            else:
                return 1
                
        elif args.command == 'list':
            cli.list_documents()
            return 0
            
        elif args.command == 'clear':
            cli.clear_documents()
            return 0
            
        elif args.command == 'stats':
            cli.system_stats()
            return 0
            
        elif args.command == 'test':
            cli.test_system()
            return 0
            
    except KeyboardInterrupt:
        logger.info("\nOperation cancelled by user")
        return 1
    except Exception as e:
        logger.error(f"[FAILED] Unexpected error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
