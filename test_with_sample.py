#!/usr/bin/env python3
"""
Test MEDRAG with sample medical data (no PDF required).
"""

import os
import sys
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_with_sample_data():
    """Test MEDRAG with sample medical text."""
    print("Testing MEDRAG with Sample Medical Data")
    print("=" * 50)
    
    try:
        from medrag.core.vector_store import VectorStore
        from medrag.core.llm_client import LLMClient
        from medrag.core.rag_system import RAGSystem
        from langchain.schema import Document
        
        # Initialize components
        print("1. Initializing components...")
        vector_store = VectorStore()
        
        # Try to initialize LLM client
        try:
            llm_client = LLMClient()
            rag_system = RAGSystem(vector_store, llm_client)
            llm_available = True
            print("   [OK] LLM client initialized")
        except Exception as e:
            print(f"   [FAILED] LLM client error: {e}")
            print("   [INFO] Make sure you have:")
            print("      - Added your OpenAI API key to .env file")
            print("      - Added billing information to your OpenAI account")
            return False
        
        # Sample medical documents
        sample_docs = [
            Document(
                page_content="""
                Diabetes mellitus is a chronic metabolic disorder characterized by high blood glucose levels.
                Common symptoms include increased thirst, frequent urination, extreme hunger, unexplained weight loss,
                fatigue, blurred vision, and slow-healing sores. Type 2 diabetes is the most common form,
                often related to lifestyle factors such as obesity and physical inactivity.
                """,
                metadata={"source": "sample_diabetes_info", "chunk_id": 0}
            ),
            Document(
                page_content="""
                Hypertension, or high blood pressure, is a condition where the force of blood against artery walls is too high.
                Normal blood pressure is typically below 120/80 mmHg. Symptoms may include headaches, shortness of breath,
                nosebleeds, chest pain, dizziness, and vision problems. Treatment often involves lifestyle changes
                such as diet modification, exercise, and stress management, along with medications if necessary.
                """,
                metadata={"source": "sample_hypertension_info", "chunk_id": 1}
            ),
            Document(
                page_content="""
                Common medications for diabetes include Metformin, which helps reduce glucose production in the liver,
                and Insulin, which helps glucose enter cells. For hypertension, medications like ACE inhibitors,
                beta-blockers, and calcium channel blockers are commonly prescribed. Side effects may include
                dizziness, fatigue, and gastrointestinal issues. Regular monitoring of blood pressure and blood
                glucose levels is essential for effective management.
                """,
                metadata={"source": "sample_medications_info", "chunk_id": 2}
            )
        ]
        
        # Add sample documents to vector store
        print("\n2. Adding sample medical documents...")
        vector_store.add_documents(sample_docs, "sample_medical_data")
        print("   [OK] Added 3 sample documents")
        
        # Test queries
        test_queries = [
            "What are the symptoms of diabetes?",
            "What is hypertension and what are its symptoms?",
            "What medications are mentioned and what are their side effects?",
            "What are the normal blood pressure ranges?",
            "How is diabetes treated?"
        ]
        
        print("\n3. Testing queries...")
        for i, query in enumerate(test_queries, 1):
            print(f"\n   Query {i}: {query}")
            try:
                response = rag_system.query(query, top_k=3, temperature=0.7)
                print(f"   Answer: {response['answer'][:200]}...")
                print(f"   Processing time: {response['processing_time']:.2f}s")
                print(f"   Documents used: {response['documents_used']}")
            except Exception as e:
                print(f"   [FAILED] Error: {e}")
        
        print("\n[OK] Sample data test completed!")
        print("\n[INFO] Now you can:")
        print("   1. Add your own medical PDF documents")
        print("   2. Use the web interface: uv run streamlit run main.py")
        print("   3. Use the CLI: uv run python cli.py query 'your question'")
        
        return True
        
    except Exception as e:
        logger.error(f"[FAILED] Test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_with_sample_data()
    sys.exit(0 if success else 1)
