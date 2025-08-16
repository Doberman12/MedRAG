"""
MEDRAG - Medical RAG (Retrieval-Augmented Generation) Application

This application allows users to upload medical PDF documents and query them
using natural language through an LLM-powered interface.
"""

import os
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any
import streamlit as st
from dotenv import load_dotenv

from medrag.core.pdf_processor import PDFProcessor
from medrag.core.vector_store import VectorStore
from medrag.core.rag_system import RAGSystem
from medrag.core.llm_client import LLMClient
from medrag.utils.config import Config

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MEDRAGApp:
    """Main MEDRAG application class."""
    
    def __init__(self):
        """Initialize the MEDRAG application."""
        self.config = Config()
        self.pdf_processor = PDFProcessor()
        self.vector_store = VectorStore()
        self.llm_client = LLMClient()
        self.rag_system = RAGSystem(self.vector_store, self.llm_client)
        
    def setup_streamlit(self):
        """Setup Streamlit page configuration."""
        st.set_page_config(
            page_title="MEDRAG - Medical Document Q&A",
            page_icon="📄",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        st.title("MEDRAG - Medical Document Q&A System")
        st.markdown("""
        Upload medical PDF documents and ask questions about their content using AI.
        """)
        
    def upload_documents(self) -> List[Path]:
        """Handle document uploads."""
        st.header("Upload Medical Documents")
        
        uploaded_files = st.file_uploader(
            "Choose PDF files",
            type=['pdf'],
            accept_multiple_files=True,
            help="Upload one or more medical PDF documents"
        )
        
        if uploaded_files:
            st.success(f"Uploaded {len(uploaded_files)} document(s)")
            return uploaded_files
        return []
    
    def process_documents(self, uploaded_files: List[Any]):
        """Process uploaded documents and add to vector store."""
        if not uploaded_files:
            return
            
        with st.spinner("Processing documents..."):
            progress_bar = st.progress(0)
            
            for i, uploaded_file in enumerate(uploaded_files):
                try:
                    # Save uploaded file temporarily
                    temp_path = Path(f"temp_{uploaded_file.name}")
                    with open(temp_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    
                    # Process PDF
                    chunks = self.pdf_processor.process_pdf(temp_path)
                    
                    # Add to vector store
                    self.vector_store.add_documents(chunks, uploaded_file.name)
                    
                    # Clean up temp file
                    temp_path.unlink()
                    
                    progress_bar.progress((i + 1) / len(uploaded_files))
                    st.success(f"Processed: {uploaded_file.name}")
                    
                except Exception as e:
                    st.error(f"Error processing {uploaded_file.name}: {str(e)}")
                    logger.error(f"Error processing {uploaded_file.name}: {str(e)}")
    
    def query_interface(self):
        """Provide query interface for users."""
        st.header("Ask Questions About Your Documents")
        
        # Query input
        query = st.text_area(
            "Enter your medical question:",
            placeholder="e.g., What are the symptoms of diabetes?",
            height=100
        )
        
        # Advanced options
        with st.expander("Advanced Options"):
            col1, col2 = st.columns(2)
            with col1:
                top_k = st.slider("Number of relevant chunks", 3, 10, 5)
            with col2:
                temperature = st.slider("Response creativity", 0.0, 1.0, 0.7)
        
        # Submit button
        if st.button("Search & Answer", type="primary"):
            if query.strip():
                self.process_query(query, top_k, temperature)
            else:
                st.warning("Please enter a question.")
    
    def process_query(self, query: str, top_k: int, temperature: float):
        """Process user query and display results."""
        with st.spinner("Searching documents and generating answer..."):
            try:
                # Get RAG response
                response = self.rag_system.query(
                    query=query,
                    top_k=top_k,
                    temperature=temperature
                )
                
                # Display results
                st.subheader("AI Response")
                st.write(response['answer'])
                
                # Display sources
                if response.get('sources'):
                    st.subheader("Sources")
                    for i, source in enumerate(response['sources'], 1):
                        with st.expander(f"Source {i}: {source['document']}"):
                            st.write(source['content'])
                            st.caption(f"Relevance score: {source.get('score', 'N/A')}")
                            
            except Exception as e:
                st.error(f"Error processing query: {str(e)}")
                logger.error(f"Error processing query: {str(e)}")
    
    def show_document_info(self):
        """Display information about processed documents."""
        st.sidebar.header("Document Information")
        
        doc_count = self.vector_store.get_document_count()
        st.sidebar.metric("Documents Processed", doc_count)
        
        if doc_count > 0:
            documents = self.vector_store.get_document_list()
            st.sidebar.subheader("Processed Documents:")
            for doc in documents:
                st.sidebar.text(f"• {doc}")
    
    def run(self):
        """Run the MEDRAG application."""
        self.setup_streamlit()
        
        # Sidebar
        with st.sidebar:
            self.show_document_info()
            
            st.header("Settings")
            if st.button("Clear All Documents"):
                self.vector_store.clear_all()
                st.success("All documents cleared!")
                st.rerun()
        
        # Main content
        uploaded_files = self.upload_documents()
        
        if uploaded_files:
            self.process_documents(uploaded_files)
        
        self.query_interface()


def main():
    """Main entry point for the MEDRAG application."""
    try:
        app = MEDRAGApp()
        app.run()
    except Exception as e:
        st.error(f"Application error: {str(e)}")
        logger.error(f"Application error: {str(e)}")


if __name__ == "__main__":
    main()
