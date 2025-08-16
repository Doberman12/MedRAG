"""
RAG (Retrieval-Augmented Generation) system for MEDRAG application.
"""

import logging
from typing import List, Dict, Any, Optional
import time

from medrag.core.vector_store import VectorStore
from medrag.core.llm_client import LLMClient
from medrag.utils.config import Config

logger = logging.getLogger(__name__)


class RAGSystem:
    """Orchestrates the RAG pipeline combining retrieval and generation."""
    
    def __init__(self, vector_store: VectorStore, llm_client: LLMClient, config: Config = None):
        """Initialize RAG system."""
        self.vector_store = vector_store
        self.llm_client = llm_client
        self.config = config or Config()
        
        logger.info("RAG system initialized successfully")
    
    def query(
        self,
        query: str,
        top_k: int = None,
        temperature: float = None,
        filter_dict: Dict = None,
        include_sources: bool = True
    ) -> Dict[str, Any]:
        """Complete RAG pipeline: retrieve relevant documents and generate response."""
        start_time = time.time()
        
        try:
            # Step 1: Validate query
            query_validation = self.llm_client.validate_medical_query(query)
            
            # Step 2: Retrieve relevant documents
            retrieved_docs = self.vector_store.search(
                query=query,
                top_k=top_k or self.config.top_k_retrieval,
                filter_dict=filter_dict
            )
            
            # Step 3: Filter documents by similarity threshold
            filtered_docs = self._filter_by_similarity(retrieved_docs)
            
            # Step 4: Generate response
            response = self.llm_client.generate_response(
                question=query,
                context=filtered_docs,
                temperature=temperature,
                max_tokens=self.config.openai_max_tokens
            )
            
            # Step 5: Prepare final response
            final_response = {
                'answer': response['answer'],
                'query': query,
                'processing_time': time.time() - start_time,
                'documents_retrieved': len(retrieved_docs),
                'documents_used': len(filtered_docs),
                'query_validation': query_validation,
                'model_info': response.get('model_info', {}),
                'tokens_used': response.get('tokens_used', 0)
            }
            
            # Add sources if requested
            if include_sources and response.get('sources'):
                final_response['sources'] = response['sources']
            
            logger.info(f"RAG query completed in {final_response['processing_time']:.2f}s")
            return final_response
            
        except Exception as e:
            logger.error(f"Error in RAG query: {e}")
            return {
                'answer': f"Sorry, I encountered an error while processing your query: {str(e)}",
                'error': str(e),
                'processing_time': time.time() - start_time
            }
    
    def _filter_by_similarity(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter documents by similarity threshold."""
        filtered = []
        for doc in documents:
            score = doc.get('score', 0)
            if score >= self.config.similarity_threshold:
                filtered.append(doc)
        
        # If no documents meet the threshold, return the top ones anyway
        if not filtered and documents:
            logger.warning(f"No documents met similarity threshold {self.config.similarity_threshold}, using top documents")
            filtered = documents[:3]  # Use top 3 as fallback
        
        return filtered
    
    def batch_query(
        self,
        queries: List[str],
        top_k: int = None,
        temperature: float = None
    ) -> List[Dict[str, Any]]:
        """Process multiple queries in batch."""
        results = []
        
        for i, query in enumerate(queries):
            logger.info(f"Processing batch query {i+1}/{len(queries)}")
            result = self.query(query, top_k, temperature)
            result['query_index'] = i
            results.append(result)
        
        return results
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get comprehensive system statistics."""
        try:
            vector_stats = self.vector_store.get_statistics()
            model_info = self.llm_client.get_model_info()
            
            return {
                'vector_store': vector_stats,
                'model': model_info,
                'config': {
                    'similarity_threshold': self.config.similarity_threshold,
                    'top_k_retrieval': self.config.top_k_retrieval,
                    'chunk_size': self.config.chunk_size,
                    'chunk_overlap': self.config.chunk_overlap
                }
            }
        except Exception as e:
            logger.error(f"Error getting system stats: {e}")
            return {}
    
    def test_system(self) -> Dict[str, Any]:
        """Test the complete RAG system."""
        test_results = {
            'vector_store': False,
            'llm_client': False,
            'overall': False
        }
        
        try:
            # Test vector store
            doc_count = self.vector_store.get_document_count()
            test_results['vector_store'] = True
            
            # Test LLM client
            llm_test = self.llm_client.test_connection()
            test_results['llm_client'] = llm_test
            
            # Test complete pipeline if documents exist
            if doc_count > 0:
                test_query = "What is the main topic of these documents?"
                test_response = self.query(test_query, top_k=1)
                test_results['overall'] = 'answer' in test_response
            else:
                test_results['overall'] = True  # No documents to test with
                
        except Exception as e:
            logger.error(f"System test failed: {e}")
            test_results['error'] = str(e)
        
        return test_results
    
    def optimize_query(self, query: str) -> str:
        """Optimize query for better retrieval."""
        # Remove common stop words that don't add semantic value
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those'
        }
        
        words = query.lower().split()
        optimized_words = [word for word in words if word not in stop_words and len(word) > 2]
        
        return ' '.join(optimized_words) if optimized_words else query
    
    def get_relevant_documents(self, query: str, top_k: int = None) -> List[Dict[str, Any]]:
        """Get relevant documents without generating a response."""
        try:
            # Optimize query
            optimized_query = self.optimize_query(query)
            
            # Retrieve documents
            documents = self.vector_store.search(
                query=optimized_query,
                top_k=top_k or self.config.top_k_retrieval
            )
            
            # Filter by similarity
            filtered_docs = self._filter_by_similarity(documents)
            
            return filtered_docs
            
        except Exception as e:
            logger.error(f"Error retrieving relevant documents: {e}")
            return []
    
    def explain_retrieval(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """Explain why certain documents were retrieved for a query."""
        try:
            # Get relevant documents
            documents = self.get_relevant_documents(query, top_k)
            
            # Analyze retrieval
            explanation = {
                'query': query,
                'optimized_query': self.optimize_query(query),
                'documents_retrieved': len(documents),
                'similarity_threshold': self.config.similarity_threshold,
                'document_analysis': []
            }
            
            for i, doc in enumerate(documents):
                analysis = {
                    'rank': i + 1,
                    'source': doc.get('metadata', {}).get('source', 'Unknown'),
                    'similarity_score': doc.get('score', 0),
                    'content_preview': doc.get('content', '')[:200] + "..." if len(doc.get('content', '')) > 200 else doc.get('content', ''),
                    'chunk_id': doc.get('metadata', {}).get('chunk_id', 'N/A')
                }
                explanation['document_analysis'].append(analysis)
            
            return explanation
            
        except Exception as e:
            logger.error(f"Error explaining retrieval: {e}")
            return {'error': str(e)}
