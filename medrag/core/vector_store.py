"""
Vector store module for MEDRAG application.
"""

import logging
import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import uuid

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import numpy as np

from langchain.schema import Document
from medrag.utils.config import Config

logger = logging.getLogger(__name__)


class VectorStore:
    """Manages document embeddings and similarity search."""
    
    def __init__(self, config: Config = None):
        """Initialize vector store."""
        self.config = config or Config()
        self.embedding_model = SentenceTransformer(self.config.embedding_model)
        
        # Initialize ChromaDB
        self._init_chroma_db()
        
        # Create collection
        self.collection = self.client.get_or_create_collection(
            name="medrag_documents",
            metadata={"hnsw:space": "cosine"}
        )
        
        logger.info("Vector store initialized successfully")
    
    def _init_chroma_db(self):
        """Initialize ChromaDB client."""
        try:
            # Create vector store directory if it doesn't exist
            os.makedirs(self.config.vector_store_path, exist_ok=True)
            
            # Initialize ChromaDB client
            self.client = chromadb.PersistentClient(
                path=self.config.vector_store_path,
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            logger.info(f"ChromaDB initialized at {self.config.vector_store_path}")
            
        except Exception as e:
            logger.error(f"Error initializing ChromaDB: {e}")
            raise
    
    def add_documents(self, documents: List[Document], source_name: str = None):
        """Add documents to the vector store."""
        if not documents:
            logger.warning("No documents to add")
            return
        
        try:
            # Prepare documents for insertion
            ids = []
            texts = []
            metadatas = []
            
            for doc in documents:
                # Generate unique ID
                doc_id = str(uuid.uuid4())
                ids.append(doc_id)
                texts.append(doc.page_content)
                
                # Prepare metadata
                metadata = doc.metadata.copy()
                if source_name:
                    metadata['source'] = source_name
                metadata['doc_id'] = doc_id
                metadatas.append(metadata)
            
            # Add to collection
            self.collection.add(
                documents=texts,
                metadatas=metadatas,
                ids=ids
            )
            
            logger.info(f"Added {len(documents)} documents to vector store")
            
        except Exception as e:
            logger.error(f"Error adding documents to vector store: {e}")
            raise
    
    def search(self, query: str, top_k: int = None, filter_dict: Dict = None) -> List[Dict[str, Any]]:
        """Search for similar documents."""
        if top_k is None:
            top_k = self.config.top_k_retrieval
        
        try:
            # Perform similarity search
            results = self.collection.query(
                query_texts=[query],
                n_results=top_k,
                where=filter_dict
            )
            
            # Format results
            formatted_results = []
            if results['documents'] and results['documents'][0]:
                for i, (doc, metadata, distance) in enumerate(zip(
                    results['documents'][0],
                    results['metadatas'][0],
                    results['distances'][0]
                )):
                    # Convert distance to similarity score (ChromaDB uses cosine distance)
                    similarity_score = 1 - distance
                    
                    formatted_results.append({
                        'content': doc,
                        'metadata': metadata,
                        'score': similarity_score,
                        'rank': i + 1
                    })
            
            logger.info(f"Found {len(formatted_results)} similar documents for query")
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error searching vector store: {e}")
            raise
    
    def get_document_count(self) -> int:
        """Get total number of documents in the store."""
        try:
            return self.collection.count()
        except Exception as e:
            logger.error(f"Error getting document count: {e}")
            return 0
    
    def get_document_list(self) -> List[str]:
        """Get list of unique document sources."""
        try:
            # Get all documents
            results = self.collection.get()
            
            # Extract unique sources
            sources = set()
            for metadata in results['metadatas']:
                if metadata and 'source' in metadata:
                    sources.add(metadata['source'])
            
            return list(sources)
            
        except Exception as e:
            logger.error(f"Error getting document list: {e}")
            return []
    
    def delete_documents(self, filter_dict: Dict = None):
        """Delete documents from the store."""
        try:
            if filter_dict:
                self.collection.delete(where=filter_dict)
                logger.info(f"Deleted documents matching filter: {filter_dict}")
            else:
                self.collection.delete()
                logger.info("Deleted all documents from vector store")
                
        except Exception as e:
            logger.error(f"Error deleting documents: {e}")
            raise
    
    def clear_all(self):
        """Clear all documents from the vector store."""
        try:
            self.collection.delete()
            logger.info("Cleared all documents from vector store")
        except Exception as e:
            logger.error(f"Error clearing vector store: {e}")
            raise
    
    def get_document_by_id(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific document by ID."""
        try:
            results = self.collection.get(ids=[doc_id])
            
            if results['documents'] and results['documents'][0]:
                return {
                    'content': results['documents'][0],
                    'metadata': results['metadatas'][0] if results['metadatas'] else {}
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting document by ID {doc_id}: {e}")
            return None
    
    def update_document(self, doc_id: str, new_content: str, new_metadata: Dict = None):
        """Update an existing document."""
        try:
            # Delete old document
            self.collection.delete(ids=[doc_id])
            
            # Add updated document
            metadata = new_metadata or {}
            metadata['doc_id'] = doc_id
            metadata['updated'] = True
            
            self.collection.add(
                documents=[new_content],
                metadatas=[metadata],
                ids=[doc_id]
            )
            
            logger.info(f"Updated document {doc_id}")
            
        except Exception as e:
            logger.error(f"Error updating document {doc_id}: {e}")
            raise
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get vector store statistics."""
        try:
            total_docs = self.get_document_count()
            sources = self.get_document_list()
            
            return {
                'total_documents': total_docs,
                'unique_sources': len(sources),
                'sources': sources,
                'embedding_model': self.config.embedding_model,
                'vector_store_path': self.config.vector_store_path
            }
            
        except Exception as e:
            logger.error(f"Error getting statistics: {e}")
            return {}
