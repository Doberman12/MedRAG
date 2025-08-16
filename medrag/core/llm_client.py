"""
LLM client module for MEDRAG application.
"""

import logging
from typing import List, Dict, Any, Optional
import openai
from openai import OpenAI

from medrag.utils.config import Config

logger = logging.getLogger(__name__)


class LLMClient:
    """Handles communication with OpenAI language models."""
    
    def __init__(self, config: Config = None):
        """Initialize LLM client."""
        self.config = config or Config()
        self.config.validate()
        
        # Initialize OpenAI client
        self.client = OpenAI(api_key=self.config.openai_api_key)
        
        # Medical context prompt template
        self.medical_prompt_template = """
You are a medical AI assistant with access to medical documents. Your role is to help answer questions based on the provided medical information.

IMPORTANT GUIDELINES:
1. Only provide information that is explicitly stated in the provided medical documents
2. If the information is not available in the documents, clearly state this
3. Always cite the source document when providing information
4. Be precise and accurate with medical terminology
5. If you're unsure about any medical information, recommend consulting a healthcare professional
6. Do not provide medical advice beyond what's in the documents

Context from medical documents:
{context}

User Question: {question}

Please provide a comprehensive answer based on the medical documents provided:
"""
        
        logger.info("LLM client initialized successfully")
    
    def generate_response(
        self,
        question: str,
        context: List[Dict[str, Any]],
        temperature: float = None,
        max_tokens: int = None
    ) -> Dict[str, Any]:
        """Generate a response based on the question and context."""
        try:
            # Prepare context
            context_text = self._prepare_context(context)
            
            # Prepare prompt
            prompt = self.medical_prompt_template.format(
                context=context_text,
                question=question
            )
            
            # Set parameters
            temp = temperature if temperature is not None else self.config.openai_temperature
            max_toks = max_tokens if max_tokens is not None else self.config.openai_max_tokens
            
            # Generate response
            response = self.client.chat.completions.create(
                model=self.config.openai_model,
                messages=[
                    {"role": "system", "content": "You are a medical AI assistant that provides accurate information based on medical documents."},
                    {"role": "user", "content": prompt}
                ],
                temperature=temp,
                max_tokens=max_toks,
                top_p=0.9,
                frequency_penalty=0.1,
                presence_penalty=0.1
            )
            
            # Extract response
            answer = response.choices[0].message.content.strip()
            
            # Prepare sources
            sources = self._extract_sources(context)
            
            return {
                'answer': answer,
                'sources': sources,
                'model': self.config.openai_model,
                'tokens_used': response.usage.total_tokens,
                'prompt_tokens': response.usage.prompt_tokens,
                'completion_tokens': response.usage.completion_tokens
            }
            
        except Exception as e:
            logger.error(f"Error generating LLM response: {e}")
            raise
    
    def _prepare_context(self, context: List[Dict[str, Any]]) -> str:
        """Prepare context from retrieved documents."""
        if not context:
            return "No relevant medical documents found."
        
        context_parts = []
        for i, doc in enumerate(context, 1):
            content = doc.get('content', '').strip()
            metadata = doc.get('metadata', {})
            source = metadata.get('source', 'Unknown source')
            score = doc.get('score', 0)
            
            context_part = f"Document {i} (Source: {source}, Relevance: {score:.3f}):\n{content}\n"
            context_parts.append(context_part)
        
        return "\n".join(context_parts)
    
    def _extract_sources(self, context: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract source information from context."""
        sources = []
        for doc in context:
            metadata = doc.get('metadata', {})
            source_info = {
                'document': metadata.get('source', 'Unknown'),
                'content': doc.get('content', '')[:500] + "..." if len(doc.get('content', '')) > 500 else doc.get('content', ''),
                'score': doc.get('score', 0),
                'chunk_id': metadata.get('chunk_id', 'N/A'),
                'file_path': metadata.get('file_path', 'N/A')
            }
            sources.append(source_info)
        
        return sources
    
    def validate_medical_query(self, query: str) -> Dict[str, Any]:
        """Validate if a query is medical-related."""
        try:
            # Medical keywords for validation
            medical_keywords = [
                'symptom', 'diagnosis', 'treatment', 'medication', 'disease', 'condition',
                'patient', 'clinical', 'medical', 'health', 'doctor', 'nurse', 'hospital',
                'blood', 'test', 'lab', 'prescription', 'dose', 'dosage', 'side effect',
                'allergy', 'infection', 'pain', 'fever', 'cough', 'headache', 'nausea',
                'diabetes', 'hypertension', 'cancer', 'heart', 'lung', 'kidney', 'liver',
                'brain', 'bone', 'muscle', 'skin', 'eye', 'ear', 'nose', 'throat'
            ]
            
            query_lower = query.lower()
            found_keywords = [keyword for keyword in medical_keywords if keyword in query_lower]
            
            is_medical = len(found_keywords) > 0
            
            return {
                'is_medical': is_medical,
                'found_keywords': found_keywords,
                'confidence': len(found_keywords) / len(medical_keywords) if medical_keywords else 0
            }
            
        except Exception as e:
            logger.error(f"Error validating medical query: {e}")
            return {'is_medical': True, 'found_keywords': [], 'confidence': 0}
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model."""
        return {
            'model': self.config.openai_model,
            'max_tokens': self.config.openai_max_tokens,
            'temperature': self.config.openai_temperature,
            'embedding_model': self.config.embedding_model
        }
    
    def test_connection(self) -> bool:
        """Test the connection to OpenAI API."""
        try:
            # Simple test request
            response = self.client.chat.completions.create(
                model=self.config.openai_model,
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=5
            )
            return True
        except Exception as e:
            logger.error(f"OpenAI API connection test failed: {e}")
            return False
