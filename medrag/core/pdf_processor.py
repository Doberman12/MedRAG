"""
PDF processing module for MEDRAG application.
"""

import logging
from pathlib import Path
from typing import List, Dict, Any
import re

from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

from medrag.utils.config import Config

logger = logging.getLogger(__name__)


class PDFProcessor:
    """Handles PDF document processing and text chunking."""
    
    def __init__(self, config: Config = None):
        """Initialize PDF processor."""
        self.config = config or Config()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
    
    def extract_text_from_pdf(self, pdf_path: Path) -> str:
        """Extract text content from PDF file."""
        try:
            reader = PdfReader(pdf_path)
            
            # Limit number of pages to process
            max_pages = min(len(reader.pages), self.config.max_pdf_pages)
            
            text_content = []
            for page_num in range(max_pages):
                try:
                    page = reader.pages[page_num]
                    text = page.extract_text()
                    if text.strip():
                        text_content.append(text)
                except Exception as e:
                    logger.warning(f"Error extracting text from page {page_num}: {e}")
                    continue
            
            full_text = "\n\n".join(text_content)
            
            # Clean up the text
            cleaned_text = self._clean_text(full_text)
            
            logger.info(f"Extracted {len(cleaned_text)} characters from {pdf_path.name}")
            return cleaned_text
            
        except Exception as e:
            logger.error(f"Error processing PDF {pdf_path}: {e}")
            raise
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize extracted text."""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove page numbers and headers/footers
        text = re.sub(r'\b\d+\s*of\s*\d+\b', '', text)  # "Page X of Y"
        text = re.sub(r'\bPage\s+\d+\b', '', text)      # "Page X"
        
        # Remove common PDF artifacts
        text = re.sub(r'[^\w\s\.\,\;\:\!\?\-\(\)\[\]\{\}]', '', text)
        
        # Normalize medical abbreviations and terms
        text = self._normalize_medical_terms(text)
        
        return text.strip()
    
    def _normalize_medical_terms(self, text: str) -> str:
        """Normalize common medical terms and abbreviations."""
        # Common medical abbreviations mapping
        medical_abbreviations = {
            'BP': 'blood pressure',
            'HR': 'heart rate',
            'RR': 'respiratory rate',
            'Temp': 'temperature',
            'WBC': 'white blood cell count',
            'RBC': 'red blood cell count',
            'Hb': 'hemoglobin',
            'Hct': 'hematocrit',
            'Na+': 'sodium',
            'K+': 'potassium',
            'Cl-': 'chloride',
            'CO2': 'carbon dioxide',
            'BUN': 'blood urea nitrogen',
            'Cr': 'creatinine',
            'Gluc': 'glucose',
            'Ca++': 'calcium',
            'Mg++': 'magnesium',
            'PO4': 'phosphate',
        }
        
        # Replace abbreviations with full terms
        for abbr, full_term in medical_abbreviations.items():
            text = re.sub(rf'\b{re.escape(abbr)}\b', full_term, text, flags=re.IGNORECASE)
        
        return text
    
    def chunk_text(self, text: str, metadata: Dict[str, Any] = None) -> List[Document]:
        """Split text into chunks for vector storage."""
        if not text.strip():
            return []
        
        # Create initial document
        doc = Document(page_content=text, metadata=metadata or {})
        
        # Split into chunks
        chunks = self.text_splitter.split_documents([doc])
        
        # Filter out chunks that are too short
        filtered_chunks = []
        for chunk in chunks:
            if len(chunk.page_content.strip()) >= self.config.min_chunk_length:
                # Add chunk-specific metadata
                chunk.metadata.update({
                    'chunk_id': len(filtered_chunks),
                    'chunk_length': len(chunk.page_content),
                    'processed': True
                })
                filtered_chunks.append(chunk)
        
        logger.info(f"Created {len(filtered_chunks)} chunks from text")
        return filtered_chunks
    
    def process_pdf(self, pdf_path: Path) -> List[Document]:
        """Complete PDF processing pipeline."""
        # Extract text
        text = self.extract_text_from_pdf(pdf_path)
        
        # Create metadata
        from datetime import datetime
        metadata = {
            'source': pdf_path.name,
            'file_path': str(pdf_path),
            'file_size': pdf_path.stat().st_size,
            'processing_timestamp': str(datetime.now())
        }
        
        # Chunk text
        chunks = self.chunk_text(text, metadata)
        
        return chunks
    
    def get_pdf_info(self, pdf_path: Path) -> Dict[str, Any]:
        """Get basic information about a PDF file."""
        try:
            reader = PdfReader(pdf_path)
            return {
                'filename': pdf_path.name,
                'pages': len(reader.pages),
                'file_size': pdf_path.stat().st_size,
                'is_encrypted': reader.is_encrypted
            }
        except Exception as e:
            logger.error(f"Error getting PDF info for {pdf_path}: {e}")
            return {}
