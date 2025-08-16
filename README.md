# MEDRAG - Medical RAG Application

A comprehensive Medical Retrieval-Augmented Generation (RAG) application that allows users to upload medical PDF documents and query them using natural language through an AI-powered interface.

## Features

- **PDF Processing**: Extract and process medical documents with intelligent text cleaning
- **Vector Database**: Store document embeddings using ChromaDB for efficient similarity search
- **Medical Context Awareness**: Specialized prompt engineering for medical document queries
- **Interactive Web Interface**: User-friendly Streamlit interface for document upload and querying
- **Source Attribution**: Always cite the source documents when providing answers
- **Medical Term Normalization**: Automatically expand common medical abbreviations
- **Advanced Search**: Configurable similarity thresholds and retrieval parameters
- **Batch Processing**: Support for multiple document uploads
- **System Monitoring**: Comprehensive logging and system statistics

## Quick Start

### Prerequisites

- Python 3.12 or higher
- OpenAI API key
- Medical PDF documents to query

### Installation

1. **Clone and navigate to the project:**
   ```bash
   cd /path/to/your/project
   ```

2. **Install dependencies:**
   ```bash
   pip install -e .
   ```

3. **Set up environment variables:**
   Create a `.env` file in the project root:
   ```env
   OPENAI_API_KEY=your_openai_api_key_here
   OPENAI_MODEL=gpt-3.5-turbo
   OPENAI_MAX_TOKENS=1000
   OPENAI_TEMPERATURE=0.7
   ```

4. **Run the application:**
   ```bash
   streamlit run main.py
   ```

5. **Open your browser** and navigate to `http://localhost:8501`

## Usage

### 1. Upload Medical Documents

- Click "Browse files" to select one or more PDF documents
- Supported format: PDF files
- Documents are automatically processed and added to the vector database

### 2. Ask Questions

- Type your medical question in the text area
- Examples:
  - "What are the symptoms of diabetes?"
  - "What medications are mentioned in the documents?"
  - "What are the treatment options for hypertension?"

### 3. View Results

- **AI Response**: The main answer based on your documents
- **Sources**: Expandable sections showing the source documents and relevance scores
- **Advanced Options**: Adjust retrieval parameters for better results

### 4. System Information

- **Sidebar**: Shows document count and processed files
- **Settings**: Clear all documents or view system statistics

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | Required | Your OpenAI API key |
| `OPENAI_MODEL` | `gpt-3.5-turbo` | OpenAI model to use |
| `OPENAI_MAX_TOKENS` | `1000` | Maximum tokens for responses |
| `OPENAI_TEMPERATURE` | `0.7` | Response creativity (0-1) |
| `VECTOR_STORE_PATH` | `./vector_store` | Path for ChromaDB storage |
| `EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Embedding model |
| `CHUNK_SIZE` | `1000` | Text chunk size for processing |
| `CHUNK_OVERLAP` | `200` | Overlap between chunks |
| `TOP_K_RETRIEVAL` | `5` | Number of documents to retrieve |
| `SIMILARITY_THRESHOLD` | `0.7` | Minimum similarity score |

### Advanced Configuration

You can modify the configuration in `medrag/utils/config.py` or set environment variables for runtime configuration.

## Architecture

```
MEDRAG Application
├── main.py                 # Main Streamlit application
├── medrag/
│   ├── core/
│   │   ├── pdf_processor.py    # PDF text extraction and chunking
│   │   ├── vector_store.py     # ChromaDB vector database management
│   │   ├── llm_client.py       # OpenAI API integration
│   │   └── rag_system.py       # RAG pipeline orchestration
│   └── utils/
│       └── config.py           # Configuration management
└── vector_store/          # ChromaDB data storage
```

### Core Components

1. **PDFProcessor**: Handles PDF text extraction, cleaning, and chunking
2. **VectorStore**: Manages document embeddings and similarity search
3. **LLMClient**: Communicates with OpenAI for response generation
4. **RAGSystem**: Orchestrates the complete RAG pipeline

## Testing

### System Test

The application includes built-in system testing:

```python
from medrag.core.rag_system import RAGSystem

# Test the complete system
test_results = rag_system.test_system()
print(test_results)
```

### Manual Testing

1. Upload a medical PDF document
2. Ask a question about the document content
3. Verify the response cites the source document
4. Check that the answer is based on the uploaded content

## Security & Privacy

- **Local Processing**: All document processing happens locally
- **No Data Sharing**: Documents are not sent to external services (except OpenAI for query processing)
- **Secure Storage**: Vector database is stored locally
- **API Key Protection**: Use environment variables for sensitive configuration

## Important Notes

### Medical Disclaimer

**This application is for educational and research purposes only. It should not be used for medical diagnosis or treatment decisions. Always consult qualified healthcare professionals for medical advice.**

### Limitations

- The quality of responses depends on the uploaded documents
- Complex medical queries may require multiple documents for comprehensive answers
- The system works best with well-structured medical documents
- Large PDF files may take longer to process

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

If you encounter any issues:

1. Check the console logs for error messages
2. Verify your OpenAI API key is valid
3. Ensure your PDF documents are not password-protected
4. Check that you have sufficient disk space for the vector database

## Updates

- **v0.1.0**: Initial release with basic RAG functionality
- Future versions will include:
  - Support for more document formats
  - Advanced medical term recognition
  - Multi-language support
  - Enhanced UI features
