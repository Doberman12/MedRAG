# MEDRAG Quick Start Guide

Get up and running with MEDRAG in minutes!

## Prerequisites

- Python 3.12 or higher
- UV package manager
- OpenAI API key (for LLM functionality)

## Step 1: Install Dependencies

```bash
# Install all dependencies using UV
uv sync
```

## Step 2: Set Up Environment

Create a `.env` file in the project root:

```bash
# Copy the example environment file
cp env.example .env

# Edit the .env file and add your OpenAI API key
echo "OPENAI_API_KEY=your_actual_api_key_here" >> .env
```

## Step 3: Test the System

```bash
# Run the system test
uv run python test_medrag.py

# Or use the CLI test command
uv run python cli.py test
```

You should see all tests passing!

## Step 4: Choose Your Interface

### Option A: Web Interface (Recommended for beginners)

```bash
# Start the Streamlit web app
uv run streamlit run main.py
```

Then open your browser to `http://localhost:8501`

### Option B: Command Line Interface

```bash
# Add PDF documents
uv run python cli.py add path/to/your/medical_document.pdf

# Query documents
uv run python cli.py query "What are the symptoms of diabetes?"

# List documents
uv run python cli.py list

# Get system stats
uv run python cli.py stats
```

### Option C: Programmatic Usage

```python
from medrag.core.pdf_processor import PDFProcessor
from medrag.core.vector_store import VectorStore
from medrag.core.llm_client import LLMClient
from medrag.core.rag_system import RAGSystem

# Initialize components
pdf_processor = PDFProcessor()
vector_store = VectorStore()
llm_client = LLMClient()
rag_system = RAGSystem(vector_store, llm_client)

# Process a PDF
chunks = pdf_processor.process_pdf(Path("medical_document.pdf"))
vector_store.add_documents(chunks, "medical_document.pdf")

# Query the documents
response = rag_system.query("What are the symptoms of diabetes?")
print(response['answer'])
```

## Step 5: Add Your Medical Documents

1. **Prepare your PDF files** - Make sure they're readable and not password-protected
2. **Upload via web interface** - Drag and drop files in the browser
3. **Or use CLI** - `uv run python cli.py add document1.pdf document2.pdf`

## Step 6: Start Querying!

### Example Questions to Try:

- "What are the symptoms of diabetes?"
- "What medications are mentioned in the documents?"
- "What are the treatment options for hypertension?"
- "What are the normal ranges for blood pressure?"
- "What are the side effects of the medications mentioned?"

## Troubleshooting

### Common Issues:

1. **"OPENAI_API_KEY is required"**
   - Set your OpenAI API key in the `.env` file
   - Or export it: `export OPENAI_API_KEY=your_key_here`

2. **"No module named 'pypdf'"**
   - Run `uv sync` to install dependencies
   - Use `uv run python` instead of just `python`

3. **PDF processing fails**
   - Ensure the PDF is not password-protected
   - Check that the PDF contains extractable text
   - Try a different PDF file

4. **No relevant documents found**
   - Add more documents to the vector store
   - Try rephrasing your question
   - Check that your documents contain relevant medical information

### Getting Help:

- Run `uv run python demo.py` for a comprehensive demo
- Check the full README.md for detailed documentation
- Use `uv run python cli.py --help` for CLI options

## Next Steps

- Explore advanced features in the web interface
- Try different embedding models by modifying the config
- Experiment with different chunk sizes and similarity thresholds
- Add more medical documents for better coverage

## Example Workflow

```bash
# 1. Test the system
uv run python cli.py test

# 2. Add medical documents
uv run python cli.py add medical_report.pdf patient_notes.pdf

# 3. Check what's in the store
uv run python cli.py list

# 4. Query the documents
uv run python cli.py query "What are the main findings in the medical report?"

# 5. Get detailed response with sources
uv run python cli.py query "What medications were prescribed and what are their side effects?"
```

Happy querying!
