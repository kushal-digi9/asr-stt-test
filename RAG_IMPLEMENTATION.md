# RAG Pipeline Implementation Guide

## Overview

This document describes the RAG (Retrieval-Augmented Generation) pipeline implementation for the Voice AI system. The RAG pipeline enhances the LLM responses with relevant context retrieved from a vector database.

## Architecture

```
┌───────────────┐
│  Documents    │ (PDF, DOCX, TXT, Images)
└───────┬───────┘
        ↓
┌───────────────┐
│  ETL/Parser   │ ← DocumentProcessor
│  + OCR        │   (Tesseract/EasyOCR)
└───────┬───────┘
        ↓
┌───────────────┐
│  Chunking     │ ← Text splitting with overlap
└───────┬───────┘
        ↓
┌───────────────┐
│  Embedding    │ ← SentenceTransformer
│               │   (Multilingual MiniLM)
└───────┬───────┘
        ↓
┌───────────────┐
│  Qdrant DB    │ ← Vector storage
└───────┬───────┘
        ↓
┌───────────────┐
│  Retrieval    │ ← Semantic search
└───────┬───────┘
        ↓
┌───────────────┐
│  LLM + RAG    │ ← Context-enhanced generation
│  (Llama)      │
└───────────────┘
```

## Components

### 1. Document Processing (`utils/document_processor.py`)

Handles document ingestion and text extraction:
- **PDF**: PyPDF for text extraction
- **DOCX**: python-docx for Word documents
- **TXT**: Plain text files
- **Images**: Tesseract OCR for image-based content

Features:
- Configurable chunk size and overlap
- Metadata preservation
- Sentence-boundary aware chunking

### 2. RAG Model (`models/rag_model.py`)

Core RAG functionality:
- **Embedding**: SentenceTransformer multilingual model
- **Vector Store**: Qdrant for efficient similarity search
- **Retrieval**: Top-K semantic search with optional filters

### 3. LLM with RAG (`models/llm_rag_model.py`)

Extended LLM model with RAG capabilities:
- Retrieves relevant context before generation
- Builds context-aware prompts
- Falls back to base LLM if retrieval fails

### 4. RAG Evaluation (`utils/rag_evaluator.py`)

Evaluation using RAGAS metrics:
- **Faithfulness**: Answer accuracy to context
- **Answer Relevancy**: Relevance to question
- **Context Precision**: Relevance of retrieved context
- **Context Recall**: Coverage of ground truth

### 5. Celery Tasks (`tasks_rag.py`)

Async document processing:
- `ingest_document`: Process single document
- `batch_ingest_documents`: Batch processing
- `update_embeddings`: Re-embed documents

### 6. RAG API (`api_rag.py`)

FastAPI endpoints for RAG operations:
- Document upload and ingestion
- Document retrieval
- RAG-enhanced generation
- Evaluation
- Collection management

## Installation

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Install Tesseract (for OCR)

**Ubuntu/Debian:**
```bash
sudo apt-get install tesseract-ocr tesseract-ocr-hin tesseract-ocr-eng
```

**macOS:**
```bash
brew install tesseract tesseract-lang
```

**Windows:**
Download from: https://github.com/UB-Mannheim/tesseract/wiki

### 3. Start Services

```bash
docker-compose up -d
```

This starts:
- Redis (task queue)
- Qdrant (vector database)
- Ollama (LLM runtime)
- FastAPI (main app)
- Celery Worker (async tasks)

## Configuration

Update `.env` file:

```env
# RAG Configuration
QDRANT_URL=http://localhost:6333
QDRANT_COLLECTION=documents
EMBEDDING_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
RAG_TOP_K=3
RAG_CHUNK_SIZE=500
RAG_CHUNK_OVERLAP=50
```

## Usage

### 1. Start RAG API Server

```bash
uvicorn api_rag:app --host 0.0.0.0 --port 8001
```

### 2. Upload Documents

```bash
curl -X POST "http://localhost:8001/ingest/upload" \
  -F "file=@hospital_info.pdf"
```

Response:
```json
{
  "doc_id": "uuid-here",
  "task_id": "celery-task-id",
  "filename": "hospital_info.pdf",
  "status": "processing"
}
```

### 3. Check Ingestion Status

```bash
curl "http://localhost:8001/ingest/status/{task_id}"
```

### 4. Query with RAG

```bash
curl -X POST "http://localhost:8001/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "अस्पताल के खुलने का समय क्या है?",
    "use_rag": true,
    "top_k": 3
  }'
```

Response:
```json
{
  "query": "अस्पताल के खुलने का समय क्या है?",
  "answer": "अस्पताल सुबह 8 बजे से रात 8 बजे तक खुला रहता है।",
  "contexts": [
    {
      "id": "chunk-id",
      "text": "Hospital timings: 8 AM to 8 PM...",
      "score": 0.89,
      "metadata": {...}
    }
  ],
  "rag_enabled": true,
  "docs_retrieved": 3
}
```

### 5. Retrieve Documents Only

```bash
curl -X POST "http://localhost:8001/retrieve" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "hospital timings",
    "top_k": 5
  }'
```

### 6. Evaluate RAG Response

```bash
curl -X POST "http://localhost:8001/evaluate" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What are hospital timings?",
    "answer": "Hospital is open 8 AM to 8 PM",
    "contexts": ["Hospital timings: 8 AM to 8 PM..."],
    "ground_truth": "8 AM to 8 PM"
  }'
```

### 7. Collection Management

**Get collection info:**
```bash
curl "http://localhost:8001/collection/info"
```

**Delete document:**
```bash
curl -X DELETE "http://localhost:8001/collection/document/{doc_id}"
```

## Integration with Voice Pipeline

To integrate RAG with the existing voice pipeline, modify `main.py`:

```python
from models.llm_rag_model import LLMRAGModel

# Replace LLMModel with LLMRAGModel
llm = LLMRAGModel(
    ollama_url=OLLAMA_URL,
    model_name=OLLAMA_MODEL,
    qdrant_url=QDRANT_URL,
    enable_rag=True
)

# In pipeline endpoint, use RAG-enhanced generation
response_text_result = await llm.generate_response_with_rag(transcribed_text)
response_text = response_text_result["answer"]
```

## Supported Document Formats

| Format | Extension | Processor | Notes |
|--------|-----------|-----------|-------|
| PDF | .pdf | PyPDF | Text-based PDFs |
| Word | .docx, .doc | python-docx | Modern Word docs |
| Text | .txt | Built-in | Plain text |
| Images | .png, .jpg, .jpeg | Tesseract OCR | Requires OCR |

## Performance Optimization

### 1. Batch Processing

Use batch ingestion for multiple documents:

```python
from tasks_rag import batch_ingest_documents_task

file_paths = ["doc1.pdf", "doc2.pdf", "doc3.pdf"]
batch_id = "batch_001"
task = batch_ingest_documents_task.delay(file_paths, batch_id)
```

### 2. Embedding Caching

Embeddings are automatically cached in Qdrant. No re-computation needed.

### 3. GPU Acceleration

SentenceTransformer automatically uses GPU if available:

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
```

### 4. Chunk Size Tuning

Adjust chunk size based on your use case:
- **Smaller chunks (200-300)**: More precise retrieval
- **Larger chunks (500-1000)**: More context per chunk

## Evaluation Metrics

### RAGAS Metrics

1. **Faithfulness** (0-1): How factually accurate is the answer based on context?
2. **Answer Relevancy** (0-1): How relevant is the answer to the question?
3. **Context Precision** (0-1): How relevant are the retrieved contexts?
4. **Context Recall** (0-1): How much of the ground truth is covered?

### Running Evaluation

```python
from utils.rag_evaluator import RAGEvaluator

evaluator = RAGEvaluator()

scores = evaluator.evaluate_single(
    question="What are hospital timings?",
    answer="8 AM to 8 PM",
    contexts=["Hospital is open from 8 AM to 8 PM daily"],
    ground_truth="8 AM to 8 PM"
)

print(scores)
```

## Troubleshooting

### Qdrant Connection Issues

```bash
# Check if Qdrant is running
curl http://localhost:6333/health

# View Qdrant logs
docker logs <qdrant-container-id>
```

### OCR Not Working

```bash
# Verify Tesseract installation
tesseract --version

# Test OCR
tesseract test_image.png output
```

### Embedding Model Download

First run will download the model (~500MB):
```
Downloading sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2...
```

### Memory Issues

Reduce batch size or chunk size if running out of memory:
```python
doc_processor = DocumentProcessor(chunk_size=300, chunk_overlap=30)
```

## API Endpoints Summary

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/ingest/upload` | POST | Upload document for ingestion |
| `/ingest/status/{task_id}` | GET | Check ingestion status |
| `/ingest/document/{doc_id}` | GET | Get document info |
| `/retrieve` | POST | Retrieve relevant documents |
| `/generate` | POST | Generate with RAG |
| `/evaluate` | POST | Evaluate RAG response |
| `/collection/info` | GET | Get collection stats |
| `/collection/document/{doc_id}` | DELETE | Delete document |
| `/health` | GET | Health check |

## Next Steps

1. **Phase 1**: Test document ingestion with sample PDFs
2. **Phase 2**: Integrate OCR for image-based documents
3. **Phase 3**: Test retrieval quality with various queries
4. **Phase 4**: Integrate with voice pipeline
5. **Phase 5**: Run evaluation on test dataset
6. **Phase 6**: Optimize chunk size and retrieval parameters

## References

- [Qdrant Documentation](https://qdrant.tech/documentation/)
- [SentenceTransformers](https://www.sbert.net/)
- [RAGAS](https://docs.ragas.io/)
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract)
