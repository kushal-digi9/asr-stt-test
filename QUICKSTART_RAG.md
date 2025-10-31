# RAG Pipeline Quick Start

Get your RAG pipeline up and running in 5 minutes!

## Prerequisites

- Docker and Docker Compose installed
- Python 3.9+ (if running locally)
- At least 8GB RAM
- GPU recommended (but not required)

## Step 1: Start Services

```bash
# Start all services (Redis, Qdrant, Ollama, FastAPI, Celery)
docker-compose up -d

# Check services are running
docker-compose ps
```

Expected output:
```
NAME                STATUS
redis               Up
qdrant              Up
ollama              Up
fastapi             Up
worker              Up
```

## Step 2: Install Dependencies (if running locally)

```bash
pip install -r requirements.txt

# Install Tesseract for OCR (Ubuntu/Debian)
sudo apt-get install tesseract-ocr tesseract-ocr-hin tesseract-ocr-eng
```

## Step 3: Test RAG Components

```bash
# Run test script
python test_rag_pipeline.py
```

This will test:
- ✅ Document processing and chunking
- ✅ Vector embedding and storage
- ✅ Semantic retrieval
- ✅ RAG-enhanced generation

## Step 4: Start RAG API

```bash
# Start RAG API server
uvicorn api_rag:app --host 0.0.0.0 --port 8001 --reload
```

API will be available at: http://localhost:8001

View docs at: http://localhost:8001/docs

## Step 5: Upload Your First Document

### Option A: Using cURL

```bash
curl -X POST "http://localhost:8001/ingest/upload" \
  -F "file=@your_document.pdf"
```

### Option B: Using Python

```python
import requests

with open("your_document.pdf", "rb") as f:
    response = requests.post(
        "http://localhost:8001/ingest/upload",
        files={"file": f}
    )

print(response.json())
# Output: {"doc_id": "...", "task_id": "...", "status": "processing"}
```

### Option C: Using Swagger UI

1. Go to http://localhost:8001/docs
2. Click on `/ingest/upload`
3. Click "Try it out"
4. Upload your file
5. Click "Execute"

## Step 6: Check Ingestion Status

```bash
# Replace {task_id} with the task_id from upload response
curl "http://localhost:8001/ingest/status/{task_id}"
```

Wait until status is "SUCCESS" and you see:
```json
{
  "task_id": "...",
  "state": "SUCCESS",
  "ready": true,
  "result": {
    "doc_id": "...",
    "status": "completed",
    "chunks_count": 15
  }
}
```

## Step 7: Query with RAG

```bash
curl -X POST "http://localhost:8001/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are the hospital timings?",
    "use_rag": true,
    "top_k": 3
  }'
```

Response:
```json
{
  "query": "What are the hospital timings?",
  "answer": "The hospital is open from 8 AM to 8 PM daily...",
  "contexts": [
    {
      "text": "Hospital timings: 8 AM to 8 PM...",
      "score": 0.89
    }
  ],
  "rag_enabled": true,
  "docs_retrieved": 3
}
```

## Step 8: Integrate with Voice Pipeline

Update `main.py` to use RAG:

```python
from models.llm_rag_model import LLMRAGModel

# Replace LLMModel initialization
llm = LLMRAGModel(
    ollama_url=OLLAMA_URL,
    model_name=OLLAMA_MODEL,
    qdrant_url=os.getenv("QDRANT_URL", "http://localhost:6333"),
    enable_rag=True
)

# In /pipeline endpoint, use RAG
response_result = await llm.generate_response_with_rag(transcribed_text)
response_text = response_result["answer"]
```

## Common Commands

### Check Collection Status
```bash
curl "http://localhost:8001/collection/info"
```

### Retrieve Documents Only
```bash
curl -X POST "http://localhost:8001/retrieve" \
  -H "Content-Type: application/json" \
  -d '{"query": "hospital timings", "top_k": 5}'
```

### Delete Document
```bash
curl -X DELETE "http://localhost:8001/collection/document/{doc_id}"
```

### Health Check
```bash
curl "http://localhost:8001/health"
```

## Troubleshooting

### Qdrant not connecting
```bash
# Check Qdrant is running
docker logs <qdrant-container-name>

# Test Qdrant directly
curl http://localhost:6333/health
```

### Ollama not responding
```bash
# Check Ollama is running
docker logs <ollama-container-name>

# Pull model manually
docker exec <ollama-container> ollama pull llama3.2:1b
```

### Celery worker not processing
```bash
# Check worker logs
docker logs <worker-container-name>

# Restart worker
docker-compose restart worker
```

### Out of memory
```bash
# Reduce chunk size in .env
RAG_CHUNK_SIZE=300
RAG_CHUNK_OVERLAP=30

# Or use smaller embedding model
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
```

## Sample Documents to Test

Create a sample hospital info document:

```bash
cat > hospital_info.txt << 'EOF'
Hospital Information

Timings:
- OPD: 8 AM to 8 PM (Monday to Saturday)
- Emergency: 24/7
- Pharmacy: 7 AM to 10 PM

Departments:
- General Medicine
- Pediatrics
- Orthopedics
- Cardiology
- Radiology

Contact:
- Phone: 1234567890
- Email: info@hospital.com
- Address: 123 Medical Street

Appointment Booking:
Call 1234567890 or visit our website.
Walk-in appointments available for emergencies.
EOF

# Upload it
curl -X POST "http://localhost:8001/ingest/upload" \
  -F "file=@hospital_info.txt"
```

## Performance Tips

1. **Use GPU**: Ensure GPU is available for faster embeddings
2. **Batch uploads**: Use batch ingestion for multiple documents
3. **Tune chunk size**: Smaller chunks = more precise, larger = more context
4. **Cache embeddings**: Embeddings are automatically cached in Qdrant
5. **Monitor memory**: Watch Docker container memory usage

## Next Steps

1. ✅ Upload your domain-specific documents
2. ✅ Test retrieval quality with various queries
3. ✅ Tune chunk size and top_k parameters
4. ✅ Integrate with voice pipeline
5. ✅ Run evaluation on test dataset
6. ✅ Deploy to production

## Support

- Full documentation: See `RAG_IMPLEMENTATION.md`
- API docs: http://localhost:8001/docs
- Test script: `python test_rag_pipeline.py`

Happy RAG-ing! 🚀
