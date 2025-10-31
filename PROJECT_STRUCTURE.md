# Project Structure

## Complete Directory Layout

```
voice-ai-project/
│
├── 📁 data/
│   ├── 📁 documents/          # NEW: Uploaded documents for RAG
│   ├── 📁 input_samples/      # Sample audio inputs
│   └── 📁 outputs/            # Generated audio outputs
│
├── 📁 models/
│   ├── asr_model.py           # ASR (Speech-to-Text)
│   ├── llm_model.py           # Base LLM model
│   ├── llm_rag_model.py       # NEW: LLM with RAG support
│   ├── rag_model.py           # NEW: RAG core (embeddings + retrieval)
│   ├── tts_model.py           # TTS (Text-to-Speech)
│   └── __init__.py
│
├── 📁 utils/
│   ├── audio_utils.py         # Audio processing utilities
│   ├── document_processor.py  # NEW: Document ingestion & chunking
│   ├── logger.py              # Logging utilities
│   ├── rag_evaluator.py       # NEW: RAG evaluation (RAGAS)
│   ├── redis_client.py        # Redis utilities
│   └── __init__.py
│
├── 📁 logs/                   # Application logs
│
├── 📄 main.py                 # Main FastAPI app (Voice Pipeline)
├── 📄 api_rag.py              # NEW: RAG API endpoints
├── 📄 celery_app.py           # Celery configuration (UPDATED)
├── 📄 tasks.py                # Original Celery tasks
├── 📄 tasks_rag.py            # NEW: RAG Celery tasks
│
├── 📄 requirements.txt        # Python dependencies (UPDATED)
├── 📄 .env                    # Environment variables (UPDATED)
├── 📄 docker-compose.yml      # Docker services (UPDATED)
├── 📄 Dockerfile              # Docker build config
├── 📄 deploy.sh               # Deployment script
│
├── 📄 test_rag_pipeline.py    # NEW: RAG test script
│
├── 📄 RAG_IMPLEMENTATION.md   # NEW: Full RAG documentation
├── 📄 QUICKSTART_RAG.md       # NEW: Quick start guide
├── 📄 RAG_CHANGES_SUMMARY.md  # NEW: Changes summary
└── 📄 PROJECT_STRUCTURE.md    # NEW: This file
```

## Component Relationships

```
┌─────────────────────────────────────────────────────────────┐
│                     VOICE AI SYSTEM                          │
└─────────────────────────────────────────────────────────────┘

┌──────────────────────┐         ┌──────────────────────┐
│   Voice Pipeline     │         │    RAG Pipeline      │
│   (main.py)          │         │    (api_rag.py)      │
│   Port: 8000         │         │    Port: 8001        │
└──────────┬───────────┘         └──────────┬───────────┘
           │                                 │
           ├─────────────────────────────────┤
           │                                 │
    ┌──────▼──────┐                  ┌──────▼──────┐
    │   Models    │                  │   Models    │
    ├─────────────┤                  ├─────────────┤
    │ ASR         │                  │ RAG         │
    │ LLM/LLM-RAG │◄─────────────────┤ LLM-RAG     │
    │ TTS         │                  │ Embeddings  │
    └─────────────┘                  └─────────────┘
           │                                 │
    ┌──────▼──────┐                  ┌──────▼──────┐
    │   Utils     │                  │   Utils     │
    ├─────────────┤                  ├─────────────┤
    │ Audio       │                  │ Documents   │
    │ Logger      │                  │ Evaluator   │
    │ Redis       │                  │ Redis       │
    └─────────────┘                  └─────────────┘
           │                                 │
           └─────────────┬───────────────────┘
                         │
                  ┌──────▼──────┐
                  │  Services   │
                  ├─────────────┤
                  │ Redis       │
                  │ Qdrant      │
                  │ Ollama      │
                  │ Celery      │
                  └─────────────┘
```

## Data Flow

### 1. Voice Pipeline (Original)
```
User Audio
    ↓
[ASR Model] → Transcribed Text
    ↓
[LLM Model] → Response Text
    ↓
[TTS Model] → Response Audio
    ↓
User
```

### 2. RAG Pipeline (New)
```
Documents
    ↓
[Document Processor] → Text Chunks
    ↓
[Embedding Model] → Vectors
    ↓
[Qdrant DB] → Stored
    ↓
Query → [Retrieval] → Relevant Chunks
    ↓
[LLM + Context] → Enhanced Response
```

### 3. Integrated Pipeline (Voice + RAG)
```
User Audio
    ↓
[ASR Model] → Transcribed Text (Query)
    ↓
[RAG Retrieval] → Relevant Context
    ↓
[LLM + RAG] → Context-Aware Response
    ↓
[TTS Model] → Response Audio
    ↓
User
```

## Service Ports

| Service | Port | Purpose |
|---------|------|---------|
| FastAPI (Voice) | 8000 | Main voice pipeline API |
| FastAPI (RAG) | 8001 | RAG operations API |
| Redis | 6379 | Task queue & caching |
| Qdrant | 6333 | Vector database (HTTP) |
| Qdrant | 6334 | Vector database (gRPC) |
| Ollama | 11434 | LLM inference |

## File Sizes (Approximate)

| Component | Size | Notes |
|-----------|------|-------|
| Embedding Model | ~500MB | First download |
| ASR Model | ~600MB | Indic Conformer |
| TTS Model | ~1GB | Parler TTS |
| LLM Model | ~1-7GB | Depends on model |
| Qdrant Data | Variable | Grows with documents |
| Document Storage | Variable | Original files |

## Key Configuration Files

### 1. `.env` - Environment Variables
```env
# Original
HF_ACCESS_TOKEN=...
ASR_MOCK_MODE=...
LLM_ECHO_MODE=...
TTS_MOCK_MODE=...
OLLAMA_URL=...
OLLAMA_MODEL=...

# NEW: RAG Configuration
QDRANT_URL=http://localhost:6333
QDRANT_COLLECTION=documents
EMBEDDING_MODEL=sentence-transformers/...
RAG_TOP_K=3
RAG_CHUNK_SIZE=500
RAG_CHUNK_OVERLAP=50
```

### 2. `docker-compose.yml` - Services
```yaml
services:
  redis:        # Task queue
  qdrant:       # NEW: Vector DB
  ollama:       # LLM runtime
  fastapi:      # Main app
  worker:       # Celery worker
```

### 3. `requirements.txt` - Dependencies
```
# Original
fastapi, uvicorn, transformers, etc.

# NEW: RAG
qdrant-client
sentence-transformers
pypdf
python-docx
pytesseract
ragas
```

## Module Dependencies

```
main.py
├── models/asr_model.py
├── models/llm_model.py (or llm_rag_model.py)
├── models/tts_model.py
├── utils/audio_utils.py
├── utils/logger.py
└── celery_app.py

api_rag.py
├── models/rag_model.py
├── models/llm_rag_model.py
├── utils/document_processor.py
├── utils/rag_evaluator.py
├── utils/redis_client.py
├── tasks_rag.py
└── celery_app.py

tasks_rag.py
├── utils/document_processor.py
├── models/rag_model.py
└── utils/redis_client.py
```

## Storage Locations

| Data Type | Location | Managed By |
|-----------|----------|------------|
| Audio Input | `data/input_samples/` | User |
| Audio Output | `data/outputs/` | FastAPI |
| Documents | `data/documents/` | RAG API |
| Vector Data | Docker volume | Qdrant |
| Model Cache | `~/.cache/huggingface/` | Transformers |
| Ollama Models | Docker volume | Ollama |
| Logs | `logs/` | Application |

## Development Workflow

### 1. Local Development
```bash
# Terminal 1: Start services
docker-compose up -d redis qdrant ollama

# Terminal 2: Run voice API
uvicorn main:app --reload --port 8000

# Terminal 3: Run RAG API
uvicorn api_rag:app --reload --port 8001

# Terminal 4: Run Celery worker
celery -A celery_app.celery_app worker --loglevel=INFO
```

### 2. Testing
```bash
# Test RAG components
python test_rag_pipeline.py

# Test voice pipeline
curl -X POST http://localhost:8000/pipeline \
  -F "file=@test_audio.wav"

# Test RAG API
curl -X POST http://localhost:8001/generate \
  -H "Content-Type: application/json" \
  -d '{"query": "test", "use_rag": true}'
```

### 3. Production Deployment
```bash
# Build and start all services
docker-compose up -d

# Check logs
docker-compose logs -f

# Scale workers
docker-compose up -d --scale worker=3
```

## Monitoring & Debugging

### Logs
```bash
# Application logs
tail -f logs/app.log

# Docker logs
docker-compose logs -f fastapi
docker-compose logs -f worker
docker-compose logs -f qdrant

# Celery tasks
docker-compose logs -f worker | grep "Task"
```

### Health Checks
```bash
# Voice API
curl http://localhost:8000/health

# RAG API
curl http://localhost:8001/health

# Qdrant
curl http://localhost:6333/health

# Ollama
curl http://localhost:11434/api/tags
```

### Metrics
```bash
# Qdrant metrics
curl http://localhost:6333/metrics

# Collection info
curl http://localhost:8001/collection/info

# Task status
curl http://localhost:8000/status/{task_id}
```

## Backup & Recovery

### Backup Qdrant Data
```bash
docker-compose exec qdrant tar -czf /tmp/qdrant_backup.tar.gz /qdrant/storage
docker cp <container_id>:/tmp/qdrant_backup.tar.gz ./backups/
```

### Restore Qdrant Data
```bash
docker cp ./backups/qdrant_backup.tar.gz <container_id>:/tmp/
docker-compose exec qdrant tar -xzf /tmp/qdrant_backup.tar.gz -C /
docker-compose restart qdrant
```

## Security Considerations

1. **API Keys**: Store in `.env`, never commit
2. **File Uploads**: Validate file types and sizes
3. **Rate Limiting**: Add to production deployment
4. **Authentication**: Add JWT/OAuth for production
5. **CORS**: Configure properly for production

## Performance Tuning

### Memory
- Adjust Docker memory limits
- Tune chunk size for documents
- Use smaller embedding models if needed

### Speed
- Enable GPU for embeddings
- Use batch processing
- Cache frequent queries
- Optimize chunk size

### Scalability
- Scale Celery workers
- Use Redis cluster
- Shard Qdrant collections
- Load balance FastAPI instances

---

**Last Updated**: Based on RAG implementation
**Version**: 1.0.0 with RAG support
