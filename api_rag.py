from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import os
import uuid
import logging
from celery.result import AsyncResult
from celery_app import celery_app
from tasks_rag import ingest_document_task, batch_ingest_documents_task
from models.rag_model import RAGModel
from models.llm_rag_model import LLMRAGModel
from utils.rag_evaluator import RAGEvaluator
from utils.redis_client import get_json
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# Initialize FastAPI app for RAG endpoints
app = FastAPI(
    title="RAG Pipeline API",
    description="Document ingestion, retrieval, and RAG-enhanced generation",
    version="1.0.0"
)

# Configuration
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:1b")
UPLOAD_DIR = "data/documents"

os.makedirs(UPLOAD_DIR, exist_ok=True)

# Initialize models (lazy loading)
_rag_model = None
_llm_rag_model = None
_evaluator = None

def get_rag_model():
    global _rag_model
    if _rag_model is None:
        _rag_model = RAGModel(qdrant_url=QDRANT_URL)
    return _rag_model

def get_llm_rag_model():
    global _llm_rag_model
    if _llm_rag_model is None:
        _llm_rag_model = LLMRAGModel(
            ollama_url=OLLAMA_URL,
            model_name=OLLAMA_MODEL,
            qdrant_url=QDRANT_URL,
            enable_rag=True
        )
    return _llm_rag_model

def get_evaluator():
    global _evaluator
    if _evaluator is None:
        _evaluator = RAGEvaluator()
    return _evaluator


# Pydantic models
class QueryRequest(BaseModel):
    query: str
    top_k: Optional[int] = 3
    use_rag: bool = True
    filter: Optional[Dict[str, Any]] = None

class EvaluationRequest(BaseModel):
    question: str
    answer: str
    contexts: List[str]
    ground_truth: Optional[str] = None


# === Document Ingestion Endpoints ===

@app.post("/ingest/upload")
async def upload_document(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = None
) -> Dict[str, Any]:
    """
    Upload and ingest a document (async via Celery).
    
    Supported formats: PDF, DOCX, TXT, images (for OCR)
    """
    logger.info(f"📄 Received document upload: {file.filename}")
    
    # Validate file type
    allowed_extensions = ['.pdf', '.docx', '.doc', '.txt', '.png', '.jpg', '.jpeg']
    file_ext = os.path.splitext(file.filename)[1].lower()
    
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type. Allowed: {allowed_extensions}"
        )
    
    # Save uploaded file
    doc_id = str(uuid.uuid4())
    file_path = os.path.join(UPLOAD_DIR, f"{doc_id}_{file.filename}")
    
    with open(file_path, "wb") as f:
        content = await file.read()
        f.write(content)
    
    logger.info(f"📄 Saved file: {file_path}")
    
    # Enqueue ingestion task
    task = ingest_document_task.delay(file_path, doc_id)
    
    return {
        "doc_id": doc_id,
        "task_id": task.id,
        "filename": file.filename,
        "status": "processing"
    }


@app.get("/ingest/status/{task_id}")
async def get_ingestion_status(task_id: str) -> Dict[str, Any]:
    """Check status of document ingestion task."""
    result = AsyncResult(task_id, app=celery_app)
    
    response = {
        "task_id": task_id,
        "state": result.state,
        "ready": result.ready()
    }
    
    if result.ready():
        try:
            response["result"] = result.get(timeout=0)
        except Exception as e:
            response["error"] = str(e)
    
    return response


@app.get("/ingest/document/{doc_id}")
async def get_document_info(doc_id: str) -> Dict[str, Any]:
    """Get information about an ingested document."""
    info = get_json(f"ingestion:{doc_id}")
    
    if not info:
        raise HTTPException(status_code=404, detail="Document not found")
    
    return info


# === Retrieval Endpoints ===

@app.post("/retrieve")
async def retrieve_documents(request: QueryRequest) -> Dict[str, Any]:
    """
    Retrieve relevant documents for a query.
    """
    logger.info(f"🔍 Retrieval request: {request.query[:100]}")
    
    rag_model = get_rag_model()
    
    documents = rag_model.retrieve(
        query=request.query,
        top_k=request.top_k,
        filter_dict=request.filter
    )
    
    return {
        "query": request.query,
        "documents": documents,
        "count": len(documents)
    }


# === RAG Generation Endpoints ===

@app.post("/generate")
async def generate_with_rag(request: QueryRequest) -> Dict[str, Any]:
    """
    Generate response using RAG pipeline.
    """
    logger.info(f"🧠 RAG generation request: {request.query[:100]}")
    
    llm_rag = get_llm_rag_model()
    
    if request.use_rag:
        result = await llm_rag.generate_response_with_rag(
            prompt=request.query,
            filter_dict=request.filter
        )
    else:
        answer = await llm_rag.generate_response(request.query)
        result = {
            "answer": answer,
            "contexts": [],
            "rag_enabled": False
        }
    
    return {
        "query": request.query,
        **result
    }


# === Evaluation Endpoints ===

@app.post("/evaluate")
async def evaluate_rag_response(request: EvaluationRequest) -> Dict[str, Any]:
    """
    Evaluate a RAG response using RAGAS metrics.
    """
    logger.info(f"📊 Evaluation request")
    
    evaluator = get_evaluator()
    
    scores = evaluator.evaluate_single(
        question=request.question,
        answer=request.answer,
        contexts=request.contexts,
        ground_truth=request.ground_truth
    )
    
    return {
        "question": request.question,
        "scores": scores
    }


# === Collection Management ===

@app.get("/collection/info")
async def get_collection_info() -> Dict[str, Any]:
    """Get information about the vector store collection."""
    rag_model = get_rag_model()
    return rag_model.get_collection_info()


@app.delete("/collection/document/{doc_id}")
async def delete_document(doc_id: str) -> Dict[str, Any]:
    """Delete a document and its chunks from the vector store."""
    rag_model = get_rag_model()
    
    # Get document info to find chunk IDs
    doc_info = get_json(f"ingestion:{doc_id}")
    
    if not doc_info:
        raise HTTPException(status_code=404, detail="Document not found")
    
    chunk_ids = doc_info.get("chunk_ids", [])
    
    if chunk_ids:
        success = rag_model.delete_documents(chunk_ids)
        if success:
            return {
                "doc_id": doc_id,
                "deleted_chunks": len(chunk_ids),
                "status": "success"
            }
    
    return {
        "doc_id": doc_id,
        "status": "no_chunks_found"
    }


# === Health Check ===

@app.get("/health")
async def health_check() -> Dict[str, Any]:
    """Health check for RAG services."""
    try:
        rag_model = get_rag_model()
        collection_info = rag_model.get_collection_info()
        
        llm_rag = get_llm_rag_model()
        rag_stats = llm_rag.get_rag_stats()
        
        return {
            "status": "healthy",
            "qdrant": {
                "url": QDRANT_URL,
                "collection": collection_info
            },
            "rag_model": rag_stats,
            "ollama": {
                "url": OLLAMA_URL,
                "model": OLLAMA_MODEL
            }
        }
    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "error": str(e)
            }
        )
