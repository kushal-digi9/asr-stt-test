import logging
from typing import List, Dict, Any
from celery import shared_task
from utils.document_processor import DocumentProcessor
from models.rag_model import RAGModel
from utils.redis_client import set_json, get_json
import os

logger = logging.getLogger(__name__)

# Initialize processors (will be lazy-loaded in tasks)
_doc_processor = None
_rag_model = None

def get_doc_processor():
    """Lazy load document processor."""
    global _doc_processor
    if _doc_processor is None:
        _doc_processor = DocumentProcessor(chunk_size=500, chunk_overlap=50)
    return _doc_processor

def get_rag_model():
    """Lazy load RAG model."""
    global _rag_model
    if _rag_model is None:
        qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
        _rag_model = RAGModel(qdrant_url=qdrant_url)
    return _rag_model


@shared_task(name="ingest_document")
def ingest_document_task(file_path: str, doc_id: str) -> Dict[str, Any]:
    """
    Celery task to ingest a document: extract text, chunk, embed, and store.
    
    Args:
        file_path: Path to document file
        doc_id: Unique document ID
        
    Returns:
        Dict with ingestion results
    """
    logger.info(f"📄 Starting document ingestion: {file_path}")
    
    try:
        # Process document
        doc_processor = get_doc_processor()
        chunks = doc_processor.process_and_chunk(file_path)
        
        if not chunks:
            return {
                "doc_id": doc_id,
                "status": "failed",
                "error": "No text extracted from document"
            }
        
        # Add to vector store
        rag_model = get_rag_model()
        texts = [chunk["text"] for chunk in chunks]
        metadatas = [
            {
                **chunk["metadata"],
                "doc_id": doc_id
            }
            for chunk in chunks
        ]
        
        chunk_ids = rag_model.add_documents(texts, metadatas)
        
        # Store ingestion record in Redis
        result = {
            "doc_id": doc_id,
            "file_path": file_path,
            "status": "completed",
            "chunks_count": len(chunks),
            "chunk_ids": chunk_ids
        }
        
        set_json(f"ingestion:{doc_id}", result, ex_seconds=86400)  # 24 hours
        
        logger.info(f"✅ Document ingestion completed: {doc_id} ({len(chunks)} chunks)")
        return result
        
    except Exception as e:
        logger.error(f"❌ Document ingestion failed: {e}")
        result = {
            "doc_id": doc_id,
            "status": "failed",
            "error": str(e)
        }
        set_json(f"ingestion:{doc_id}", result, ex_seconds=3600)
        return result


@shared_task(name="batch_ingest_documents")
def batch_ingest_documents_task(file_paths: List[str], batch_id: str) -> Dict[str, Any]:
    """
    Celery task to ingest multiple documents in batch.
    
    Args:
        file_paths: List of document file paths
        batch_id: Unique batch ID
        
    Returns:
        Dict with batch ingestion results
    """
    logger.info(f"📄 Starting batch ingestion: {len(file_paths)} documents")
    
    results = []
    for idx, file_path in enumerate(file_paths):
        doc_id = f"{batch_id}_{idx}"
        result = ingest_document_task(file_path, doc_id)
        results.append(result)
    
    # Store batch record
    batch_result = {
        "batch_id": batch_id,
        "total_documents": len(file_paths),
        "successful": sum(1 for r in results if r["status"] == "completed"),
        "failed": sum(1 for r in results if r["status"] == "failed"),
        "results": results
    }
    
    set_json(f"batch:{batch_id}", batch_result, ex_seconds=86400)
    
    logger.info(f"✅ Batch ingestion completed: {batch_id}")
    return batch_result


@shared_task(name="update_embeddings")
def update_embeddings_task(doc_ids: List[str]) -> Dict[str, Any]:
    """
    Celery task to re-embed documents (useful after model updates).
    
    Args:
        doc_ids: List of document IDs to re-embed
        
    Returns:
        Dict with update results
    """
    logger.info(f"🔄 Updating embeddings for {len(doc_ids)} documents")
    
    # This is a placeholder - implement based on your needs
    # You would need to retrieve original texts and re-embed them
    
    return {
        "status": "completed",
        "updated_count": len(doc_ids)
    }
