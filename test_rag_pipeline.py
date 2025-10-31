"""
Test script for RAG pipeline components.
Run this to verify your RAG setup is working correctly.
"""

import asyncio
import logging
from pathlib import Path
from models.rag_model import RAGModel
from models.llm_rag_model import LLMRAGModel
from utils.document_processor import DocumentProcessor
from utils.rag_evaluator import RAGEvaluator
import os
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:1b")

async def test_document_processing():
    """Test document processing and chunking."""
    logger.info("=" * 60)
    logger.info("TEST 1: Document Processing")
    logger.info("=" * 60)
    
    processor = DocumentProcessor(chunk_size=200, chunk_overlap=20)
    
    # Create a sample text file
    sample_text = """
    अस्पताल की जानकारी
    
    हमारा अस्पताल सुबह 8 बजे से रात 8 बजे तक खुला रहता है।
    आपातकालीन सेवाएं 24 घंटे उपलब्ध हैं।
    
    विभाग:
    - सामान्य चिकित्सा
    - बाल रोग
    - हड्डी रोग
    - हृदय रोग
    
    अपॉइंटमेंट के लिए कृपया 1234567890 पर कॉल करें।
    """
    
    # Save sample file
    sample_file = "data/outputs/test_hospital_info.txt"
    os.makedirs("data/outputs", exist_ok=True)
    with open(sample_file, "w", encoding="utf-8") as f:
        f.write(sample_text)
    
    # Process document
    chunks = processor.process_and_chunk(sample_file)
    
    logger.info(f"✅ Created {len(chunks)} chunks")
    for i, chunk in enumerate(chunks[:3]):  # Show first 3 chunks
        logger.info(f"Chunk {i+1}: {chunk['text'][:100]}...")
    
    return chunks

async def test_rag_model(chunks):
    """Test RAG model - embedding and retrieval."""
    logger.info("=" * 60)
    logger.info("TEST 2: RAG Model (Embedding & Retrieval)")
    logger.info("=" * 60)
    
    try:
        rag_model = RAGModel(qdrant_url=QDRANT_URL, collection_name="test_collection")
        
        # Add documents
        texts = [chunk["text"] for chunk in chunks]
        metadatas = [chunk["metadata"] for chunk in chunks]
        
        doc_ids = rag_model.add_documents(texts, metadatas)
        logger.info(f"✅ Added {len(doc_ids)} documents to vector store")
        
        # Test retrieval
        query = "अस्पताल का समय क्या है?"
        results = rag_model.retrieve(query, top_k=3)
        
        logger.info(f"✅ Retrieved {len(results)} documents for query: '{query}'")
        for i, doc in enumerate(results):
            logger.info(f"Result {i+1} (score: {doc['score']:.3f}): {doc['text'][:100]}...")
        
        # Get collection info
        info = rag_model.get_collection_info()
        logger.info(f"✅ Collection info: {info}")
        
        return rag_model, results
        
    except Exception as e:
        logger.error(f"❌ RAG model test failed: {e}")
        logger.error("Make sure Qdrant is running: docker-compose up -d qdrant")
        return None, []

async def test_llm_rag():
    """Test LLM with RAG integration."""
    logger.info("=" * 60)
    logger.info("TEST 3: LLM with RAG")
    logger.info("=" * 60)
    
    try:
        llm_rag = LLMRAGModel(
            ollama_url=OLLAMA_URL,
            model_name=OLLAMA_MODEL,
            qdrant_url=QDRANT_URL,
            enable_rag=True,
            echo_mode=False
        )
        
        # Test query
        query = "अस्पताल कब खुलता है?"
        
        logger.info(f"Query: {query}")
        result = await llm_rag.generate_response_with_rag(query)
        
        logger.info(f"✅ Answer: {result['answer']}")
        logger.info(f"✅ Retrieved {result.get('docs_retrieved', 0)} documents")
        
        if result.get('contexts'):
            logger.info("Retrieved contexts:")
            for i, ctx in enumerate(result['contexts'][:2]):
                logger.info(f"  Context {i+1}: {ctx['text'][:100]}...")
        
        return result
        
    except Exception as e:
        logger.error(f"❌ LLM RAG test failed: {e}")
        logger.error("Make sure Ollama is running: docker-compose up -d ollama")
        return None

async def test_evaluation(result):
    """Test RAG evaluation."""
    logger.info("=" * 60)
    logger.info("TEST 4: RAG Evaluation")
    logger.info("=" * 60)
    
    if not result or not result.get('contexts'):
        logger.warning("⚠️ Skipping evaluation - no RAG result available")
        return
    
    try:
        evaluator = RAGEvaluator()
        
        question = "अस्पताल कब खुलता है?"
        answer = result['answer']
        contexts = [ctx['text'] for ctx in result['contexts']]
        ground_truth = "अस्पताल सुबह 8 बजे से रात 8 बजे तक खुला रहता है"
        
        scores = evaluator.evaluate_single(
            question=question,
            answer=answer,
            contexts=contexts,
            ground_truth=ground_truth
        )
        
        logger.info("✅ Evaluation scores:")
        for metric, score in scores.items():
            logger.info(f"  {metric}: {score:.4f}")
        
        return scores
        
    except Exception as e:
        logger.error(f"❌ Evaluation test failed: {e}")
        logger.error("Note: RAGAS evaluation requires OpenAI API or compatible LLM")
        return None

async def main():
    """Run all tests."""
    logger.info("🚀 Starting RAG Pipeline Tests")
    logger.info("")
    
    # Test 1: Document Processing
    chunks = await test_document_processing()
    logger.info("")
    
    # Test 2: RAG Model
    rag_model, retrieval_results = await test_rag_model(chunks)
    logger.info("")
    
    # Test 3: LLM with RAG
    if rag_model:
        rag_result = await test_llm_rag()
        logger.info("")
        
        # Test 4: Evaluation
        if rag_result:
            await test_evaluation(rag_result)
            logger.info("")
    
    logger.info("=" * 60)
    logger.info("✅ All tests completed!")
    logger.info("=" * 60)
    logger.info("")
    logger.info("Next steps:")
    logger.info("1. Start RAG API: uvicorn api_rag:app --port 8001")
    logger.info("2. Upload documents via API")
    logger.info("3. Test queries with RAG")
    logger.info("4. Integrate with voice pipeline")

if __name__ == "__main__":
    asyncio.run(main())
