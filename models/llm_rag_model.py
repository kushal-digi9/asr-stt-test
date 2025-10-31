import httpx
import logging
from typing import Optional, List, Dict, Any
from models.llm_model import LLMModel
from models.rag_model import RAGModel

logger = logging.getLogger(__name__)

class LLMRAGModel(LLMModel):
    """Extended LLM model with RAG capabilities."""
    
    def __init__(
        self,
        ollama_url: str = "http://localhost:11434",
        model_name: str = "llama3.2:1b",
        echo_mode: bool = False,
        timeout: float = 30.0,
        qdrant_url: str = "http://localhost:6333",
        enable_rag: bool = True,
        top_k_docs: int = 3
    ):
        """
        Initialize LLM with RAG support.
        
        Args:
            ollama_url: URL of Ollama service
            model_name: Name of the model
            echo_mode: Echo mode for testing
            timeout: Request timeout
            qdrant_url: URL of Qdrant service
            enable_rag: Whether to enable RAG
            top_k_docs: Number of documents to retrieve
        """
        super().__init__(ollama_url, model_name, echo_mode, timeout)
        
        self.enable_rag = enable_rag
        self.top_k_docs = top_k_docs
        
        if self.enable_rag:
            logger.info(f"🔍 Initializing RAG capabilities")
            self.rag_model = RAGModel(
                qdrant_url=qdrant_url,
                top_k=top_k_docs
            )
            logger.info(f"✅ RAG enabled with top_k={top_k_docs}")
        else:
            self.rag_model = None
            logger.info(f"⚠️ RAG disabled")
    
    async def generate_response_with_rag(
        self,
        prompt: str,
        max_tokens: int = 150,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Generate response using RAG pipeline.
        
        Args:
            prompt: User query
            max_tokens: Maximum tokens to generate
            filter_dict: Optional metadata filters for retrieval
            
        Returns:
            Dict with answer and retrieved contexts
        """
        logger.info(f"🔍 Generating RAG response")
        logger.info(f"🔍 Query: '{prompt[:100]}{'...' if len(prompt) > 100 else ''}'")
        
        if not self.enable_rag or self.rag_model is None:
            logger.warning("⚠️ RAG not enabled, falling back to standard generation")
            answer = await self.generate_response(prompt, max_tokens)
            return {
                "answer": answer,
                "contexts": [],
                "rag_enabled": False
            }
        
        try:
            # Retrieve relevant documents
            logger.info(f"🔍 Retrieving relevant documents...")
            retrieved_docs = self.rag_model.retrieve(
                query=prompt,
                top_k=self.top_k_docs,
                filter_dict=filter_dict
            )
            
            if not retrieved_docs:
                logger.warning("⚠️ No documents retrieved, using base LLM")
                answer = await self.generate_response(prompt, max_tokens)
                return {
                    "answer": answer,
                    "contexts": [],
                    "rag_enabled": True,
                    "docs_retrieved": 0
                }
            
            # Build context from retrieved documents
            context_texts = [doc["text"] for doc in retrieved_docs]
            context_str = "\n\n".join([
                f"संदर्भ {i+1}:\n{text}"
                for i, text in enumerate(context_texts)
            ])
            
            logger.info(f"🔍 Retrieved {len(retrieved_docs)} documents")
            logger.info(f"🔍 Context length: {len(context_str)} characters")
            
            # Create RAG prompt
            rag_prompt = self._build_rag_prompt(prompt, context_str)
            
            # Generate response
            answer = await self.generate_response(rag_prompt, max_tokens)
            
            logger.info(f"✅ RAG response generated")
            
            return {
                "answer": answer,
                "contexts": retrieved_docs,
                "rag_enabled": True,
                "docs_retrieved": len(retrieved_docs)
            }
            
        except Exception as e:
            logger.error(f"❌ RAG generation failed: {e}")
            # Fallback to base LLM
            answer = await self.generate_response(prompt, max_tokens)
            return {
                "answer": answer,
                "contexts": [],
                "rag_enabled": True,
                "error": str(e)
            }
    
    def _build_rag_prompt(self, query: str, context: str) -> str:
        """
        Build prompt with retrieved context.
        
        Args:
            query: User query
            context: Retrieved context documents
            
        Returns:
            Formatted prompt
        """
        prompt = f"""आप एक सहायक अस्पताल रिसेप्शनिस्ट हैं। नीचे दिए गए संदर्भ का उपयोग करके रोगी के प्रश्न का उत्तर दें।

संदर्भ जानकारी:
{context}

रोगी का प्रश्न: {query}

कृपया संदर्भ के आधार पर संक्षिप्त और स्पष्ट उत्तर दें। यदि संदर्भ में जानकारी नहीं है, तो विनम्रता से बताएं।

उत्तर:"""
        
        return prompt
    
    def get_rag_stats(self) -> Dict[str, Any]:
        """Get RAG model statistics."""
        if not self.enable_rag or self.rag_model is None:
            return {"rag_enabled": False}
        
        try:
            collection_info = self.rag_model.get_collection_info()
            return {
                "rag_enabled": True,
                "collection_info": collection_info,
                "top_k": self.top_k_docs
            }
        except Exception as e:
            logger.error(f"❌ Failed to get RAG stats: {e}")
            return {"rag_enabled": True, "error": str(e)}
