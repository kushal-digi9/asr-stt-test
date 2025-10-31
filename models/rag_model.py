import logging
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
import uuid

logger = logging.getLogger(__name__)

class RAGModel:
    def __init__(
        self,
        qdrant_url: str = "http://localhost:6333",
        collection_name: str = "documents",
        embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        top_k: int = 3
    ):
        """
        Initialize RAG model with Qdrant vector store and embedding model.
        
        Args:
            qdrant_url: URL of Qdrant service
            collection_name: Name of the collection in Qdrant
            embedding_model: SentenceTransformer model name
            top_k: Number of documents to retrieve
        """
        self.qdrant_url = qdrant_url
        self.collection_name = collection_name
        self.top_k = top_k
        
        logger.info(f"🔍 Initializing RAG Model")
        logger.info(f"🔍 Qdrant URL: {qdrant_url}")
        logger.info(f"🔍 Collection: {collection_name}")
        logger.info(f"🔍 Embedding model: {embedding_model}")
        
        # Initialize Qdrant client
        self.client = QdrantClient(url=qdrant_url)
        
        # Initialize embedding model
        logger.info(f"🔍 Loading embedding model...")
        self.encoder = SentenceTransformer(embedding_model)
        self.embedding_dim = self.encoder.get_sentence_embedding_dimension()
        logger.info(f"🔍 Embedding dimension: {self.embedding_dim}")
        
        # Create collection if it doesn't exist
        self._ensure_collection()
        
        logger.info(f"✅ RAG Model initialized successfully")
    
    def _ensure_collection(self):
        """Create collection if it doesn't exist."""
        try:
            collections = self.client.get_collections().collections
            collection_names = [col.name for col in collections]
            
            if self.collection_name not in collection_names:
                logger.info(f"🔍 Creating collection: {self.collection_name}")
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.embedding_dim,
                        distance=Distance.COSINE
                    )
                )
                logger.info(f"✅ Collection created: {self.collection_name}")
            else:
                logger.info(f"✅ Collection already exists: {self.collection_name}")
        except Exception as e:
            logger.error(f"❌ Failed to ensure collection: {e}")
            raise
    
    def embed_text(self, text: str) -> List[float]:
        """Generate embedding for text."""
        return self.encoder.encode(text).tolist()
    
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for batch of texts."""
        return self.encoder.encode(texts).tolist()
    
    def add_documents(
        self,
        texts: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None
    ) -> List[str]:
        """
        Add documents to vector store.
        
        Args:
            texts: List of text chunks to add
            metadatas: Optional metadata for each chunk
            ids: Optional IDs for each chunk
            
        Returns:
            List of document IDs
        """
        logger.info(f"🔍 Adding {len(texts)} documents to vector store")
        
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in texts]
        
        if metadatas is None:
            metadatas = [{} for _ in texts]
        
        # Generate embeddings
        embeddings = self.embed_batch(texts)
        
        # Create points
        points = []
        for idx, (doc_id, text, embedding, metadata) in enumerate(zip(ids, texts, embeddings, metadatas)):
            payload = {
                "text": text,
                **metadata
            }
            points.append(
                PointStruct(
                    id=doc_id,
                    vector=embedding,
                    payload=payload
                )
            )
        
        # Upload to Qdrant
        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )
        
        logger.info(f"✅ Added {len(texts)} documents successfully")
        return ids
    
    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents for a query.
        
        Args:
            query: Query text
            top_k: Number of documents to retrieve
            filter_dict: Optional metadata filters
            
        Returns:
            List of retrieved documents with scores
        """
        if top_k is None:
            top_k = self.top_k
        
        logger.info(f"🔍 Retrieving top {top_k} documents for query")
        
        # Generate query embedding
        query_embedding = self.embed_text(query)
        
        # Build filter if provided
        query_filter = None
        if filter_dict:
            conditions = []
            for key, value in filter_dict.items():
                conditions.append(
                    FieldCondition(key=key, match=MatchValue(value=value))
                )
            query_filter = Filter(must=conditions)
        
        # Search
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            limit=top_k,
            query_filter=query_filter
        )
        
        # Format results
        documents = []
        for result in results:
            documents.append({
                "id": result.id,
                "text": result.payload.get("text", ""),
                "score": result.score,
                "metadata": {k: v for k, v in result.payload.items() if k != "text"}
            })
        
        logger.info(f"Retrieved {len(documents)} documents")
        return documents
    
    def delete_documents(self, ids: List[str]) -> bool:
        """Delete documents by IDs."""
        try:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=ids
            )
            logger.info(f"Deleted {len(ids)} documents")
            return True
        except Exception as e:
            logger.error(f"Failed to delete documents: {e}")
            return False
    
    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the collection."""
        try:
            info = self.client.get_collection(self.collection_name)
            return {
                "name": self.collection_name,
                "vectors_count": info.vectors_count,
                "points_count": info.points_count,
                "status": info.status
            }
        except Exception as e:
            logger.error(f"Failed to get collection info: {e}")
            return {}
