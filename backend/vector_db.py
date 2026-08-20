"""
Vector database module using ChromaDB.

Handles storage and retrieval of document chunks with embeddings.
"""

import chromadb
from chromadb.config import Settings
from typing import List, Dict, Any, Optional
from text_chunker import TextChunk
import numpy as np
from datetime import datetime


class VectorDatabase:
    """
    ChromaDB wrapper for vector storage and semantic search.
    """
    
    def __init__(self, persist_directory: str = "./chroma_db"):
        """
        Initialize ChromaDB client.
        
        Args:
            persist_directory: Directory for persistent storage
        """
        self.client = chromadb.Client(Settings(
            persist_directory=persist_directory,
            anonymized_telemetry=False
        ))
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name="medical_documents",
            metadata={"hnsw:space": "cosine"}  # Use cosine similarity
        )
    
    def store_chunks(self, chunks: List[TextChunk], embeddings: np.ndarray) -> None:
        """
        Store document chunks with embeddings.
        
        Args:
            chunks: List of TextChunk objects
            embeddings: Array of embedding vectors
        """
        if len(chunks) == 0:
            return
        
        # Prepare data for ChromaDB
        ids = [f"{chunk.document_name}_{chunk.chunk_index}" for chunk in chunks]
        documents = [chunk.text for chunk in chunks]
        metadatas = [
            {
                "document_name": chunk.document_name,
                "page_number": chunk.page_number,
                "chunk_index": chunk.chunk_index,
                "token_count": chunk.token_count,
                "created_at": datetime.now().isoformat()
            }
            for chunk in chunks
        ]
        
        # Convert embeddings to list format
        embeddings_list = embeddings.tolist() if isinstance(embeddings, np.ndarray) else embeddings
        
        # Add to collection
        self.collection.add(
            ids=ids,
            embeddings=embeddings_list,
            documents=documents,
            metadatas=metadatas
        )
    
    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        threshold: float = 0.7
    ) -> List[Dict[str, Any]]:
        """
        Perform semantic search.
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return
            threshold: Minimum similarity score
            
        Returns:
            List of search results with metadata and scores
        """
        # Convert to list format
        query_embedding_list = query_embedding.tolist() if isinstance(query_embedding, np.ndarray) else query_embedding
        
        # Query collection
        results = self.collection.query(
            query_embeddings=[query_embedding_list],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )
        
        # Format results
        formatted_results = []
        if results and results['ids'] and len(results['ids'][0]) > 0:
            for i in range(len(results['ids'][0])):
                # Convert distance to similarity score (cosine distance to similarity)
                distance = results['distances'][0][i]
                similarity = 1 - distance  # ChromaDB returns cosine distance
                
                # Filter by threshold
                if similarity >= threshold:
                    formatted_results.append({
                        'id': results['ids'][0][i],
                        'text': results['documents'][0][i],
                        'metadata': results['metadatas'][0][i],
                        'similarity_score': similarity
                    })
        
        return formatted_results
    
    def get_by_id(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a chunk by ID.
        
        Args:
            chunk_id: Chunk identifier
            
        Returns:
            Chunk data or None if not found
        """
        try:
            result = self.collection.get(
                ids=[chunk_id],
                include=["documents", "metadatas", "embeddings"]
            )
            
            if result and result['ids'] and len(result['ids']) > 0:
                return {
                    'id': result['ids'][0],
                    'text': result['documents'][0],
                    'metadata': result['metadatas'][0],
                    'embedding': result['embeddings'][0] if result['embeddings'] else None
                }
        except Exception:
            pass
        
        return None
    
    def count(self) -> int:
        """Get total number of chunks in database."""
        return self.collection.count()
    
    def delete_all(self) -> None:
        """Delete all chunks from the collection."""
        # Get all IDs
        all_data = self.collection.get()
        if all_data and all_data['ids']:
            self.collection.delete(ids=all_data['ids'])
