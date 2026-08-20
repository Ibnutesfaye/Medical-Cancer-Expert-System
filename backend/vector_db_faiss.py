"""
Vector database module using FAISS.

Handles storage and retrieval of document chunks with embeddings.
"""

import faiss
import numpy as np
import pickle
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
from text_chunker import TextChunk

logger = logging.getLogger(__name__)


class VectorDatabase:
    """
    FAISS wrapper for vector storage and semantic search.
    """
    
    def __init__(self, persist_directory: str = "./faiss_db"):
        """
        Initialize FAISS index.
        
        Args:
            persist_directory: Directory for persistent storage
        """
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(exist_ok=True)
        
        self.index_path = self.persist_directory / "index.faiss"
        self.metadata_path = self.persist_directory / "metadata.pkl"
        
        # Initialize or load index
        self.dimension = 384  # all-MiniLM-L6-v2 dimension
        self.index = None
        self.metadata_store = []  # List of metadata dicts
        
        self._load_or_create_index()
    
    def _load_or_create_index(self):
        """Load existing index or create new one."""
        if self.index_path.exists() and self.metadata_path.exists():
            # Load existing
            self.index = faiss.read_index(str(self.index_path))
            with open(self.metadata_path, 'rb') as f:
                self.metadata_store = pickle.load(f)
        else:
            # Create new index (L2 distance, will convert to cosine similarity)
            self.index = faiss.IndexFlatIP(self.dimension)  # Inner product for cosine
            self.metadata_store = []
    
    def _save_index(self):
        """Save index and metadata to disk."""
        faiss.write_index(self.index, str(self.index_path))
        with open(self.metadata_path, 'wb') as f:
            pickle.dump(self.metadata_store, f)
    
    def store_chunks(self, chunks: List[TextChunk], embeddings: np.ndarray) -> None:
        """
        Store document chunks with embeddings.
        
        Args:
            chunks: List of TextChunk objects
            embeddings: Array of embedding vectors
        """
        if len(chunks) == 0:
            return
        
        # Normalize embeddings for cosine similarity
        embeddings_array = np.array(embeddings).astype('float32')
        faiss.normalize_L2(embeddings_array)
        
        # Add to index
        self.index.add(embeddings_array)
        
        # Store metadata
        for chunk in chunks:
            self.metadata_store.append({
                'id': f"{chunk.document_name}_{chunk.chunk_index}",
                'text': chunk.text,
                'document_name': chunk.document_name,
                'page_number': chunk.page_number,
                'chunk_index': chunk.chunk_index,
                'token_count': chunk.token_count
            })
        
        # Persist to disk
        self._save_index()
    
    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        threshold: float = 0.2
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
        if self.index.ntotal == 0:
            logger.warning("FAISS index is empty!")
            return []
            
        logger.info(f"FAISS index size: {self.index.ntotal} vectors")
        
        # Normalize query for cosine similarity
        query_array = np.array([query_embedding]).astype('float32')
        faiss.normalize_L2(query_array)
        
        # Search
        k = min(top_k, self.index.ntotal)
        distances, indices = self.index.search(query_array, k)
        
        logger.info(f"Raw FAISS similarity scores: {distances[0]}")
        logger.info(f"Raw FAISS retrieved indices: {indices[0]}")
        
        # Format results
        all_results = []
        for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
            if idx < len(self.metadata_store) and idx >= 0:
                similarity = float(dist)  # Already cosine similarity due to normalization
                metadata = self.metadata_store[idx]
                all_results.append({
                    'id': metadata['id'],
                    'text': metadata['text'],
                    'metadata': {
                        'document_name': metadata['document_name'],
                        'page_number': metadata['page_number'],
                        'chunk_index': metadata['chunk_index']
                    },
                    'similarity_score': similarity
                })
        
        # Apply threshold conditionally: avoid returning [] if possible
        filtered_results = [r for r in all_results if r['similarity_score'] >= threshold]
        
        if not filtered_results and all_results:
            logger.info("No chunks met the threshold requirement. Returning top 1 result to avoid empty response.")
            results = [all_results[0]]  # Return top 1 result
        else:
            results = filtered_results
            
        logger.info(f"Returning {len(results)} chunks after processing.")
        
        return results
    
    def get_by_id(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a chunk by ID.
        
        Args:
            chunk_id: Chunk identifier
            
        Returns:
            Chunk data or None if not found
        """
        for metadata in self.metadata_store:
            if metadata['id'] == chunk_id:
                return {
                    'id': metadata['id'],
                    'text': metadata['text'],
                    'metadata': {
                        'document_name': metadata['document_name'],
                        'page_number': metadata['page_number'],
                        'chunk_index': metadata['chunk_index']
                    }
                }
        return None
    
    def count(self) -> int:
        """Get total number of chunks in database."""
        return self.index.ntotal
    
    def delete_all(self) -> None:
        """Delete all chunks from the index."""
        self.index = faiss.IndexFlatIP(self.dimension)
        self.metadata_store = []
        self._save_index()
