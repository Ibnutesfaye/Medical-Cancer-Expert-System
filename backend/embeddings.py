"""
Embedding generation module for Medical Cancer RAG Chatbot.

Uses sentence-transformers to generate 384-dimensional embeddings.
"""

from sentence_transformers import SentenceTransformer
from typing import List
import numpy as np


class EmbeddingModel:
    """
    Wrapper for sentence-transformers embedding model.
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the embedding model.
        
        Args:
            model_name: Name of the sentence-transformers model
        """
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
    
    def encode(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text.
        
        Args:
            text: Input text
            
        Returns:
            384-dimensional embedding vector
        """
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding
    
    def encode_batch(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for multiple texts.
        
        Args:
            texts: List of input texts
            
        Returns:
            Array of embedding vectors
        """
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        return embeddings
    
    def get_dimension(self) -> int:
        """Get embedding dimension."""
        return self.dimension


# Global embedding model instance (cached)
_embedding_model = None


def get_embedding_model(model_name: str = "all-MiniLM-L6-v2") -> EmbeddingModel:
    """
    Get or create the global embedding model instance.
    
    Args:
        model_name: Name of the model
        
    Returns:
        EmbeddingModel instance
    """
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = EmbeddingModel(model_name)
    return _embedding_model
