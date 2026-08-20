"""
Configuration management for the Medical Cancer Expert System.

Loads configuration from environment variables with sensible defaults.
"""

import os
from typing import Optional
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class RAGConfig(BaseModel):
    """Configuration for RAG pipeline parameters."""
    
    # Retrieval settings
    top_k: int = Field(default=5, description="Number of chunks to retrieve")
    similarity_threshold: float = Field(default=0.7, description="Minimum similarity score")
    
    # Chunking settings
    chunk_size: int = Field(default=1000, description="Maximum tokens per chunk")
    chunk_overlap: int = Field(default=200, description="Token overlap between chunks")
    
    # LLM settings
    temperature: float = Field(default=0.2, description="LLM temperature for factual responses")
    max_tokens: int = Field(default=4096, description="Maximum context window tokens")
    model_name: str = Field(default="llama-3.3-70b-versatile", description="LLM model identifier")
    
    # Conversation settings
    max_history_pairs: int = Field(default=10, description="Maximum conversation history pairs")
    context_history_pairs: int = Field(default=3, description="History pairs to include in context")


class AppConfig(BaseModel):
    """Application-wide configuration."""
    
    # API Keys
    groq_api_key: Optional[str] = Field(default=None, description="Groq API key")
    openai_api_key: Optional[str] = Field(default=None, description="OpenAI API key")
    
    # Database
    vector_db_path: str = Field(default="./chroma_db", description="ChromaDB storage path")
    
    # Embedding Model
    embedding_model: str = Field(default="all-MiniLM-L6-v2", description="Sentence transformer model")
    
    # Authentication
    jwt_secret_key: str = Field(default="change-me-in-production", description="JWT secret key")
    jwt_algorithm: str = Field(default="HS256", description="JWT algorithm")
    jwt_expiration_hours: int = Field(default=24, description="JWT token expiration in hours")
    
    # Admin credentials
    admin_username: str = Field(default="admin", description="Admin username")
    admin_password: Optional[str] = Field(
        default=None,
        description="Admin bootstrap password; no admin is seeded when unset",
    )

    # External search fallback
    fallback_enabled: bool = Field(default=True, description="Enable external search fallback")

    # RAG configuration
    rag: RAGConfig = Field(default_factory=RAGConfig)
    
    @classmethod
    def from_env(cls) -> "AppConfig":
        """Load configuration from environment variables."""
        return cls(
            groq_api_key=os.getenv("GROQ_API_KEY"),
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            vector_db_path=os.getenv("VECTOR_DB_PATH", "./chroma_db"),
            embedding_model=os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
            jwt_secret_key=os.getenv("JWT_SECRET_KEY", "change-me-in-production"),
            jwt_algorithm=os.getenv("JWT_ALGORITHM", "HS256"),
            jwt_expiration_hours=int(os.getenv("JWT_EXPIRATION_HOURS", "24")),
            admin_username=os.getenv("ADMIN_USERNAME", "admin"),
            admin_password=os.getenv("ADMIN_PASSWORD") or None,
            fallback_enabled=os.getenv("FALLBACK_ENABLED", "true").lower() == "true",
            rag=RAGConfig(
                top_k=int(os.getenv("RAG_TOP_K", "5")),
                similarity_threshold=float(os.getenv("RAG_SIMILARITY_THRESHOLD", "0.7")),
                chunk_size=int(os.getenv("RAG_CHUNK_SIZE", "1000")),
                chunk_overlap=int(os.getenv("RAG_CHUNK_OVERLAP", "200")),
                temperature=float(os.getenv("RAG_TEMPERATURE", "0.2")),
                max_tokens=int(os.getenv("RAG_MAX_TOKENS", "4096")),
                model_name=os.getenv("RAG_MODEL_NAME", "llama-3.3-70b-versatile"),
                max_history_pairs=int(os.getenv("RAG_MAX_HISTORY_PAIRS", "10")),
                context_history_pairs=int(os.getenv("RAG_CONTEXT_HISTORY_PAIRS", "3")),
            )
        )


# Global configuration instance
config = AppConfig.from_env()
