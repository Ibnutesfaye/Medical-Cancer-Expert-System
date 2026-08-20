"""
Document ingestion orchestration service.

Combines PDF parsing, chunking, embedding, and storage.
"""

from typing import Optional
from pydantic import BaseModel
from pdf_parser import PDFParser, PDFParserError
from text_chunker import TextChunker
from embeddings import get_embedding_model
from vector_db_faiss import VectorDatabase


class IngestionResult(BaseModel):
    """Result of document ingestion."""
    success: bool
    document_name: str
    chunks_created: int
    error: Optional[str] = None


class IngestionService:
    """
    Orchestrates the document ingestion pipeline.
    """
    
    def __init__(
        self,
        vector_db: VectorDatabase,
        chunk_size: int = 1000,
        overlap: int = 200,
        embedding_model_name: str = "all-MiniLM-L6-v2"
    ):
        """
        Initialize ingestion service.
        
        Args:
            vector_db: VectorDatabase instance
            chunk_size: Maximum tokens per chunk
            overlap: Token overlap between chunks
            embedding_model_name: Embedding model name
        """
        self.pdf_parser = PDFParser()
        self.text_chunker = TextChunker(chunk_size=chunk_size, overlap=overlap)
        self.embedding_model = get_embedding_model(embedding_model_name)
        self.vector_db = vector_db
    
    def ingest_pdf(self, pdf_bytes: bytes, filename: str) -> IngestionResult:
        """
        Ingest a PDF document into the vector database.
        
        Args:
            pdf_bytes: PDF file content
            filename: Document filename
            
        Returns:
            IngestionResult with success status and statistics
        """
        try:
            # Step 1: Parse PDF
            text = self.pdf_parser.parse_pdf(pdf_bytes, filename)
            
            if not text or len(text.strip()) == 0:
                return IngestionResult(
                    success=False,
                    document_name=filename,
                    chunks_created=0,
                    error="No text content could be extracted from PDF"
                )
            
            # Step 2: Chunk text
            chunks = self.text_chunker.chunk_text(text, filename)
            
            if len(chunks) == 0:
                return IngestionResult(
                    success=False,
                    document_name=filename,
                    chunks_created=0,
                    error="No chunks created from document"
                )
            
            # Step 3: Generate embeddings
            chunk_texts = [chunk.text for chunk in chunks]
            embeddings = self.embedding_model.encode_batch(chunk_texts)
            
            # Step 4: Store in vector database
            self.vector_db.store_chunks(chunks, embeddings)
            
            return IngestionResult(
                success=True,
                document_name=filename,
                chunks_created=len(chunks),
                error=None
            )
            
        except PDFParserError as e:
            return IngestionResult(
                success=False,
                document_name=filename,
                chunks_created=0,
                error=str(e)
            )
        except Exception as e:
            return IngestionResult(
                success=False,
                document_name=filename,
                chunks_created=0,
                error=f"Unexpected error during ingestion: {str(e)}"
            )
    
    def ingest_file(self, file_path: str) -> IngestionResult:
        """
        Ingest a PDF file from disk.
        
        Args:
            file_path: Path to PDF file
            
        Returns:
            IngestionResult
        """
        try:
            with open(file_path, 'rb') as f:
                pdf_bytes = f.read()
            
            from pathlib import Path
            filename = Path(file_path).name
            
            return self.ingest_pdf(pdf_bytes, filename)
        except Exception as e:
            return IngestionResult(
                success=False,
                document_name=file_path,
                chunks_created=0,
                error=f"Failed to read file: {str(e)}"
            )
