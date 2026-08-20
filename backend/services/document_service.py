"""
Document service — saves PDF metadata and chunk references to MySQL,
while embeddings live in FAISS.
"""

from sqlalchemy.orm import Session
from fastapi import HTTPException
from models.document import Document, DocumentChunk
from typing import List, Optional


def create_document(
    db: Session,
    filename: str,
    original_name: str,
    uploaded_by: Optional[int],
    file_size_bytes: Optional[int] = None,
) -> Document:
    doc = Document(
        filename=filename,
        original_name=original_name,
        uploaded_by=uploaded_by,
        file_size_bytes=file_size_bytes,
        status="processing",
    )
    db.add(doc)
    db.commit()
    db.refresh(doc)
    return doc


def mark_document_ready(
    db: Session,
    doc_id: int,
    total_chunks: int,
    total_pages: Optional[int] = None,
) -> Document:
    doc = db.query(Document).filter(Document.id == doc_id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    doc.status = "ready"
    doc.total_chunks = total_chunks
    doc.total_pages = total_pages
    db.commit()
    db.refresh(doc)
    return doc


def mark_document_failed(db: Session, doc_id: int, error: str) -> None:
    doc = db.query(Document).filter(Document.id == doc_id).first()
    if doc:
        doc.status = "failed"
        doc.error_message = error
        db.commit()


def save_chunks(
    db: Session,
    doc_id: int,
    chunks: list,          # list of TextChunk objects from text_chunker
    faiss_ids: List[int],  # FAISS internal IDs assigned during storage
) -> None:
    """Persist chunk metadata to MySQL after FAISS storage."""
    for i, (chunk, faiss_id) in enumerate(zip(chunks, faiss_ids)):
        db_chunk = DocumentChunk(
            document_id=doc_id,
            chunk_index=i,
            page_number=getattr(chunk, "page_number", None),
            text=chunk.text,
            token_count=getattr(chunk, "token_count", None),
            faiss_index_id=faiss_id,
        )
        db.add(db_chunk)
    db.commit()


def list_documents(db: Session, skip: int = 0, limit: int = 50) -> List[Document]:
    return db.query(Document).order_by(Document.created_at.desc()).offset(skip).limit(limit).all()


def get_document(db: Session, doc_id: int) -> Document:
    doc = db.query(Document).filter(Document.id == doc_id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    return doc


def delete_document(db: Session, doc_id: int) -> None:
    doc = get_document(db, doc_id)
    db.delete(doc)
    db.commit()
