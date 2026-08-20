"""
Document and DocumentChunk models.

Document      — metadata for an ingested PDF.
DocumentChunk — individual text chunks with FAISS vector index reference.
"""

from sqlalchemy import Column, Integer, String, Text, ForeignKey, DateTime, Float, func, BigInteger
from sqlalchemy.orm import relationship
from db.database import Base


class Document(Base):
    __tablename__ = "documents"

    id             = Column(Integer, primary_key=True, index=True)
    uploaded_by    = Column(Integer, ForeignKey("users.id", ondelete="SET NULL"), nullable=True, index=True)
    filename       = Column(String(256), nullable=False)
    original_name  = Column(String(256), nullable=False)
    file_size_bytes= Column(BigInteger, nullable=True)
    total_pages    = Column(Integer, nullable=True)
    total_chunks   = Column(Integer, default=0)
    status         = Column(String(32), default="processing")  # processing | ready | failed
    error_message  = Column(Text, nullable=True)
    created_at     = Column(DateTime, server_default=func.now(), nullable=False)

    # Relationships
    uploaded_by_user = relationship("User", back_populates="documents")
    chunks           = relationship("DocumentChunk", back_populates="document", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<Document id={self.id} filename={self.filename!r} status={self.status}>"


class DocumentChunk(Base):
    __tablename__ = "document_chunks"

    id              = Column(Integer, primary_key=True, index=True)
    document_id     = Column(Integer, ForeignKey("documents.id", ondelete="CASCADE"), nullable=False, index=True)
    chunk_index     = Column(Integer, nullable=False)          # position within document
    page_number     = Column(Integer, nullable=True)
    text            = Column(Text, nullable=False)
    token_count     = Column(Integer, nullable=True)
    faiss_index_id  = Column(BigInteger, nullable=True, index=True)  # ID in FAISS index
    created_at      = Column(DateTime, server_default=func.now(), nullable=False)

    # Relationships
    document = relationship("Document", back_populates="chunks")

    def __repr__(self):
        return f"<DocumentChunk id={self.id} doc_id={self.document_id} chunk={self.chunk_index}>"
