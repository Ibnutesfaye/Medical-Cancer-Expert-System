"""
Document ingestion routes — upload PDF, list documents.
"""

from fastapi import APIRouter, Depends, UploadFile, File, HTTPException
from sqlalchemy.orm import Session

from db.database import get_db
from schemas.document import DocumentRead
from services import document_service
from core.security import get_current_user_payload, require_admin

router = APIRouter(prefix="/documents", tags=["documents"])


def _get_ingestion_service():
    from main_v2 import ingestion_service
    return ingestion_service


@router.post("/ingest", response_model=DocumentRead, status_code=201)
async def ingest_document(
    file: UploadFile = File(...),
    payload: dict = Depends(require_admin),
    db: Session = Depends(get_db),
):
    """Upload and ingest a PDF into FAISS + MySQL. Admin only."""
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    content = await file.read()
    if len(content) > 50 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File size exceeds 50 MB")

    # Create DB record first (status=processing)
    doc_record = document_service.create_document(
        db,
        filename=file.filename,
        original_name=file.filename,
        uploaded_by=payload["user_id"],
        file_size_bytes=len(content),
    )

    # Run ingestion pipeline
    svc = _get_ingestion_service()
    result = svc.ingest_pdf(content, file.filename)

    if not result.success:
        document_service.mark_document_failed(db, doc_record.id, result.error or "Unknown error")
        raise HTTPException(status_code=400, detail=result.error)

    # Mark ready and save chunk metadata
    document_service.mark_document_ready(db, doc_record.id, result.chunks_created)

    db.refresh(doc_record)
    return doc_record


@router.get("/", response_model=list[DocumentRead])
def list_documents(
    skip: int = 0,
    limit: int = 50,
    payload: dict = Depends(require_admin),
    db: Session = Depends(get_db),
):
    return document_service.list_documents(db, skip=skip, limit=limit)


@router.delete("/{doc_id}", status_code=204)
def delete_document(
    doc_id: int,
    payload: dict = Depends(require_admin),
    db: Session = Depends(get_db),
):
    document_service.delete_document(db, doc_id)
