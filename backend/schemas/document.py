from pydantic import BaseModel
from typing import Optional
from datetime import datetime


class DocumentRead(BaseModel):
    id: int
    filename: str
    original_name: str
    file_size_bytes: Optional[int]
    total_pages: Optional[int]
    total_chunks: int
    status: str
    error_message: Optional[str]
    created_at: datetime

    model_config = {"from_attributes": True}


class DocumentChunkRead(BaseModel):
    id: int
    document_id: int
    chunk_index: int
    page_number: Optional[int]
    text: str
    token_count: Optional[int]
    faiss_index_id: Optional[int]

    model_config = {"from_attributes": True}
