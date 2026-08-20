"""
Chat routes — streaming RAG chat with MySQL persistence.
"""

import json
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from db.database import get_db
from schemas.chat import ChatRequest, ChatRead, ChatSummary, MessageRead
from services import chat_service
from core.security import get_current_user_payload

# These are the existing RAG services — imported from the original modules
from rag_pipeline import RAGPipeline, Message as RAGMessage
from vector_db_faiss import VectorDatabase
from llm_service import LLMService
from config import config

router = APIRouter(prefix="/chat", tags=["chat"])


def _get_rag_pipeline() -> RAGPipeline:
    """Build RAG pipeline (singleton via module-level import in main)."""
    from main_v2 import rag_pipeline
    return rag_pipeline


# ── Chat session CRUD ─────────────────────────────────────────────────────────

@router.get("/sessions", response_model=list[ChatSummary])
def list_sessions(
    payload: dict = Depends(get_current_user_payload),
    db: Session = Depends(get_db),
):
    chats = chat_service.list_chats(db, payload["user_id"])
    result = []
    for c in chats:
        result.append(ChatSummary(
            id=c.id,
            title=c.title,
            created_at=c.created_at,
            updated_at=c.updated_at,
            message_count=len(c.messages),
        ))
    return result


@router.get("/sessions/{chat_id}", response_model=ChatRead)
def get_session(
    chat_id: int,
    payload: dict = Depends(get_current_user_payload),
    db: Session = Depends(get_db),
):
    return chat_service.get_chat(db, chat_id, payload["user_id"])


@router.delete("/sessions/{chat_id}", status_code=204)
def delete_session(
    chat_id: int,
    payload: dict = Depends(get_current_user_payload),
    db: Session = Depends(get_db),
):
    chat_service.delete_chat(db, chat_id, payload["user_id"])


# ── Streaming chat ────────────────────────────────────────────────────────────

@router.post("")
async def chat(
    request: ChatRequest,
    payload: dict = Depends(get_current_user_payload),
    db: Session = Depends(get_db),
):
    """
    Stream a RAG response and persist both user message and assistant reply to MySQL.
    """
    if not request.query or not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    user_id = payload["user_id"]

    # Get or create chat session
    chat_obj = chat_service.get_or_create_chat(db, user_id, request.chat_id, request.query)

    # Save user message immediately
    chat_service.add_message(db, chat_obj.id, "user", request.query)

    # Build RAG history from request
    history = [RAGMessage(role=m.role, content=m.content) for m in (request.conversation_history or [])]

    rag = _get_rag_pipeline()
    chat_id = chat_obj.id

    async def generate():
        full_content = ""
        source = "document"
        citations = []

        try:
            async for token in rag.process_query(request.query, history):
                if token.startswith("[SOURCE_MARKER]"):
                    source = token.replace("[SOURCE_MARKER]", "")
                    continue
                full_content += token
                escaped = token.replace('\n', '\\n').replace('\r', '')
                yield f"data: {escaped}\n\n"

            # Fetch citations
            citations = [c.dict() for c in rag.get_citations(request.query)]

            yield f"data: [CITATIONS]{json.dumps(citations)}\n\n"
            yield f"data: [SOURCE]{source}\n\n"
            yield f"data: [CHAT_ID]{chat_id}\n\n"
            yield "data: [DONE]\n\n"

        except Exception as e:
            yield f"data: [ERROR]Failed to generate response: {str(e)}\n\n"
            return

        # Persist assistant reply after streaming completes
        from db.database import SessionLocal
        try:
            with SessionLocal() as async_db:
                chat_service.add_message(
                    async_db, chat_id, "assistant", full_content,
                    source=source, citations=citations,
                )
        except Exception as e:
            print(f"Failed to save assistant message: {e}")

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
    )
