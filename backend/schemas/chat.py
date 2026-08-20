from pydantic import BaseModel
from typing import Optional, List, Any
from datetime import datetime


class MessageCreate(BaseModel):
    role: str
    content: str
    source: Optional[str] = None
    citations: Optional[List[Any]] = None


class MessageRead(BaseModel):
    id: int
    chat_id: int
    role: str
    content: str
    source: Optional[str]
    citations: Optional[List[Any]]
    created_at: datetime

    model_config = {"from_attributes": True}


class ChatCreate(BaseModel):
    title: str = "New Chat"


class ChatRead(BaseModel):
    id: int
    user_id: int
    title: str
    created_at: datetime
    updated_at: Optional[datetime]
    messages: List[MessageRead] = []

    model_config = {"from_attributes": True}


class ChatSummary(BaseModel):
    """Lightweight chat listing — no messages."""
    id: int
    title: str
    created_at: datetime
    updated_at: Optional[datetime]
    message_count: int = 0

    model_config = {"from_attributes": True}


class ChatRequest(BaseModel):
    query: str
    chat_id: Optional[int] = None          # None = start new chat
    conversation_history: Optional[List[MessageCreate]] = []
