"""
Chat service — manages chat sessions and messages in MySQL.
"""

from sqlalchemy.orm import Session
from sqlalchemy import func
from fastapi import HTTPException
from models.chat import Chat, Message
from schemas.chat import ChatCreate, MessageCreate
from typing import List, Optional


def create_chat(db: Session, user_id: int, title: str = "New Chat") -> Chat:
    chat = Chat(user_id=user_id, title=title)
    db.add(chat)
    db.commit()
    db.refresh(chat)
    return chat


def get_chat(db: Session, chat_id: int, user_id: int) -> Chat:
    chat = db.query(Chat).filter(Chat.id == chat_id, Chat.user_id == user_id).first()
    if not chat:
        raise HTTPException(status_code=404, detail="Chat not found")
    return chat


def list_chats(db: Session, user_id: int, skip: int = 0, limit: int = 50) -> List[Chat]:
    return (
        db.query(Chat)
        .filter(Chat.user_id == user_id)
        .order_by(Chat.updated_at.desc())
        .offset(skip)
        .limit(limit)
        .all()
    )


def delete_chat(db: Session, chat_id: int, user_id: int) -> None:
    chat = get_chat(db, chat_id, user_id)
    db.delete(chat)
    db.commit()


def update_chat_title(db: Session, chat_id: int, user_id: int, title: str) -> Chat:
    chat = get_chat(db, chat_id, user_id)
    chat.title = title
    db.commit()
    db.refresh(chat)
    return chat


def add_message(
    db: Session,
    chat_id: int,
    role: str,
    content: str,
    source: Optional[str] = None,
    citations: Optional[list] = None,
) -> Message:
    msg = Message(
        chat_id=chat_id,
        role=role,
        content=content,
        source=source,
        citations=citations,
    )
    db.add(msg)
    # bump chat updated_at
    db.query(Chat).filter(Chat.id == chat_id).update({"updated_at": func.now()})
    db.commit()
    db.refresh(msg)
    return msg


def get_chat_messages(db: Session, chat_id: int, user_id: int) -> List[Message]:
    chat = get_chat(db, chat_id, user_id)
    return chat.messages


def get_or_create_chat(
    db: Session, user_id: int, chat_id: Optional[int], first_message: str
) -> Chat:
    """Return existing chat or create a new one with auto-generated title."""
    if chat_id:
        return get_chat(db, chat_id, user_id)
    title = first_message[:60] + ("…" if len(first_message) > 60 else "")
    return create_chat(db, user_id, title)
