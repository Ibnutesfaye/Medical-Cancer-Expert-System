"""
Chat and Message models.

Chat     — a conversation session belonging to a user.
Message  — individual messages within a chat (user or assistant).
"""

from sqlalchemy import Column, Integer, String, Text, ForeignKey, DateTime, Float, func, JSON
from sqlalchemy.orm import relationship
from db.database import Base


class Chat(Base):
    __tablename__ = "chats"

    id         = Column(Integer, primary_key=True, index=True)
    user_id    = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    title      = Column(String(256), nullable=False, default="New Chat")
    created_at = Column(DateTime, server_default=func.now(), nullable=False)
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())

    # Relationships
    user     = relationship("User", back_populates="chats")
    messages = relationship("Message", back_populates="chat", cascade="all, delete-orphan", order_by="Message.created_at")

    def __repr__(self):
        return f"<Chat id={self.id} user_id={self.user_id} title={self.title!r}>"


class Message(Base):
    __tablename__ = "messages"

    id           = Column(Integer, primary_key=True, index=True)
    chat_id      = Column(Integer, ForeignKey("chats.id", ondelete="CASCADE"), nullable=False, index=True)
    role         = Column(String(16), nullable=False)          # "user" | "assistant"
    content      = Column(Text, nullable=False)
    source       = Column(String(32), nullable=True)           # "document" | "wikipedia" | "pubmed" | "llm"
    citations    = Column(JSON, nullable=True)                 # serialised list of citation dicts
    created_at   = Column(DateTime, server_default=func.now(), nullable=False)

    # Relationships
    chat = relationship("Chat", back_populates="messages")

    def __repr__(self):
        return f"<Message id={self.id} chat_id={self.chat_id} role={self.role}>"
