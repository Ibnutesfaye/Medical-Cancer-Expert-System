"""
User model — stores registered users with hashed passwords.
"""

from sqlalchemy import Column, Integer, String, Boolean, DateTime, func
from sqlalchemy.orm import relationship
from db.database import Base


class User(Base):
    __tablename__ = "users"

    id         = Column(Integer, primary_key=True, index=True)
    username   = Column(String(64), unique=True, nullable=False, index=True)
    email      = Column(String(128), unique=True, nullable=True, index=True)
    full_name  = Column(String(128), nullable=True)
    password_hash = Column(String(256), nullable=False)
    is_admin   = Column(Boolean, default=False, nullable=False)
    is_active  = Column(Boolean, default=True, nullable=False)
    created_at = Column(DateTime, server_default=func.now(), nullable=False)
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())

    # Relationships
    chats            = relationship("Chat", back_populates="user", cascade="all, delete-orphan")
    documents        = relationship("Document", back_populates="uploaded_by_user", cascade="all, delete-orphan")
    image_analyses   = relationship("ImageAnalysis", back_populates="user", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<User id={self.id} username={self.username}>"
