"""
User service — CRUD operations for users.
"""

from sqlalchemy.orm import Session
from fastapi import HTTPException
from models.user import User
from schemas.user import UserCreate, UserUpdate
from core.security import hash_password, verify_password, create_access_token
from datetime import timedelta
import os


EXPIRE_HOURS = int(os.getenv("JWT_EXPIRATION_HOURS", "24"))


def get_user_by_id(db: Session, user_id: int) -> User:
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user


def get_user_by_username(db: Session, username: str) -> User | None:
    return db.query(User).filter(User.username == username).first()


def create_user(db: Session, data: UserCreate) -> User:
    if get_user_by_username(db, data.username):
        raise HTTPException(status_code=409, detail="Username already taken")
    user = User(
        username=data.username,
        email=data.email,
        full_name=data.full_name,
        password_hash=hash_password(data.password),
        is_admin=data.is_admin,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


def authenticate_user(db: Session, username: str, password: str) -> User:
    user = get_user_by_username(db, username)
    if not user or not verify_password(password, user.password_hash):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    if not user.is_active:
        raise HTTPException(status_code=403, detail="Account is disabled")
    return user


def login_user(db: Session, username: str, password: str) -> dict:
    user = authenticate_user(db, username, password)
    token = create_access_token(
        data={
            "sub": user.username,
            "username": user.username,
            "user_id": user.id,
            "is_admin": user.is_admin,
            "role": "admin" if user.is_admin else "user"
        },
        expires_delta=timedelta(hours=EXPIRE_HOURS),
    )
    return {
        "access_token": token,
        "token_type": "bearer",
        "expires_in": EXPIRE_HOURS * 3600,
        "user": user,
    }


def update_user(db: Session, user_id: int, data: UserUpdate) -> User:
    user = get_user_by_id(db, user_id)
    if data.email is not None:
        user.email = data.email
    if data.full_name is not None:
        user.full_name = data.full_name
    if data.password is not None:
        user.password_hash = hash_password(data.password)
    if data.is_active is not None:
        user.is_active = data.is_active
    db.commit()
    db.refresh(user)
    return user


def list_users(db: Session, skip: int = 0, limit: int = 50):
    return db.query(User).offset(skip).limit(limit).all()


def delete_user(db: Session, user_id: int) -> None:
    user = get_user_by_id(db, user_id)
    db.delete(user)
    db.commit()
