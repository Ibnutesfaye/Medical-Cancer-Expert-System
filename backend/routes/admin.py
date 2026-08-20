"""
Admin routes — user management.
"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from db.database import get_db
from schemas.user import UserCreate, UserRead, UserUpdate
from services import user_service
from core.security import require_admin

router = APIRouter(prefix="/admin", tags=["admin"])


@router.get("/users", response_model=list[UserRead])
def list_users(
    skip: int = 0,
    limit: int = 50,
    payload: dict = Depends(require_admin),
    db: Session = Depends(get_db),
):
    return user_service.list_users(db, skip=skip, limit=limit)


@router.post("/users", response_model=UserRead, status_code=201)
def create_user(
    data: UserCreate,
    payload: dict = Depends(require_admin),
    db: Session = Depends(get_db),
):
    return user_service.create_user(db, data)


@router.patch("/users/{user_id}", response_model=UserRead)
def update_user(
    user_id: int,
    data: UserUpdate,
    payload: dict = Depends(require_admin),
    db: Session = Depends(get_db),
):
    return user_service.update_user(db, user_id, data)


@router.delete("/users/{user_id}", status_code=204)
def delete_user(
    user_id: int,
    payload: dict = Depends(require_admin),
    db: Session = Depends(get_db),
):
    user_service.delete_user(db, user_id)
