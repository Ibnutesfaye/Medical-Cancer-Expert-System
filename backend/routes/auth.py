"""
Auth routes — login, register, profile.
"""

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from db.database import get_db
from schemas.user import LoginRequest, LoginResponse, UserCreate, UserRead, UserUpdate
from services import user_service
from core.security import get_current_user_payload

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/login", response_model=LoginResponse)
def login(request: LoginRequest, db: Session = Depends(get_db)):
    """Authenticate and return JWT token."""
    result = user_service.login_user(db, request.username, request.password)
    return LoginResponse(
        access_token=result["access_token"],
        token_type="bearer",
        expires_in=result["expires_in"],
        user=UserRead.model_validate(result["user"]),
    )


@router.post("/register", response_model=UserRead, status_code=201)
def register(data: UserCreate, db: Session = Depends(get_db)):
    """Register a new user account."""
    # Only allow admin creation via env-seeded admin; regular users are non-admin
    data.is_admin = False
    return user_service.create_user(db, data)


@router.get("/me", response_model=UserRead)
def me(
    payload: dict = Depends(get_current_user_payload),
    db: Session = Depends(get_db),
):
    """Return the currently authenticated user."""
    return user_service.get_user_by_id(db, payload["user_id"])


@router.patch("/me", response_model=UserRead)
def update_me(
    data: UserUpdate,
    payload: dict = Depends(get_current_user_payload),
    db: Session = Depends(get_db),
):
    """Update own profile."""
    return user_service.update_user(db, payload["user_id"], data)


@router.post("/logout")
def logout():
    """Stateless JWT — client just discards the token."""
    return {"message": "Logged out successfully"}
