"""
Authentication service for Medical Cancer RAG Chatbot.
Handles JWT token generation, validation, and user registry.
"""

from datetime import datetime, timedelta
from typing import Optional
from jose import JWTError, jwt
import hashlib
from pydantic import BaseModel
from fastapi import HTTPException, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

security = HTTPBearer()


def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()


def verify_password(plain_password: str, hashed_password: str) -> bool:
    return hash_password(plain_password) == hashed_password


class TokenData(BaseModel):
    username: str
    is_admin: bool


class AuthService:
    def __init__(
        self,
        secret_key: str,
        algorithm: str = "HS256",
        expiration_hours: int = 24,
        admin_username: str = "admin",
        admin_password: str = "admin",
    ):
        self.secret_key = secret_key
        self.algorithm = algorithm
        self.expiration_hours = expiration_hours
        self.admin_username = admin_username
        self.admin_password_hash = hash_password(admin_password)
        # In-memory user store {username: {...}}
        self._users: dict = {
            admin_username: {
                "password_hash": hash_password(admin_password),
                "email": None,
                "full_name": "Admin",
                "is_admin": True,
                "is_active": True,
            }
        }

    def register_user(self, username: str, password: str, email: str = None, full_name: str = None) -> dict:
        if username in self._users:
            raise HTTPException(status_code=409, detail="Username already taken")
        if len(password) < 6:
            raise HTTPException(status_code=400, detail="Password must be at least 6 characters")
        self._users[username] = {
            "password_hash": hash_password(password),
            "email": email,
            "full_name": full_name or username,
            "is_admin": False,
            "is_active": True,
        }
        return {"username": username, "email": email, "full_name": full_name or username, "is_admin": False}

    def list_users(self) -> list:
        return [
            {
                "id": i + 1,
                "username": uname,
                "email": udata.get("email"),
                "full_name": udata.get("full_name"),
                "is_admin": udata.get("is_admin", False),
                "is_active": udata.get("is_active", True),
                "created_at": "2024-01-01T00:00:00",
                "updated_at": "2024-01-01T00:00:00",
            }
            for i, (uname, udata) in enumerate(self._users.items())
        ]

    def authenticate_user(self, username: str, password: str) -> Optional[TokenData]:
        user = self._users.get(username)
        if user and verify_password(password, user["password_hash"]):
            if not user.get("is_active", True):
                raise HTTPException(status_code=403, detail="Account is disabled")
            return TokenData(username=username, is_admin=user.get("is_admin", False))
        return None

    def create_token(self, token_data: TokenData) -> str:
        expires = datetime.utcnow() + timedelta(hours=self.expiration_hours)
        to_encode = {
            "sub": token_data.username,
            "is_admin": token_data.is_admin,
            "exp": expires,
        }
        return jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)

    def validate_token(self, token: str) -> TokenData:
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            username = payload.get("sub")
            is_admin = payload.get("is_admin", False)
            if username is None:
                raise HTTPException(status_code=401, detail="Invalid token")
            return TokenData(username=username, is_admin=is_admin)
        except JWTError:
            raise HTTPException(status_code=401, detail="Invalid token")

    def invalidate_token(self, token: str) -> None:
        pass

    def get_current_user(self, credentials: HTTPAuthorizationCredentials = Security(security)) -> TokenData:
        return self.validate_token(credentials.credentials)
