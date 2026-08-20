"""
Role-Based Access Control (RBAC)
==================================
Extends the existing JWT auth with fine-grained role checks.

Roles:
  patient   — upload images, use chat, view own history
  doctor    — all patient permissions + doctor dashboard
  researcher — all patient permissions + benchmark/research APIs
  admin     — all permissions + user/system management

Usage in routes:
    from core.rbac import require_role

    @router.get("/doctor/patients")
    def list_patients(payload: dict = Depends(require_role("doctor"))):
        ...
"""

from __future__ import annotations
from fastapi import Depends, HTTPException, status
from core.security import get_current_user_payload

# Role hierarchy — each role inherits all permissions of roles below it
ROLE_HIERARCHY = {
    "admin":      4,
    "researcher": 3,
    "doctor":     2,
    "patient":    1,
}


def require_role(minimum_role: str):
    """
    FastAPI dependency factory — enforces minimum role level.

    Usage:
        @router.get("/path")
        def endpoint(payload = Depends(require_role("doctor"))):
            ...
    """
    min_level = ROLE_HIERARCHY.get(minimum_role, 1)

    def _check(payload: dict = Depends(get_current_user_payload)) -> dict:
        # Admins always pass
        if payload.get("is_admin"):
            return payload

        # Check user role from JWT payload
        user_role  = payload.get("role", "patient")
        user_level = ROLE_HIERARCHY.get(user_role, 1)

        if user_level < min_level:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Requires '{minimum_role}' role or higher. "
                       f"Your role: '{user_role}'",
            )
        return payload

    return _check


def require_admin(payload: dict = Depends(get_current_user_payload)) -> dict:
    """Shortcut for admin-only endpoints (compatible with existing code)."""
    if not payload.get("is_admin"):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required",
        )
    return payload


def get_user_role(payload: dict) -> str:
    """Extract user role from JWT payload."""
    if payload.get("is_admin"):
        return "admin"
    return payload.get("role", "patient")
