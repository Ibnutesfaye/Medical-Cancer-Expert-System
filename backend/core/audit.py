"""
Audit Logging Middleware
=========================
Records every significant API call to the audit_logs table.
Used for security compliance, HIPAA audit trails, and debugging.

Automatically captures:
  - user_id (from JWT)
  - action (HTTP method + path)
  - resource (path params)
  - ip_address
  - user_agent
  - status_code
  - timestamp

Usage:
    # Add to main_v2.py after app creation:
    from core.audit import AuditMiddleware
    app.add_middleware(AuditMiddleware)
"""

from __future__ import annotations

import json
import time
from typing import Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

# Paths that should NOT be logged (too noisy or sensitive)
SKIP_PATHS = {
    "/health", "/metrics", "/docs", "/openapi.json",
    "/assets/", "/favicon.ico",
}

# Paths that ARE logged (security-relevant endpoints)
AUDIT_PATHS_PREFIX = (
    "/auth/", "/images/", "/inference/", "/benchmark/",
    "/doctor/", "/admin/", "/documents/",
)


class AuditMiddleware(BaseHTTPMiddleware):
    """
    Starlette middleware that logs security-relevant requests
    to the audit_logs table after each response.
    """

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Skip non-audit paths
        path = request.url.path
        if any(path.startswith(skip) for skip in SKIP_PATHS):
            return await call_next(request)
        if not any(path.startswith(p) for p in AUDIT_PATHS_PREFIX):
            return await call_next(request)

        t0 = time.time()
        response = await call_next(request)
        elapsed_ms = int((time.time() - t0) * 1000)

        # Extract user_id from JWT (best effort)
        user_id = None
        try:
            auth = request.headers.get("Authorization", "")
            if auth.startswith("Bearer "):
                import base64, json as _json
                token   = auth.split(" ")[1]
                payload = _json.loads(
                    base64.b64decode(token.split(".")[1] + "==").decode()
                )
                user_id = payload.get("user_id")
        except Exception:
            pass

        # Write to DB asynchronously (don't block the response)
        try:
            _write_audit_log(
                user_id    = user_id,
                action     = f"{request.method} {path}",
                resource   = path,
                ip_address = request.client.host if request.client else None,
                user_agent = request.headers.get("user-agent", "")[:256],
                status_code = response.status_code,
                details    = {"elapsed_ms": elapsed_ms},
            )
        except Exception:
            pass   # never let audit logging break the request

        return response


def _write_audit_log(**kwargs):
    """Write one audit log entry to the database."""
    try:
        from db.database import SessionLocal
        from models.audit_log import AuditLog
        with SessionLocal() as db:
            log = AuditLog(**kwargs)
            db.add(log)
            db.commit()
    except Exception:
        pass  # silently ignore DB errors in audit logging
