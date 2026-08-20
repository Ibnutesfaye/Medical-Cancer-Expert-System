"""
AuditLog ORM Model
Records every significant action for security and compliance.
"""
from sqlalchemy import Column, BigInteger, Integer, String, DateTime, JSON
from sqlalchemy.sql import func
from db.database import Base


class AuditLog(Base):
    __tablename__ = "audit_logs"

    id          = Column(BigInteger, primary_key=True, index=True)
    user_id     = Column(Integer, nullable=True)
    action      = Column(String(128), nullable=False, index=True)
    resource    = Column(String(256), nullable=True)
    ip_address  = Column(String(45), nullable=True)
    user_agent  = Column(String(256), nullable=True)
    status_code = Column(Integer, nullable=True)
    details     = Column(JSON, nullable=True)
    created_at  = Column(DateTime, server_default=func.now(), index=True)
