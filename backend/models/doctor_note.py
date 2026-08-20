"""
DoctorNote ORM Model
Stores clinical notes written by doctors about patient analyses.
"""
from sqlalchemy import Column, Integer, String, Text, Boolean, DateTime, ForeignKey
from sqlalchemy.sql import func
from db.database import Base


class DoctorNote(Base):
    __tablename__ = "doctor_notes"

    id          = Column(Integer, primary_key=True, index=True)
    doctor_id   = Column(Integer, ForeignKey("users.id"), nullable=False)
    patient_id  = Column(Integer, ForeignKey("users.id"), nullable=False)
    analysis_id = Column(Integer, ForeignKey("image_analyses.id"), nullable=True)
    note_type   = Column(String(32), default="observation")
    content     = Column(Text, nullable=False)
    is_private  = Column(Boolean, default=True)
    created_at  = Column(DateTime, server_default=func.now())
    updated_at  = Column(DateTime, server_default=func.now(), onupdate=func.now())
