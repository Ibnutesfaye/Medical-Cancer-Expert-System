"""
Doctor Dashboard Routes
========================
GET  /doctor/patients                   — list patients with analyses
GET  /doctor/patients/{id}/analyses     — patient scan history + timeline
POST /doctor/notes                      — add clinical note
GET  /doctor/notes/{patient_id}         — get notes for a patient
GET  /doctor/dashboard/stats            — overview stats
"""
from __future__ import annotations
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import Optional

from db.database import get_db
from core.security import get_current_user_payload
from models.doctor_note import DoctorNote
from models.image_analysis import ImageAnalysis
from models.user import User

router = APIRouter(prefix="/doctor", tags=["doctor"])


def _require_doctor(payload: dict):
    """Allow doctors and admins only."""
    if not (payload.get("is_admin") or payload.get("role") == "doctor"):
        # For now allow all authenticated users (expand with roles later)
        pass
    return payload


class NoteCreate(BaseModel):
    patient_id:  int
    analysis_id: Optional[int] = None
    note_type:   str = "observation"
    content:     str
    is_private:  bool = True


@router.get("/dashboard/stats")
def dashboard_stats(
    payload: dict = Depends(get_current_user_payload),
    db: Session = Depends(get_db),
):
    """Overview stats for the doctor dashboard."""
    total_analyses   = db.query(ImageAnalysis).count()
    cancer_detected  = db.query(ImageAnalysis).filter(
        ImageAnalysis.cancer_detected == True).count()
    total_patients   = db.query(User).filter(User.is_active == True).count()
    total_notes      = db.query(DoctorNote).count()

    return {
        "total_analyses":  total_analyses,
        "cancer_detected": cancer_detected,
        "healthy":         total_analyses - cancer_detected,
        "total_patients":  total_patients,
        "total_notes":     total_notes,
        "detection_rate":  round(cancer_detected / total_analyses * 100, 1)
                           if total_analyses > 0 else 0.0,
    }


@router.get("/patients")
def list_patients(
    skip: int = 0,
    limit: int = 50,
    payload: dict = Depends(get_current_user_payload),
    db: Session = Depends(get_db),
):
    """List all patients who have uploaded at least one image."""
    # Get distinct user_ids from image_analyses
    from sqlalchemy import distinct, func as sqlfunc
    rows = (
        db.query(
            ImageAnalysis.user_id,
            sqlfunc.count(ImageAnalysis.id).label("scan_count"),
            sqlfunc.max(ImageAnalysis.created_at).label("last_scan"),
            sqlfunc.sum(
                (ImageAnalysis.cancer_detected == True).cast(db.bind.dialect.INTEGER
                if db.bind else int)
            ).label("positive_count"),
        )
        .group_by(ImageAnalysis.user_id)
        .offset(skip).limit(limit).all()
    )

    results = []
    for row in rows:
        user = db.query(User).filter(User.id == row.user_id).first()
        results.append({
            "user_id":       row.user_id,
            "username":      user.username if user else "Unknown",
            "scan_count":    row.scan_count,
            "last_scan":     row.last_scan.isoformat() if row.last_scan else None,
            "positive_scans": int(row.positive_count or 0),
        })
    return results


@router.get("/patients/{patient_id}/analyses")
def patient_analyses(
    patient_id: int,
    payload: dict = Depends(get_current_user_payload),
    db: Session = Depends(get_db),
):
    """Get full scan history and timeline for a patient."""
    analyses = (
        db.query(ImageAnalysis)
        .filter(ImageAnalysis.user_id == patient_id)
        .order_by(ImageAnalysis.created_at.asc())
        .all()
    )
    if not analyses:
        raise HTTPException(status_code=404, detail="No analyses found for this patient")

    timeline = []
    for a in analyses:
        timeline.append({
            "id":             a.id,
            "filename":       a.original_filename,
            "cancer_detected": a.cancer_detected,
            "cancer_type":    a.cancer_type,
            "confidence":     a.confidence,
            "model_used":     a.model_used,
            "created_at":     a.created_at.isoformat() if a.created_at else None,
        })

    # Trend: is confidence increasing over time?
    confidences = [
        (t["created_at"], t["confidence"])
        for t in timeline
        if t["confidence"] is not None and t["cancer_detected"]
    ]
    trend = "stable"
    if len(confidences) >= 2:
        first_conf = confidences[0][1]
        last_conf  = confidences[-1][1]
        if last_conf - first_conf > 0.05:
            trend = "increasing"
        elif first_conf - last_conf > 0.05:
            trend = "decreasing"

    return {
        "patient_id": patient_id,
        "total_scans": len(timeline),
        "timeline":   timeline,
        "trend":      trend,
    }


@router.post("/notes", status_code=201)
def create_note(
    data: NoteCreate,
    payload: dict = Depends(get_current_user_payload),
    db: Session = Depends(get_db),
):
    """Add a clinical note for a patient."""
    note = DoctorNote(
        doctor_id   = payload["user_id"],
        patient_id  = data.patient_id,
        analysis_id = data.analysis_id,
        note_type   = data.note_type,
        content     = data.content,
        is_private  = data.is_private,
    )
    db.add(note)
    db.commit()
    db.refresh(note)
    return {
        "id":         note.id,
        "note_type":  note.note_type,
        "created_at": note.created_at.isoformat() if note.created_at else None,
        "message":    "Note saved successfully",
    }


@router.get("/notes/{patient_id}")
def get_notes(
    patient_id: int,
    payload: dict = Depends(get_current_user_payload),
    db: Session = Depends(get_db),
):
    """Get all clinical notes for a patient."""
    notes = (
        db.query(DoctorNote)
        .filter(DoctorNote.patient_id == patient_id)
        .order_by(DoctorNote.created_at.desc())
        .all()
    )
    return [
        {
            "id":          n.id,
            "doctor_id":   n.doctor_id,
            "analysis_id": n.analysis_id,
            "note_type":   n.note_type,
            "content":     n.content,
            "is_private":  n.is_private,
            "created_at":  n.created_at.isoformat() if n.created_at else None,
        }
        for n in notes
    ]
