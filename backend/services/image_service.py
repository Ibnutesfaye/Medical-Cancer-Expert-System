"""
Image analysis service — saves prediction results to MySQL.
"""

import json
from sqlalchemy.orm import Session
from models.image_analysis import ImageAnalysis
from typing import Optional, List


def save_analysis(
    db: Session,
    user_id: Optional[int],
    original_filename: Optional[str],
    file_size_bytes: Optional[int],
    result: dict,
) -> ImageAnalysis:
    """Persist an image analysis result returned by the analyzer."""
    record = ImageAnalysis(
        user_id=user_id,
        original_filename=original_filename,
        file_size_bytes=file_size_bytes,
        cancer_detected=result.get("cancer_detected", False),
        cancer_type=result.get("cancer_type"),
        confidence=result.get("confidence"),
        safety_message=result.get("safety_message"),
        model_used=result.get("model_used", "ResNet18"),
        raw_result=json.dumps(result),
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    return record


def list_analyses(
    db: Session,
    user_id: Optional[int] = None,
    skip: int = 0,
    limit: int = 50,
) -> List[ImageAnalysis]:
    q = db.query(ImageAnalysis)
    if user_id:
        q = q.filter(ImageAnalysis.user_id == user_id)
    return q.order_by(ImageAnalysis.created_at.desc()).offset(skip).limit(limit).all()


def get_analysis(db: Session, analysis_id: int) -> Optional[ImageAnalysis]:
    return db.query(ImageAnalysis).filter(ImageAnalysis.id == analysis_id).first()


def delete_analysis(db: Session, analysis_id: int) -> bool:
    """Delete an image analysis record."""
    record = get_analysis(db, analysis_id)
    if record:
        db.delete(record)
        db.commit()
        return True
    return False

