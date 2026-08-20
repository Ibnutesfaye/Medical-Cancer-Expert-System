from pydantic import BaseModel
from typing import Optional
from datetime import datetime


class ImageAnalysisRead(BaseModel):
    id: int
    user_id: Optional[int]
    original_filename: Optional[str]
    cancer_detected: bool
    cancer_type: Optional[str]
    confidence: Optional[float]
    safety_message: Optional[str]
    model_used: str
    message: Optional[str] = None
    match_source: Optional[str] = None
    created_at: datetime
    
    training_accuracy: Optional[float] = None
    validation_accuracy: Optional[float] = None
    training_loss: Optional[float] = None
    evaluation_info: Optional[dict] = None
    unknown_image: Optional[bool] = False
    low_confidence: Optional[bool] = False

    model_config = {"from_attributes": True}
