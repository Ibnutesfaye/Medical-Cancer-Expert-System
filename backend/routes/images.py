"""
Image analysis routes — upload image, get prediction, history.
"""

from fastapi import APIRouter, Depends, UploadFile, File, HTTPException
from sqlalchemy.orm import Session

from db.database import get_db
from schemas.image_analysis import ImageAnalysisRead
from services import image_service
from core.security import get_current_user_payload

router = APIRouter(prefix="/images", tags=["images"])


def _get_analyzer():
    from image_analyzer import get_image_analyzer
    return get_image_analyzer()


@router.post("/analyze", response_model=ImageAnalysisRead, status_code=201)
async def analyze_image(
    file: UploadFile = File(...),
    payload: dict = Depends(get_current_user_payload),
    db: Session = Depends(get_db),
):
    """Analyze a medical image and persist the result."""
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Only image files are supported")

    content = await file.read()
    if len(content) > 20 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="Image size exceeds 20 MB")

    analyzer = _get_analyzer()
    result = analyzer.analyze(content)

    # Only generate LLM explanation for valid, confident cancer detections
    if (
        result.get("cancer_detected")
        and not result.get("unknown_image")
        and not result.get("low_confidence")
    ):
        cancer_type = result.get("cancer_type", "Unknown Cancer")
        confidence = result.get("confidence", 0) * 100
        
        prompt = f"""You are a medical AI assistant specialized in cancer education.

A user uploaded a medical image and the AI model predicted:
* Cancer Detected: YES
* Cancer Type: {cancer_type}
* Confidence: {confidence:.0f}%

Your task is to explain this result in a helpful, safe, and educational way.

⚠️ IMPORTANT RULES:
* Do NOT act as a doctor
* Do NOT give definitive diagnosis
* Do NOT prescribe medications
* Always recommend consulting a healthcare professional

---

Structure your response EXACTLY like this:

1. **What This Means**
   Explain in simple terms what a {cancer_type} is.

2. 🔬 **Possible Symptoms**
   List common symptoms (3–5 bullet points).

3. ⚠️ **What You Should Do Next**
   Give safe, practical next steps (doctor visit, tests like MRI, biopsy, etc.)

4. 🚫 **What NOT To Do**
   List 3–5 important warnings (ignore symptoms, self-diagnosis, random drugs, etc.)

5. 💊 **Common Treatment Options (General Info Only)**
   Explain typical treatments like:
   * Surgery
   * Radiation therapy
   * Chemotherapy
   (No specific prescriptions)

6. ❤️ **Final Advice**
   Encourage the user calmly and recommend professional consultation.

Keep the tone calm, supportive, and easy to understand."""

        from main_v2 import llm_service
        try:
            explanation = await llm_service.generate(prompt)
            result["message"] = explanation
        except Exception as e:
            print(f"Error generating explanation: {e}")

    record = image_service.save_analysis(
        db,
        user_id=payload["user_id"],
        original_filename=file.filename,
        file_size_bytes=len(content),
        result=result,
    )
    
    response_data = {
        "id": record.id,
        "user_id": record.user_id,
        "original_filename": record.original_filename,
        "cancer_detected": record.cancer_detected,
        "cancer_type": record.cancer_type,
        "confidence": record.confidence,
        "safety_message": record.safety_message,
        "model_used": record.model_used,
        "created_at": record.created_at,
        "message": result.get("message", ""),
        "match_source": result.get("match_source", ""),
        "training_accuracy": result.get("training_accuracy"),
        "validation_accuracy": result.get("validation_accuracy"),
        "training_loss": result.get("training_loss"),
        "evaluation_info": result.get("evaluation_info"),
        "unknown_image": result.get("unknown_image", False),
        "low_confidence": result.get("low_confidence", False),
    }
    return response_data


@router.get("/history", response_model=list[ImageAnalysisRead])
def image_history(
    skip: int = 0,
    limit: int = 20,
    payload: dict = Depends(get_current_user_payload),
    db: Session = Depends(get_db),
):
    """Return the current user's image analysis history."""
    return image_service.list_analyses(db, user_id=payload["user_id"], skip=skip, limit=limit)


@router.get("/admin/all", response_model=list[ImageAnalysisRead])
def all_analyses(
    skip: int = 0,
    limit: int = 50,
    payload: dict = Depends(get_current_user_payload),
    db: Session = Depends(get_db),
):
    """Admin: list all image analyses across all users."""
    if not payload.get("is_admin"):
        raise HTTPException(status_code=403, detail="Admin access required")
    return image_service.list_analyses(db, skip=skip, limit=limit)


@router.delete("/{image_id}")
def delete_image_analysis(
    image_id: int,
    payload: dict = Depends(get_current_user_payload),
    db: Session = Depends(get_db),
):
    """Admin: delete an image analysis record."""
    if not payload.get("is_admin"):
        raise HTTPException(status_code=403, detail="Admin access required")
    
    success = image_service.delete_analysis(db, image_id)
    if not success:
        raise HTTPException(status_code=404, detail="Image analysis not found")
        
    return {"message": "Image analysis deleted successfully"}

