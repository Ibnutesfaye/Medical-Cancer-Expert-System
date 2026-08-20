"""
ImageAnalysis model — stores results of cancer image predictions.
"""

from sqlalchemy import Column, Integer, String, Text, ForeignKey, DateTime, Float, Boolean, func
from sqlalchemy.orm import relationship
from db.database import Base


class ImageAnalysis(Base):
    __tablename__ = "image_analyses"

    id               = Column(Integer, primary_key=True, index=True)
    user_id          = Column(Integer, ForeignKey("users.id", ondelete="SET NULL"), nullable=True, index=True)
    original_filename= Column(String(256), nullable=True)
    file_size_bytes  = Column(Integer, nullable=True)
    cancer_detected  = Column(Boolean, nullable=False)
    cancer_type      = Column(String(128), nullable=True)
    confidence       = Column(Float, nullable=True)            # 0.0 – 1.0
    safety_message   = Column(Text, nullable=True)
    model_used       = Column(String(64), default="ResNet18")
    raw_result       = Column(Text, nullable=True)             # full JSON result as string
    created_at       = Column(DateTime, server_default=func.now(), nullable=False)

    # Relationships
    user = relationship("User", back_populates="image_analyses")

    @property
    def training_accuracy(self):
        import json
        if self.raw_result:
            return json.loads(self.raw_result).get("training_accuracy")
        return None

    @property
    def validation_accuracy(self):
        import json
        if self.raw_result:
            return json.loads(self.raw_result).get("validation_accuracy")
        return None

    @property
    def training_loss(self):
        import json
        if self.raw_result:
            return json.loads(self.raw_result).get("training_loss")
        return None

    @property
    def evaluation_info(self):
        import json
        if self.raw_result:
            return json.loads(self.raw_result).get("evaluation_info")
        return None

    def __repr__(self):
        return f"<ImageAnalysis id={self.id} cancer={self.cancer_detected} type={self.cancer_type}>"
