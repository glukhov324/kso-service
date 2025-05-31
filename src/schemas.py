from pydantic import BaseModel



class ClassificationPrediction(BaseModel):
    confidence: float
    product_class: int

class PipelinePrediction(ClassificationPrediction):
    mask_base_64: str