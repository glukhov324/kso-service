from pydantic import BaseModel



class ClassificationPrediction(BaseModel):
    confidence: float
    is_true_product: int

class PipelinePrediction(ClassificationPrediction):
    mask_base_64: str