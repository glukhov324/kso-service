from pydantic import BaseModel



class PipelinePrediction(BaseModel):
    product_class: int
    mask_base_64: str