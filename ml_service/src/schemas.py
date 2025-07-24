from pydantic import BaseModel



class PipelinePrediction(BaseModel):
    product_class: int
    added_base_64: str

class Msg(BaseModel):
    msg: str