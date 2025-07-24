from fastapi import (
    APIRouter, 
    UploadFile, 
    Form, 
    status
)
from fastapi.responses import JSONResponse
from src.classification.model import cls_models_dict
from src.pipeline import pipeline_prediction
from src.schemas import PipelinePrediction, Msg



router = APIRouter(prefix="/api")


@router.post("/predict", response_model=PipelinePrediction)
async def a_b_label_predict(file_a: UploadFile, 
                            file_b: UploadFile, 
                            user_label: str = Form(...)):

    if user_label not in cls_models_dict.keys():
        return JSONResponse(content=Msg(msg="Для указанного названия товара нет классификатора").model_dump(),
                            status_code=status.HTTP_404_NOT_FOUND)

    data_a = await file_a.read()
    data_b = await file_b.read()

    pipeline_response = pipeline_prediction(
        data_a, 
        data_b, 
        user_label
    )

    return pipeline_response