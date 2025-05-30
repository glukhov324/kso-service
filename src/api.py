from fastapi.templating import Jinja2Templates
from fastapi import APIRouter, Request, UploadFile, Form
from fastapi.responses import JSONResponse
from src.classification.model import cls_models_dict
from src.pipeline import pipeline_prediction
from src.schemas import PipelinePrediction



router = APIRouter(prefix="/pair_images")
templates = Jinja2Templates(directory="frontend/templates")


@router.get("/")
async def get_index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@router.post('/predict')
async def a_b_label_predict(file_a: UploadFile, 
                            file_b: UploadFile, 
                            user_label: str = Form(...)) -> PipelinePrediction:
    """
    file_a: изображение до добавления товара

    file_b: изображение после добавления товара

    label: название товара, которое пользователь указал в интерфейсе кассы самообслуживания

    return:

    model_answer: указанный товар действительно оказался на кассе самообслуживания - 1, указанный товар не оказался на кассе самообслуживания - 0

    img_with_mask: маска модели сегментации изменений, закодированная в base64 
    """

    if user_label not in cls_models_dict.keys():
        return JSONResponse(status_code=404, content={"message": "Для указанного названия товара нет классификатора"})

    data_a = await file_a.read()
    data_b = await file_b.read()

    pipeline_response = pipeline_prediction(data_a, data_b, user_label)

    return pipeline_response