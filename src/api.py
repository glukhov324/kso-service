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
    Обрабатывает изображения до и после добавления товара на кассу самообслуживания
    и проверяет, соответствует ли товар, добавленный покупателем в интерфейсе кассы самообслуживания,
    товару, который был положен на весы кассы самообслуживания

    Args:
        file_a (UploadFile): Изображение до добавления товара
        file_b (UploadFile): Изображение после добавления товара
        user_label (str): Название товара, указанное пользователем в интерфейсе кассы самообслуживания

    Returns:
        PipelinePrediction: Объект, содержащий результаты с полями:
            - confidence (float): Вероятность положительного класса
            - product_class (int): Предсказанный класс
            - mask_base_64 (str): Бинарная маска изменений в формат Base64
    
    Raises:
        JSONResponse (404): Если для указанного названия товара не найдено модели классификации
    """

    if user_label not in cls_models_dict.keys():
        return JSONResponse(status_code=404, content={"message": "Для указанного названия товара нет классификатора"})

    data_a = await file_a.read()
    data_b = await file_b.read()

    pipeline_response = pipeline_prediction(data_a, data_b, user_label)

    return pipeline_response