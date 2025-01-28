from fastapi import FastAPI, UploadFile, File, Request, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates

from loguru import logger
from PIL import Image
import base64
import io
import numpy as np
import cv2

from src.datasets.data_utils import transform_test
from src.init_models import cd_model, cls_models_match, kernel, dilate_iter, device
from src.classification_transforms import cls_data_transforms
from src.classification_predict import cls_predict



app = FastAPI()

# Настройка CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Подключение Jinja2
templates = Jinja2Templates(directory="frontend/templates")


@app.get("/")
async def get_index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post('/a_b_label_predict')
async def a_b_label_predict(file_a: UploadFile, 
                            file_b: UploadFile, 
                            label: str = Form(...)) -> JSONResponse:
    """
    file_a: изображение до добавления товара

    file_b: изображение после добавления товара

    label: название товара, которое пользователь указал в интерфейсе кассы самообслуживания

    return:

    model_answer: оказался ли указанный товар на кассе (1) или нет (0)

    img_with_mask: маска модели сегментации изменений, закодированная в base64 
    """

    data_a = await file_a.read()
    data_b = await file_b.read()
    img_a = Image.open(io.BytesIO(data_a)).convert('RGB')
    img_b = Image.open(io.BytesIO(data_b)).convert('RGB')

    # инференс модели сегментации изменений
    [img_A, img_B]= transform_test(imgs=[img_a, img_b], img_size=256, to_tensor=True, img_size_dynamic=False)
    img_A = img_A.unsqueeze(0).to(device)
    img_B = img_B.unsqueeze(0).to(device)
    batch = {
        'A': img_A,
        'B': img_B
    }

    mask = cd_model._forward_pass(batch).cpu().numpy().squeeze((0, 1))
    mask = cv2.dilate(mask.astype(np.uint8), kernel, iterations=dilate_iter)
    resized = np.array(img_b)
    resized[mask == 0] = 1

    # инференс модели классификации 

    img = cls_data_transforms(Image.fromarray(resized)).unsqueeze(0).to(device)
    
    if label in cls_models_match.keys():
        confidence, predicted = cls_predict(model=cls_models_match[label], transformed_image=img)
    else:
        return JSONResponse(status_code=404, content={"message": "Для указанного названия товара нет классификатора"})

    logger.info(f'model confidence: {confidence}, model answer: {predicted}')

    resized_mask = Image.fromarray(resized)
    img_byte_arr = io.BytesIO()
    resized_mask.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)
    img_base64 = base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')

    return JSONResponse(content={
        "model_answer": predicted,
        "img_with_mask": img_base64
    })