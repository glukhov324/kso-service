from PIL import Image
import io
import numpy as np
from loguru import logger
import base64
from src.change_detection import cd_predictor, transform_test
from src.classification import (
    cls_data_transforms,
    get_clf_predict,
    cls_models_dict
)
from src.schemas import PipelinePrediction
from src.settings import settings



def pipeline_prediction(data_a: str, 
                        data_b:str, 
                        user_label: str) -> PipelinePrediction:
    """
    Обрабатывает изображения до и после добавления товара на кассу самообслуживания
    и проверяет, соответствует ли товар, добавленный покупателем в интерфейсе кассы самообслуживания,
    товару, который был положен на весы кассы самообслуживания

    Args:
        data_a (str): Изображение до добавления товара
        data_b (str): Изображение после добавления товара
        user_label (str): Название товара, указанное пользователем в интерфейсе кассы самообслуживания

    Returns:
        PipelinePrediction: Объект, содержащий результаты с полями:
            - product_class (int): Предсказанный класс
            - mask_base_64 (str): Бинарная маска изменений в формат Base64
    
    Raises:
        JSONResponse (404): Если для указанного названия товара не найдено модели классификации
    """

    pil_img_a = Image.open(io.BytesIO(data_a)).convert('RGB')
    pil_img_b = Image.open(io.BytesIO(data_b)).convert('RGB')

    logger.info("Start change detection model prediction process")
    img_a_tr, img_b_tr = transform_test(imgs=[pil_img_a, pil_img_b], 
                                        img_size=256)
    cd_mask_raw = cd_predictor.inference(img_a_tr, img_b_tr)
    logger.info("End change detection model prediction process")

    cd_mask = np.array(pil_img_b)
    cd_mask[cd_mask_raw == 0] = 1
    cd_mask_pil = Image.fromarray(cd_mask)
    cd_mask_tr = cls_data_transforms(cd_mask_pil).unsqueeze(0).to(settings.DEVICE)

    logger.info("Start classification model prediction process")
    product_class = get_clf_predict(model=cls_models_dict[user_label], 
                                    transformed_image=cd_mask_tr)
    logger.info("End classification model prediction process")
    
    cd_mask_byte_arr = io.BytesIO()
    cd_mask_pil.save(cd_mask_byte_arr, format='PNG')
    cd_mask_byte_arr.seek(0)
    mask_base64 = base64.b64encode(cd_mask_byte_arr.getvalue()).decode('utf-8')

    return PipelinePrediction(product_class=product_class,
                              mask_base_64=mask_base64)