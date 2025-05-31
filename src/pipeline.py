from PIL import Image
import io
import numpy as np
from src.change_detection.data_transforms import transform_test
from loguru import logger
import base64
from src.change_detection.predictor import cd_predictor
from src.classification.data_transforms import cls_data_transforms
from src.classification.prediction import get_clf_predict
from src.classification.model import cls_models_dict
from src.schemas import PipelinePrediction
from src.settings import settings



def pipeline_prediction(data_a: str, 
                        data_b:str, 
                        user_label: str) -> PipelinePrediction:

    pil_img_a = Image.open(io.BytesIO(data_a)).convert('RGB')
    pil_img_b = Image.open(io.BytesIO(data_b)).convert('RGB')

    logger.info("Start change detection model prediction process")
    img_a_tr, img_b_tr = transform_test(imgs=[pil_img_a, pil_img_b], 
                                        img_size=256)
    cd_mask_raw = cd_predictor.inference(img_a_tr, img_b_tr)
    logger.info("End change detection model prediction process")

    cd_mask = np.array(pil_img_b)
    cd_mask_pil = Image.fromarray(cd_mask)
    cd_mask[cd_mask_raw == 0] = 1
    cd_mask_tr = cls_data_transforms(cd_mask_pil).unsqueeze(0).to(settings.DEVICE)

    logger.info("Start classification model prediction process")
    classification_response = get_clf_predict(model=cls_models_dict[user_label], 
                                              transformed_image=cd_mask_tr)
    logger.info("End classification model prediction process")
    
    cd_mask_byte_arr = io.BytesIO()
    cd_mask_pil.save(cd_mask_byte_arr, format='PNG')
    cd_mask_byte_arr.seek(0)
    mask_base64 = base64.b64encode(cd_mask_byte_arr.getvalue()).decode('utf-8')

    return PipelinePrediction(confidence=classification_response.confidence,
                              product_class=classification_response.product_class,
                              mask_base_64=mask_base64)