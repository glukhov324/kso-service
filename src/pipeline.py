from PIL import Image
import io
import cv2
import numpy as np
from src.change_detection.data_transforms import transform_test
from loguru import logger
import base64
from src.change_detection.predictor import cd_predictor
from src.classification.data_transforms import cls_data_transforms
from src.classification.predict import cls_predict
from src.classification.model import cls_models_dict
from src.settings import settings

dilate_iter = 7 
kernel = np.array([[0,1,0],[1,1,1],[0,1,0]]).astype('uint8')

def pipeline_prediction(data_a: str, data_b:str, label: str):

    img_a = Image.open(io.BytesIO(data_a)).convert('RGB')
    img_b = Image.open(io.BytesIO(data_b)).convert('RGB')

    # инференс модели сегментации изменений
    [img_A, img_B]= transform_test(imgs=[img_a, img_b], img_size=256, to_tensor=True, img_size_dynamic=False)
    img_A = img_A.unsqueeze(0).to(settings.DEVICE)
    img_B = img_B.unsqueeze(0).to(settings.DEVICE)
    batch = {
        'A': img_A,
        'B': img_B
    }

    mask = cd_predictor._inference_helper(batch).cpu().numpy().squeeze((0, 1))
    mask = cv2.dilate(mask.astype(np.uint8), kernel, iterations=dilate_iter)
    resized = np.array(img_b)
    resized[mask == 0] = 1

    # инференс модели классификации 

    img = cls_data_transforms(Image.fromarray(resized)).unsqueeze(0).to(settings.DEVICE)
    confidence, predicted = cls_predict(model=cls_models_dict[label], transformed_image=img)
    

    logger.info(f'model confidence: {confidence}, model answer: {predicted}')

    resized_mask = Image.fromarray(resized)
    img_byte_arr = io.BytesIO()
    resized_mask.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)
    img_base64 = base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')

    return predicted, img_base64