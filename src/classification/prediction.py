import torch
from torch import nn
from src.schemas import ClassificationPrediction



@torch.inference_mode()
def get_clf_predict(model: nn.Module, 
                transformed_image: torch.Tensor) -> ClassificationPrediction:
    """
    Выполняет предсказание модели бинарной классификации на одном изображении

    Args:
        model (nn.Module): Vодель бинарной классификации
        transformed_image (torch.Tensor): Тензор изображения после применения преобразований

    Returns:
        ClassificationPrediction: Объект, содержащий результаты предсказания с полями:
            - confidence (float): Вероятность положительного класса
            - product_class (int): Предсказанный класс
    """

    y_pred = model(transformed_image).squeeze()
    pred_class = (y_pred > 0.5).cpu().item()

    return ClassificationPrediction(confidence=y_pred.cpu().item(),
                                    product_class=int(pred_class)) 