import torch
from torch import nn
from src.settings import settings



@torch.inference_mode()
def get_clf_predict(model: nn.Module, 
                    transformed_image: torch.Tensor) -> int:
    """
    Выполняет предсказание модели бинарной классификации на одном изображении

    Args:
        model (nn.Module): Модель бинарной классификации
        transformed_image (torch.Tensor): Тензор изображения после применения преобразований

    Returns:
        product_class (int): Предсказанный класс
    """

    y_pred = model(transformed_image).squeeze()
    product_class = (y_pred > settings.CLF_CONFIDENCE_THRESHOLD).cpu().item()

    return product_class 