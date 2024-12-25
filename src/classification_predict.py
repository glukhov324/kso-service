import torch
from torch import nn
from loguru import logger

@torch.inference_mode()
def cls_predict(model: nn.Module, 
                transformed_image: torch.Tensor) -> tuple[float, int]:
  """
  model - классификационная модель
  
  transformed_image - изображение после применения трансформаций для классификации

  return: confidence: float, pred_class: int (0 - нет искомого объекта, 1 - есть искомый объект)
  """

  y_pred = model(transformed_image).squeeze()
  pred_class = (y_pred > 0.5).cpu().item()

  return (y_pred.cpu().item(), int(pred_class))