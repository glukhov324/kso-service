import torch
from torch import nn
from src.schemas import ClassificationPrediction

@torch.inference_mode()
def clf_predict(model: nn.Module, 
                transformed_image: torch.Tensor) -> ClassificationPrediction:

    y_pred = model(transformed_image).squeeze()
    pred_class = (y_pred > 0.5).cpu().item()

    return ClassificationPrediction(confidence=y_pred.cpu().item(),
                                    is_true_product=int(pred_class)) 