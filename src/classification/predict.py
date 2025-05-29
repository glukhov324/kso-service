import torch
from torch import nn

@torch.inference_mode()
def cls_predict(model: nn.Module, 
                transformed_image: torch.Tensor) -> tuple[float, int]:

    y_pred = model(transformed_image).squeeze()
    pred_class = (y_pred > 0.5).cpu().item()

    return (y_pred.cpu().item(), int(pred_class))