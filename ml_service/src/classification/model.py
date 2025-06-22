from torchvision import models
from torch import nn
import torch
from src.settings import settings



def init_cls_model(model_wts_path: str) -> nn.Module:
    
    model = models.mobilenet_v3_large()
    model.classifier[3] = nn.Linear(in_features=1280, out_features=1, bias=True)
    model.classifier.add_module('4', nn.Sigmoid())
    model.load_state_dict(torch.load(model_wts_path, map_location=settings.DEVICE))
    model.to(settings.DEVICE)
    model.eval()

    return model


model_onion = init_cls_model(settings.ONION_CLF_WTS_PATH)
model_potato = init_cls_model(settings.POTATO_CLF_WTS_PATH)
model_cabbage = init_cls_model(settings.CABBAGE_CLF_WTS_PATH)
model_pollock = init_cls_model(settings.POLLOCK_CLF_WTS_PATH)
model_beet = init_cls_model(settings.BEET_CLF_WTS_PATH)

cls_models_dict = {
    "ЛУК РЕПЧАТЫЙ": model_onion,
    "КАРТОФЕЛЬ": model_potato,
    "КАПУСТА БЕЛОКОЧАННАЯ": model_cabbage,
    "МИНТАЙ": model_pollock,
    "СВЕКЛА": model_beet
}