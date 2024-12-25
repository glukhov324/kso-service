from argparse import ArgumentParser
from src.models.basic_model import CDEvaluator
from torchvision import models
import torch
from torch import nn
import numpy as np

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

# init ChangeFormer

dilate_iter = 7 
kernel = np.array([[0,1,0],[1,1,1],[0,1,0]]).astype('uint8')

infer_parser = ArgumentParser()
infer_parser.add_argument('--checkpoint_dir', type=str, default='src/checkpoints')
infer_parser.add_argument('--n_class', type=int, default=2)
infer_parser.add_argument('--output_folder', type=str, default='output_preds')
infer_parser.add_argument('--gpu_ids', type=str, default='0')
infer_parser.add_argument('--net_G', default='ChangeFormerV6', type=str)
infer_parser.add_argument('--embed_dim', default=256, type=int)
infer_parser.add_argument('--checkpoint_name', default='best_ckpt.pt', type=str)

infer_args, data_unknown = infer_parser.parse_known_args()
cd_model = CDEvaluator(infer_args)
cd_model.load_checkpoint(infer_args.checkpoint_name)
cd_model.eval()


# init classification models function

def init_cls_model(models_dir: str = "src/classification_models",
                   model_file_name: str = "best_model_onion.pt") -> nn.Module:
    
    model = models.mobilenet_v3_large()
    model.classifier[3] = nn.Linear(in_features=1280, out_features=1, bias=True)
    model.classifier.add_module('4', nn.Sigmoid())
    model = model.to(device)
    model.load_state_dict(torch.load(f"{models_dir}/{model_file_name}"))
    model.eval()

    return model

# init classification models for onion, potato, cabbage, pollock

model_onion = init_cls_model()
model_potato = init_cls_model(model_file_name="best_model_potato.pt")
model_cabbage = init_cls_model(model_file_name="best_model_cabbage.pt")
model_pollock = init_cls_model(model_file_name="best_model_pollock.pt")

cls_models_match = {
    "ЛУК РЕПЧАТЫЙ": model_onion,
    "КАРТОФЕЛЬ": model_potato,
    "КАПУСТА БЕЛОКОЧАННАЯ": model_cabbage,
    "МИНТАЙ": model_pollock
}