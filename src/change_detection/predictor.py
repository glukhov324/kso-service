from src.change_detection.model import ChangeFormerV6
import os
from src.settings import settings
import torch
import torch.nn.functional as F


class ChangeDetectionPredictor():
    def __init__(self):
        self.cd_model = ChangeFormerV6()
    
    def load_checkpoint(self, checkpoint_path: str):
        if os.path.exists(checkpoint_path):
            # load the entire checkpoint
            checkpoint = torch.load(checkpoint_path, map_location=settings.DEVICE)
            print('Checkpoint loaded!')
            self.cd_model.load_state_dict(checkpoint['model_G_state_dict'])
            self.cd_model.to(settings.DEVICE)

        else:
            raise FileNotFoundError('no such checkpoint %s' % checkpoint_path)
        return self.cd_model.eval()

    def _visualize_pred(self):
        pred = F.interpolate(self.G_pred, size=(480, 640), mode="bilinear", align_corners=False)
        pred = torch.argmax(pred, dim=1, keepdim=True)
        pred_vis = pred * 255
        return pred_vis

    def _inference_helper(self, batch):
        self.batch = batch
        img_in1 = batch['A'].to(settings.DEVICE)
        img_in2 = batch['B'].to(settings.DEVICE)
        self.G_pred = self.cd_model(img_in1, img_in2)[-1]
        
        return self._visualize_pred()


cd_predictor = ChangeDetectionPredictor()
cd_predictor.load_checkpoint(checkpoint_path=settings.CD_MODEL_WTS_PATH)