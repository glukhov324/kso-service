import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from src.change_detection.model import ChangeFormerV6
from src.settings import settings


class ChangeDetectionPredictor():
    def __init__(self):
        self.cd_model = ChangeFormerV6()
        self.dilate_kernel = np.array([[0,1,0],[1,1,1],[0,1,0]]).astype('uint8')
    
    def load_checkpoint(self, checkpoint_path: str):

        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=settings.DEVICE)
            self.cd_model.load_state_dict(checkpoint['model_G_state_dict'])
            self.cd_model.to(settings.DEVICE)

        else:
            raise FileNotFoundError('no such checkpoint %s' % checkpoint_path)
        return self.cd_model.eval()

    def _postprocessing_mask(self, prediction: torch.Tensor) -> np.ndarray:
        pred = F.interpolate(prediction, size=(480, 640), mode="bilinear", align_corners=False)
        pred = torch.argmax(pred, dim=1, keepdim=True).cpu().numpy()
        pred_vis = (pred * 255).squeeze((0, 1))
        pred_vis = cv2.dilate(src=pred_vis.astype(np.uint8), 
                              kernel=self.dilate_kernel, 
                              iterations=settings.DILATE_ITER)
   
        return pred_vis

    def inference(self, 
                  img_a: torch.Tensor, 
                  img_b: torch.Tensor) -> np.ndarray:
        
        img_a = img_a.to(settings.DEVICE)
        img_b = img_b.to(settings.DEVICE)
        prediction = self.cd_model(img_a, img_b)[-1]
   
        return self._postprocessing_mask(prediction)


cd_predictor = ChangeDetectionPredictor()
cd_predictor.load_checkpoint(checkpoint_path=settings.CD_MODEL_WTS_PATH)