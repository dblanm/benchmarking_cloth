import cv2 as cv
import numpy as np
import torch

from torchvision.transforms import Resize

from .model import UNet


class ODNet:
    def __init__(self, *args, **kwargs):
        self.model = UNet(n_channels=3)
        if "device" in kwargs:
            self.model.to(kwargs["device"])

        if "checkpoint_path" in kwargs:
            self.model.load_state_dict(torch.load(kwargs["checkpoint_path"]))

        self.resize = Resize((224, 224))

    def __call__(self, input_torch_img):
        prob = self.model(input_torch_img)
        torch_mask = torch.transpose(torch.transpose(prob[0], 0, 1), 1, 2) * 255
        mask = torch_mask.data.cpu().numpy().astype(np.uint8)
        blur = cv.GaussianBlur(
            mask, (5, 5), 0
        )  # gaussian blur (parameters have to be adjusted)
        ret3, mask = cv.threshold(
            blur, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU
        )  # Otsu binarization

        se = np.ones((50, 24), dtype="uint8")

        # coutour filling
        image_close = cv.morphologyEx(mask, cv.MORPH_CLOSE, se)
        cnt = cv.findContours(image_close, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)[0]
        mask = np.zeros(mask.shape[:2], np.uint8)
        # Fill mask matrix
        cv.drawContours(mask, cnt, -1, 255, -1)

        torch_mask = self.resize(
            torch.from_numpy((mask / 255).astype(bool))
            .unsqueeze(0)
            .to(input_torch_img.device)
        )
        input_torch_img = self.resize(input_torch_img) * torch_mask
        return input_torch_img, torch_mask
