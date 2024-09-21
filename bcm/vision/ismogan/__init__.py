import cv2
import torch

from bcm.vision import ShapeReconstructor
from bcm.vision.odnet import ODNet
from omegaconf import DictConfig

from .model import Decoder_residual_real, Encoder_residual


class IsMOGAN(ShapeReconstructor):
    def __init__(self, cfg: DictConfig, device, **kwargs):
        super().__init__(cfg, device)
        self.segmentation = ODNet(cfg, device=device, **kwargs)
        self.encoder = Encoder_residual().to(self._device)
        self.decoder = Decoder_residual_real().to(self._device)

        self.encoder.load_state_dict(torch.load(cfg.encoder_checkpoint_file))
        self.decoder.load_state_dict(torch.load(cfg.decoder_checkpoint_file))

        self.device = device
        self.img_size = (224, 224)

    def _process_img(self, img, mask):
        masked_img = cv2.bitwise_and(img, img, mask=mask)
        masked_img = cv2.resize(masked_img, self.img_size).reshape(
            3, self.img_size[0], self.img_size[1]
        )
        return torch.from_numpy(masked_img).float().to(self.device).unsqueeze(0)

    def __call__(self, img, info=None, visualize=False, **kwargs):
        assert "mask" in info
        masked_img = self._process_img(img, info["mask"])
        outputs, indices, size = self.encoder(masked_img)
        torch_vertices = self.decoder(outputs, indices, size)

        # From (batch_size = 1, 3, n_rows, n_cols)
        # To (n_vertices = n_rows * n_cols, 3)
        torch_vertices = torch_vertices.squeeze().permute(1, 2, 0).reshape(-1, 3)

        if visualize:
            self._visualize(torch_vertices, info)
        return torch_vertices
