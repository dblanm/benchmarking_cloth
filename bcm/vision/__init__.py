import kornia as K

from .utils import scatter_points


def get_shape_reconstructor(cfg, *args, **kwargs):
    if cfg.name == "mp_sft":
        from .mp_sft import MetricPreservationSfT

        return MetricPreservationSfT(cfg, *args, **kwargs)
    elif cfg.name == "ismogan":
        from .ismogan import IsMOGAN

        return IsMOGAN(cfg, *args, **kwargs)
    else:
        raise ValueError(f"Shape reconstructor {cfg.name} not recognized")


class ShapeReconstructor:
    def __init__(self, cfg, device):
        self._cfg = cfg
        self._device = device

    def _process_img(self, img, gray=False):
        img = K.image_to_tensor(img, False).float() / 255
        if gray:
            img = K.color.rgb_to_grayscale(img)
        return img.to(self._device)

    def _visualize(self, torch_vertices, info, colors="r"):
        scatter_points(info, torch_vertices.cpu().numpy(), c=colors)
