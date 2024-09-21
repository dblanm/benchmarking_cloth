from .utils import get_default_device


def get_model_by_name(model_name, *args, **kwargs):
    if model_name == "occlusion_fusion":
        from .occlusion_fusion import OcclusionFusion

        return OcclusionFusion(*args, **kwargs)
    else:
        raise NotImplementedError(f"Model {model_name} not supported")


class BaseModel:
    def __init__(self, opt):
        if "device" in opt:
            self._device = opt["device"]
        else:
            self._device = get_default_device()
