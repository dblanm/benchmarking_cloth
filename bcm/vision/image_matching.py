import cv2
import kornia.feature as KF
import numpy as np
import torch


def get_image_matcher(method, *args, **kwargs):
    method = method.lower()
    if method == "loftr":
        return LoFTR(*args, **kwargs)
    elif method == "synthetic":
        return Synthetic(*args, **kwargs)
    else:
        raise NotImplementedError(f"Image matching method {method} not implemented")


class ImageMatching:
    def __call__(self, img1, img2, mask1=None, mask2=None, **kwargs):
        raise NotImplementedError


class LoFTR(ImageMatching):
    def __init__(self, *args, **kwargs):
        self.matcher = KF.LoFTR(pretrained="indoor")
        self.outlier_rejection = kwargs.get("outlier_rejection", True)
        if "device" in kwargs:
            self.matcher.to(kwargs["device"])

    def __call__(self, img1, img2, mask1=None, mask2=None, **kwargs):
        input_dict = {"image0": img1, "image1": img2}
        with torch.inference_mode():
            correspondences = self.matcher(input_dict)
        mkpts0 = correspondences["keypoints0"]
        mkpts1 = correspondences["keypoints1"]
        if self.outlier_rejection:
            Fm, inliers = cv2.findFundamentalMat(
                mkpts0.cpu().numpy(),
                mkpts1.cpu().numpy(),
                cv2.USAC_MAGSAC,
                0.5,
                0.999,
                100000,
            )
            inliers = (inliers > 0).reshape(-1)
            mkpts0 = mkpts0[inliers]
            mkpts1 = mkpts1[inliers]
        return mkpts0, mkpts1


class Synthetic(ImageMatching):
    def __init__(self, *args, **kwargs):
        assert "control_points" in kwargs and "texture_shape" in kwargs
        self.up_s = kwargs["control_points"] * kwargs["texture_shape"]

    def __call__(self, img1, img2, mask1=None, mask2=None, info=None):
        assert info is not None
        points = info["vertices"]
        proj_points = (
            info["intr"] @ info["extr"] @ np.r_[points.T, np.ones((1, points.shape[0]))]
        )
        proj_points = proj_points / proj_points[-1]
        proj_points = torch.from_numpy(proj_points[:2].T).float().to(self.up_s.device)
        return self.up_s, proj_points
