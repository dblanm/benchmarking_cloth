import numpy as np
import torch
from omegaconf import DictConfig
from sim_utils.kornia import get_kornia_pinhole_camera

from . import ShapeReconstructor

from .evaluate import Evaluator
from .image_matching import get_image_matcher
from .mesh import Mesh
from .utils import loss_euc, metric, MLP


class MetricPreservationSfT(ShapeReconstructor):
    def __init__(
        self,
        cfg: DictConfig,
        template_mesh_file,
        template_texture_file,
        camera_intrinsics,
        camera_extrinsics,
        control_points,
        device: torch.device,
    ):
        super().__init__(cfg, device)

        self.model = MLP().to(self._device)
        self.template = Mesh(
            mesh_file=template_mesh_file,
            texture_file=template_texture_file,
            device=self._device,
        )
        self.param_colors = np.c_[
            self.template.control_points, np.zeros(len(self.template))
        ]

        width, height, _ = self.template.texture_img.shape
        self.camera = get_kornia_pinhole_camera(
            intrinsics_np=camera_intrinsics,
            extrinsics_np=camera_extrinsics,
            height=height,
            width=width,
            device=self._device,
        )

        self._evaluator = Evaluator(["chamfer_distance", "mean_rmse"])

        self.torch_texture_img = self._process_img(self.template.texture_img)
        self.torch_texture_shape = torch.FloatTensor(
            self.template.texture_img.shape[:2]
        ).to(self._device)
        self.feature_matcher = get_image_matcher(
            self._cfg.image_matcher,
            device=self._device,
            control_points=self.template.torch_control_points,
            texture_shape=self.torch_texture_shape,
        )

        self.prev_xyz, self.template_metric = self._register_template()

    def _register_template(self):
        # Over-fit to template, for which we know the ground truth
        optimizer = torch.optim.Adam(self.model.parameters())

        min_loss = torch.inf
        counter = 0

        while True:
            prediction = self.model(self.template.torch_control_points)

            loss = loss_euc(prediction, self.template.torch_vertices)
            if loss < min_loss:
                min_loss = loss
                counter = 0
            else:
                counter += 1
                if counter >= self._cfg.patience_template:
                    break

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            prediction = self.model(self.template.torch_control_points)
            prev_xyz = prediction
            template_metric = metric(self.model, self.template.torch_control_points)
        return prev_xyz, template_metric

    def __call__(self, image, info=None, visualize=False):
        img = self._process_img(image, gray=True)
        up_s, i_s = self.feature_matcher(self.torch_texture_img, img, info=info)
        # Convert coordinates in texture image to coordinates in parametrization space
        p_s = self._texture_coords_to_param(up_s)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self._cfg.lr_update)

        min_loss = torch.inf
        counter = 0
        best_state_dict = None

        if info is None:
            target = None
        else:
            eval_loss = {k: [] for k in self._evaluator.metrics.keys()}
            target = torch.from_numpy(info["vertices"]).float().to(self._device)

        for _ in range(self._cfg.max_it):
            optimizer.zero_grad()
            loss_proj = torch.nn.functional.mse_loss(
                self.camera.project(self.model(p_s)), i_s
            )
            loss = loss_proj
            prediction = self.model(self.template.torch_control_points)
            if self._cfg.lambda_time != 0:
                loss_time = torch.nn.functional.mse_loss(prediction, self.prev_xyz)
                loss += self._cfg.lambda_time * loss_time

            if self._cfg.lambda_metric != 0:
                loss_metric = torch.nn.functional.mse_loss(
                    metric(self.model, self.template.torch_control_points),
                    self.template_metric,
                )
                loss += self._cfg.lambda_metric * loss_metric

            if loss < min_loss:
                min_loss = loss
                best_state_dict = self.model.state_dict()
                counter = 0
            else:
                counter += 1
                if counter >= self._cfg.patience_update:
                    break

            loss.backward()
            optimizer.step()

            if target is not None:
                for k, v in self._evaluator(prediction.detach(), target).items():
                    eval_loss[k].append(v.item())

        self.model.load_state_dict(best_state_dict)
        with torch.no_grad():
            self.prev_xyz = self.model(self.template.torch_control_points)

        if visualize:
            self._visualize(self.prev_xyz, info, colors=self.param_colors)
        return self.prev_xyz, (up_s, i_s)

    def _texture_coords_to_param(self, p_s):
        # TODO: Check correctness
        return p_s / self.torch_texture_shape
