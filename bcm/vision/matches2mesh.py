import hydra
import torch
from bcm.vision.utils import get_default_device, loss_euc, metric, MLP
from kornia.geometry.camera.pinhole import PinholeCamera
from omegaconf import DictConfig


@hydra.main(config_path="config", config_name="metric_preservation")
class MLPSolver:
    def __init__(self, mesh, template, camera: PinholeCamera, cfg: DictConfig):
        self.seed = cfg.seed
        self.patience_template = cfg.patience_template
        self.patience_update = cfg.patience_update
        self.max_it = cfg.max_it
        self.lr_update = cfg.lr_update
        self.lambda_time = cfg.lambda_time
        self.lambda_metric = cfg.lambda_metric

        self.setup_torch()

        self.camera = camera
        self.model = MLP().to(self.device)

        assert mesh.parameters is not None
        self.control_points = torch.from_numpy(mesh.parameters).float().to(self.device)

        self.prev_xyz = None
        self.template_metric = None
        self.register_template(template)

    def setup_torch(self):
        # Reproducibility
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)

        self.device = get_default_device()

    def register_template(self, template):
        # Over-fit to template, for which we know the ground truth
        ground_truth = torch.from_numpy(template.ground_truth).float().to(self.device)

        optimizer = torch.optim.Adam(self.model.parameters())

        min_loss = torch.inf
        counter = 0

        while True:
            prediction = self.model(self.control_points)

            loss = loss_euc(prediction, ground_truth)
            if loss < min_loss:
                min_loss = loss
                counter = 0
            else:
                counter += 1
                if counter >= self.patience_template:
                    break

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            prediction = self.model(self.control_points)
        print(f"Final loss is {loss_euc(prediction, ground_truth)}")
        self.prev_xyz = prediction.detach()
        self.template_metric = metric(self.model, self.control_points).detach()

    def update(self, data_point):
        num_matches = len(data_point.matches.barycentric_coords)
        idx1 = []
        idx2 = []
        values = []
        for i, bary in enumerate(data_point.matches.barycentric_coords):
            for vertex_idx, weight in bary.items():
                idx1.append(i)
                idx2.append(vertex_idx)
                values.append(weight)
        weights = torch.sparse_coo_tensor(
            torch.tensor([idx1, idx2]),
            values,
            size=(num_matches, len(self.mesh)),
            device=self.device,
            dtype=torch.float32,
        )
        tracked_image_true = (
            torch.from_numpy(data_point.matches.tracked_points).float().to(self.device)
        )

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr_update)

        min_loss = torch.inf
        counter = 0
        best_state_dict = None

        for _ in range(self.max_it):
            optimizer.zero_grad()
            prediction = self.model(self.control_points)
            tracked_surface_prediction = torch.sparse.mm(weights, prediction)
            tracked_image_prediction = self.projector(tracked_surface_prediction)
            loss_proj = torch.nn.functional.mse_loss(
                tracked_image_prediction, tracked_image_true
            )
            loss = loss_proj
            if self.lambda_time != 0:
                loss_time = torch.nn.functional.mse_loss(prediction, self.prev_xyz)
                loss += self.lambda_time * loss_time

            if self.lambda_metric != 0:
                loss_metric = torch.nn.functional.mse_loss(
                    metric(self.model, self.control_points), self.template_metric
                )
                loss += self.lambda_metric * loss_metric

            if loss < min_loss:
                min_loss = loss
                best_state_dict = self.model.state_dict()
                counter = 0
            else:
                counter += 1
                if counter >= self.patience_update:
                    break

            loss.backward()
            optimizer.step()

        self.model.load_state_dict(best_state_dict)
        with torch.no_grad():
            self.prev_xyz = self.model(self.control_points).detach()
        self._update_mesh(self.prev_xyz.cpu().numpy())
