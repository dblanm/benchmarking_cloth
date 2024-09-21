import torch
from chamferdist import ChamferDistance


class BidirectionalChamferDistance(ChamferDistance):
    def forward(
        self,
        source_cloud: torch.Tensor,
        target_cloud: torch.Tensor,
    ):
        return super().forward(
            source_cloud.unsqueeze(0), target_cloud.unsqueeze(0), bidirectional=True
        )


class MeanRMSE(torch.nn.Module):
    def forward(
        self,
        source_cloud: torch.Tensor,
        target_cloud: torch.Tensor,
    ):
        diff = source_cloud - target_cloud
        return torch.mean(
            torch.sqrt(diff[:, 0] ** 2 + diff[:, 1] ** 2 + diff[:, 2] ** 2)
        )


class Evaluator:
    available_metrics = {
        "chamfer_distance": BidirectionalChamferDistance(),
        "mean_rmse": MeanRMSE(),
    }

    def __init__(self, metrics):
        self.metrics = {metric: self.available_metrics[metric] for metric in metrics}

    def __call__(
        self,
        source_cloud: torch.Tensor,
        target_cloud: torch.Tensor,
    ):
        return {
            name: function(source_cloud, target_cloud)
            for name, function in self.metrics.items()
        }
