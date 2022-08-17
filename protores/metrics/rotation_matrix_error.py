import torch
from pytorch_lightning.metrics import Metric

from protores.geometry.rotations import compute_geodesic_distance_from_two_matrices


class RotationMatrixError(Metric):
    def __init__(self, compute_on_step=True, dist_sync_on_step=False):
        super().__init__(compute_on_step=compute_on_step, dist_sync_on_step=dist_sync_on_step)

        self.add_state("accumulated", default=torch.FloatTensor([0.0]), dist_reduce_fx="sum")
        self.add_state("count", default=torch.LongTensor([0]), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        preds = preds.view(-1, 3, 3)
        target = target.view(-1, 3, 3)

        assert preds.shape == target.shape

        self.accumulated += torch.sum(compute_geodesic_distance_from_two_matrices(preds, target))
        self.count += target.shape[0]

    def compute(self):
        return self.accumulated / self.count