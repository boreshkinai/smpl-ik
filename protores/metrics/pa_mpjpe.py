import torch
from torchmetrics import Metric

from protores.metrics.metric_utils import calc_pampjpe


class PA_MPJPE(Metric):
    def __init__(self, compute_on_step=True, dist_sync_on_step=False):
        super().__init__(compute_on_step=compute_on_step, dist_sync_on_step=dist_sync_on_step)

        self.add_state("accumulated", default=torch.FloatTensor([0.0]), dist_reduce_fx="sum")
        self.add_state("count", default=torch.LongTensor([0]), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        assert preds.shape == target.shape
        assert preds.dim() == 3
        self.accumulated += torch.sum(calc_pampjpe(preds, target))
        self.count += target.shape[0]

    def compute(self):
        return (self.accumulated / self.count)[0]