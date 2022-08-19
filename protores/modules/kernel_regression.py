import torch
from torch import nn


class GaussianKernelRegression(nn.Module):
    # source: source data -> NxK
    # target: target feature -> NxF
    def __init__(self, source: torch.Tensor, target: torch.Tensor, p: float = 2.0):
        super().__init__()
        self.register_buffer("source", source)
        self.register_buffer("target", target)
        self.p = p

    # target: target feature -> BxF
    # std: std of kernel for each sample -> B
    def forward(self, target: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        # Note: torch.cdist does not export to ONNX
        # dist = torch.cdist(target.unsqueeze(0), self.target.unsqueeze(0), self.p).squeeze(0)  # BxN
        dist = target.unsqueeze(1).repeat(1, self.target.shape[0], 1) - self.target.unsqueeze(0)  # BxNxF
        dist = (dist ** 2).sum(dim=2, keepdim=False)  # BxN

        std = std.unsqueeze(-1)  # Bx1
        x = -dist / (2 * (std ** 2))  # BxN

        logsumexp = x.logsumexp(dim=1, keepdim=True)  # Bx1
        w = torch.exp(x - logsumexp)  # BxN
        y = self.source.unsqueeze(0) * w.unsqueeze(-1)  # BxNxF
        y = y.sum(dim=1, keepdim=False)  # BxF

        return y
