import torch


def weighted_mse(input: torch.Tensor, target: torch.Tensor, weights: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    ret = ((input - target) ** 2).sum(dim=1) / input.shape[1]
    ret = ret * weights
    return torch.sum(ret) / (weights.sum() + eps)


def margin_mse_loss(input: torch.Tensor, target: torch.Tensor, margins: torch.Tensor):
    mse = ((input - target) ** 2).sum(dim=1) / input.shape[1]
    return torch.maximum(torch.zeros_like(mse), mse - margins).mean()

