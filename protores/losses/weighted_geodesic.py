import torch

from protores.geometry.rotations import compute_geodesic_distance_from_two_matrices


def weighted_geodesic_loss(input: torch.Tensor, target: torch.Tensor, weights: torch.Tensor, eps: float = 1e-6):
    theta = compute_geodesic_distance_from_two_matrices(target, input)
    error = torch.sum(theta * weights) / (weights.sum() + eps)
    return error


def margin_geodesic_loss(input: torch.Tensor, target: torch.Tensor, margins: torch.Tensor):
    theta = compute_geodesic_distance_from_two_matrices(target, input)
    return torch.maximum(torch.zeros_like(theta), theta - margins).mean()
