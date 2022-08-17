import torch

from protores.geometry.vector import normalize_vector


# Compute the angular loss between 3D vectors
# input: predicted 3D vectors, size BxN
# target: target 3D vectors, size BxN
# unit_input: if set to True, input vectors are considered unit vectors and will not be normalized (saves computation)
# unit_target: if set to True, target vectors are considered unit vectors and will not be normalized (saves computation)
# eps: epsilon values used to avoid infinite gradients of acos() near -1 and +1
def angular_loss(input: torch.Tensor, target: torch.Tensor, unit_input: bool = False, unit_target: bool = False,
                 eps: float = 1e-6) -> torch.Tensor:

    if not unit_input:
        input = normalize_vector(input, eps=eps)

    if not unit_target:
        target = normalize_vector(target, eps=eps)

    loss = (target * input).sum(dim=1)
    loss = torch.acos(loss.clamp(-1.0 + eps, 1.0 - eps))
    loss = loss.mean()
    return loss
