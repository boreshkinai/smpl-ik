import torch

from typing import Optional

from protores.geometry.quaternions import multiply_quaternions
from .augmentation import BaseAugmentation
from protores.geometry.rotations import get_random_rotation_matrices_around_random_axis, get_random_rotation_around_axis


class Quaternion_XYZW_to_WXYZ(BaseAugmentation):
    def quaternion(self, quaterion_tensor, feature_name):
        return quaterion_tensor[:, :, [3, 0, 1, 2]]


class RandomTranslation(BaseAugmentation):
    axis = torch.FloatTensor([1.0, 1.0, 1.0])
    range = [-1.0, 1.0]
    random_vec = None

    def __init__(self, axis: Optional[list] = None, range: Optional[list] = None, features: Optional[list] = None):
        super().__init__(features)
        if axis is not None:
            self.axis = torch.FloatTensor(axis)
        if range is not None:
            self.range = torch.FloatTensor(range)

    def begin_batch(self, batch):
        self.random_vec = torch.FloatTensor(batch.shape[0], 3).uniform_(self.range[0], self.range[1]) * self.axis
        self.random_vec = self.random_vec.type_as(batch)

    def vector3(self, vec3, feature_name=None):
        p = vec3.view(vec3.shape[0], -1) + self.random_vec.repeat(1, vec3.shape[1])
        return p.view_as(vec3)


class RandomRotation(BaseAugmentation):
    rotation_axis = None
    random_rot = None
    random_quat = None

    def __init__(self, axis: Optional[list] = None, features: Optional[list] = None):
        super().__init__(features)
        if axis is not None:
            self.rotation_axis = torch.FloatTensor(axis).view(1, 3)

    def begin_batch(self, batch):
        if self.rotation_axis is None:
            self.random_rot, self.random_quat = get_random_rotation_matrices_around_random_axis(batch, return_quaternion=True)
        else:
            self.random_rot, self.random_quat = get_random_rotation_around_axis(
                self.rotation_axis.repeat(batch.shape[0], 1), return_quaternion=True)

        self.random_rot = self.random_rot.type_as(batch)
        self.random_quat = self.random_quat.type_as(batch)  # order is w, x, y, z

    def vector3(self, vector3_tensor: torch.Tensor, feature_name=None) -> torch.Tensor:
        new_vec3 = torch.matmul(self.random_rot, vector3_tensor.transpose(2, 1)).transpose(2, 1)
        return new_vec3

    def quaternion(self, quat4_tensor: torch.Tensor, feature_name=None) -> torch.Tensor:
        # The hip quaternion will always be the first index of the second axis
        # Rotations are all local wrt to parent so we need to only rotate hip bone
        # input / output quat order is x, y, z, w
        quat4_tensor[:, 0] = multiply_quaternions(self.random_quat, quat4_tensor[:, 0, [3, 0, 1, 2]])[:, [1, 2, 3, 0]]
        return quat4_tensor
