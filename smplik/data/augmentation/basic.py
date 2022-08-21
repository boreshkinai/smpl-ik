import torch
from .augmentation import BaseAugmentation


class Ones(BaseAugmentation):
    ''' makes everything all 1.0 '''

    def scalar(self, scal, feature_name=None):
        return torch.ones_like(scal)

    def vector2(self, vec2, feature_name=None):
        return torch.ones_like(vec2)

    def vector3(self, vec3, feature_name=None):
        return torch.ones_like(vec3)

    def quaternion(self, quat, feature_name=None):
        return torch.ones_like(quat)


class Zeros(BaseAugmentation):
    ''' makes everything all 0.0 '''

    def scalar(self, scal, feature_name=None):
        return torch.zeros_like(scal)

    def vector2(self, vec2, feature_name=None):
        return torch.zeros_like(vec2)

    def vector3(self, vec3, feature_name=None):
        return torch.zeros_like(vec3)

    def quaternion(self, quat, feature_name=None):
        return torch.zeros_like(quat)
