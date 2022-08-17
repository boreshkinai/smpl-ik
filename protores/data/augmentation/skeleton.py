import torch
from typing import Optional
from .augmentation import BaseAugmentation
from protores.geometry.skeleton import Skeleton


class MirrorSkeleton(BaseAugmentation):
    axis = [1.0, 0.0, 0.0]
    skeleton = None
    reflection_matrix_2 = None
    reflection_matrix_3 = None
    quat_indices = None
    mirror = False

    def __init__(self, skeleton: Skeleton, axis: Optional[list] = None, features: Optional[list] = None):
        super().__init__(features)
        self.axis = [float(i) for i in axis] if axis is not None else self.axis
        self._build_reflection_matrices(self.axis)
        self.swap_index_list = torch.LongTensor(skeleton.bone_pair_indices)

    def begin_batch(self, batch):
        self.swap_index_list = self.swap_index_list.to(batch.device)
        if self.reflection_matrix_2 is not None:
            self.reflection_matrix_2 = self.reflection_matrix_2.type_as(batch)
        if self.reflection_matrix_3 is not None:
            self.reflection_matrix_3 = self.reflection_matrix_3.type_as(batch)

        self.mirror = False
        if torch.rand(1)[0] > 0.5:
            self.mirror = True

    def _build_reflection_matrices(self, axis: list):
        assert len(axis) >= 2 and len(axis) <= 3, "Please ensure the specified mirror axis is either 2 dimensional or 3 dimensional"
        axis2D = axis[:2] if len(axis) == 3 else axis
        axis3D = axis

        self.reflection_matrix_2 = None
        self.reflection_matrix_3 = None

        if len(axis2D) == 2:
            self.reflection_matrix_2 = torch.eye(2)
            if axis2D == [0.0, 1.0]:
                # Reflect about Y
                self.reflection_matrix_2[0, 0] *= -1
            elif axis2D == [1.0, 0.0]:
                # Reflect about X
                self.reflection_matrix_2[1, 1] *= -1
            else:
                if len(axis3D) != 3:
                    # If only 2D axis passed we know it should be correct
                    raise ValueError("Invalid axis for 2D mirroring, please use one of: \n {}".format(
                        torch.eye(2).cpu().detach().numpy()))

        if len(axis3D) == 3:
            # Reflection for vector3
            self.reflection_matrix_3 = torch.eye(3)
            if axis3D == [1.0, 0.0, 0.0]:
                # YZ plane
                self.reflection_matrix_3[0, 0] *= -1
                self.quat_indices = torch.tensor([1, 2])
            else:
                raise ValueError("Invalid axis for 3D mirroring, please use one of: \n {}".format(
                    torch.eye(3)[-2].cpu().detach().numpy()))

    def _flip_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(tensor).index_copy_(1, self.swap_index_list, tensor)

    def vector2(self, vector2_tensor: torch.Tensor, feature_name=None) -> torch.Tensor:
        if self.reflection_matrix_2 is None:
            return vector2_tensor

        if not self.mirror:
            return vector2_tensor
        flipped = self._flip_tensor(vector2_tensor)
        return torch.matmul(self.reflection_matrix_2, flipped.transpose(2, 1)).transpose(1, 2)

    def vector3(self, vector3_tensor: torch.Tensor, feature_name=None) -> torch.Tensor:
        if self.reflection_matrix_3 is None:
            return vector3_tensor

        if not self.mirror:
            return vector3_tensor
        flipped = self._flip_tensor(vector3_tensor)
        return torch.matmul(self.reflection_matrix_3, flipped.permute(1, 2, 0)).permute(2, 0, 1)

    def quaternion(self, quat4_tensor: torch.Tensor, feature_name=None) -> torch.Tensor:
        if self.quat_indices is None:
            return quat4_tensor

        if not self.mirror:
            return quat4_tensor
        flipped = self._flip_tensor(quat4_tensor)
        flipped[:, :, self.quat_indices] *= -1
        return flipped
