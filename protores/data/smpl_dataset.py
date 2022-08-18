import torch
from torch.utils.data import Dataset

from typing import List

from protores.data.augmentation import BaseAugmentation
from protores.data.dataset.typed_table import TypedColumnDataset
from protores.smpl.smpl_info import SMPL_JOINT_NAMES, SMPL_SHAPE_NAMES


class SmplDataset(Dataset):
    def __init__(self, source: TypedColumnDataset):
        super().__init__()

        self.source = source

        # get all joints in skeleton order
        self.all_joint_names = SMPL_JOINT_NAMES[:24]
        self.all_shape_names = SMPL_SHAPE_NAMES
        self.nb_joints = len(self.all_joint_names)
        self.nb_shapes = len(self.all_shape_names)

        # compute feature indices
        self.joint_positions_idx = self.source.get_feature_indices(["BonePositions"], self.all_joint_names)
        self.joint_rotations_idx = self.source.get_feature_indices(["BoneRotations"], self.all_joint_names)
        self.shape_idx = self.source.get_feature_indices(self.all_shape_names)
        self.gender_idx = self.source.get_feature_indices("Gender")
        self.translation_idx = self.source.get_feature_indices("RootPosition")

    def set_transforms(self, transforms: List[BaseAugmentation]):
        for transform in transforms:
            self.source.add_transform(transform)

    def __getitem__(self, index):
        if not isinstance(index, list):
            index = [index]

        data = self.source.__getitem__(index)

        # Note: we convert quaternions from x, y, z, w to w, x, y, z
        item = {
            "joint_positions": data[:, self.joint_positions_idx].view(-1, self.nb_joints, 3),
            "joint_rotations": data[:, self.joint_rotations_idx].view(-1, self.nb_joints, 4),
            "betas": data[:, self.shape_idx].view(-1, self.nb_shapes),
            "gender": data[:, self.gender_idx].view(-1, 1).to(dtype=torch.int64)
        }

        return item

    def __len__(self):
        return len(self.source)
