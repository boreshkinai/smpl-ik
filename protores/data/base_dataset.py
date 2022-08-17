from typing import List
import torch
from torch.utils.data import Dataset

from protores.data.augmentation import BaseAugmentation
from protores.data.dataset.typed_table import FlatTypedColumnDataset
from protores.geometry.skeleton import Skeleton


class BatchedDataset(Dataset):
    def __init__(self, source: Dataset, batch_size: int = 1024, shuffle: bool = True, drop_last: bool = True, seed: int = 0):
        super().__init__()

        self.source = source
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.batch_indices = None
        self.seed = seed
        self.epoch = 0
        self._shuffle()

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch
        self._shuffle()

    def _shuffle(self):
        if self.shuffle:
            # seeding is needed to make it deterministic across processes and epochs
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            self.batch_indices = torch.randperm(len(self.source), generator=g).tolist()
        else:
            self.batch_indices = list(range(len(self.source)))

    def __getitem__(self, index):
        start_idx = index * self.batch_size
        end_idx = min(len(self.source), start_idx + self.batch_size)
        batch_indices = self.batch_indices[start_idx:end_idx]
        batch = self.source.__getitem__(batch_indices)
        return batch

    def __len__(self):
        if self.drop_last:
            return len(self.source) // self.batch_size
        else:
            return (len(self.source) + self.batch_size - 1) // self.batch_size


class BaseDataset(Dataset):
    def __init__(self, source: FlatTypedColumnDataset, skeleton: Skeleton, subsampling: int = 1):
        super().__init__()

        self.source = source
        self.skeleton = skeleton
        self.subsampling = subsampling

        # add computed features
        # TODO: is this the best place to do that? Should that be in the data module?
        #       or a subclass of the dataset or data module?
        self.source.add_calculated_feature(character_position, "BonePositions", "RootPosition", "Vector3")
        self.source.add_calculated_feature(character_rotation, "BoneRotations", "RootRotation", "Quaternion")

        # get all joints in skeleton order
        self.all_joints = []
        for i in range(self.skeleton.nb_joints):
            self.all_joints.append(self.skeleton.index_bones[i])

        # compute feature indices
        self.joint_positions_idx = self.source.get_feature_indices(["BonePositions"], self.all_joints)
        self.joint_rotations_idx = self.source.get_feature_indices(["BoneRotations"], self.all_joints)
        self.character_position_idx_in = self.source.select_features("RootPosition")
        self.character_rotation_idx_in = self.source.select_features("RootRotation")

    def set_transforms(self, transforms: List[BaseAugmentation]):
        for transform in transforms:
            self.source.add_transform(transform)

    def __getitem__(self, index):
        if self.subsampling > 1:
            if isinstance(index, list):
                for i in range(len(index)):
                    index[i] = self.subsampling * index[i]
            else:
                index = self.subsampling * index

        data = self.source.__getitem__(index)

        #Note: we convert quaternions from x, y, z, w to w, x, y, z
        item = {
            "joint_positions": data[:, self.joint_positions_idx].view(-1, self.skeleton.nb_joints, 3),
            "joint_rotations": data[:, self.joint_rotations_idx].view(-1, self.skeleton.nb_joints, 4)[:, :, [3, 0, 1, 2]],
            "root_position": data[:, self.character_position_idx_in].view(-1, 3),
            "root_rotation": data[:, self.character_rotation_idx_in].view(-1, 4)[:, [3, 0, 1, 2]]
        }

        return item

    def __len__(self):
        if self.subsampling > 1:
            return len(self.source) // self.subsampling
        return len(self.source)


# create character frame position features (0, 0, 0)
def character_position(features):
    length = features[0].shape[0]
    pos = torch.zeros((length, 3)).type_as(features[0])
    return pos


# create character frame rotation features (identity)
def character_rotation(features):
    length = features[0].shape[0]
    rot = torch.zeros((length, 4)).type_as(features[0])
    rot[:, 3] = 1  # identity quaternion x, y, z, w  (to follow unity convention)
    return rot