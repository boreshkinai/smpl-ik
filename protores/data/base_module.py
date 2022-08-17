import pytorch_lightning as pl
from torch.utils.data import DataLoader

from protores.geometry.skeleton import Skeleton
from protores.data.base_dataset import BaseDataset, BatchedDataset

import hydra
from dataclasses import dataclass

from protores.data.dataset.typed_table import TypedColumnDataset
from protores.data.augmentation import MirrorSkeleton, RandomRotation, RandomTranslation
from protores.utils.python import get_full_class_reference
from protores.data.datasets import DatasetLoader


# custom collate function for batched dataset
# Note: would be nicer as lambdas, but lambdas cannot be pickled and will thus break distributed training
def _batched_collate(batch):
    return batch[0]


class BaseDataModule(pl.LightningDataModule):
    def __init__(self, path: str, name: str, batch_size: int, num_workers: int = 0, mirror: bool = True,
                 rotate: bool = True, translate: bool = True,
                 augment_training: bool = True, augment_validation: bool = True, train_subsampling: int = 1):
        super().__init__()

        self.path = hydra.utils.to_absolute_path(path)
        self.name = name
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.skeleton = None
        self.mirror = mirror
        self.rotate = rotate
        self.translate = translate
        self.augment_training = augment_training
        self.augment_validation = augment_validation
        self.train_subsampling = train_subsampling

    def prepare_data(self):
        # download dataset
        dataset_loader = DatasetLoader(self.path)
        dataset_loader.pull(self.name)

    def get_skeleton(self) -> Skeleton:
        dataset_settings = DatasetLoader(self.path).get_settings(self.name)
        assert "skeleton" in dataset_settings, "No skeleton data could be found in dataset settings"
        return Skeleton(dataset_settings["skeleton"])

    def setup(self, stage=None):
        # retrieve train / validation split
        self.split = DatasetLoader(self.path).get_split(self.name)

        # retrieve skeleton
        self.skeleton = self.get_skeleton()

        # load datasets
        training_dataset, validation_dataset = TypedColumnDataset.FromSplit(self.split)
        self.training_dataset = BaseDataset(source=training_dataset, skeleton=self.skeleton, subsampling=self.train_subsampling)
        self.validation_dataset = BaseDataset(source=validation_dataset, skeleton=self.skeleton)

        # setup data augmentation
        # TODO: make this parameters
        # Note: this should ideally be done in the init function but mirroring requires the skeleton
        self.transforms = []
        if self.mirror:
            self.transforms.append(MirrorSkeleton(self.skeleton, features=['BonePositions', 'BoneRotations'])) # TODO: 'RootPosition' 'RootRotation'
        if self.rotate:
            self.transforms.append(RandomRotation(axis=[0, 1, 0], features=['BonePositions', 'BoneRotations', 'RootPosition', 'RootRotation']))
        if self.translate:
            self.transforms.append(RandomTranslation(range=[-2, 2], features=['BonePositions', 'RootPosition']))

        if self.augment_training:
            self.training_dataset.set_transforms(self.transforms)
        if self.augment_validation:
            self.validation_dataset.set_transforms(self.transforms)

        # wrap datasets into their batched version
        # We need to do that as a wrapped dataset as Lightning doesn't support well custom samplers with distributed training
        # As a consequence of that, we skipp collapsing in the data loader
        self.training_dataset = BatchedDataset(self.training_dataset, batch_size=self.batch_size)
        self.validation_dataset = BatchedDataset(self.validation_dataset, batch_size=self.batch_size)

    def train_dataloader(self):
        return DataLoader(self.training_dataset, batch_size=1, shuffle=True, num_workers=self.num_workers, collate_fn=_batched_collate)

    def val_dataloader(self):
        return DataLoader(self.validation_dataset, batch_size=1, shuffle=False, num_workers=self.num_workers, collate_fn=_batched_collate)

    def test_dataloader(self):
        return self.val_dataloader()


@dataclass
class BaseDataModuleOptions:
    _target_: str = get_full_class_reference(BaseDataModule)
    path: str = "./datasets"
    name: str = "deeppose_master_v1_fps60"
    mirror: bool = True
    rotate: bool = True
    translate: bool = True
    augment_training: bool = True
    augment_validation: bool = True
    batch_size: int = 2048
    num_workers: int = 0
    train_subsampling: int = 1
