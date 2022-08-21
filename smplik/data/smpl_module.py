import os
import pytorch_lightning as pl
from torch.utils.data import DataLoader

import hydra
from dataclasses import dataclass

from smplik.data.augmentation import RandomRotationLocal, RandomTranslation
from smplik.data.base_dataset import BatchedDataset
from smplik.data.dataset.typed_table import TypedColumnDataset
from smplik.data.datasets import DatasetLoader
from smplik.data.smpl_dataset import SmplDataset
from smplik.utils.python import get_full_class_reference


# custom collate function for batched dataset
# Note: would be nicer as lambdas, but lambdas cannot be pickled and will thus break distributed training
def _batched_collate(batch):
    return batch[0]


class SmplDataModule(pl.LightningDataModule):
    def __init__(self, path: str, name: str, batch_size: int, num_workers: int = 0,
                 rotate: bool = True, translate: bool = True,
                 augment_training: bool = True, augment_validation: bool = True):
        super().__init__()

        self.path = hydra.utils.to_absolute_path(path)
        self.name = name
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.rotate = rotate
        self.translate = translate
        self.augment_training = augment_training
        self.augment_validation = augment_validation

    def prepare_data(self):
        # download dataset
        dataset_loader = DatasetLoader(self.path)
        dataset_loader.pull(self.name)

    def get_data_specific_components(self):
        return None

    def setup(self, stage=None):
        self.initialize_datasets()
        self.setup_augmentations()
        self.wrap_datasets()

    def initialize_datasets(self):
        # retrieve train / validation / test split
        self.split = DatasetLoader(self.path).get_split(self.name)

        training_dataset = TypedColumnDataset(self.split, subset="Training")
        self.training_dataset = SmplDataset(source=training_dataset)

        validation_dataset = TypedColumnDataset(self.split, subset="Validation")
        self.validation_dataset = SmplDataset(source=validation_dataset)

        test_dataset = TypedColumnDataset(self.split, subset="Test")
        self.test_dataset = SmplDataset(source=test_dataset)

    def setup_augmentations(self):
        # Note: this should ideally be done in the init function but mirroring requires the skeleton
        self.transforms = []
        if self.rotate:
            self.transforms.append(
                RandomRotationLocal(axis=[0, 0, 1],  # None if used in conjunction with pose estimation
                                    features=['BonePositions', 'BoneRotations'],
                                    is_xyzw_quat=False))
        if self.translate:
            self.transforms.append(RandomTranslation(range=[-2, 2], features=['BonePositions']))

        if self.augment_training:
            self.training_dataset.set_transforms(self.transforms)
        if self.augment_validation:
            self.validation_dataset.set_transforms(self.transforms)
            self.test_dataset.set_transforms(self.transforms)

    def wrap_datasets(self):
        # wrap datasets into their batched version
        # We need to do that as a wrapped dataset as Lightning doesn't support well custom samplers with distributed training
        # As a consequence of that, we skipp collapsing in the data loader
        self.training_dataset = BatchedDataset(self.training_dataset, batch_size=self.batch_size)
        self.validation_dataset = BatchedDataset(self.validation_dataset, batch_size=self.batch_size)
        self.test_dataset = BatchedDataset(self.test_dataset, batch_size=self.batch_size)

    def train_dataloader(self):
        return DataLoader(self.training_dataset, batch_size=1, shuffle=True, num_workers=self.num_workers,
                          collate_fn=_batched_collate)

    def val_dataloader(self):
        loaders = [DataLoader(self.validation_dataset, batch_size=1,
                              shuffle=False, num_workers=self.num_workers, collate_fn=_batched_collate),
                   DataLoader(self.test_dataset, batch_size=1,
                              shuffle=False, num_workers=self.num_workers, collate_fn=_batched_collate)]
        return loaders

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=1, shuffle=False, num_workers=self.num_workers,
                          collate_fn=_batched_collate)


@dataclass
class SmplDataModuleOptions:
    _target_: str = get_full_class_reference(SmplDataModule)
    path: str = "./datasets"
    name: str = "AMASS_SMPL"
    batch_size: int = 2048
    num_workers: int = 0
    rotate: bool = True
    translate: bool = False
    augment_training: bool = True
    augment_validation: bool = False
