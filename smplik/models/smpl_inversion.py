from typing import Any, Optional

import torch
import pytorch_lightning as pl

from torch.utils.data import Dataset, DataLoader
from dataclasses import dataclass

from pytorch_lightning.utilities.types import EPOCH_OUTPUT
from torchmetrics.regression.mean_squared_error import MeanSquaredError
from smplik.models.base_task import *
from smplik.smpl.smpl_fk import SmplFK
from smplik.smpl.smpl_info import SMPL_JOINT_NAMES
from smplik.utils.python import get_full_class_reference


class DummyDataset(Dataset):
    def __init__(self, epoch_length: int):
        super().__init__()

        self.epoch_length = epoch_length
        self.dummy_data = torch.zeros(1)

    def __getitem__(self, index):
        return self.dummy_data

    def __len__(self):
        return self.epoch_length


class DummyDataModule(pl.LightningDataModule):
    def __init__(self, training_size: int, validation_size: int, test_size: int, batch_size: int, num_workers: int):
        super().__init__()

        self.training_size = training_size
        self.validation_size = validation_size
        self.test_size = test_size
        self.batch_size = batch_size
        self.num_workers = num_workers

    def get_data_specific_components(self):
        return None

    def train_dataloader(self):
        return DataLoader(DummyDataset(self.training_size), batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(DummyDataset(self.validation_size), batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(DummyDataset(self.test_size), batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)


@dataclass
class DummyDataModuleOptions:
    _target_: str = get_full_class_reference(DummyDataModule)
    training_size: int = 100000
    validation_size: int = 10000
    test_size: int = 10000
    batch_size: int = 2048
    num_workers: int = 0


@dataclass
class SmplInversionTaskOptions(AbstractTaskOptions):
    dataset: DummyDataModuleOptions = DummyDataModuleOptions()
    smpl_models_path: str = "./tools/smpl/models/"
    smpl_male_name: str = "basicModel_m_lbs_10_207_0_v1.0.0"
    smpl_female_name: str = "basicModel_f_lbs_10_207_0_v1.0.0"
    smpl_neutral_name: str = "basicModel_neutral_lbs_10_207_0_v1.0.0"
    betas_range: float = 5.0
    betas_uniform: bool = False
    scale_min: float = 0.5
    scale_max: float = 2.0
    uniform_scale: bool = True
    predict_scale: bool = True


class SmplInversionTask(AbstractTask):
    @staticmethod
    def get_metrics():
        metrics = AbstractTask.get_metrics()
        metrics["hp_metrics/length"] = -1
        return metrics

    def __init__(self, data_components: Any, opts: SmplInversionTaskOptions):
        super().__init__(opts=opts)

        self.smpl_male = SmplFK(models_path=opts.smpl_models_path, model_name=opts.smpl_male_name)
        self.smpl_female = SmplFK(models_path=opts.smpl_models_path, model_name=opts.smpl_female_name)
        self.smpl_neutral = SmplFK(models_path=opts.smpl_models_path, model_name=opts.smpl_neutral_name)

        # indices of each joint in SMPL
        smpl_joint_idx = {}
        for i in range(len(SMPL_JOINT_NAMES)):
            smpl_joint_idx[SMPL_JOINT_NAMES[i]] = i

        self.num_betas = self.smpl_male.num_betas
        self.input_joints = ['head', 'right_hip', 'right_knee', 'right_ankle', 'right_shoulder', 'right_elbow', 'right_wrist']
        self.smpl_input_joints_idx = [smpl_joint_idx[name] for name in self.input_joints]

        # indices of each joint in this model
        joint_idx = {}
        for i in range(len(self.input_joints)):
            joint_idx[self.input_joints[i]] = i

        # here we define "virtual" bones that will be used to compute distance, so that the estimation work independently of the pose
        # Note: we only use Unity mandatory humanoid joints here to make sure this is compatible with all characters
        self.bones = [['right_hip', 'right_knee'],
                      ['right_knee', 'right_ankle'],
                      ['head', 'right_ankle'],
                      ['head', 'right_wrist'],
                      ['right_shoulder', 'right_elbow'],
                      ['right_elbow', 'right_wrist']]
        self.bones_parent_idx = [joint_idx[pair[0]] for pair in self.bones]
        self.bones_child_idx = [joint_idx[pair[1]] for pair in self.bones]

        self.gender_multinomial = torch.ones((1, 3))

        self.test_length_metric = MeanSquaredError(compute_on_step=False)

    def test_step(self, batch, batch_idx):
        input_data, target_data = self.get_data_from_batch(batch)
        predicted = self(input_data)
        self.update_test_metrics(predicted, target_data, input_data)

    def test_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
        super().test_epoch_end(outputs=outputs)
        self.logger.log_metrics({
            "hp_metrics/position": self.test_length_metric.compute()
        })
        self.test_length_metric.reset()

    def update_test_metrics(self, predicted, target, input):
        input_gender = input["gender"]
        target_joint_positions = target["positions"]
        predicted_betas = predicted["betas"]

        target_lengths = self.get_bone_lengths(target_joint_positions)

        predicted_joint_positions = self.apply_smpl(betas=predicted_betas, gender=input_gender)

        if self.hparams.predict_scale:
            predicted_scale = predicted["scale"]
            predicted_joint_positions = self.scale(predicted_joint_positions, predicted_scale)

        predicted_lengths = self.get_bone_lengths(predicted_joint_positions)

        self.test_length_metric(predicted_lengths, target_lengths)

    def get_bone_lengths(self, positions):
        offsets = positions[:, self.bones_parent_idx, :] - positions[:, self.bones_child_idx, :]
        lengths = torch.sqrt(torch.pow(offsets, 2).sum(dim=-1, keepdim=False))
        return lengths

    def scale(self, positions, scale):
        return positions * scale.unsqueeze(1)

    def apply_smpl(self, betas, gender):
        male_positions = self._apply_smpl(betas, self.smpl_male)
        female_positions = self._apply_smpl(betas, self.smpl_female)
        neutral_positions = self._apply_smpl(betas, self.smpl_neutral)

        is_male = gender == 0
        is_female = gender == 1
        positions = torch.where(is_male.unsqueeze(1), male_positions, torch.where(is_female.unsqueeze(1), female_positions, neutral_positions))

        # only keep input positions
        positions = positions[:, self.smpl_input_joints_idx, :]

        return positions

    def _apply_smpl(self, betas, smpl_model):
        body_pose = torch.zeros((betas.shape[0], smpl_model.NUM_BODY_JOINTS, 3)).type_as(betas)
        global_orient = torch.zeros((betas.shape[0], 1, 3)).type_as(betas)
        smpl_out = smpl_model.forward(betas=betas, body_pose=body_pose, global_orient=global_orient, transl=None, return_verts=False)
        return smpl_out.joints

    @torch.no_grad()
    def get_data_from_batch(self, batch):
        batch_size = batch.shape[0]
        batch = self.generate_batch(batch_size)
        input_data = self.get_input_data_from_batch(batch)
        target_data = self.get_target_data_from_batch(batch)
        return input_data, target_data

    def generate_batch(self, batch_size: int, force_gender: Optional[int] = None):
        if self.hparams.betas_uniform:
            betas = 2 * self.hparams.betas_range * (torch.rand((batch_size, self.num_betas), device=self.device) - 0.5)
        else:
            betas = self.hparams.betas_range * torch.randn((batch_size, self.num_betas), device=self.device)

        if force_gender is None:
            gender = torch.multinomial(input=self.gender_multinomial.repeat(batch_size, 1), num_samples=1, replacement=False).to(self.device)
        else:
            gender = force_gender * torch.ones((batch_size, 1), dtype=torch.int64, device=self.device)

        positions = self.apply_smpl(betas=betas, gender=gender)

        # generate character scale
        if self.hparams.predict_scale:
            if self.hparams.uniform_scale:
                scale = self.hparams.scale_min + (self.hparams.scale_max - self.hparams.scale_min) * torch.rand((batch_size, 1), device=self.device)
            else:
                scale = self.hparams.scale_min + (self.hparams.scale_max - self.hparams.scale_min) * torch.rand((batch_size, 3), device=self.device)
            positions = self.scale(positions, scale)

        batch = {
            "betas": betas,
            "gender": gender,
            "positions": positions
        }

        if self.hparams.predict_scale:
            batch["scale"] = scale

        return batch

    def get_input_data_from_batch(self, batch):
        return {
            "gender": batch["gender"],
            "positions": batch["positions"]
        }

    def get_target_data_from_batch(self, batch):
        target = {
            "betas": batch["betas"],
            "positions": batch["positions"]
        }
        if self.hparams.predict_scale:
            target["scale"] = batch["scale"]
        return target

    def get_dummy_input(self):
        input = {
            "positions": torch.zeros((1, len(self.input_joints), 3)),
            "gender": torch.zeros((1, 1), dtype=torch.int64)
        }
        return input

    def get_dynamic_axes(self):
        return {}

    def get_dummy_output(self):
        out = {
            "betas": torch.randn((1, self.num_betas))
        }
        if self.hparams.predict_scale:
            out["scale"] = torch.randn((1, 1))
        return out

    def get_metadata(self):
        return {
            "num_betas": self.smpl_male.num_betas,
            "gender_ids": [0, 1, 2],
            "gender_names": ["male", "female", "neutral"],
            "input_joints": self.input_joints,
            "predict_scale": self.hparams.predict_scale
        }

