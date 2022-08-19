from dataclasses import dataclass
from typing import Any, Optional

import torch
from hydra.utils import instantiate

from torchmetrics.regression.mean_squared_error import MeanSquaredError

from protores.data.data_components import DataComponents
from protores.modules.SmplInversionNet import SmplInversionGaussian
from protores.models.smpl_inversion import SmplInversionTaskOptions, SmplInversionTask
from protores.utils.model_factory import ModelFactory



@dataclass
class SmplInversionModelOptions(SmplInversionTaskOptions):
    backbone: Optional[Any] = None
    optimizer: Optional[Any] = None
    betas_loss_scale: float = 0.0
    position_loss_scale: float = 0.0
    length_loss_scale: float = 1.0
    betas_l1_loss_scale: float = 1.0
    scale_loss_scale: float = 1.0
    input_bone_lengths: bool = True


@ModelFactory.register(SmplInversionModelOptions, schema_name="SmplInversion")
class SmplInversionModel(SmplInversionTask):
    def __init__(self, data_components: DataComponents, opts: SmplInversionModelOptions):
        super().__init__(data_components=data_components, opts=opts)

        self.create_backbone()

        self.validation_length_metric = MeanSquaredError(compute_on_step=False)
        self.validation_betas_metric = MeanSquaredError(compute_on_step=False)
        self.validation_scale_metric = MeanSquaredError(compute_on_step=False)

    def create_backbone(self):
        size_in = len(self.bones) if self.hparams.input_bone_lengths else len(self.input_joints) * 3
        size_out = self.num_betas + 1 if self.hparams.predict_scale else self.num_betas
        self.net = instantiate(self.hparams.backbone, size_in=size_in, size_out=size_out)

    def forward(self, input_data):
        gender = input_data["gender"]
        positions = input_data["positions"]

        x = self.get_input_features(positions)
        betas = self.net(x, gender)

        if self.hparams.predict_scale:
            scale = betas[:, self.num_betas:]
            betas = betas[:, :self.num_betas]

        out = {
            "betas": betas
        }
        if self.hparams.predict_scale:
            out["scale"] = scale
        return out

    def get_input_features(self, positions):
        if self.hparams.input_bone_lengths:
            # make model translation and rotation invariant
            x = self.get_bone_lengths(positions)
        else:
            # make translation invariant
            # Notes: assume root (or reference) joint is provide at index 0
            x = positions - positions[:, 0, :].unsqueeze(1)
            x = x.view(-1, len(self.input_joints) * 3)
        return x

    def training_step(self, batch, batch_idx):
        losses = self.shared_step(batch, step="training")
        self.log_train_losses(losses)
        return losses["total"]

    def validation_step(self, batch, batch_idx):
        losses = self.shared_step(batch, step="validation")
        self.log_validation_losses(losses)
        return losses

    def validation_step_end(self, *args, **kwargs):
        super().validation_step_end(args, kwargs)
        self.log("validation/metrics/length", self.validation_length_metric, on_step=False, on_epoch=True)
        self.log("validation/metrics/betas", self.validation_betas_metric, on_step=False, on_epoch=True)
        if self.hparams.predict_scale:
            self.log("validation/metrics/scale", self.validation_scale_metric, on_step=False, on_epoch=True)

    def shared_step(self, batch, step: str):
        input_data, target_data = self.get_data_from_batch(batch)

        in_positions = input_data["positions"]
        in_gender = input_data["gender"]

        target_positions = target_data["positions"]
        target_betas = target_data["betas"]
        target_lengths = self.get_bone_lengths(target_positions)
        if self.hparams.predict_scale:
            target_scale = target_data["scale"]

        predicted = self.forward(input_data)

        predicted_betas = predicted["betas"]
        predicted_positions = self.apply_smpl(betas=predicted_betas, gender=in_gender)
        if self.hparams.predict_scale:
            predicted_scale = predicted["scale"]
            predicted_positions = self.scale(predicted_positions, predicted_scale)
        predicted_lengths = self.get_bone_lengths(predicted_positions)

        betas_loss = torch.nn.functional.mse_loss(predicted_betas, target_betas)
        positions_loss = torch.nn.functional.mse_loss(predicted_positions.view(-1, 3), target_positions.view(-1, 3))
        length_loss = torch.nn.functional.mse_loss(predicted_lengths, target_lengths)
        betas_l1_loss = torch.nn.functional.l1_loss(predicted_betas, torch.zeros_like(predicted_betas))

        total_loss = self.hparams.betas_loss_scale * betas_loss + self.hparams.position_loss_scale * positions_loss\
                     + self.hparams.length_loss_scale * length_loss + self.hparams.betas_l1_loss_scale

        losses = {
            "total": total_loss,
            "position": positions_loss,
            "betas": betas_loss,
            "betas_l1": betas_l1_loss,
            "length": length_loss
        }

        if self.hparams.predict_scale:
            scale_loss = torch.nn.functional.mse_loss(predicted_scale, target_scale)
            total_loss += self.hparams.scale_loss_scale * scale_loss
            losses["scale"] = scale_loss
        else:
            target_scale = None
            predicted_scale = None

        self.update_validation_metrics(predicted_betas, target_betas, predicted_positions, target_positions, predicted_scale, target_scale)
        return losses

    def update_validation_metrics(self, predicted_betas, target_betas, predicted_positions, target_positions, predicted_scale, target_scale):
        predicted_lengths = self.get_bone_lengths(predicted_positions)
        target_lengths = self.get_bone_lengths(target_positions)

        self.validation_length_metric(predicted_lengths, target_lengths)
        self.validation_betas_metric(predicted_betas, target_betas)
        if self.hparams.predict_scale:
            self.validation_scale_metric(predicted_scale, target_scale)

    def configure_optimizers(self):
        return instantiate(self.hparams.optimizer, params=self.parameters())


@dataclass
class SmplInversionKernelModelOptions(SmplInversionModelOptions):
    kernel_std: float = 0.01
    num_samples: int = 100000


@ModelFactory.register(SmplInversionKernelModelOptions, schema_name="SmplInversionKernel")
class SmplInversionKernelModel(SmplInversionModel):
    def __init__(self, data_components: DataComponents, opts: SmplInversionModelOptions):
        super().__init__(data_components=data_components, opts=opts)

    def create_backbone(self):
        male_batch = self.generate_batch(batch_size=self.hparams.num_samples, force_gender=0)
        female_batch = self.generate_batch(batch_size=self.hparams.num_samples, force_gender=1)
        neutral_batch = self.generate_batch(batch_size=self.hparams.num_samples, force_gender=2)

        male_batch["features"] = self.get_input_features(male_batch["positions"])
        female_batch["features"] = self.get_input_features(female_batch["positions"])
        neutral_batch["features"] = self.get_input_features(neutral_batch["positions"])

        if self.hparams.predict_scale:
            male_batch["output"] = torch.cat([male_batch["betas"], male_batch["scale"]], dim=1)
            female_batch["output"] = torch.cat([female_batch["betas"], female_batch["scale"]], dim=1)
            neutral_batch["output"] = torch.cat([neutral_batch["betas"], neutral_batch["scale"]], dim=1)
        else:
            male_batch["output"] = male_batch["betas"]
            female_batch["output"] = female_batch["betas"]
            neutral_batch["output"] = neutral_batch["betas"]

        self.net = SmplInversionGaussian(male_batch, female_batch, neutral_batch)

    def forward(self, input_data):
        gender = input_data["gender"]
        positions = input_data["positions"]
        kernel_std = self.hparams.kernel_std * torch.ones((positions.shape[0]), device=positions.device, dtype=positions.dtype)

        x = self.get_input_features(positions)

        betas = self.net(x, gender, std=kernel_std)

        if self.hparams.predict_scale:
            scale = betas[:, self.num_betas:]
            betas = betas[:, :self.num_betas]

        out = {
            "betas": betas
        }
        if self.hparams.predict_scale:
            out["scale"] = scale
        return out

    def training_step(self, batch, batch_idx):
        super().training_step(batch, batch_idx)
        return None

    def configure_optimizers(self):
        return None

    def export(self, filepath, **kwargs):
        super().export(filepath, opset_version=12, **kwargs)
