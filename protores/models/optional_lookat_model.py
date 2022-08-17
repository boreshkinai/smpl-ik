import errno
import os
from dataclasses import dataclass
from typing import Any, Dict, Union, List

import torch
import numpy as np
from hydra.utils import instantiate
from omegaconf import DictConfig
import pytorch_lightning as pl

from protores.losses.angular_loss import angular_loss
from protores.metrics.rotation_matrix_error import RotationMatrixError
from protores.geometry.rotations import compute_rotation_matrix_from_quaternion, compute_rotation_matrix_from_ortho6d, \
    compute_rotation_matrix_from_euler, compute_ortho6d_from_rotation_matrix, geodesic_loss_matrix3x3_matrix3x3
from protores.geometry.skeleton import Skeleton
from protores.losses.weighted_geodesic import weighted_geodesic_loss
from protores.losses.weighted_mse import weighted_mse
from protores.geometry.vector import normalize_vector
from protores.data.base_module import BaseDataModuleOptions
from protores.utils.model_factory import ModelFactory
from protores.utils.onnx_export import export_named_model_to_onnx
from protores.utils.options import BaseOptions

TYPE_VALUES = {'position': 0, 'rotation': 1, 'lookat': 2}

from protores.evaluation.eval_model import RANDOM_EFFECTOR_COUNTS, BenchmarkEvaluator


def compute_weights_from_std(std: torch.Tensor, max_weight: float, std_at_max: float = 1e-3) -> torch.Tensor:
    m = max_weight * std_at_max
    return m / std.clamp(min=std_at_max)


@dataclass
class OptionalLookAtModelOptions(BaseOptions):
    dataset: BaseDataModuleOptions = BaseDataModuleOptions()
    max_effector_weight: float = 1000.0
    use_fk_loss: bool = True
    use_rot_loss: bool = True
    use_pos_loss: bool = True
    use_lookat_loss: bool = True
    use_true_lookat_loss: bool = True
    fk_loss_scale: float = 1e2
    rot_loss_scale: float = 1.0
    pos_loss_scale: float = 1e2
    lookat_loss_scale: float = 1.0
    true_lookat_loss_scale: float = 1.0
    loss_scales_learnable: bool = False
    lookat_distance_std: float = 5.0  # The std used to sample the look-at target distance from the joint. Note that this is not a normal distribution as we then take the absolute value
    max_effector_noise_scale: float = 0.1  # the maximum std of the noise added to effectors
    effector_noise_exp: float = 13.0  # exponent applied to uniformed noise sampling to bias sampling toward less/more noise
    min_effectors_count: int = 3  # Minimum number of effectors to sample per batch
    max_effectors_count: int = 16  # Maximum number of effectors to sample per batch
    pos_effector_probability: float = 1.0  # Relative probability of drawing a position effector
    rot_effector_probability: float = 1.0  # Relative probability of drawing a rotation effector
    lookat_effector_probability: float = 1.0  # Relative probability of drawing a look-at effector
    backbone: Any = None
    optimizer: Any = None
    use_ux_effectors: bool = True  # If True, uses only the main 18 effectors instead of any joint
    head_lookat_only: bool = False  # If True, only the head will be considered for look-at
    generalized_lookat: bool = True  # If True, any local vector can be used for look-at, otherwise Z vector is used
    min_pos_effectors: int = 3  # Minimum number of position effectors that will be sample. Must be smaller than the minimum number of effectors
    add_effector_noise: bool = True
    weighted_losses: bool = True

    datasets_path: str = ''
    benchmark: str = 'minimixamo'
    repeat: int = 0


@ModelFactory.register(OptionalLookAtModelOptions, schema_name="PosingOptionalLookAt")
class OptionalLookAtModel(pl.LightningModule):
    def __init__(self, skeleton: Skeleton, opts: OptionalLookAtModelOptions):
        super().__init__()

        assert isinstance(opts,
                          DictConfig), f"opt constructor argument must be of type DictConfig but got {type(opts)} instead."
        assert skeleton is not None, "You must provide a valid skeleton"

        self.save_hyperparameters(opts)
        self.skeleton = skeleton

        self.root_idx = self.get_joint_indices('Hips')

        self.test_fk_metric = pl.metrics.regression.MeanSquaredError(compute_on_step=False)
        self.test_position_metric = pl.metrics.regression.MeanSquaredError(compute_on_step=False)
        self.test_rotation_metric = RotationMatrixError(compute_on_step=False)

        self.validation_effectors = ['Hips', 'Neck', 'HandLeft', 'HandRight', 'FootLeft', 'FootRight']
        self.validation_effector_indices = self.get_joint_indices(self.validation_effectors)
        self.validation_multinomial_input = torch.zeros((1, self.skeleton.nb_joints))
        self.validation_multinomial_input[:, self.validation_effector_indices] = 1

        self.create_backbone()

        self.rot_loss_scale = torch.nn.Parameter(torch.tensor(-0.5 * np.log(self.hparams.rot_loss_scale)),
                                                 requires_grad=self.hparams.loss_scales_learnable) if self.hparams.use_rot_loss else None
        self.fk_loss_scale = torch.nn.Parameter(torch.tensor(-0.5 * np.log(self.hparams.fk_loss_scale)),
                                                requires_grad=self.hparams.loss_scales_learnable) if self.hparams.use_fk_loss else None
        self.pos_loss_scale = torch.nn.Parameter(torch.tensor(-0.5 * np.log(self.hparams.pos_loss_scale)),
                                                 requires_grad=self.hparams.loss_scales_learnable) if self.hparams.use_pos_loss else None
        self.lookat_loss_scale = torch.nn.Parameter(torch.tensor(-0.5 * np.log(self.hparams.lookat_loss_scale)),
                                                    requires_grad=self.hparams.loss_scales_learnable) if self.hparams.use_lookat_loss else None
        self.true_lookat_loss_scale = torch.nn.Parameter(
            torch.tensor(-0.5 * np.log(self.hparams.true_lookat_loss_scale)),
            requires_grad=self.hparams.loss_scales_learnable) if self.hparams.use_true_lookat_loss else None

        if self.hparams.use_ux_effectors:
            effectors = ['Hips', 'Neck', 'HandLeft', 'HandRight', 'FootLeft', 'FootRight', 'CalfLeft', 'CalfRight',
                         'ForarmLeft', 'ForarmRight', 'Head', 'ToeLeft', 'ToeRight', 'Chest', 'BicepLeft', 'BicepRight',
                         'ThighLeft', 'ThighRight']
        else:
            effectors = self.skeleton.all_joints
        effector_indices = self.get_joint_indices(effectors)
        self.pos_multinomial = torch.zeros((1, self.skeleton.nb_joints))
        self.rot_multinomial = torch.zeros((1, self.skeleton.nb_joints))
        self.lookat_multinomial = torch.zeros((1, self.skeleton.nb_joints))
        self.pos_multinomial[:, effector_indices] = 1
        self.rot_multinomial[:, effector_indices] = 1

        if self.hparams.head_lookat_only:
            self.lookat_multinomial[:, self.skeleton.bone_indexes["Head"]] = 1
        else:
            self.lookat_multinomial[:, effector_indices] = 1

        self.type_multinomial = torch.tensor([self.hparams.pos_effector_probability,
                                              self.hparams.rot_effector_probability,
                                              self.hparams.lookat_effector_probability], dtype=torch.float)

        # validation metrics
        # note: we must duplicate for random/fixed setup as they are used during the same step
        self.validation_fixed_fk_metric = pl.metrics.regression.MeanSquaredError(compute_on_step=False)
        self.validation_fixed_position_metric = pl.metrics.regression.MeanSquaredError(compute_on_step=False)
        self.validation_fixed_rotation_metric = RotationMatrixError(compute_on_step=False)
        self.validation_random_fk_metric = pl.metrics.regression.MeanSquaredError(compute_on_step=False)
        self.validation_random_position_metric = pl.metrics.regression.MeanSquaredError(compute_on_step=False)
        self.validation_random_rotation_metric = RotationMatrixError(compute_on_step=False)

        self.evaluator = None
        if opts.benchmark != 'None':
            self.evaluator = BenchmarkEvaluator(datasets_path=opts.datasets_path,
                                                random_effector_counts=RANDOM_EFFECTOR_COUNTS,
                                                benchmark=opts.benchmark, device='cpu', verbose=False)

    def create_backbone(self):

        if self.hparams.backbone._target_.split('.')[-1] == 'MaskedFcr':
            size_in = 7 * self.skeleton.nb_joints * 3
        else:
            size_in = 7

        self.net = instantiate(self.hparams.backbone, size_in=size_in, size_out=self.skeleton.nb_joints * 6,
                               size_out_stage1=self.skeleton.nb_joints * 3)

    @torch.no_grad()
    def get_data_from_batch(self, batch, fixed_effector_setup: bool = True):
        input_data = {}

        device = batch["joint_positions"].device
        batch_size = batch["joint_positions"].shape[0]

        # forward rotations
        joint_rotations = batch["joint_rotations"]
        joint_rotations_mat = compute_rotation_matrix_from_quaternion(joint_rotations.view(-1, 4)).view(batch_size, -1,
                                                                                                        3, 3)
        _, joint_world_rotations_mat = self.skeleton.forward(joint_rotations_mat)

        # ======================
        # EFFECTOR TYPE SAMPLING
        # ======================
        if not fixed_effector_setup:
            # note: we must have at least one positional effector (for translation invariance)
            num_random_effectors = \
            torch.randint(low=self.hparams.min_effectors_count, high=self.hparams.max_effectors_count + 1,
                          size=(1,)).numpy()[0]
            random_effector_types = torch.multinomial(input=self.type_multinomial, num_samples=num_random_effectors,
                                                      replacement=True)

            num_pos_effectors = (random_effector_types == 0).sum()
            num_rot_effectors = (random_effector_types == 1).sum()
            num_lookat_effectors = (random_effector_types == 2).sum()

            # Forces a minimum number of position effector without shifting the expected number of position effectors
            # There are probably better ways to do this... This can also cause the actual number of effectors to be higher than expected
            num_pos_effectors = max(self.hparams.min_pos_effectors, num_pos_effectors)

            # if look-at is not generalized, we cannot draw more look-at effector than the number of possible look-at joints
            if not self.hparams.generalized_lookat:
                num_lookat_effectors = min((self.lookat_multinomial != 0).sum(), num_lookat_effectors)
        else:
            num_pos_effectors = len(self.validation_effectors)
            num_rot_effectors = 0
            num_lookat_effectors = 0

        # ====================
        # POSITIONAL EFFECTORS
        # ====================
        if not fixed_effector_setup:
            pos_effector_ids = torch.multinomial(input=self.pos_multinomial.repeat(batch_size, 1),
                                                 num_samples=num_pos_effectors, replacement=False).to(device)
            pos_effector_tolerances = self.hparams.max_effector_noise_scale * torch.pow(
                torch.rand(size=pos_effector_ids.shape).to(device), self.hparams.effector_noise_exp)
        else:
            pos_effector_ids = torch.multinomial(input=self.validation_multinomial_input.repeat(batch_size, 1),
                                                 num_samples=num_pos_effectors, replacement=False).to(device)
            pos_effector_tolerances = torch.zeros(size=pos_effector_ids.shape).to(device)
        pos_effector_weights = torch.ones(size=pos_effector_ids.shape).to(
            device)  # blending weights are always set to 1 during training
        pos_effectors_in = torch.gather(batch["joint_positions"], dim=1,
                                        index=pos_effector_ids.unsqueeze(2).repeat(1, 1, 3))
        if self.hparams.add_effector_noise:
            pos_noise = pos_effector_tolerances.unsqueeze(2) * torch.randn((batch_size, num_pos_effectors, 3)).type_as(
                pos_effectors_in)
            pos_effectors_in = pos_effectors_in + pos_noise

        input_data["position_data"] = pos_effectors_in
        input_data["position_weight"] = pos_effector_weights
        input_data["position_tolerance"] = pos_effector_tolerances
        input_data["position_id"] = pos_effector_ids

        # ====================
        # ROTATIONAL EFFECTORS
        # ====================
        if num_rot_effectors > 0:
            joint_world_rotations_ortho6d = compute_ortho6d_from_rotation_matrix(
                joint_world_rotations_mat.view(-1, 3, 3)).view(batch_size, -1, 6)
            rot_effector_ids = torch.multinomial(input=self.rot_multinomial.repeat(batch_size, 1),
                                                 num_samples=num_rot_effectors, replacement=False).to(device)
            rot_effectors_in = torch.gather(joint_world_rotations_ortho6d, dim=1,
                                            index=rot_effector_ids.unsqueeze(2).repeat(1, 1, 6))
            rot_effector_weight = torch.ones(size=rot_effector_ids.shape).to(
                device)  # blending weights are always set to 1 during training
            rot_effector_tolerances = self.hparams.max_effector_noise_scale * torch.pow(
                torch.rand(size=rot_effector_ids.shape).to(device), self.hparams.effector_noise_exp)
            if self.hparams.add_effector_noise:
                rot_noise = rot_effector_tolerances.unsqueeze(2) * torch.randn(
                    (batch_size, num_rot_effectors, 3)).type_as(rot_effectors_in)
                rot_noise = compute_rotation_matrix_from_euler(
                    rot_noise.view(-1, 3))  # TODO: pick std that makes more sense for angles
                rot_effectors_in_mat = compute_rotation_matrix_from_ortho6d(rot_effectors_in.view(-1, 6))
                rot_effectors_in_mat = torch.matmul(rot_noise, rot_effectors_in_mat)
                rot_effectors_in = compute_ortho6d_from_rotation_matrix(rot_effectors_in_mat).view(batch_size,
                                                                                                   num_rot_effectors, 6)

            input_data["rotation_data"] = rot_effectors_in
            input_data["rotation_weight"] = rot_effector_weight
            input_data["rotation_tolerance"] = rot_effector_tolerances
            input_data["rotation_id"] = rot_effector_ids
        else:
            input_data["rotation_data"] = torch.zeros((batch_size, 0, 6)).type_as(joint_rotations)
            input_data["rotation_weight"] = torch.zeros((batch_size, 0)).type_as(joint_rotations)
            input_data["rotation_tolerance"] = torch.zeros((batch_size, 0)).type_as(joint_rotations)
            input_data["rotation_id"] = torch.zeros((batch_size, 0), dtype=torch.int64).to(device)

        # =================
        # LOOK-AT EFFECTORS
        # =================
        if num_lookat_effectors > 0:
            # Note: we set replacement to True forthe generalized look-at as we expect user to be able to provide
            #       multiple look-at constraints on the same joint, for instance for simulating an Aim constraint
            lookat_effector_ids = torch.multinomial(input=self.lookat_multinomial.repeat(batch_size, 1),
                                                    num_samples=num_lookat_effectors,
                                                    replacement=self.hparams.generalized_lookat).to(device)
            lookat_effector_weights = torch.ones(size=lookat_effector_ids.shape).to(
                device)  # blending weights are always set to 1 during training
            lookat_positions = torch.gather(batch["joint_positions"], dim=1,
                                            index=lookat_effector_ids.unsqueeze(2).repeat(1, 1, 3))
            lookat_effector_tolerances = torch.zeros(size=lookat_effector_ids.shape).to(
                device)  # TODO: self.hparams.max_effector_noise_scale * torch.pow(torch.rand(size=lookat_effector_ids.shape).to(device), self.hparams.effector_noise_exp)
            lookat_rotations_mat = torch.gather(joint_world_rotations_mat, dim=1,
                                                index=lookat_effector_ids.unsqueeze(2).unsqueeze(3).repeat(1, 1, 3, 3))
            if self.hparams.generalized_lookat:
                local_lookat_directions = torch.randn((batch_size, num_lookat_effectors, 3)).type_as(lookat_positions)
            else:
                local_lookat_directions = torch.zeros((batch_size, num_lookat_effectors, 3)).type_as(lookat_positions)
                local_lookat_directions[:, :, 2] = 1.0
            local_lookat_directions = normalize_vector(local_lookat_directions.view(-1, 3), eps=1e-5).view(batch_size,
                                                                                                           num_lookat_effectors,
                                                                                                           3)
            lookat_directions = torch.matmul(lookat_rotations_mat.view(-1, 3, 3),
                                             local_lookat_directions.view(-1, 3).unsqueeze(2)).squeeze(1).view(
                batch_size, num_lookat_effectors, 3)
            lookat_distance = 1e-3 + self.hparams.lookat_distance_std * torch.abs(
                torch.randn((batch_size, num_lookat_effectors, 1)).type_as(lookat_directions))
            lookat_positions = lookat_positions + lookat_distance * lookat_directions
            if self.hparams.add_effector_noise:
                # lookat_noise = lookat_effector_tolerances.unsqueeze(2) * torch.randn((batch_size, num_lookat_effectors, 3)).type_as(lookat_positions)
                lookat_positions = lookat_positions  # + lookat_noise
            lookat_effectors_in = torch.cat([lookat_positions, local_lookat_directions], dim=2)

            input_data["lookat_data"] = lookat_effectors_in
            input_data["lookat_weight"] = lookat_effector_weights
            input_data["lookat_tolerance"] = lookat_effector_tolerances
            input_data["lookat_id"] = lookat_effector_ids
        else:
            input_data["lookat_data"] = torch.zeros((batch_size, 0, 6)).type_as(joint_rotations)
            input_data["lookat_weight"] = torch.zeros((batch_size, 0)).type_as(joint_rotations)
            input_data["lookat_tolerance"] = torch.zeros((batch_size, 0)).type_as(joint_rotations)
            input_data["lookat_id"] = torch.zeros((batch_size, 0), dtype=torch.int64).to(device)

        target_data = {
            "joint_positions": batch["joint_positions"],
            "root_joint_position": batch["joint_positions"][:, self.root_idx, :],
            "joint_rotations": joint_rotations,
            "joint_rotations_mat": joint_rotations_mat,
            "joint_world_rotations_mat": joint_world_rotations_mat
        }

        return input_data, target_data

    def pack_data(self, input_data):
        effector_data = []  # effector + tolerance
        effector_ids = []
        effector_types = []
        effector_weight = []

        # POSITIONS
        pos_effectors_in = input_data["position_data"]
        pos_effectors_in = torch.cat([pos_effectors_in,
                                      torch.zeros((pos_effectors_in.shape[0], pos_effectors_in.shape[1], 3)).type_as(
                                          pos_effectors_in),
                                      input_data["position_tolerance"].unsqueeze(2)], dim=2)  # padding with zeros
        effector_data.append(pos_effectors_in)
        effector_ids.append(input_data["position_id"])
        effector_types.append(torch.zeros_like(input_data["position_id"]))
        effector_weight.append(input_data["position_weight"])

        # ROTATIONS
        effector_data.append(
            torch.cat([input_data["rotation_data"], input_data["rotation_tolerance"].unsqueeze(2)], dim=2))
        effector_ids.append(input_data["rotation_id"])
        effector_types.append(torch.ones_like(input_data["rotation_id"]))
        effector_weight.append(input_data["rotation_weight"])

        # LOOK-AT
        effector_data.append(torch.cat([input_data["lookat_data"], input_data["lookat_tolerance"].unsqueeze(2)], dim=2))
        effector_ids.append(input_data["lookat_id"])
        effector_types.append(2 * torch.ones_like(input_data["lookat_id"]))
        effector_weight.append(input_data["lookat_weight"])

        return {
            "effectors": torch.cat(effector_data, dim=1),
            "effector_type": torch.cat(effector_types, dim=1),
            "effector_id": torch.cat(effector_ids, dim=1),
            "effector_weight": torch.cat(effector_weight, dim=1)
        }

    def make_packed_translation_invariant(self, input_data):
        """
        This is different from self.make_translation_invariant, because it assumes that the input data are packed
        """
        # re-reference with WEIGHTED centroid of positional effectors
        # IMPORTANT NOTE 1: we create a new data structure and tensors to avoid side-effects of modifying input data
        # IMPORTANT NOTE 2: centroid is weighted so that effectors with null blending weights don't impact computations in any way
        referenced_input_data = input_data.copy()
        batch_size = referenced_input_data["effectors"].shape[0]

        effector_types = input_data["effector_type"]
        position_mask = (effector_types == TYPE_VALUES['position']).type_as(referenced_input_data["effectors"])
        lookat_mask = (effector_types == TYPE_VALUES['lookat']).type_as(referenced_input_data["effectors"])
        reference_mask = ((position_mask + lookat_mask) > 0).type_as(referenced_input_data["effectors"])

        effectors = referenced_input_data["effectors"]

        pos_weights = (position_mask * input_data["effector_weight"]).unsqueeze(-1)
        pos_weights_sum = pos_weights.sum(dim=1, keepdim=True)

        reference_pos = (effectors[..., 0:3] * pos_weights).sum(dim=1, keepdim=True) / pos_weights_sum

        referenced_input_data["effectors"] = torch.cat(
            [effectors[..., 0:3] - reference_pos * reference_mask.unsqueeze(-1),
             effectors[..., 3:]], dim=2)

        return referenced_input_data, reference_pos

    def make_translation_invariant(self, input_data):
        # re-reference with WEIGHTED centroid of positional effectors
        # IMPORTANT NOTE 1: we create a new data structure and tensors to avoid side-effects of modifying input data
        # IMPORTANT NOTE 2: centroid is weighted so that effectors with null blending weights don't impact computations in any way
        referenced_input_data = input_data.copy()
        pos_weights = referenced_input_data["position_weight"].unsqueeze(2)
        pos_weights_sum = pos_weights.sum(dim=1, keepdim=True)
        reference_pos = (referenced_input_data["position_data"] * pos_weights).sum(dim=1,
                                                                                   keepdim=True) / pos_weights_sum
        referenced_input_data["position_data"] = referenced_input_data["position_data"] - reference_pos
        referenced_input_data["lookat_data"] = torch.cat(
            [referenced_input_data["lookat_data"][:, :, 0:3] - reference_pos,
             referenced_input_data["lookat_data"][:, :, 3:6]], dim=2)
        return referenced_input_data, reference_pos

    def forward_packed(self, input_data):
        """
        This implements the forward pass of the model assuming that the data have
        already been packed using self.pack_data
        """

        referenced_input_data, reference_pos = self.make_packed_translation_invariant(input_data)

        effectors_in = referenced_input_data["effectors"]
        effector_ids = referenced_input_data["effector_id"]
        effector_types = referenced_input_data["effector_type"]
        effector_weights = referenced_input_data["effector_weight"]

        out_positions, out_rotations = self.net(effectors_in, effector_weights, effector_ids, effector_types)

        joint_positions = out_positions.view(-1, self.skeleton.nb_joints, 3) + reference_pos
        joint_rotations = out_rotations.view(-1, self.skeleton.nb_joints, 6)

        return {
            "joint_positions": joint_positions,
            "joint_rotations": joint_rotations,
            "root_joint_position": joint_positions[:, self.root_idx, :]
        }

    def forward(self, input_data):
        referenced_input_data, reference_pos = self.make_translation_invariant(input_data)

        # contenate all
        packed_data = self.pack_data(referenced_input_data)

        effectors_in = packed_data["effectors"]
        effector_ids = packed_data["effector_id"]
        effector_types = packed_data["effector_type"]
        effector_weights = packed_data["effector_weight"]

        out_positions, out_rotations = self.net(effectors_in, effector_weights, effector_ids, effector_types)

        joint_positions = out_positions.view(-1, self.skeleton.nb_joints, 3) + reference_pos
        joint_rotations = out_rotations.view(-1, self.skeleton.nb_joints, 6)

        return {
            "joint_positions": joint_positions,
            "joint_rotations": joint_rotations,
            "root_joint_position": joint_positions[:, self.root_idx, :]
        }

    def training_step(self, batch, batch_idx):
        losses = self.shared_step(batch, step="training")
        self.log_train_losses(losses)
        return losses["total"]

    def validation_step(self, batch, batch_idx):
        fixed_losses = self.shared_step(batch, step="validation_fixed")
        self.log_validation_losses(fixed_losses, prefix="fixed/")

        random_losses = self.shared_step(batch, step="validation_random")
        self.log_validation_losses(random_losses, prefix="random/")

        # log a total loss for early stopping
        # TODO: make this less hacky
        self.log_validation_losses({"total": random_losses["total"]})

        return {
            "fixed": fixed_losses,
            "random": random_losses
        }

    def validation_step_end(self, *args, **kwargs):
        super().validation_step_end(args, kwargs)

        self.log("validation/metrics/fixed/fk", self.validation_fixed_fk_metric.compute(), on_step=False, on_epoch=True)
        self.log("validation/metrics/fixed/position", self.validation_fixed_position_metric.compute(), on_step=False,
                 on_epoch=True)
        self.log("validation/metrics/fixed/rotation", self.validation_fixed_rotation_metric.compute(), on_step=False,
                 on_epoch=True)

        self.log("validation/metrics/random/fk", self.validation_random_fk_metric.compute(), on_step=False,
                 on_epoch=True)
        self.log("validation/metrics/random/position", self.validation_random_position_metric.compute(), on_step=False,
                 on_epoch=True)
        self.log("validation/metrics/random/rotation", self.validation_random_rotation_metric.compute(), on_step=False,
                 on_epoch=True)

        if self.evaluator is not None:
            metrics = self.evaluator.evaluate_random(self, split='validation')
            for k, m in metrics.items():
                self.log(f"benchmark/validation/random/{k}", m, on_step=False, on_epoch=True)
            metrics = self.evaluator.evaluate_random(self, split='test')
            for k, m in metrics.items():
                self.log(f"benchmark/test/random/{k}", m, on_step=False, on_epoch=True)

            metrics = self.evaluator.evaluate_sixpoint(self, split='validation')
            for k, m in metrics.items():
                self.log(f"benchmark/validation/sixpoint/{k}", m, on_step=False, on_epoch=True)
            metrics = self.evaluator.evaluate_sixpoint(self, split='test')
            for k, m in metrics.items():
                self.log(f"benchmark/test/sixpoint/{k}", m, on_step=False, on_epoch=True)

            metrics = self.evaluator.evaluate_fivepoint(self, split='validation')
            for k, m in metrics.items():
                self.log(f"benchmark/validation/fivepoint/{k}", m, on_step=False, on_epoch=True)
            metrics = self.evaluator.evaluate_fivepoint(self, split='test')
            for k, m in metrics.items():
                self.log(f"benchmark/test/fivepoint/{k}", m, on_step=False, on_epoch=True)

    def shared_step(self, batch, step: str):
        # determine if we are going to get fixed effector data or not
        # TODO: completely remove fixed-effector setup at some point?
        fixed_effector_setup = False
        if step == "test" or step == "validation_fixed":
            fixed_effector_setup = True

        input_data, target_data = self.get_data_from_batch(batch, fixed_effector_setup=fixed_effector_setup)

        in_position_data = input_data["position_data"]
        in_position_ids = input_data["position_id"]
        in_position_tolerance = input_data["position_tolerance"]

        in_rotation_data = input_data["rotation_data"]
        in_rotation_ids = input_data["rotation_id"]
        in_rotation_tolerance = input_data["rotation_tolerance"]

        in_lookat_data = input_data["lookat_data"]
        in_lookat_ids = input_data["lookat_id"]
        in_lookat_tolerance = input_data["lookat_tolerance"]

        target_joint_positions = target_data["joint_positions"]
        target_root_joint_positions = target_data["root_joint_position"]
        target_joint_rotations_mat = target_data["joint_rotations_mat"]
        target_joint_rotations_fk = target_data["joint_world_rotations_mat"]

        predicted = self.forward(input_data)

        predicted_root_joint_position = predicted["root_joint_position"]
        predicted_joint_positions = predicted["joint_positions"]
        predicted_joint_rotations = predicted["joint_rotations"]

        batch_size = target_joint_positions.shape[0]

        # compute rotation matrices
        predicted_joint_rotations_mat = compute_rotation_matrix_from_ortho6d(
            predicted_joint_rotations.view(-1, 6)).view(-1, self.skeleton.nb_joints, 3, 3)

        # apply forward kinematics
        predicted_joint_positions_fk, predicted_joint_rotations_fk = self.skeleton.forward(
            predicted_joint_rotations_mat, predicted_root_joint_position)

        # ==================
        # POSITION EFFECTORS
        # ==================
        if self.hparams.weighted_losses:
            pos_effector_w = compute_weights_from_std(std=in_position_tolerance,
                                                      max_weight=self.hparams.max_effector_weight, std_at_max=1e-3)
            joint_positions_weights = torch.ones(predicted_joint_positions_fk.shape[0:2]).type_as(
                in_position_tolerance)  # to mimick the old behaviour, we apply a 1 weight to all joints by default
            joint_positions_weights.scatter_(dim=1, index=in_position_ids.view(batch_size, -1),
                                             src=pos_effector_w.view(batch_size, -1))
        else:
            joint_positions_weights = torch.ones(predicted_joint_positions_fk.shape[0:2]).type_as(in_position_tolerance)

        # ==================
        # ROTATION EFFECTORS
        # ==================
        if self.hparams.weighted_losses:
            rot_effector_w = compute_weights_from_std(std=in_rotation_tolerance,
                                                      max_weight=self.hparams.max_effector_weight, std_at_max=1e-3)
            joint_rotations_weights = torch.zeros(predicted_joint_rotations_fk.shape[0:2]).type_as(
                in_rotation_tolerance)  # to mimick the old behaviour, we apply a 0 weight to all joints by default
            joint_rotations_weights.scatter_(dim=1, index=in_rotation_ids.view(batch_size, -1),
                                             src=rot_effector_w.view(batch_size, -1))
        else:
            joint_rotations_weights = torch.ones(predicted_joint_rotations_fk.shape[0:2]).type_as(in_rotation_tolerance)

        # =================
        # LOOK-AT EFFECTORS
        # =================
        # TODO: look-at margin
        if in_lookat_data.shape[1] > 0:
            lookat_target = in_lookat_data[:, :, 0:3]
            local_lookat_directions = in_lookat_data[:, :, 3:6]
            lookat_rotations_mat = torch.gather(predicted_joint_rotations_fk, dim=1,
                                                index=in_lookat_ids.unsqueeze(2).unsqueeze(3).repeat(1, 1, 3, 3))
            predicted_lookat_directions = torch.matmul(lookat_rotations_mat.view(-1, 3, 3),
                                                       local_lookat_directions.view(-1, 3).unsqueeze(2)).squeeze(
                1).view_as(local_lookat_directions)
            predicted_lookat_positions = torch.gather(predicted_joint_positions_fk, dim=1,
                                                      index=in_lookat_ids.unsqueeze(2).repeat(1, 1, 3))
            predicted_target_directions = lookat_target - predicted_lookat_positions
            predicted_target_directions = normalize_vector(predicted_target_directions.view(-1, 3), eps=1e-5).view_as(
                predicted_target_directions)
            true_lookat_loss = angular_loss(predicted_lookat_directions.view(-1, 3),
                                            predicted_target_directions.view(-1, 3),
                                            unit_input=True, unit_target=True)
        else:
            true_lookat_loss = torch.zeros((1)).type_as(in_lookat_data)

        # =================
        # LOSSES
        # =================
        fk_loss = weighted_mse(input=predicted_joint_positions_fk.view(-1, 3),
                               target=target_joint_positions.view(-1, 3), weights=joint_positions_weights.view(-1))
        if predicted_joint_positions is None:
            pos_loss = None
        else:
            pos_loss = weighted_mse(input=predicted_joint_positions.view(-1, 3),
                                    target=target_joint_positions.view(-1, 3), weights=joint_positions_weights.view(-1))
        rot_loss = self.rotation_matrix_loss(predicted_joint_rotations_mat, target_joint_rotations_mat)
        lookat_loss = weighted_geodesic_loss(input=predicted_joint_rotations_fk.view(-1, 3, 3),
                                             target=target_joint_rotations_fk.view(-1, 3, 3),
                                             weights=joint_rotations_weights.view(-1))

        # This is loss weighting as per https://arxiv.org/pdf/1705.07115.pdf
        total_loss = 0.0
        if self.hparams.use_rot_loss:
            rot_loss_scale_exp = torch.exp(-2 * self.rot_loss_scale)
            total_loss = total_loss + rot_loss_scale_exp * rot_loss + self.rot_loss_scale
        if self.hparams.use_fk_loss:
            fk_loss_scale_exp = torch.exp(-2 * self.fk_loss_scale)
            total_loss = total_loss + fk_loss_scale_exp * fk_loss + self.fk_loss_scale
        if self.hparams.use_pos_loss and pos_loss is not None:
            pos_loss_scale_exp = torch.exp(-2 * self.pos_loss_scale)
            total_loss = total_loss + pos_loss_scale_exp * pos_loss + self.pos_loss_scale
        if self.hparams.use_lookat_loss:
            lookat_loss_scale_exp = torch.exp(-2 * self.lookat_loss_scale)
            total_loss = total_loss + lookat_loss_scale_exp * lookat_loss + self.lookat_loss_scale
        if self.hparams.use_true_lookat_loss:
            true_lookat_loss_scale_exp = torch.exp(-2 * self.true_lookat_loss_scale)
            total_loss = total_loss + true_lookat_loss_scale_exp * true_lookat_loss + self.true_lookat_loss_scale

        if step == "training":
            # plot loss scale values (note: these are log values)
            if self.hparams.use_rot_loss:
                self.log("train/rot_scale", rot_loss_scale_exp, on_step=False, on_epoch=True)
            if self.hparams.use_fk_loss:
                self.log("train/fk_scale", fk_loss_scale_exp, on_step=False, on_epoch=True)
            if self.hparams.use_pos_loss and pos_loss is not None:
                self.log("train/pos_scale", pos_loss_scale_exp, on_step=False, on_epoch=True)
            if self.hparams.use_lookat_loss:
                self.log("train/lookat_scale", lookat_loss_scale_exp, on_step=False, on_epoch=True)
            if self.hparams.use_true_lookat_loss:
                self.log("train/true_lookat_scale", true_lookat_loss_scale_exp, on_step=False, on_epoch=True)
            self.log("train/rotation_max", predicted_joint_rotations.max(), on_step=False, on_epoch=True)

        if step == "validation_random":
            self.validation_random_fk_metric(predicted_joint_positions_fk.view(-1, 3),
                                             target_joint_positions.view(-1, 3))
            self.validation_random_position_metric(predicted_root_joint_position.view(-1, 3),
                                                   target_root_joint_positions.view(-1, 3))
            self.validation_random_rotation_metric(predicted_joint_rotations_mat, target_joint_rotations_mat)

        if step == "validation_fixed":
            self.validation_fixed_fk_metric(predicted_joint_positions_fk.view(-1, 3),
                                            target_joint_positions.view(-1, 3))
            self.validation_fixed_position_metric(predicted_root_joint_position.view(-1, 3),
                                                  target_root_joint_positions.view(-1, 3))
            self.validation_fixed_rotation_metric(predicted_joint_rotations_mat, target_joint_rotations_mat)

        return {
            "total": total_loss,
            "fk": fk_loss,
            "position": pos_loss,
            "rotation": rot_loss,
            "lookat": lookat_loss,
            "true_lookat": true_lookat_loss
        }

    def configure_optimizers(self):
        return instantiate(self.hparams.optimizer, params=self.parameters())

    def export(self, filepath, **kwargs):
        dynamic_axes = self.get_dynamic_axes()

        dirpath = os.path.dirname(filepath)
        if not os.path.exists(dirpath):
            try:
                os.makedirs(dirpath)
            except OSError as exc:  # Guard against race condition
                if exc.errno != errno.EEXIST:
                    raise

        dummy_input = self.get_dummy_input()
        # opset 11 needed for Round operator
        export_named_model_to_onnx(self, dummy_input, filepath, verbose=True, opset_version=11,
                                   dynamic_axes=dynamic_axes, **kwargs)

    def test_step(self, batch, batch_idx):
        pass

    def test_step_end(self, *args, **kwargs):
        pass

    def update_test_metrics(self, predicted, target):
        target_joint_positions = target["joint_positions"]
        target_root_joint_position = target["root_joint_position"]
        target_joint_rotations = target["joint_rotations"]

        predicted_root_joint_position = predicted["root_joint_position"]
        predicted_joint_rotations = predicted["joint_rotations"]

        # compute rotation matrices
        target_joint_rotations_mat = compute_rotation_matrix_from_quaternion(target_joint_rotations.view(-1, 4)).view(-1, self.skeleton.nb_joints, 3, 3)
        predicted_joint_rotations_mat = compute_rotation_matrix_from_ortho6d(predicted_joint_rotations.view(-1, 6)).view(-1, self.skeleton.nb_joints, 3, 3)

        # forward kinematics
        predicted_joint_positions, _ = self.skeleton.forward(predicted_joint_rotations_mat, predicted_root_joint_position)

        self.test_fk_metric(predicted_joint_positions, target_joint_positions)
        self.test_position_metric(predicted_root_joint_position, target_root_joint_position)
        self.test_rotation_metric(predicted_joint_rotations_mat, target_joint_rotations_mat)

    def on_train_epoch_start(self) -> None:
        try:
            dataset = self.trainer.train_dataloader.dataset
            dataset.set_epoch(self.current_epoch)
        except Exception:
            pass
        return super().on_train_epoch_start()

    def log_train_losses(self, losses: Dict[str, Any], prefix: str = ""):
        for k, v in losses.items():
            if v is not None:
                self.log("train/" + prefix + k, v, on_step=False, on_epoch=True, prog_bar=True, logger=True,
                         sync_dist=True)

    def get_joint_indices(self, joint_names: Union[str, List[str]]):
        if isinstance(joint_names, str):
            return self.skeleton.bone_indexes[joint_names]
        else:
            return [self.skeleton.bone_indexes[name] for name in joint_names]

    @staticmethod
    def rotation_matrix_loss(input_matrix3x3, target_matrix3x3):
        return geodesic_loss_matrix3x3_matrix3x3(input_matrix3x3.view(-1, 3, 3), target_matrix3x3.view(-1, 3, 3))

    def log_validation_losses(self, losses: Dict[str, Any], prefix: str = ""):
        for k, v in losses.items():
            if v is not None:
                self.log("validation/" + prefix + k, v, on_step=False, on_epoch=True, prog_bar=True, logger=True,
                         sync_dist=True)

    @staticmethod
    def get_dynamic_axes():
        return {
            'position_data': {1: 'num_pos_effectors'},
            'position_weight': {1: 'num_pos_effectors'},
            'position_tolerance': {1: 'num_pos_effectors'},
            'position_id': {1: 'num_pos_effectors'},

            'rotation_data': {1: 'num_rot_effectors'},
            'rotation_weight': {1: 'num_rot_effectors'},
            'rotation_tolerance': {1: 'num_rot_effectors'},
            'rotation_id': {1: 'num_rot_effectors'},

            'lookat_data': {1: 'num_lookat_effectors'},
            'lookat_weight': {1: 'num_lookat_effectors'},
            'lookat_tolerance': {1: 'num_lookat_effectors'},
            'lookat_id': {1: 'num_lookat_effectors'}
        }

    @staticmethod
    def get_dummy_input():
        num_effectors = 1
        return {
            "position_data": torch.zeros((1, num_effectors, 3)),
            "position_weight": torch.zeros((1, num_effectors)),
            "position_tolerance": torch.zeros((1, num_effectors)),
            "position_id": torch.zeros((1, num_effectors), dtype=torch.int64),

            "rotation_data": torch.zeros((1, num_effectors, 6)),
            "rotation_weight": torch.zeros((1, num_effectors)),
            "rotation_tolerance": torch.zeros((1, num_effectors)),
            "rotation_id": torch.zeros((1, num_effectors), dtype=torch.int64),

            "lookat_data": torch.zeros((1, num_effectors, 6)),
            "lookat_weight": torch.zeros((1, num_effectors)),
            "lookat_tolerance": torch.zeros((1, num_effectors)),
            "lookat_id": torch.zeros((1, num_effectors), dtype=torch.int64)
        }


# no additional parameter => use alias
OptionalLookAtModelSingleStageOptions = OptionalLookAtModelOptions


@ModelFactory.register(OptionalLookAtModelSingleStageOptions, schema_name="PosingOptionalLookAtSingleStage")
class OptionalLookAtModelSingleStage(OptionalLookAtModel):
    def __init__(self, skeleton: Skeleton, opts: OptionalLookAtModelSingleStageOptions):
        super().__init__(skeleton=skeleton, opts=opts)

    def create_backbone(self):
        self.net = instantiate(self.hparams.backbone, size_in=7, size_out=self.skeleton.nb_joints * 6 + 3)

    def forward_packed(self, input_data):
        """
        This implements the forward pass of the model assuming that the data have
        already been packed using self.pack_data
        """

        referenced_input_data, reference_pos = self.make_packed_translation_invariant(input_data)

        effectors_in = referenced_input_data["effectors"]
        effector_ids = referenced_input_data["effector_id"]
        effector_types = referenced_input_data["effector_type"]
        effector_weights = referenced_input_data["effector_weight"]

        out = self.net(effectors_in, effector_weights, effector_ids, effector_types)
        out_positions = out[:, 0:3].contiguous()
        out_rotations = out[:, 3:].contiguous()

        root_joint_position = out_positions.view(-1, 3) + reference_pos.squeeze(1)
        joint_rotations = out_rotations.view(-1, self.skeleton.nb_joints, 6)

        return {
            "joint_positions": None,
            "joint_rotations": joint_rotations,
            "root_joint_position": root_joint_position
        }

    def forward(self, input_data):
        referenced_input_data, reference_pos = self.make_translation_invariant(input_data)

        # contenate all
        packed_data = self.pack_data(referenced_input_data)

        effectors_in = packed_data["effectors"]
        effector_ids = packed_data["effector_id"]
        effector_types = packed_data["effector_type"]
        effector_weights = packed_data["effector_weight"]

        out = self.net(effectors_in, effector_weights, effector_ids, effector_types).view(effectors_in.shape[0], -1)
        out_positions = out[:, 0:3].contiguous()
        out_rotations = out[:, 3:].contiguous()

        root_joint_position = out_positions.view(-1, 3) + reference_pos.squeeze(1)
        joint_rotations = out_rotations.view(-1, self.skeleton.nb_joints, 6)

        return {
            "joint_positions": None,
            "joint_rotations": joint_rotations,
            "root_joint_position": root_joint_position
        }
