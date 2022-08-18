# IDs of each effector type
from dataclasses import dataclass, field
from typing import Any, Union, List

import torch
import numpy as np
from hydra.utils import instantiate
import pytorch_lightning as pl
from pytorch3d.transforms import quaternion_to_axis_angle, matrix_to_rotation_6d, euler_angles_to_matrix, \
    quaternion_to_matrix, rotation_6d_to_matrix, matrix_to_quaternion
from pytorch_lightning.metrics import MeanSquaredError


from protores.losses.angular_loss import angular_loss
from protores.losses.weighted_geodesic import weighted_geodesic_loss
from protores.losses.weighted_mse import weighted_mse
from protores.metrics.rotation_matrix_error import RotationMatrixError
from protores.metrics.mpjpe import MPJPE
from protores.metrics.pa_mpjpe import PA_MPJPE
from protores.geometry.rotations import geodesic_loss_matrix3x3_matrix3x3
from protores.geometry.vector import normalize_vector
from protores.geometry.forward_kinematics import extract_translation_rotation
from protores.utils.model_factory import ModelFactory
from protores.utils.options import BaseOptions
from protores.data.smpl_module import SmplDataModuleOptions

from protores.smpl.smpl_fk import SmplFK
from protores.smpl.smpl_info import SMPL_JOINT_NAMES


TYPE_VALUES = {'position': 0, 'rotation': 1, 'lookat': 2}
POSITION_EFFECTOR_ID = TYPE_VALUES["position"]
ROTATION_EFFECTOR_ID = TYPE_VALUES["rotation"]
LOOKAT_EFFECTOR_ID = TYPE_VALUES["lookat"]


def compute_weights_from_std(std: torch.Tensor, max_weight: float, std_at_max: float = 1e-3) -> torch.Tensor:
    m = max_weight * std_at_max
    return m / std.clamp(min=std_at_max)


@dataclass
class SmplModelOptions(BaseOptions):
    dataset: SmplDataModuleOptions = SmplDataModuleOptions()
    validation_effectors: List[str] = field(
        default_factory=lambda: ['pelvis', 'neck', 'left_wrist', 'right_wrist', 'left_ankle', 'right_ankle'])
    smpl_models_path: str = "./tools/smpl/models/"
    smpl_male_name: str = "basicModel_m_lbs_10_207_0_v1.0.0"
    smpl_female_name: str = "basicModel_f_lbs_10_207_0_v1.0.0"
    smpl_neutral_name: str = "basicModel_neutral_lbs_10_207_0_v1.0.0"
    max_effector_weight: float = 1000.0
    use_fk_loss: bool = True
    use_rot_loss: bool = True
    use_pos_loss: bool = True
    use_lookat_loss: bool = True
    use_true_lookat_loss: bool = True
    use_ortho6d_hinge_loss: bool = False
    fk_loss_scale: float = 1e2
    rot_loss_scale: float = 1.0
    pos_loss_scale: float = 1e2
    lookat_loss_scale: float = 1.0
    true_lookat_loss_scale: float = 1.0
    ortho6d_hinge_loss_scale: float = 1.0
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
    generalized_lookat: bool = True  # If True, any local vector can be used for look-at, otherwise Z vector is used
    min_pos_effectors: int = 3  # Minimum number of position effectors that will be sample. Must be smaller than the minimum number of effectors
    add_effector_noise: bool = True
    weighted_losses: bool = True
    head_lookat_only: bool = False  # If True, only the head will be considered for look-at


@ModelFactory.register(SmplModelOptions, schema_name="PosingSmpl")
class SmplModel(pl.LightningModule):
    def __init__(self, data_components: Any, opts: SmplModelOptions):
        super().__init__()

        self.save_hyperparameters(opts)

        self.smpl_male = SmplFK(models_path=opts.smpl_models_path, model_name=opts.smpl_male_name)
        self.smpl_female = SmplFK(models_path=opts.smpl_models_path, model_name=opts.smpl_female_name)
        self.smpl_neutral = SmplFK(models_path=opts.smpl_models_path, model_name=opts.smpl_neutral_name)

        self.all_joint_names = SMPL_JOINT_NAMES[:24]
        self.nb_joints = len(self.all_joint_names)
        self.joint_indexes = {}
        self.index_bones = {}
        for joint_idx in range(len(self.all_joint_names)):
            self.joint_indexes[self.all_joint_names[joint_idx]] = joint_idx
            self.index_bones[joint_idx] = self.all_joint_names[joint_idx]

        self.root_idx = self.get_joint_indices(SMPL_JOINT_NAMES[0])

        self.validation_effector_indices = self.get_joint_indices(self.hparams.validation_effectors)
        self.validation_multinomial_input = torch.zeros((1, self.nb_joints))
        self.validation_multinomial_input[:, self.validation_effector_indices] = 1

        self.test_fk_metric = MeanSquaredError(compute_on_step=False)
        self.test_position_metric = MeanSquaredError(compute_on_step=False)
        self.test_rotation_metric = RotationMatrixError(compute_on_step=False)


        self.before_create_backbone()
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
        self.ortho6d_hinge_loss_scale = torch.nn.Parameter(
            torch.tensor(-0.5 * np.log(self.hparams.ortho6d_hinge_loss_scale)),
            requires_grad=self.hparams.loss_scales_learnable) if self.hparams.use_ortho6d_hinge_loss else None

        effectors = self.all_joint_names
        effector_indices = self.get_joint_indices(effectors)
        self.pos_multinomial = torch.zeros((1, self.nb_joints))
        self.rot_multinomial = torch.zeros((1, self.nb_joints))
        self.lookat_multinomial = torch.zeros((1, self.nb_joints))
        self.pos_multinomial[:, effector_indices] = 1
        self.rot_multinomial[:, effector_indices] = 1

        if self.hparams.head_lookat_only:
            self.lookat_multinomial[:, self.get_joint_indices('Head')] = 1
        else:
            self.lookat_multinomial[:, effector_indices] = 1

        self.type_multinomial = torch.tensor([self.hparams.pos_effector_probability,
                                              self.hparams.rot_effector_probability,
                                              self.hparams.lookat_effector_probability], dtype=torch.float)

        # validation metrics
        # note: we must duplicate for random/fixed setup as they are used during the same step
        self.validation_fixed_fk_metric = MeanSquaredError(compute_on_step=False)
        self.validation_fixed_position_metric = MeanSquaredError(compute_on_step=False)
        self.validation_fixed_rotation_metric = RotationMatrixError(compute_on_step=False)
        self.validation_fixed_mpjpe_metric = MPJPE(compute_on_step=False)
        self.validation_fixed_pampjpe_metric = PA_MPJPE(compute_on_step=False)
        self.validation_random_fk_metric = MeanSquaredError(compute_on_step=False)
        self.validation_random_position_metric = MeanSquaredError(compute_on_step=False)
        self.validation_random_rotation_metric = RotationMatrixError(compute_on_step=False)
        self.validation_random_mpjpe_metric = MPJPE(compute_on_step=False)
        self.validation_random_pampjpe_metric = PA_MPJPE(compute_on_step=False)
        # test metrics
        self.test_fixed_fk_metric = MeanSquaredError(compute_on_step=False)
        self.test_fixed_rotation_metric = RotationMatrixError(compute_on_step=False)
        self.test_fixed_mpjpe_metric = MPJPE(compute_on_step=False)
        self.test_fixed_pampjpe_metric = PA_MPJPE(compute_on_step=False)
        self.test_random_fk_metric = MeanSquaredError(compute_on_step=False)
        self.test_random_rotation_metric = RotationMatrixError(compute_on_step=False)
        self.test_random_mpjpe_metric = MPJPE(compute_on_step=False)
        self.test_random_pampjpe_metric = PA_MPJPE(compute_on_step=False)

    def before_create_backbone(self):
        pass

    def create_backbone(self):
        self.net = instantiate(self.hparams.backbone, size_in=7, size_out=self.nb_joints * 6,
                               size_out_stage1=self.nb_joints * 3, shape_size=10 + 1)  # betas + gender

    def apply_smpl_quat(self, betas, joint_rotations_quat, gender, root_position=None, predict_verts: bool = False):
        joint_rotations_axis_angle = quaternion_to_axis_angle(joint_rotations_quat)
        global_orient = joint_rotations_axis_angle[:, [0], :]
        body_pose = joint_rotations_axis_angle[:, 1:, :]
        return self.apply_smpl(betas=betas, global_orient=global_orient, body_pose=body_pose, gender=gender,
                               predict_verts=predict_verts, root_position=root_position)

    def apply_smpl(self, betas, global_orient, body_pose, gender, predict_verts: bool = False, root_position=None):
        if predict_verts:
            male_positions, male_rotations, male_vertices = self._apply_smpl(betas, global_orient, body_pose,
                                                                             self.smpl_male, True)
            female_positions, female_rotations, female_vertices = self._apply_smpl(betas, global_orient, body_pose,
                                                                                   self.smpl_female, True)
            neutral_positions, neutral_rotations, neutral_vertices = self._apply_smpl(betas, global_orient,
                                                                                      body_pose, self.smpl_neutral,
                                                                                      True)
        else:
            male_positions, male_rotations = self._apply_smpl(betas, global_orient, body_pose, self.smpl_male,
                                                              False)
            female_positions, female_rotations = self._apply_smpl(betas, global_orient, body_pose, self.smpl_female,
                                                                  False)
            neutral_positions, neutral_rotations = self._apply_smpl(betas, global_orient, body_pose,
                                                                    self.smpl_neutral, False)

        is_male = gender == 0
        is_female = gender == 1
        positions = torch.where(is_male.unsqueeze(1), male_positions,
                                torch.where(is_female.unsqueeze(1), female_positions, neutral_positions))
        rotations = torch.where(is_male.unsqueeze(1).unsqueeze(1), male_rotations,
                                torch.where(is_female.unsqueeze(1).unsqueeze(1), female_rotations,
                                            neutral_rotations))

        if root_position is not None:
            #             print(root_position.shape, positions.shape)
            positions = positions + root_position.unsqueeze(1)

        if predict_verts:
            vertices = torch.where(is_male.unsqueeze(1), male_vertices,
                                   torch.where(is_female.unsqueeze(1), female_vertices, neutral_vertices))
            vertices = vertices + root_position.unsqueeze(1)
            return positions, rotations, vertices
        else:
            return positions, rotations

    def _apply_smpl(self, betas, global_orient, body_pose, smpl_model, predict_verts: bool = False):
        # TODO: Male VS Female
        # Note: for some reason, Transl is not used in dataset
        smpl_out = smpl_model.forward(betas=betas, body_pose=body_pose, global_orient=global_orient, transl=None,
                                      return_verts=predict_verts)
        smpl_positions, smpl_rotations = extract_translation_rotation(smpl_out.transforms)

        # keep only our joints
        smpl_positions = smpl_positions[:, :len(self.all_joint_names), :]
        smpl_rotations = smpl_rotations[:, :len(self.all_joint_names), :]

        if predict_verts:
            return smpl_positions, smpl_rotations, smpl_out.vertices

        return smpl_positions, smpl_rotations

    def get_target_data_from_batch(self, batch):
        return {
            "joint_positions": batch["joint_positions"],
            "root_joint_position": batch["joint_positions"][:, self.root_idx, :],
            "joint_rotations": batch["joint_rotations"]
        }

    def get_dummy_input(self):
        num_effectors = 1
        num_betas = 10
        input = {
            "betas": torch.zeros((1, num_betas)),
            "gender": torch.zeros((1, 1)),

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
        return input

    def get_dynamic_axes(self):
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

    def get_dummy_output(self):
        return {
            "joint_rotations": torch.randn((1, self.nb_joints, 6)),
            "root_joint_position": torch.randn((1, 3))
        }

    def get_joint_indices(self, joint_names: Union[str, List[str]]):
        if isinstance(joint_names, str):
            return self.joint_indexes[joint_names]
        else:
            return [self.joint_indexes[name] for name in joint_names]

    @torch.no_grad()
    def get_data_from_batch(self, batch, fixed_effector_setup: bool = True):
        device = batch["joint_positions"].device
        batch_size = batch["joint_positions"].shape[0]

        betas = batch["betas"]
        gender = batch["gender"]

        input_data = {
            "betas": betas,
            "gender": gender
        }

        # forward rotations
        joint_rotations = batch["joint_rotations"]
        joint_rotations_mat = quaternion_to_matrix(joint_rotations)
        _, joint_world_rotations_mat = self.apply_smpl_quat(betas=betas, joint_rotations_quat=joint_rotations,
                                                            gender=gender)

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
            num_pos_effectors = len(self.hparams.validation_effectors)
            num_rot_effectors = 0
            num_lookat_effectors = 0

        # ====================
        # POSITIONAL EFFECTORS
        # ====================
        if num_pos_effectors > 0:
            joint_position_data = batch["joint_positions"]
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
            pos_effectors_in = torch.gather(joint_position_data, dim=1, index=pos_effector_ids.unsqueeze(2).repeat(1, 1,
                                                                                                                   joint_position_data.shape[
                                                                                                                       2]))
            if self.hparams.add_effector_noise:
                pos_noise = pos_effector_tolerances.unsqueeze(2) * torch.randn(
                    (batch_size, num_pos_effectors, 3)).type_as(pos_effectors_in)
                pos_effectors_in = pos_effectors_in + pos_noise

            input_data["position_data"] = pos_effectors_in
            input_data["position_weight"] = pos_effector_weights
            input_data["position_tolerance"] = pos_effector_tolerances
            input_data["position_id"] = pos_effector_ids
        else:
            input_data["position_data"] = torch.zeros((batch_size, 0, 3)).type_as(joint_rotations)
            input_data["position_weight"] = torch.zeros((batch_size, 0)).type_as(joint_rotations)
            input_data["position_tolerance"] = torch.zeros((batch_size, 0)).type_as(joint_rotations)
            input_data["position_id"] = torch.zeros((batch_size, 0), dtype=torch.int64).to(device)

        # ====================
        # ROTATIONAL EFFECTORS
        # ====================
        if num_rot_effectors > 0:
            joint_world_rotations_ortho6d = matrix_to_rotation_6d(joint_world_rotations_mat.view(-1, 3, 3)).view(
                batch_size, -1, 6)
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
                rot_noise = euler_angles_to_matrix(rot_noise.view(-1, 3),
                                                   convention="XYZ")  # TODO: pick std that makes more sense for angles
                rot_effectors_in_mat = rotation_6d_to_matrix(rot_effectors_in.view(-1, 6))
                rot_effectors_in_mat = torch.matmul(rot_noise, rot_effectors_in_mat)
                rot_effectors_in = matrix_to_rotation_6d(rot_effectors_in_mat).view(batch_size, num_rot_effectors, 6)

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
                                      torch.zeros((pos_effectors_in.shape[0], pos_effectors_in.shape[1],
                                                   6 - pos_effectors_in.shape[2])).type_as(pos_effectors_in),
                                      input_data["position_tolerance"].unsqueeze(2)], dim=2)  # padding with zeros
        effector_data.append(pos_effectors_in)
        effector_ids.append(input_data["position_id"])
        effector_types.append(POSITION_EFFECTOR_ID * torch.ones_like(input_data["position_id"]))
        effector_weight.append(input_data["position_weight"])

        # ROTATIONS
        effector_data.append(
            torch.cat([input_data["rotation_data"], input_data["rotation_tolerance"].unsqueeze(2)], dim=2))
        effector_ids.append(input_data["rotation_id"])
        effector_types.append(ROTATION_EFFECTOR_ID * torch.ones_like(input_data["rotation_id"]))
        effector_weight.append(input_data["rotation_weight"])

        # LOOK-AT
        effector_data.append(torch.cat([input_data["lookat_data"], input_data["lookat_tolerance"].unsqueeze(2)], dim=2))
        effector_ids.append(input_data["lookat_id"])
        effector_types.append(LOOKAT_EFFECTOR_ID * torch.ones_like(input_data["lookat_id"]))
        effector_weight.append(input_data["lookat_weight"])

        return {
            "effectors": torch.cat(effector_data, dim=1),
            "effector_type": torch.cat(effector_types, dim=1),
            "effector_id": torch.cat(effector_ids, dim=1),
            "effector_weight": torch.cat(effector_weight, dim=1)
        }

    def make_translation_invariant(self, input_data):
        # re-reference with WEIGHTED centroid of positional effectors
        # IMPORTANT NOTE 1: we create a new data structure and tensors to avoid side-effects of modifying input data
        # IMPORTANT NOTE 2: centroid is weighted so that effectors with null blending weights don't impact computations in any way
        referenced_input_data = input_data.copy()
        pos_weights = referenced_input_data["position_weight"].unsqueeze(2)
        pos_weights_sum = pos_weights.sum(dim=1, keepdim=True)
        reference_pos = (referenced_input_data["position_data"][:, :, 0:3] * pos_weights).sum(dim=1,
                                                                                              keepdim=True) / pos_weights_sum
        referenced_input_data["position_data"] = referenced_input_data["position_data"] - reference_pos
        referenced_input_data["lookat_data"] = torch.cat(
            [referenced_input_data["lookat_data"][:, :, 0:3] - reference_pos,
             referenced_input_data["lookat_data"][:, :, 3:6]], dim=2)
        return referenced_input_data, reference_pos

    def forward(self, input_data, target_data=None):
        referenced_input_data, reference_pos = self.make_translation_invariant(input_data)

        # contenate all
        packed_data = self.pack_data(referenced_input_data)

        effectors_in = packed_data["effectors"]
        effector_ids = packed_data["effector_id"]
        effector_types = packed_data["effector_type"]
        effector_weights = packed_data["effector_weight"]

        # Concatenate betas and gender as a single "shape" vector
        betas = referenced_input_data["betas"]
        gender = referenced_input_data["gender"]
        shapes = torch.cat([betas, gender], dim=1)

        out_positions, out_rotations = self.net(effectors_in, effector_weights, shapes, effector_ids, effector_types)

        joint_positions = out_positions.view(-1, self.nb_joints, 3) + reference_pos
        joint_rotations = out_rotations.view(-1, self.nb_joints, 6)

        return {
            "joint_positions": joint_positions,
            "joint_rotations": joint_rotations,
            "root_joint_position": joint_positions[:, self.root_idx, :]
        }

    def training_step(self, batch, batch_idx):
        losses = self.shared_step(batch, step="training")
        self.log_train_losses(losses)
        return losses["total"]

    def validation_step(self, batch, batch_idx, dataloader_idx):
        if dataloader_idx == 0:
            fixed_losses = self.shared_step(batch, step="validation_fixed")
            random_losses = self.shared_step(batch, step="validation_random")
        else:
            fixed_losses = self.shared_step(batch, step="test_fixed")
            random_losses = self.shared_step(batch, step="test_random")

    def test_step(self, batch, batch_idx):
        fixed_losses = self.shared_step(batch, step="test_fixed")
        random_losses = self.shared_step(batch, step="test_random")

    def validation_step_end(self, *args, **kwargs):
        pass

    def validation_epoch_end(self, val_step_outputs):
        self.log("validation/metrics/fixed/fk", self.validation_fixed_fk_metric, on_step=False, on_epoch=True)
        self.log("validation/metrics/fixed/position", self.validation_fixed_position_metric, on_step=False,
                 on_epoch=True)
        self.log("validation/metrics/fixed/rotation", self.validation_fixed_rotation_metric, on_step=False,
                 on_epoch=True)
        self.log("validation/metrics/fixed/mpjpe", self.validation_fixed_mpjpe_metric, on_step=False, on_epoch=True)
        self.log("validation/metrics/fixed/pa_mpjpe", self.validation_fixed_pampjpe_metric, on_step=False,
                 on_epoch=True)

        self.log("validation/metrics/random/fk", self.validation_random_fk_metric, on_step=False, on_epoch=True)
        self.log("validation/metrics/random/position", self.validation_random_position_metric, on_step=False,
                 on_epoch=True)
        self.log("validation/metrics/random/rotation", self.validation_random_rotation_metric, on_step=False,
                 on_epoch=True)
        self.log("validation/metrics/random/mpjpe", self.validation_random_mpjpe_metric, on_step=False, on_epoch=True)
        self.log("validation/metrics/random/pa_mpjpe", self.validation_random_pampjpe_metric, on_step=False,
                 on_epoch=True)
        # Log test metrics
        self.log("test/metrics/fixed/fk", self.test_fixed_fk_metric, on_step=False, on_epoch=True)
        self.log("test/metrics/fixed/rotation", self.test_fixed_rotation_metric, on_step=False, on_epoch=True)
        self.log("test/metrics/fixed/mpjpe", self.test_fixed_mpjpe_metric, on_step=False, on_epoch=True)
        self.log("test/metrics/fixed/pa_mpjpe", self.test_fixed_pampjpe_metric, on_step=False, on_epoch=True)

        self.log("test/metrics/random/fk", self.test_random_fk_metric, on_step=False, on_epoch=True)
        self.log("test/metrics/random/rotation", self.test_random_rotation_metric, on_step=False, on_epoch=True)
        self.log("test/metrics/random/mpjpe", self.test_random_mpjpe_metric, on_step=False, on_epoch=True)
        self.log("test/metrics/random/pa_mpjpe", self.test_random_pampjpe_metric, on_step=False, on_epoch=True)

    def test_epoch_end(self, *args, **kwargs):
        self.log("test/metrics/fixed/fk", self.test_fixed_fk_metric, on_step=False, on_epoch=True)
        self.log("test/metrics/fixed/rotation", self.test_fixed_rotation_metric, on_step=False, on_epoch=True)
        self.log("test/metrics/fixed/mpjpe", self.test_fixed_mpjpe_metric, on_step=False, on_epoch=True)
        self.log("test/metrics/fixed/pa_mpjpe", self.test_fixed_pampjpe_metric, on_step=False, on_epoch=True)

        self.log("test/metrics/random/fk", self.test_random_fk_metric, on_step=False, on_epoch=True)
        self.log("test/metrics/random/rotation", self.test_random_rotation_metric, on_step=False, on_epoch=True)
        self.log("test/metrics/random/mpjpe", self.test_random_mpjpe_metric, on_step=False, on_epoch=True)
        self.log("test/metrics/random/pa_mpjpe", self.test_random_pampjpe_metric, on_step=False, on_epoch=True)

    def shared_step(self, batch, step: str):
        # determine if we are going to get fixed effector data or not
        # TODO: completely remove fixed-effector setup at some point?
        fixed_effector_setup = False
        if step is "test_fixed" or step is "validation_fixed":
            fixed_effector_setup = True

        input_data, target_data = self.get_data_from_batch(batch, fixed_effector_setup=fixed_effector_setup)

        betas = input_data["betas"]
        gender = input_data["gender"]

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

        predicted = self.forward(input_data, target_data if step is "training" else None)

        predicted_root_joint_position = predicted["root_joint_position"]
        predicted_joint_positions = predicted["joint_positions"]
        predicted_joint_rotations = predicted["joint_rotations"]

        batch_size = target_joint_positions.shape[0]

        # compute rotation matrices
        predicted_joint_rotations_mat = rotation_6d_to_matrix(predicted_joint_rotations.view(-1, 6)).view(-1,
                                                                                                          self.nb_joints,
                                                                                                          3, 3)

        # apply forward kinematics
        predicted_joint_rotations_quat = matrix_to_quaternion(predicted_joint_rotations_mat)
        predicted_joint_positions_fk, predicted_joint_rotations_fk = self.apply_smpl_quat(betas=betas,
                                                                                          joint_rotations_quat=predicted_joint_rotations_quat,
                                                                                          root_position=predicted_root_joint_position,
                                                                                          gender=gender)

        # ==================
        # POSITION EFFECTORS
        # ==================
        if self.hparams.weighted_losses:
            pos_effector_w = compute_weights_from_std(std=in_position_tolerance,
                                                      max_weight=self.hparams.max_effector_weight, std_at_max=1e-3)
            joint_pos_weights = torch.ones(predicted_joint_positions_fk.shape[0:2]).type_as(
                in_position_tolerance)  # to mimick the old behaviour, we apply a 1 weight to all joints by default
            # joint_pos_weights.scatter_(dim=1, index=in_position_ids.view(batch_size, -1), src=pos_effector_w.view(batch_size, -1))
            joint_fk_weights = torch.zeros(predicted_joint_positions_fk.shape[0:2]).type_as(in_position_tolerance)
            joint_fk_weights.scatter_(dim=1, index=in_position_ids.view(batch_size, -1),
                                      src=pos_effector_w.view(batch_size, -1))
        else:
            joint_pos_weights = torch.ones(predicted_joint_positions_fk.shape[0:2]).type_as(in_position_tolerance)
            joint_fk_weights = torch.ones(predicted_joint_positions_fk.shape[0:2]).type_as(in_position_tolerance)

        # ==================
        # ROTATION EFFECTORS
        # ==================
        if self.hparams.weighted_losses:
            rot_effector_w = compute_weights_from_std(std=in_rotation_tolerance,
                                                      max_weight=self.hparams.max_effector_weight, std_at_max=1e-3)
            # joint_local_rot_weights = torch.ones(predicted_joint_rotations_fk.shape[0:2]).type_as(in_rotation_tolerance)
            # joint_local_rot_weights.scatter_(dim=1, index=in_rotation_ids.view(batch_size, -1), src=rot_effector_w.view(batch_size, -1))
            joint_global_rot_weights = torch.zeros(predicted_joint_rotations_fk.shape[0:2]).type_as(
                in_rotation_tolerance)
            joint_global_rot_weights.scatter_(dim=1, index=in_rotation_ids.view(batch_size, -1),
                                              src=rot_effector_w.view(batch_size, -1))
        else:
            # joint_local_rot_weights = torch.ones(predicted_joint_rotations_fk.shape[0:2]).type_as(in_rotation_tolerance)
            joint_global_rot_weights = torch.ones(predicted_joint_rotations_fk.shape[0:2]).type_as(
                in_rotation_tolerance)

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
                               target=target_joint_positions.view(-1, 3), weights=joint_fk_weights.view(-1))
        if predicted_joint_positions is None:
            pos_loss = None
        else:
            pos_loss = weighted_mse(input=predicted_joint_positions.view(-1, 3),
                                    target=target_joint_positions.view(-1, 3), weights=joint_pos_weights.view(-1))
        local_rot_loss = geodesic_loss_matrix3x3_matrix3x3(predicted_joint_rotations_mat.view(-1, 3, 3),
                                                           target_joint_rotations_mat.view(-1, 3,
                                                                                           3))  # Note: local loss is not weighted ???
        global_rot_loss = weighted_geodesic_loss(input=predicted_joint_rotations_fk.view(-1, 3, 3),
                                                 target=target_joint_rotations_fk.view(-1, 3, 3),
                                                 weights=joint_global_rot_weights.view(-1))

        if self.hparams.use_ortho6d_hinge_loss:
            predicted_joint_rotations_abs = torch.abs(predicted_joint_rotations)
            ortho6d_hinge_loss = torch.mean(
                torch.where(predicted_joint_rotations_abs > 2.0, predicted_joint_rotations_abs,
                            torch.zeros_like(predicted_joint_rotations_abs)))
        else:
            ortho6d_hinge_loss = None

        # This is loss weighting as per https://arxiv.org/pdf/1705.07115.pdf
        total_loss = 0.0
        if self.hparams.use_rot_loss:
            rot_loss_scale_exp = torch.exp(-2 * self.rot_loss_scale)
            total_loss = total_loss + rot_loss_scale_exp * local_rot_loss + self.rot_loss_scale
        if self.hparams.use_fk_loss:
            fk_loss_scale_exp = torch.exp(-2 * self.fk_loss_scale)
            total_loss = total_loss + fk_loss_scale_exp * fk_loss + self.fk_loss_scale
        if self.hparams.use_pos_loss and pos_loss is not None:
            pos_loss_scale_exp = torch.exp(-2 * self.pos_loss_scale)
            total_loss = total_loss + pos_loss_scale_exp * pos_loss + self.pos_loss_scale
        if self.hparams.use_lookat_loss:
            lookat_loss_scale_exp = torch.exp(-2 * self.lookat_loss_scale)
            total_loss = total_loss + lookat_loss_scale_exp * global_rot_loss + self.lookat_loss_scale
        if self.hparams.use_true_lookat_loss:
            true_lookat_loss_scale_exp = torch.exp(-2 * self.true_lookat_loss_scale)
            total_loss = total_loss + true_lookat_loss_scale_exp * true_lookat_loss + self.true_lookat_loss_scale
        if self.hparams.use_ortho6d_hinge_loss:
            ortho6d_hinge_loss_scale_exp = torch.exp(-2 * self.ortho6d_hinge_loss_scale)
            total_loss = total_loss + ortho6d_hinge_loss_scale_exp * ortho6d_hinge_loss + self.ortho6d_hinge_loss_scale

        if step is "training":
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
            if self.hparams.use_ortho6d_hinge_loss:
                self.log("train/ortho6d_hinge_scale", ortho6d_hinge_loss_scale_exp, on_step=False, on_epoch=True)
            self.log("train/rotation_max", predicted_joint_rotations.max(), on_step=False, on_epoch=True)

        self.update_metrics(predicted_joint_positions_fk, target_joint_positions,
                            predicted_root_joint_position, target_root_joint_positions,
                            predicted_joint_rotations_mat, target_joint_rotations_mat, step)

        return {
            "total": total_loss,
            "fk": fk_loss,
            "position": pos_loss,
            "rotation": local_rot_loss,
            "lookat": global_rot_loss,
            "true_lookat": true_lookat_loss,
            "ortho6d_hinge": ortho6d_hinge_loss
        }

    def update_metrics(self, predicted_joint_positions_fk, target_joint_positions,
                       predicted_root_joint_position, target_root_joint_positions,
                       predicted_joint_rotations_mat, target_joint_rotations_mat, step: str):
        if step is "validation_random":
            self.validation_random_fk_metric(predicted_joint_positions_fk.view(-1, 3),
                                             target_joint_positions.view(-1, 3))
            self.validation_random_position_metric(predicted_root_joint_position.view(-1, 3),
                                                   target_root_joint_positions.view(-1, 3))
            self.validation_random_rotation_metric(predicted_joint_rotations_mat, target_joint_rotations_mat)
            self.validation_random_mpjpe_metric(predicted_joint_positions_fk, target_joint_positions)
            self.validation_random_pampjpe_metric(predicted_joint_positions_fk, target_joint_positions)

        if step is "validation_fixed":
            self.validation_fixed_fk_metric(predicted_joint_positions_fk.view(-1, 3),
                                            target_joint_positions.view(-1, 3))
            self.validation_fixed_position_metric(predicted_root_joint_position.view(-1, 3),
                                                  target_root_joint_positions.view(-1, 3))
            self.validation_fixed_rotation_metric(predicted_joint_rotations_mat, target_joint_rotations_mat)
            self.validation_fixed_mpjpe_metric(predicted_joint_positions_fk, target_joint_positions)
            self.validation_fixed_pampjpe_metric(predicted_joint_positions_fk, target_joint_positions)

        if step is "test_random":
            self.test_random_fk_metric(predicted_joint_positions_fk.view(-1, 3),
                                       target_joint_positions.view(-1, 3))
            self.test_random_rotation_metric(predicted_joint_rotations_mat, target_joint_rotations_mat)
            self.test_random_mpjpe_metric(predicted_joint_positions_fk, target_joint_positions)
            self.test_random_pampjpe_metric(predicted_joint_positions_fk, target_joint_positions)

        if step is "test_fixed":
            self.test_fixed_fk_metric(predicted_joint_positions_fk.view(-1, 3),
                                      target_joint_positions.view(-1, 3))
            self.test_fixed_rotation_metric(predicted_joint_rotations_mat, target_joint_rotations_mat)
            self.test_fixed_mpjpe_metric(predicted_joint_positions_fk, target_joint_positions)
            self.test_fixed_pampjpe_metric(predicted_joint_positions_fk, target_joint_positions)

    def update_test_metrics(self, predicted, target, input):
        betas = input["betas"]
        gender = input["gender"]

        target_joint_positions = target["joint_positions"]
        target_root_joint_position = target["root_joint_position"]
        target_joint_rotations = target["joint_rotations"]

        predicted_root_joint_position = predicted["root_joint_position"]
        predicted_joint_rotations = predicted["joint_rotations"]

        # compute rotation matrices
        target_joint_rotations_mat = quaternion_to_matrix(target_joint_rotations.view(-1, 4)).view(-1, self.nb_joints, 3, 3)
        predicted_joint_rotations_mat = rotation_6d_to_matrix(predicted_joint_rotations.view(-1, 6)).view(-1, self.nb_joints, 3, 3)

        # forward kinematics
        predicted_joint_rotations_quat = matrix_to_quaternion(predicted_joint_rotations_mat)
        predicted_joint_positions, _ = self.apply_smpl_quat(betas=betas,
                                                            joint_rotations_quat=predicted_joint_rotations_quat,
                                                            gender=gender)

        self.test_fk_metric(predicted_joint_positions, target_joint_positions)
        self.test_position_metric(predicted_root_joint_position, target_root_joint_position)
        self.test_rotation_metric(predicted_joint_rotations_mat, target_joint_rotations_mat)

    def configure_optimizers(self):
        return instantiate(self.hparams.optimizer, params=self.parameters())

    def get_metadata(self):
        metadata = super().get_metadata()

        # IMPORTANT: this metadata structure MUST match the one on the C# side

        # get support effector names for each effector type
        pos_effectors = []
        rot_effectors = []
        lookat_effectors = []
        if self.hparams.pos_effector_probability > 0:
            pos_effectors = [self.index_bones[x] for x in np.nonzero(self.pos_multinomial.view(-1).cpu().numpy())[0]]
        if self.hparams.rot_effector_probability > 0:
            rot_effectors = [self.index_bones[x] for x in np.nonzero(self.rot_multinomial.view(-1).cpu().numpy())[0]]
        if self.hparams.lookat_effector_probability > 0:
            lookat_effectors = [self.index_bones[x] for x in
                                np.nonzero(self.lookat_multinomial.view(-1).cpu().numpy())[0]]

        # list all parameters useful to run the inference
        model_params = {
            "max_effector_noise_scale": self.hparams.max_effector_noise_scale,
            "generalized_lookat": self.hparams.generalized_lookat,
            "pos_effectors": pos_effectors,
            "rot_effectors": rot_effectors,
            "lookat_effectors": lookat_effectors,
            "transpose_ortho6d": True  # pytorch3d uses rows instead of columns
        }

        metadata["model_params"] = model_params

        # metadata["skeleton"] = self.skeleton.full_hierarchy

        return metadata

    def export(self, filepath, **kwargs):
        super().export(filepath, opset_version=10, **kwargs)

