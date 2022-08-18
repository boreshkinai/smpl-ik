from typing import Any

import torch
from dataclasses import dataclass, field

from pytorch3d.transforms.rotation_conversions import matrix_to_axis_angle, matrix_to_quaternion, \
    quaternion_to_axis_angle
from pytorch_lightning.utilities.types import EPOCH_OUTPUT
from torchmetrics.regression.mean_squared_error import MeanSquaredError
from protores.models.base_task import *
from protores.metrics.rotation_matrix_error import RotationMatrixError
from protores.geometry.forward_kinematics import extract_translation_rotation
from protores.smpl.smpl_fk import SmplFK
from protores.smpl.smpl_info import SMPL_JOINT_NAMES
from protores.data.smpl_module import SmplDataModuleOptions


@dataclass
class SmplPosingTaskOptions(AbstractTaskOptions):
    dataset: SmplDataModuleOptions = SmplDataModuleOptions()
    validation_effectors: List[str] = field(default_factory=lambda: ['pelvis', 'neck', 'left_wrist', 'right_wrist', 'left_ankle', 'right_ankle'])
    smpl_models_path: str = "./tools/smpl/models/"
    smpl_male_name: str = "basicModel_m_lbs_10_207_0_v1.0.0"
    smpl_female_name: str = "basicModel_f_lbs_10_207_0_v1.0.0"
    smpl_neutral_name: str = "basicModel_neutral_lbs_10_207_0_v1.0.0"


class SmplPosingTask(AbstractTask):
    @staticmethod
    def get_metrics():
        metrics = AbstractTask.get_metrics()
        metrics["hp_metrics/fk"] = -1
        metrics["hp_metrics/position"] = -1
        metrics["hp_metrics/rotation"] = -1
        return metrics

    def __init__(self, data_components: Any, opts: SmplPosingTaskOptions):
        super().__init__(opts=opts)

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

    def test_step(self, batch, batch_idx):
        input_data, target_data = self.get_data_from_batch(batch)
        predicted = self(input_data)
        self.update_test_metrics(predicted, target_data, input_data)

    def test_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
        super().test_epoch_end(outputs=outputs)
        self.logger.log_metrics({
            "hp_metrics/fk": self.test_fk_metric.compute(),
            "hp_metrics/position": self.test_position_metric.compute(),
            "hp_metrics/rotation": self.test_rotation_metric.compute()
        })
        self.test_fk_metric.reset()
        self.test_position_metric.reset()
        self.test_rotation_metric.reset()

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

    def apply_smpl_quat(self, betas, joint_rotations_quat, gender, root_position=None, predict_verts: bool = False):
        joint_rotations_axis_angle = quaternion_to_axis_angle(joint_rotations_quat)
        global_orient= joint_rotations_axis_angle[:, [0], :]
        body_pose = joint_rotations_axis_angle[:, 1:, :]
        return self.apply_smpl(betas=betas, global_orient=global_orient, body_pose=body_pose, gender=gender,
                               predict_verts=predict_verts, root_position=root_position)

    def apply_smpl(self, betas, global_orient, body_pose, gender, predict_verts: bool = False, root_position=None):
        if predict_verts:
            male_positions, male_rotations, male_vertices = self._apply_smpl(betas, global_orient, body_pose, self.smpl_male, True)
            female_positions, female_rotations, female_vertices = self._apply_smpl(betas, global_orient, body_pose, self.smpl_female, True)
            neutral_positions, neutral_rotations, neutral_vertices = self._apply_smpl(betas, global_orient, body_pose, self.smpl_neutral, True)
        else:
            male_positions, male_rotations = self._apply_smpl(betas, global_orient, body_pose, self.smpl_male, False)
            female_positions, female_rotations = self._apply_smpl(betas, global_orient, body_pose, self.smpl_female, False)
            neutral_positions, neutral_rotations = self._apply_smpl(betas, global_orient, body_pose, self.smpl_neutral, False)

        is_male = gender == 0
        is_female = gender == 1
        positions = torch.where(is_male.unsqueeze(1), male_positions, torch.where(is_female.unsqueeze(1), female_positions, neutral_positions))
        rotations = torch.where(is_male.unsqueeze(1).unsqueeze(1), male_rotations, torch.where(is_female.unsqueeze(1).unsqueeze(1), female_rotations, neutral_rotations))
        
        if root_position is not None:
#             print(root_position.shape, positions.shape)
            positions = positions + root_position.unsqueeze(1)
        
        if predict_verts:
            vertices = torch.where(is_male.unsqueeze(1), male_vertices, torch.where(is_female.unsqueeze(1), female_vertices, neutral_vertices))
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

    def get_data_from_batch(self, batch):
        input_data = self.get_input_data_from_batch(batch)
        target_data = self.get_target_data_from_batch(batch)
        return input_data, target_data

    def get_input_data_from_batch(self, batch):
        raise Exception("get_input_data_from_batch must be implemented")

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
