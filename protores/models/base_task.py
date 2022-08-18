from typing import List, Union

from pytorch3d.transforms import rotation_6d_to_matrix, quaternion_to_matrix
from torch import nn

from protores.data.data_components import DataComponents
from protores.losses.weighted_mse import weighted_L1
from protores.models.abstract_task import AbstractTaskOptions, AbstractTask
from protores.geometry.rotations import geodesic_loss_matrix3x3_matrix3x3
from protores.losses.weighted_geodesic import weighted_geodesic_loss

# Note: as the base task currently does not have any additional option
# we define it as an alias as a dataclass cannot be empty
BaseTaskOptions = AbstractTaskOptions


class BaseTask(AbstractTask):
    def __init__(self, data_components: DataComponents, opts: BaseTaskOptions):
        super().__init__(opts=opts)

        assert data_components.skeleton is not None, "You must provide a valid skeleton"
        self.skeleton = data_components.skeleton

    def position_loss(self, input_positions, target_positions):
        batch_size = input_positions.shape[0]
        pos_loss = nn.functional.mse_loss(input_positions.reshape(batch_size, -1), target_positions.reshape(batch_size, -1))
        return pos_loss

    def l1_quaternion_loss(self, input_quaternions, target_quaternions):
        return nn.functional.l1_loss(input_quaternions.view(-1, 4), target_quaternions.view(-1, 4))

    def weighted_L1_loss(self, inputs, targets, weights):
        return weighted_L1(inputs, targets, weights)

    def forward_kinematics_loss(self, predicted_joint_rotations_matrices, predicted_root_joint_position, true_joint_positions):
        assert predicted_joint_rotations_matrices.shape[0] == true_joint_positions.shape[0], "Batch sizes must be identical"
        assert true_joint_positions.shape[1] == self.skeleton.nb_joints and true_joint_positions.shape[2] == 3, "true_joint_positions must contain one position per joint"

        fk_joint_positions, _ = self.apply_forward_kinematics(predicted_root_joint_position, predicted_joint_rotations_matrices)
        fk_loss = self.position_loss(fk_joint_positions, true_joint_positions)
        return fk_loss, fk_joint_positions

    def apply_forward_kinematics(self, root_joint_position, joint_rotations):
        assert root_joint_position.shape[0] == joint_rotations.shape[0], \
            "Batch sizes must be identical"
        assert root_joint_position.shape[1] == 3, \
            "root_joint_position must contain the position of the root joint of the skeleton"
        assert joint_rotations.shape[1] == self.skeleton.nb_joints and joint_rotations.shape[2] == 3 and joint_rotations.shape[3] == 3, \
            "joint_rotations must contain one rotation matrix for each skeleton joint"

        return self.skeleton.forward(joint_rotations, root_joint_position)

    def quaternion_loss(self, input_quaternions, target_quaternions):
        input_matrix3x3 = quaternion_to_matrix(input_quaternions.view(-1, 4))
        target_matrix3x3 = quaternion_to_matrix(target_quaternions.view(-1, 4))
        return self.rotation_matrix_loss(input_matrix3x3, target_matrix3x3)

    def rotation_matrix_loss(self, input_matrix3x3, target_matrix3x3):
        return geodesic_loss_matrix3x3_matrix3x3(input_matrix3x3.view(-1, 3, 3), target_matrix3x3.view(-1, 3, 3))

    def weighted_rotation_matrix_loss(self, input_matrix3x3, target_matrix3x3, weights):
        return weighted_geodesic_loss(input_matrix3x3, target_matrix3x3, weights)

    def ortho6d_quat_loss(self, input_ortho6d, target_quaternions):
        input_matrix3x3 = rotation_6d_to_matrix(input_ortho6d.view(-1, 6))
        target_matrix3x3 = quaternion_to_matrix(target_quaternions.view(-1, 4))
        return self.rotation_matrix_loss(input_matrix3x3, target_matrix3x3)

    def get_joint_indices(self, joint_names: Union[str, List[str]]):
        if isinstance(joint_names, str):
            return self.skeleton.bone_indexes[joint_names]
        else:
            return [self.skeleton.bone_indexes[name] for name in joint_names]
