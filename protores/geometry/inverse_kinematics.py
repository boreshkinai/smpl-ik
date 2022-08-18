from typing import List

import torch
import numpy as np
from pytorch3d.transforms import quaternion_multiply, quaternion_apply, quaternion_invert

# performs differentiable inverse kinematics
# local_transforms: local transform matrix of each transform (B, J, 4, 4)
# level_transforms: a list of hierarchy levels, ordered by distance to the root transforms in the hierarchy
#                   each elements of the list is a list transform indexes
#                   for instance level_transforms[3] contains a list of indices of all the transforms that are 3 levels
#                   below the root transforms the indices should match these of the provided local transforms
#                   for instance if level_transforms[4, 1] is equal to 7, it means that local_transform[:, 7, :, :]
#                   contains the local matrix of a transform that is 4 levels deeper than the root transform
#                   (ie there are 4 transforms above it in the hierarchy)
# level_transform_parents: similar to level_transforms but contains the parent indices
#                   for instance if level_transform_parents[4, 1] is equal to 5, it means that local_transform[:, 5, :, :]
#                   contains the local matrix of the parent transform of the transform contained in level_transforms[4, 1]
# out: world transform matrix of each transform in the batch (B, J, 4, 4). Order is the same as input

def invert_transform_hierarchy(global_transforms: torch.Tensor, level_transforms: List[List[int]],
                               level_transform_parents: List[List[int]]):
    # used to store local transforms
    local_transforms = global_transforms.clone()

    # then process all children transforms
    for level in range(len(level_transforms)-1, 0, -1):
        parent_bone_indices = level_transform_parents[level]
        local_bone_indices = level_transforms[level]
        parent_level_transforms = global_transforms[..., parent_bone_indices, :, :]
        local_level_transforms = global_transforms[..., local_bone_indices, :, :]
        local_matrix = torch.matmul(torch.inverse(parent_level_transforms),
                                    local_level_transforms)
        local_transforms[..., local_bone_indices, :, :] = local_matrix.type_as(local_transforms)

    return local_transforms


# Compute a transform matrix combining a rotation and translation (rotation is applied first)
# translations: translation vectors (N, 3)
# rotations: rotation matrices (N, 3, 3)
# out: transform matrix (N, 4, 4)
def get_transform_matrix(translations: torch.Tensor, rotations: torch.Tensor):    
    out_shape = np.array(rotations.shape)
    out_shape[-2:] = 4    
    out = torch.zeros(list(out_shape)).type_as(rotations)
    
    out[..., 0:3, 0:3] = rotations
    out[..., 0:3, 3] = translations
    out[..., 3, 3] = 1.0
    return out


# performs differentiable inverse kinematics
# global_rotations: global rotation matrices of each joint in the batch (B, J, 3, 3)
# global_offset: local position offset of each joint in the batch (B, J, 3).
#               This corresponds to the offset with respect to the parent joint when no rotation is applied
# level_joints: a list of hierarchy levels, ordered by distance to the root joints in the hierarchy
#               each elements of the list is a list transform indexes
#               for instance level_joints[3] contains a list of indices of all the joints that are 3 levels
#               below the root joints the indices should match these of the provided local rotations
#               for instance if level_joints[4, 1] is equal to 7, it means that local_rotations[:, 7, :, :]
#               contains the local rotation matrix of a joint that is 4 levels deeper than the root joint
#               (ie there are 4 joints above it in the hierarchy)
# level_joint_parents: similar to level_transforms but contains the parent indices
#                      for instance if level_joint_parents[4, 1] is equal to 5, it means that local_rotations[:, 5, :, :]
#                      contains the local rotation matrix of the parent joint of the joint contained in level_joints[4, 1]
def inverse_kinematics(global_rotations: torch.Tensor, global_offsets: torch.Tensor,
                       level_joints: List[List[int]], level_joint_parents: List[List[int]]):

    # compute global transformation matrix for each joints
    global_transforms = get_transform_matrix(translations=global_offsets, rotations=global_rotations)
    # used to store local transforms
    local_transforms = global_transforms.clone()
    
    # then process all children transforms
    for level in range(len(level_joints)-1, 0, -1):
        parent_bone_indices = level_joint_parents[level]
        local_bone_indices = level_joints[level]
        parent_level_transforms = global_transforms[..., parent_bone_indices, :, :]
        local_level_transforms = global_transforms[..., local_bone_indices, :, :]
        local_matrix = torch.matmul(torch.inverse(parent_level_transforms),
                                    local_level_transforms)
        local_transforms[..., local_bone_indices, :, :] = local_matrix.type_as(local_transforms)

    return local_transforms


def inverse_kinematics_rotations_only(global_rotations: torch.Tensor,  level_joints: List[List[int]],
                                    level_joint_parents: List[List[int]]):

    local_rotations = global_rotations.clone()

    # then process all children transforms
    for level in range(len(level_joints) - 1, 0, -1):
        parent_bone_indices = level_joint_parents[level]
        local_bone_indices = level_joints[level]
        parent_level_transforms = global_rotations[..., parent_bone_indices, :, :]
        local_level_transforms = global_rotations[..., local_bone_indices, :, :]
        local_matrix = torch.matmul(torch.transpose(parent_level_transforms, -1, -2), local_level_transforms)
        local_rotations[..., local_bone_indices, :, :] = local_matrix.type_as(local_rotations)

    return local_rotations


def inverse_kinematics_quats(joints_global_rotations: torch.Tensor, joints_global_positions: torch.Tensor,
                                              level_joints: List[List[int]], level_joint_parents: List[List[int]]):
    """
    Performs differentiable inverse kinematics with quaternion-based rotations
    :param joints_global_rotations: tensor of global quaternions of shape (..., J, 4)
    :param joints_global_positions: tensor of global positions of shape (..., J, 3)
    :param a list of hierarchy levels, ordered by distance to the root joints in the hierarchy
                           each elements of the list is a list transform indexes
                           for instance level_joints[3] contains a list of indices of all the joints that are 3 levels
                           below the root joints the indices should match these of the provided local rotations
                           for instance if level_joints[4, 1] is equal to 7, it means that local_rotations[:, 7, :, :]
                           contains the local rotation matrix of a joint that is 4 levels deeper than the root joint
                           (ie there are 4 joints above it in the hierarchy)
    :param level_joint_parents: similar to level_transforms but contains the parent indices
                      for instance if level_joint_parents[4, 1] is equal to 5, it means that local_rotations[:, 5, :, :]
                      contains the local rotation matrix of the parent joint of the joint contained in level_joints[4, 1]
    :return:
        joints_local_positions: tensor of local bone offsets of shape (..., J, 3)
        joints_local_rotations: tensor of local quaternions of shape (..., J, 4)
    """

    joint_local_positions = joints_global_positions.clone()
    joint_local_quats = joints_global_rotations.clone()

    for level in range(len(level_joints) - 1, 0, -1):
        parent_level_bone_indices = level_joint_parents[level]
        level_bone_indices = level_joints[level]

        parent_level_global_quats = joints_global_rotations[..., parent_level_bone_indices, :]
        level_global_quats = joints_global_rotations[..., level_bone_indices, :]
        parent_level_global_positions = joints_global_positions[..., parent_level_bone_indices, :]
        level_global_positions = joints_global_positions[..., level_bone_indices, :]

        joint_local_quats[..., level_bone_indices, :] = quaternion_multiply(quaternion_invert(parent_level_global_quats), level_global_quats)
        joint_local_positions[..., level_bone_indices, :] = quaternion_apply(quaternion_invert(parent_level_global_quats), level_global_positions - parent_level_global_positions)

    return joint_local_positions, joint_local_quats


