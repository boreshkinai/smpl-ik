import json
import numpy as np
import torch

from protores.geometry.rotations import get_4x4_rotation_matrix_from_3x3_rotation_matrix


class Skeleton:
    # IMPORTANT: joint indexes should always be defined so that if B is a child of A then idx(A) < idx(B)
    # Also, indexes must be starting at 0 and be contiguous
    def __init__(self, file_or_skeleton):
        if not isinstance(file_or_skeleton, dict):
            with open(file_or_skeleton) as file:
                skeleton_data = json.load(file)
        else:
            skeleton_data = file_or_skeleton

        self.all_joints = []
        self.bone_indexes = {}  # dictionary bone's name => bone's index
        self.bone_parent = {}  # dictionary bone's name => parent's name
        self.bone_children = {}  # dictionary bone's name => children's name list
        self.index_bones = {}  # dictionary bone's index => bone's name
        self.bone_offsets = {}  # dictionary bone's name => local offset (3D coords)
        self.bone_pairing = {}  # dictionary bone's name => bone's pair, if no pair, no key in dictionary
        self.bone_levels = {0: 0}  # bone index => level
        self.level_bones = {}
        self.level_bones_parents = {}
        self.max_level = 0

        skeleton_data["joints"].insert(0, {
            "distal": "Hips",
            "index": 0,
            "proximal": "",
            "offset": {"x": 0.0, "y": 0.0, "z": 0.0},
            "localOffset": {"x": 0.0, "y": 0.0, "z": 0.0}
        })

        # read data and construct arrays
        for joint in skeleton_data["joints"]:
            bone_name = joint["distal"]
            bone_idx = joint["index"]
            parent_bone = joint.get("proximal", "")
            localOffset = joint.get("localOffset", {})
            paired_bone = joint.get("pairedBone", "")

            self.bone_offsets[bone_name] = np.array([localOffset.get("x", 0.0), localOffset.get("y", 0.0), localOffset.get("z", 0.0)])

            if paired_bone != "":
                self.bone_pairing[bone_idx] = paired_bone

            self.all_joints.append(bone_name)

            # Apply rotation to offsets same as in game
            # self.bone_offsets[bone_name] = np.matmul(self.rotation, self.bone_offsets[bone_name])
            self.bone_indexes[bone_name] = bone_idx
            self.index_bones[bone_idx] = bone_name
            if parent_bone != '':
                self.bone_parent[bone_name] = parent_bone
                if self.bone_children.get(parent_bone, None) is None:
                    self.bone_children[parent_bone] = []
                self.bone_children[parent_bone].append(bone_name)
            else:
                self.bone_parent[bone_name] = None

            level = 0
            if bone_idx not in self.bone_levels:
                self.bone_levels[bone_idx] = self.bone_levels[self.bone_indexes[parent_bone]] + 1  # One bone deeper than parent
                level = self.bone_levels[bone_idx]
                self.max_level = max(level, self.max_level)

            if level not in self.level_bones:
                self.level_bones[level] = []
                self.level_bones_parents[level] = []

            self.level_bones[level].append(bone_idx)
            if parent_bone == '':
                self.level_bones_parents[level].append(bone_idx)
            else:
                self.level_bones_parents[level].append(self.bone_indexes[parent_bone])

        self.nb_joints = len(self.bone_indexes)

        # compute adjacency matrix
        self._compute_adjacency_matrix()

        # Keeps an index map that when applied will swap left and right
        self.bone_pair_indices = np.array([i if i not in self.bone_pairing else self.bone_indexes[self.bone_pairing[i]]
                                  for i in range(0, self.nb_joints)])

        # compute local offsets array
        self._compute_local_offsets()

    def _compute_adjacency_matrix(self):
        self.adjacency_matrix = np.zeros((self.nb_joints, self.nb_joints))
        for bone_idx in range(self.nb_joints):
            bone_name = self.index_bones[bone_idx]
            parent_name = self.bone_parent[bone_name]
            if parent_name is not None:
                parent_idx = self.bone_indexes[parent_name]
                self.adjacency_matrix[parent_idx, bone_idx] = 1.0
                self.adjacency_matrix[bone_idx, parent_idx] = 1.0

    def _compute_local_offsets(self):
        self.joint_offsets = np.zeros((len(self.index_bones), 3))
        for i in range(len(self.index_bones)):
            bone = self.index_bones[i]
            self.joint_offsets[i, :] = np.asarray(self.bone_offsets[bone])

        # convert to pytorch
        self.joint_offsets = torch.autograd.Variable(torch.FloatTensor(self.joint_offsets))
        self.cached_offsets = None

    # convert offsets into transform matrices
    # offsets: offset for each joint in the batch (B, J, 3)
    # out: transformation matrix of each joint in the batch (B, J, 4, 4)
    @staticmethod
    def get_offset_matrices(offsets):
        batch_size = offsets.shape[0]
        joints_nbr = offsets.shape[1]

        zeros = torch.autograd.Variable(torch.zeros(batch_size, joints_nbr, 1).type_as(offsets))
        ones = torch.autograd.Variable(torch.ones(batch_size, joints_nbr, 1).type_as(offsets))

        dim0 = torch.cat((ones, zeros, zeros, offsets[:, :, 0].view(batch_size, joints_nbr, 1)), 2)
        dim1 = torch.cat((zeros, ones, zeros, offsets[:, :, 1].view(batch_size, joints_nbr, 1)), 2)
        dim2 = torch.cat((zeros, zeros, ones, offsets[:, :, 2].view(batch_size, joints_nbr, 1)), 2)
        dim3 = torch.cat((zeros, zeros, zeros, ones), 2)

        matrices = torch.cat((dim0, dim1, dim2, dim3), 2).view(batch_size, joints_nbr, 4, 4)

        return matrices

    # performs differentiable forward kinematics
    # joints_local_rotations: local rotation matrices of each joint in the batch (B, J, 3, 3)
    # out: world positions of each joint in the batch (B, J, 3)
    # note: for now only works with only one parent bone at index 0 (hips)
    def forward(self, joints_local_rotations, true_hip_offset=None):
        batch_size = joints_local_rotations.shape[0]
        joints_nbr = self.joint_offsets.shape[0]

        # assert joints_nbr == len(self.joint_offsets), "To use forward kinematics, " \
        #                                               "please ensure that all joints are being predicted. " \
        #                                               "{} != {}".format(joints_nbr, len(self.joint_offsets))

        # convert 3x3 rotation matrices to 4x4 rotation matrices
        # so that we can store local offset to perform both translation and rotation at once
        rotation_matrices = get_4x4_rotation_matrix_from_3x3_rotation_matrix(joints_local_rotations.view(-1, 3, 3))

        # get the joints local offsets for the whole batch
        if self.cached_offsets is None or self.cached_offsets.shape[0] != batch_size:
            self.cached_offsets = self.joint_offsets.type_as(joints_local_rotations).view(1, joints_nbr, 3).repeat(batch_size, 1, 1)  # (batch, joints, 3)
            self.translation_matrices = self.get_offset_matrices(self.cached_offsets).view(-1, 4, 4).type_as(joints_local_rotations)

        # compute local transformation matrix for each joints
        local_transforms = torch.matmul(self.translation_matrices, rotation_matrices).view(batch_size, joints_nbr, 4, 4)

        # build world transformation matrices
        # start with hips (must be index 0)
        world_transforms = torch.zeros_like(local_transforms)  # Initialize a matrix of zeros we will fill in
        world_transforms[:, 0] = local_transforms[:, 0, :, :]

        # then process all children joints
        for level in range(1, self.max_level + 1):
            parent_bone_indices = self.level_bones_parents[level]
            local_bone_indices = self.level_bones[level]
            parent_level_transforms = world_transforms[:, parent_bone_indices]
            local_level_transforms = local_transforms[:, local_bone_indices]
            global_matrix = torch.matmul(parent_level_transforms, local_level_transforms)
            world_transforms[:, local_bone_indices] = global_matrix

        # extract global position from transform
        joint_positions = world_transforms[:, :, 0:3, 3].contiguous()  # (batch, joints, 3)
        joint_rotations = world_transforms[:, :, 0:3, 0:3].contiguous()

        # apply hips translation
        if true_hip_offset is not None:
            joint_positions += true_hip_offset.unsqueeze(1).expand(batch_size, joints_nbr, 3)

        return joint_positions, joint_rotations


