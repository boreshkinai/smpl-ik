import copy
import json
from typing import List
import math

from smplik.geometry.forward_kinematics import forward_kinematics, \
    extract_translation_rotation, forward_kinematics_quats
from smplik.geometry.inverse_kinematics import inverse_kinematics, inverse_kinematics_quats, \
    inverse_kinematics_rotations_only
from smplik.geometry.rotations import *


class Skeleton:
    # IMPORTANT: joint indexes should always be defined so that if B is a child of A then idx(A) < idx(B)
    # Also, indexes must be starting at 0 and be contiguous
    def __init__(self, file_or_skeleton):
        if not isinstance(file_or_skeleton, dict):
            with open(file_or_skeleton) as file:
                skeleton_data = json.load(file)
        else:
            skeleton_data = file_or_skeleton

        # because of the way we export joints, root joints are usually missing
        # TODO: fix that on the Unity side, this is ugly
        fixed_skeleton_data = self._add_missing_joints(skeleton_data)

        # convert flat segment list into a full hierarchy
        self.full_hierarchy = self._build_hierarchy(fixed_skeleton_data)

        # update caches
        self._rebuild_lookup_data()

    def _build_hierarchy(self, skeleton_data):
        root_joints = []
        mapping = {}

        for joint in skeleton_data["joints"]:
            parentName = joint.get("proximal", "")
            jointName = joint.get("distal")
            jointIndex = joint.get("index")
            jointOffset = joint.get("localOffset", {"x": 0.0, "y": 0.0, "z": 0.0})
            pairedJointName = joint.get("pairedBone", "")

            assert jointName not in mapping, "Joint was already introduced in the hierarchy"

            # create dictionary for new joint
            mapping[jointName] = {
                "index": jointIndex,
                "offset": jointOffset,
                "symmetry": pairedJointName,
                "children": {}
            }

            if parentName == "":
                root_joints.append(jointName)
            else:
                assert parentName in mapping, "Parent joint not found"
                mapping[parentName]["children"][jointName] = mapping[jointName]

        assert len(root_joints) == 1, "Skeleton must have a single root"

        root_name = root_joints[0]
        return {root_name: mapping[root_name]}

    # removes a set of joints from the skeleton
    # this will also remove all their children
    # this will also create new bone indexes to ensure they are contiguous
    def remove_joints(self, joint_names: List[str]):
        self._remove_joints_recursive(self.full_hierarchy, joint_names)
        self._cleanup_pairing()
        self._generate_bone_indexes()
        self._rebuild_lookup_data()

    def _remove_joints_recursive(self, bone_dict, joint_names):
        for joint in joint_names:
            bone_dict.pop(joint, None)
        for joint in bone_dict:
            self._remove_joints_recursive(bone_dict[joint]["children"], joint_names)

    def _get_all_joint_names(self, bone_dict, all_joint_names):
        for joint in bone_dict:
            all_joint_names.append(joint)
            self._get_all_joint_names(bone_dict[joint]["children"], all_joint_names)

    def _cleanup_pairing(self):
        all_joint_names = []
        self._get_all_joint_names(self.full_hierarchy, all_joint_names)
        self._cleanup_pairing_recursive(self.full_hierarchy, all_joint_names)

    def _cleanup_pairing_recursive(self, bone_dict, all_joint_names):
        for joint in bone_dict:
            if bone_dict[joint]["symmetry"] not in all_joint_names:
                bone_dict[joint]["symmetry"] = ""
            self._cleanup_pairing_recursive(bone_dict[joint]["children"], all_joint_names)

    def _generate_bone_indexes(self):
        bone_index = 0
        self._generate_bone_indexes_recursive(self.full_hierarchy, bone_index)

    def _generate_bone_indexes_recursive(self, bone_dict, bone_index):
        for joint in bone_dict:
            bone_dict[joint]["index"] = bone_index
            bone_index += 1
            bone_index = self._generate_bone_indexes_recursive(bone_dict[joint]["children"], bone_index)
        return bone_index

    def is_child_of(self, child_name: str, parent_name: str):
        direct_parent = self.bone_parent.get(child_name, "")
        if direct_parent != "" and direct_parent is not None:
            if direct_parent == parent_name:
                return True
            else:
                return self.is_child_of(direct_parent, parent_name)

        return False

    def _rebuild_lookup_data(self):
        self.joints = []
        self.all_joints = []
        self.inner_joints = []
        self.end_joints = []
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

        self._rebuild_lookup_data_recursive(self.full_hierarchy, "")

        self.nb_joints = len(self.bone_indexes)

        # Keeps an index map that when applied will swap left and right
        self.bone_pair_indices = np.array([i if i not in self.bone_pairing else self.bone_indexes[self.bone_pairing[i]]
                                           for i in range(0, self.nb_joints)])

        self._compute_adjacency_matrix()
        self._compute_local_offsets()
        self._set_inner_outer_joints()

    def _rebuild_lookup_data_recursive(self, bone_dict, parent_bone):
        for bone_name in bone_dict:
            joint = bone_dict[bone_name]
            bone_idx = joint["index"]
            local_offset = joint["offset"]
            paired_bone = joint["symmetry"]

            self.bone_offsets[bone_name] = np.array(
                [local_offset.get("x", 0.0), local_offset.get("y", 0.0), local_offset.get("z", 0.0)])

            if paired_bone != "":
                self.bone_pairing[bone_idx] = paired_bone

            self.all_joints.append(bone_name)

            # Apply rotation to offsets same as in game
            # self.bone_offsets[bone_name] = np.matmul(self.rotation, self.bone_offsets[bone_name])
            self.bone_indexes[bone_name] = bone_idx
            self.index_bones[bone_idx] = bone_name
            if parent_bone != "":
                self.bone_parent[bone_name] = parent_bone
                if self.bone_children.get(parent_bone, None) is None:
                    self.bone_children[parent_bone] = []
                self.bone_children[parent_bone].append(bone_name)
            else:
                self.bone_parent[bone_name] = None

            level = 0
            if bone_idx not in self.bone_levels:
                self.bone_levels[bone_idx] = self.bone_levels[
                                                 self.bone_indexes[parent_bone]] + 1  # One bone deeper than parent
                level = self.bone_levels[bone_idx]
                self.max_level = max(level, self.max_level)

            if level not in self.level_bones:
                self.level_bones[level] = []
                self.level_bones_parents[level] = []

            self.level_bones[level].append(bone_idx)
            if parent_bone == "":
                self.level_bones_parents[level].append(bone_idx)
            else:
                self.level_bones_parents[level].append(self.bone_indexes[parent_bone])

            self._rebuild_lookup_data_recursive(joint["children"], bone_name)

    def _add_missing_joints(self, skeleton_data):
        # we make a deep-copy to not modify the original data
        fixed_skeleton_data = copy.deepcopy(skeleton_data)

        # list all proximal and distal joints
        all_distal_joints = []
        all_proximal_joints = []
        for joint in fixed_skeleton_data["joints"]:
            bone_name = joint["distal"]
            all_distal_joints.append(bone_name)
            parent_bone = joint.get("proximal", None)
            if parent_bone is not None and parent_bone != "" and parent_bone not in all_proximal_joints:
                all_proximal_joints.append(parent_bone)

        # add missing proximal joints (roots)
        added_joints = 0
        for joint in all_proximal_joints:
            if joint not in all_distal_joints:
                all_distal_joints.append(joint)
                fixed_skeleton_data["joints"].insert(0, {
                    "distal": joint,
                    "index": 0,  # default index for root is 0
                    "proximal": "",
                    "offset": {"x": 0.0, "y": 0.0, "z": 0.0},
                    "localOffset": {"x": 0.0, "y": 0.0, "z": 0.0}
                })
                added_joints += 1

        # we only support one root currently
        assert added_joints <= 1, "Multiple root joints is not supported"

        return fixed_skeleton_data

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

    def _set_inner_outer_joints(self):
        self.inner_joints = [joint for joint in self.all_joints if
                             joint in self.bone_children and len(self.bone_children[joint]) > 0]
        self.end_joints = [joint for joint in self.all_joints if joint not in self.inner_joints]

    def check_indexes(self):
        for bone in self.bone_parent:
            parent = self.bone_parent[bone]
            if parent is not None:
                assert (self.bone_indexes[parent] < self.bone_indexes[bone])

    # Computes the accumulated bone chain length of each bone of the skeleton for a given pose
    # bones_positions: vector of num_joints * 3 world positions
    def compute_bone_chain_length(self, bones_positions):
        n_bones = len(self.bone_indexes)

        bones_length = {}
        for i in range(n_bones):
            bone = self.index_bones[i]
            bones_length[bone] = 0.0

        for i in range(n_bones):
            bone_idx = n_bones - i - 1
            bone = self.index_bones[bone_idx]
            parent_bone = self.bone_parent.get(bone)
            if parent_bone is not None:
                parent_idx = self.bone_indexes[parent_bone]
                dx = bones_positions[bone_idx * 3] - bones_positions[parent_idx * 3]
                dy = bones_positions[bone_idx * 3 + 1] - bones_positions[parent_idx * 3 + 1]
                dz = bones_positions[bone_idx * 3 + 2] - bones_positions[parent_idx * 3 + 2]
                length = math.sqrt(dx * dx + dy * dy + dz * dz)
                bones_length[parent_bone] += bones_length[bone] + length

        return bones_length

    # performs differentiable forward kinematics
    # joints_local_rotations: local rotation matrices of each joint in the batch (B, J, 3, 3)
    # out: world positions of each joint in the batch (B, J, 3)
    # note: for now only works with only one parent bone at index 0 (hips)
    def forward(self, joints_local_rotations, true_hip_offset=None):
        batch_size = joints_local_rotations.shape[0]
        joints_nbr = self.joint_offsets.shape[0]

        offsets = self.joint_offsets.type_as(joints_local_rotations).view(1, joints_nbr, 3).repeat(batch_size, 1,
                                                                                                   1)  # (batch, joints, 3)
        world_transforms = forward_kinematics(joints_local_rotations, offsets, level_joints=self.level_bones,
                                              level_joint_parents=self.level_bones_parents)

        # extract global position from transform
        joint_positions, joint_rotations = extract_translation_rotation(world_transforms.view(-1, 4, 4))
        joint_positions = joint_positions.view(batch_size, joints_nbr, 3).contiguous()
        joint_rotations = joint_rotations.view(batch_size, joints_nbr, 3, 3).contiguous()

        # apply hips translation
        if true_hip_offset is not None:
            joint_positions += true_hip_offset.unsqueeze(1).expand(batch_size, joints_nbr, 3)

        return joint_positions, joint_rotations

    # performs differentiable forward kinematics on local quaternions
    # joints_local_rotations: local rotation matrices of each joint in the batch (B, J, 4)
    # out: (world positions of each joint in the batch (B, J, 3),  world quaternions of each joint in the batch (B, J, 4))
    # note: for now only works with only one parent bone at index 0 (hips)
    def forward_quats(self, joints_local_rotations, true_hip_offset=None):
        batch_size = joints_local_rotations.shape[0]
        joints_nbr = self.joint_offsets.shape[0]

        offsets = self.joint_offsets.type_as(joints_local_rotations).view(1, joints_nbr, 3).repeat(batch_size, 1,
                                                                                                   1)  # (batch, joints, 3)

        global_positions, global_quats = forward_kinematics_quats(joints_local_rotations, offsets,
                                                                  self.level_bones, self.level_bones_parents)

        if true_hip_offset is not None:
            global_positions += true_hip_offset.unsqueeze(1).expand(batch_size, joints_nbr, 3)

        return global_positions, global_quats

    # performs differentiable inverse kinematics
    # joints_local_rotations: local rotation matrices of each joint in the batch (B, J, 3, 3)
    # out: world positions of each joint in the batch (B, J, 3)
    # note: for now only works with only one parent bone at index 0 (hips)
    def invert(self, joints_global_rotations: torch.Tensor, joints_global_positions: torch.Tensor,
               true_hip_offset=None):
        batch_size = joints_global_rotations.shape[0]
        joints_nbr = self.joint_offsets.shape[0]

        if true_hip_offset is not None:
            joints_global_positions = joints_global_positions - true_hip_offset[..., None, :].expand_as(
                joints_global_positions)

        local_transforms = inverse_kinematics(joints_global_rotations,
                                              joints_global_positions,
                                              level_joints=self.level_bones,
                                              level_joint_parents=self.level_bones_parents)

        # extract global position from transform
        joint_local_positions, joint_local_rotations = extract_translation_rotation(local_transforms.view(-1, 4, 4))
        joint_local_positions = joint_local_positions.view(batch_size, joints_nbr, 3).contiguous()
        joint_local_rotations = joint_local_rotations.view(batch_size, joints_nbr, 3, 3).contiguous()

        return joint_local_positions, joint_local_rotations

    # performs differentiable inverse kinematics on rotations only
    # joints_local_rotations: local rotation matrices of each joint in the batch (B, J, 3, 3)
    # out: local rotations of each joint in the batch (B, J, 3, 3)
    # note: for now only works with only one parent bone at index 0 (hips)
    def invert_rotations(self, joints_global_rotations: torch.Tensor):
        batch_size = joints_global_rotations.shape[0]
        joints_nbr = self.joint_offsets.shape[0]

        joint_local_rotations = inverse_kinematics_rotations_only(joints_global_rotations,
                                                                  level_joints=self.level_bones,
                                                                  level_joint_parents=self.level_bones_parents)
        joint_local_rotations = joint_local_rotations.view(batch_size, joints_nbr, 3, 3).contiguous()

        return joint_local_rotations

    # performs differentiable inverse kinematics on quaternions
    # joints_local_rotations: localquaternions of each joint in the batch (..., J, 4)
    # out: world positions of each joint in the batch (..., J, 3)
    # note: for now only works with only one parent bone at index 0 (hips)
    def invert_quats(self, joints_global_rotations: torch.Tensor, joints_global_positions: torch.Tensor,
                     true_hip_offset=None):
        if true_hip_offset is not None:
            joints_global_positions = joints_global_positions - true_hip_offset[..., None, :].expand_as(
                joints_global_positions)

        joint_local_positions, joint_local_quats = inverse_kinematics_quats(joints_global_rotations,
                                                                            joints_global_positions,
                                                                            level_joints=self.level_bones,
                                                                            level_joint_parents=self.level_bones_parents)

        return joint_local_positions, joint_local_quats




