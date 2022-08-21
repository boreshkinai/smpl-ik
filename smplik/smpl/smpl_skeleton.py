import torch
from pytorch3d.transforms import quaternion_to_axis_angle
from smplx import SMPL

from smplik.geometry.forward_kinematics import forward_kinematics, extract_translation_rotation


class SmplSkeleton(torch.nn.Module):
    def __init__(self, smpl: SMPL):
        super().__init__()

        self.smpl = smpl
        self.num_joints = smpl.NUM_BODY_JOINTS + 1

        identity_quats = torch.zeros((self.num_joints, 4))
        identity_quats[:, 0] = 1
        tpose_angles = quaternion_to_axis_angle(identity_quats)
        self.tpose_body_pose = tpose_angles[1:, :].unsqueeze(0)
        self.tpose_global_orient = tpose_angles[[0], :].unsqueeze(0)

        self.level_bones = {}
        self.level_bones_parents = {}
        for i in range(self.num_joints):
            depth = self._get_joint_depth(i)
            if not depth in self.level_bones:
                self.level_bones[depth] = []
                self.level_bones_parents[depth] = []

            self.level_bones[depth].append(i)
            self.level_bones_parents[depth].append(self.smpl.parents[i].item())

    def _get_joint_depth(self, joint_idx: int):
        depth = 0
        while self.smpl.parents[joint_idx] != -1:
            joint_idx = self.smpl.parents[joint_idx]
            depth += 1

        return depth

    def _compute_local_offsets(self, betas: torch.Tensor):
        # Apply SMPL with no joint rotation
        smpl_out = self.smpl.forward(betas=betas, body_pose=self.tpose_body_pose.repeat(betas.shape[0], 1, 1), global_orient=self.tpose_global_orient.repeat(betas.shape[0], 1, 1), return_verts=False)

        # compute offset between joint and their parents (root excepted)
        child_positions = smpl_out.joints[:, 1:self.num_joints, :]
        parent_positions = smpl_out.joints[:, self.smpl.parents[1:], :]
        local_offsets = child_positions - parent_positions

        # add back the root
        local_offsets = torch.cat([smpl_out.joints[:, [0], :], local_offsets], dim=1)

        return local_offsets

    # joints_local_rotations: local rotation matrices of each joint in the batch (B, J, 3, 3)
    def forward(self, joints_local_rotations: torch.Tensor, betas: torch.Tensor, root_positions: torch.Tensor = None):
        batch_size = joints_local_rotations.shape[0]

        offsets = self._compute_local_offsets(betas=betas)

        world_transforms = forward_kinematics(joints_local_rotations, offsets, level_joints=self.level_bones,
                                              level_joint_parents=self.level_bones_parents)

        # extract global position from transform
        joint_positions, joint_rotations = extract_translation_rotation(world_transforms.view(-1, 4, 4))
        joint_positions = joint_positions.view(batch_size, self.num_joints, 3).contiguous()
        joint_rotations = joint_rotations.view(batch_size, self.num_joints, 3, 3).contiguous()

        # apply root translation
        if root_positions is not None:
            joint_positions += root_positions.unsqueeze(1).expand(batch_size, self.num_joints, 3)

        return joint_positions, joint_rotations


