from typing import Optional, Dict, Tuple, Union

import hydra
import os
import os.path as osp
import pickle
import numpy as np
import torch
import torch.nn as nn

from dataclasses import dataclass

# from smplx import SMPL
from smplx.lbs import lbs, blend_shapes, vertices2joints, vertices2landmarks, batch_rodrigues, find_dynamic_lmk_idx_and_bcoords
from smplx.vertex_ids import vertex_ids as VERTEX_IDS
from smplx.utils import (
    Struct, to_np, to_tensor, Tensor, Array,
    SMPLOutput,
    SMPLHOutput,
    SMPLXOutput,
    find_joint_kin_chain)
from smplx.vertex_joint_selector import VertexJointSelector

from protores.smpl.smpl_models import SmplModelDownloader


def transform_mat(R: Tensor, t: Tensor) -> Tensor:
    ''' Creates a batch of transformation matrices
        Args:
            - R: Bx3x3 array of a batch of rotation matrices
            - t: Bx3x1 array of a batch of translation vectors
        Returns:
            - T: Bx4x4 Transformation matrix
    '''
    # No padding left or right, only add an extra row
    # return torch.cat([F.pad(R, [0, 0, 0, 1]),
    #                   F.pad(t, [0, 0, 0, 1], value=1)], dim=2)
    return torch.cat([torch.cat([R, torch.zeros((R.shape[0], 1, R.shape[2]), dtype=R.dtype, device=R.device)], dim=1),
                      torch.cat([t, torch.ones((t.shape[0], 1, t.shape[2]), dtype=t.dtype, device=t.device)], dim=1)], dim=2)


def batch_rigid_transform(
    rot_mats: Tensor,
    joints: Tensor,
    parents: Tensor,
    dtype=torch.float32
) -> Tensor:
    """
    Applies a batch of rigid transformations to the joints
    Parameters
    ----------
    rot_mats : torch.tensor BxNx3x3
        Tensor of rotation matrices
    joints : torch.tensor BxNx3
        Locations of joints
    parents : torch.tensor BxN
        The kinematic tree of each object
    dtype : torch.dtype, optional:
        The data type of the created tensors, the default is torch.float32
    Returns
    -------
    posed_joints : torch.tensor BxNx3
        The locations of the joints after applying the pose rotations
    rel_transforms : torch.tensor BxNx4x4
        The relative (with respect to the root joint) rigid transformations
        for all the joints
    """

    joints = torch.unsqueeze(joints, dim=-1)

    rel_joints = joints.clone()
    rel_joints[:, 1:] -= joints[:, parents[1:]]

    transforms_mat = transform_mat(
        rot_mats.reshape(-1, 3, 3),
        rel_joints.reshape(-1, 3, 1)).reshape(-1, joints.shape[1], 4, 4)

    transform_chain = [transforms_mat[:, 0]]
    for i in range(1, parents.shape[0]):
        # Subtract the joint location at the rest pose
        # No need for rotation, since it's identity when at rest
        curr_res = torch.matmul(transform_chain[parents[i]],
                                transforms_mat[:, i])
        transform_chain.append(curr_res)

    transforms = torch.stack(transform_chain, dim=1)

    # The last column of the transformations contains the posed joints
    posed_joints = transforms[:, :, :3, 3]

    # joints_homogen = F.pad(joints, [0, 0, 0, 1])
    joints_homogen = torch.cat([joints, torch.zeros((joints.shape[0], joints.shape[1], 1, joints.shape[3]), dtype=joints.dtype, device=joints.device)], dim=2)

    # rel_transforms = transforms - F.pad(torch.matmul(transforms, joints_homogen), [3, 0, 0, 0, 0, 0, 0, 0])
    rel_transforms = transforms - torch.cat([torch.zeros((joints_homogen.shape[0], joints_homogen.shape[1], joints_homogen.shape[2], 3), dtype=transforms.dtype, device=transforms.device), torch.matmul(transforms, joints_homogen)], dim=3)

    return posed_joints, rel_transforms, transforms


def lbsFK(betas: Tensor,
          pose: Tensor,
          v_template: Tensor,
          shapedirs: Tensor,
          posedirs: Tensor,
          J_regressor: Tensor,
          parents: Tensor,
          lbs_weights: Tensor,
          pose2rot: bool = True,
    ) -> Tuple[Tensor, Tensor]:
    batch_size = pose.shape[0]
    device, dtype = betas.device, betas.dtype

    # Add shape contribution
    v_shaped = v_template + blend_shapes(betas, shapedirs)

    # Get the joints
    # NxJx3 array
    J = vertices2joints(J_regressor, v_shaped)

    # 3. Add pose blend shapes
    # N x J x 3 x 3
    ident = torch.eye(3, dtype=dtype, device=device)
    if pose2rot:
        rot_mats = batch_rodrigues(pose.view(-1, 3)).view(
            [batch_size, -1, 3, 3])

        pose_feature = (rot_mats[:, 1:, :, :] - ident).view([batch_size, -1])
        # (N x P) x (P, V * 3) -> N x V x 3
        pose_offsets = torch.matmul(
            pose_feature, posedirs).view(batch_size, -1, 3)
    else:
        pose_feature = pose[:, 1:].view(batch_size, -1, 3, 3) - ident
        rot_mats = pose.view(batch_size, -1, 3, 3)

        pose_offsets = torch.matmul(pose_feature.view(batch_size, -1),
                                    posedirs).view(batch_size, -1, 3)

    v_posed = pose_offsets + v_shaped
    # 4. Get the global joint location
    J_transformed, A, A_glob = batch_rigid_transform(rot_mats, J, parents, dtype=dtype)

    # 5. Do skinning:
    # W is N x V x (J + 1)
    W = lbs_weights.unsqueeze(dim=0).expand([batch_size, -1, -1])
    # (N x V x (J + 1)) x (N x (J + 1) x 16)
    num_joints = J_regressor.shape[0]
    T = torch.matmul(W, A.view(batch_size, num_joints, 16)) \
        .view(batch_size, -1, 4, 4)

    homogen_coord = torch.ones([batch_size, v_posed.shape[1], 1],
                               dtype=dtype, device=device)
    v_posed_homo = torch.cat([v_posed, homogen_coord], dim=2)
    v_homo = torch.matmul(T, torch.unsqueeze(v_posed_homo, dim=-1))

    verts = v_homo[:, :, :3, 0]

    return verts, J_transformed, A_glob


@dataclass
class SmplOutput(SMPLOutput):
    joints_h36m17: Optional[Tensor] = None


class SMPL(nn.Module):

    NUM_JOINTS = 23
    NUM_BODY_JOINTS = 23
    SHAPE_SPACE_DIM = 300

    def __init__(
        self, model_path: str,
        kid_template_path: str = '',
        data_struct: Optional[Struct] = None,
        create_betas: bool = True,
        betas: Optional[Tensor] = None,
        num_betas: int = 10,
        create_global_orient: bool = True,
        global_orient: Optional[Tensor] = None,
        create_body_pose: bool = True,
        body_pose: Optional[Tensor] = None,
        create_transl: bool = True,
        transl: Optional[Tensor] = None,
        dtype=torch.float32,
        batch_size: int = 1,
        joint_mapper=None,
        gender: str = 'neutral',
        age: str = 'adult',
        vertex_ids: Dict[str, int] = None,
        v_template: Optional[Union[Tensor, Array]] = None,
        J_reg_h36m17_path=None,
        J_reg_extra9_path=None,
        **kwargs
    ) -> None:
        ''' SMPL model constructor

            Parameters
            ----------
            model_path: str
                The path to the folder or to the file where the model
                parameters are stored
            data_struct: Strct
                A struct object. If given, then the parameters of the model are
                read from the object. Otherwise, the model tries to read the
                parameters from the given `model_path`. (default = None)
            create_global_orient: bool, optional
                Flag for creating a member variable for the global orientation
                of the body. (default = True)
            global_orient: torch.tensor, optional, Bx3
                The default value for the global orientation variable.
                (default = None)
            create_body_pose: bool, optional
                Flag for creating a member variable for the pose of the body.
                (default = True)
            body_pose: torch.tensor, optional, Bx(Body Joints * 3)
                The default value for the body pose variable.
                (default = None)
            num_betas: int, optional
                Number of shape components to use
                (default = 10).
            create_betas: bool, optional
                Flag for creating a member variable for the shape space
                (default = True).
            betas: torch.tensor, optional, Bx10
                The default value for the shape member variable.
                (default = None)
            create_transl: bool, optional
                Flag for creating a member variable for the translation
                of the body. (default = True)
            transl: torch.tensor, optional, Bx3
                The default value for the transl variable.
                (default = None)
            dtype: torch.dtype, optional
                The data type for the created variables
            batch_size: int, optional
                The batch size used for creating the member variables
            joint_mapper: object, optional
                An object that re-maps the joints. Useful if one wants to
                re-order the SMPL joints to some other convention (e.g. MSCOCO)
                (default = None)
            gender: str, optional
                Which gender to load
            vertex_ids: dict, optional
                A dictionary containing the indices of the extra vertices that
                will be selected
        '''

        self.gender = gender
        self.age = age

        if data_struct is None:
            if osp.isdir(model_path):
                model_fn = 'SMPL_{}.{ext}'.format(gender.upper(), ext='pkl')
                smpl_path = os.path.join(model_path, model_fn)
            else:
                smpl_path = model_path
            assert osp.exists(smpl_path), 'Path {} does not exist!'.format(
                smpl_path)

            with open(smpl_path, 'rb') as smpl_file:
                data_struct = Struct(**pickle.load(smpl_file,
                                                   encoding='latin1'))

        super().__init__()
        self.batch_size = batch_size
        shapedirs = data_struct.shapedirs
        if (shapedirs.shape[-1] < self.SHAPE_SPACE_DIM):
            print(f'WARNING: You are using a {self.name()} model, with only'
                  ' 10 shape coefficients.')
            num_betas = min(num_betas, 10)
        else:
            num_betas = min(num_betas, self.SHAPE_SPACE_DIM)

        if self.age=='kid':
            v_template_smil = np.load(kid_template_path)
            v_template_smil -= np.mean(v_template_smil, axis=0)
            v_template_diff = np.expand_dims(v_template_smil - data_struct.v_template, axis=2)
            shapedirs = np.concatenate((shapedirs[:, :, :num_betas], v_template_diff), axis=2)
            num_betas = num_betas + 1

        self._num_betas = num_betas
        shapedirs = shapedirs[:, :, :num_betas]
        # The shape components
        self.register_buffer(
            'shapedirs',
            to_tensor(to_np(shapedirs), dtype=dtype))

        if vertex_ids is None:
            # SMPL and SMPL-H share the same topology, so any extra joints can
            # be drawn from the same place
            vertex_ids = VERTEX_IDS['smplh']

        self.dtype = dtype

        self.joint_mapper = joint_mapper

        self.vertex_joint_selector = VertexJointSelector(
            vertex_ids=vertex_ids, **kwargs)

        self.faces = data_struct.f
        self.register_buffer('faces_tensor',
                             to_tensor(to_np(self.faces, dtype=np.int64),
                                       dtype=torch.long))

        if create_betas:
            if betas is None:
                default_betas = torch.zeros(
                    [batch_size, self.num_betas], dtype=dtype)
            else:
                if torch.is_tensor(betas):
                    default_betas = betas.clone().detach()
                else:
                    default_betas = torch.tensor(betas, dtype=dtype)

            self.register_parameter(
                'betas', nn.Parameter(default_betas, requires_grad=True))

        # The tensor that contains the global rotation of the model
        # It is separated from the pose of the joints in case we wish to
        # optimize only over one of them
        if create_global_orient:
            if global_orient is None:
                default_global_orient = torch.zeros(
                    [batch_size, 3], dtype=dtype)
            else:
                if torch.is_tensor(global_orient):
                    default_global_orient = global_orient.clone().detach()
                else:
                    default_global_orient = torch.tensor(
                        global_orient, dtype=dtype)

            global_orient = nn.Parameter(default_global_orient,
                                         requires_grad=True)
            self.register_parameter('global_orient', global_orient)

        if create_body_pose:
            if body_pose is None:
                default_body_pose = torch.zeros(
                    [batch_size, self.NUM_BODY_JOINTS * 3], dtype=dtype)
            else:
                if torch.is_tensor(body_pose):
                    default_body_pose = body_pose.clone().detach()
                else:
                    default_body_pose = torch.tensor(body_pose,
                                                     dtype=dtype)
            self.register_parameter(
                'body_pose',
                nn.Parameter(default_body_pose, requires_grad=True))

        if create_transl:
            if transl is None:
                default_transl = torch.zeros([batch_size, 3],
                                             dtype=dtype,
                                             requires_grad=True)
            else:
                default_transl = torch.tensor(transl, dtype=dtype)
            self.register_parameter(
                'transl', nn.Parameter(default_transl, requires_grad=True))

        if v_template is None:
            v_template = data_struct.v_template
        if not torch.is_tensor(v_template):
            v_template = to_tensor(to_np(v_template), dtype=dtype)
        # The vertices of the template model
        self.register_buffer('v_template', v_template)

        # # add bias
        # if self.v_template.shape[0] == 10475:
        #     bias = torch.Tensor(np.array([[0, 0.1728, 0.0218]]))
        #     self.v_template = self.v_template + bias

        j_regressor = to_tensor(to_np(
            data_struct.J_regressor), dtype=dtype)
        self.register_buffer('J_regressor', j_regressor)

        if J_reg_h36m17_path is not None:
            H36M_TO_J17 = [6, 5, 4, 1, 2, 3, 16, 15, 14, 11, 12, 13, 8, 10, 0, 7, 9]
            J_regressor_h36m17 = np.load(J_reg_h36m17_path)[H36M_TO_J17]
            J_regressor_h36m17 = to_tensor(to_np(J_regressor_h36m17), dtype=dtype)
            self.register_buffer('J_regressor_h36m17', J_regressor_h36m17)
        else:
            self.register_buffer('J_regressor_h36m17', None)

        if J_reg_extra9_path is not None:
            J_regressor_extra9 = np.load(J_reg_extra9_path)
            J_regressor_extra9 = to_tensor(to_np(J_regressor_extra9), dtype=dtype)
            self.register_buffer('J_regressor_extra9', J_regressor_extra9)
        else:
            self.register_buffer('J_regressor_extra9', None)

        # Pose blend shape basis: 6890 x 3 x 207, reshaped to 6890*3 x 207
        num_pose_basis = data_struct.posedirs.shape[-1]
        # 207 x 20670
        posedirs = np.reshape(data_struct.posedirs, [-1, num_pose_basis]).T
        self.register_buffer('posedirs',
                             to_tensor(to_np(posedirs), dtype=dtype))

        # indices of parents for each joints
        parents = to_tensor(to_np(data_struct.kintree_table[0])).long()
        parents[0] = -1
        self.register_buffer('parents', parents)

        lbs_weights = to_tensor(to_np(data_struct.weights), dtype=dtype)
        self.register_buffer('lbs_weights', lbs_weights)

    @property
    def num_betas(self):
        return self._num_betas

    @property
    def num_expression_coeffs(self):
        return 0

    def create_mean_pose(self, data_struct) -> Tensor:
        pass

    def name(self) -> str:
        return 'SMPL'

    @torch.no_grad()
    def reset_params(self, **params_dict) -> None:
        for param_name, param in self.named_parameters():
            if param_name in params_dict:
                param[:] = torch.tensor(params_dict[param_name])
            else:
                param.fill_(0)

    def get_num_verts(self) -> int:
        return self.v_template.shape[0]

    def get_num_faces(self) -> int:
        return self.faces.shape[0]

    def extra_repr(self) -> str:
        msg = [
            f'Gender: {self.gender.upper()}',
            f'Number of joints: {self.J_regressor.shape[0]}',
            f'Betas: {self.num_betas}',
        ]
        return '\n'.join(msg)

    def forward_shape(
        self,
        betas: Optional[Tensor] = None,
    ) -> SMPLOutput:
        betas = betas if betas is not None else self.betas
        v_shaped = self.v_template + blend_shapes(betas, self.shapedirs)
        return SMPLOutput(vertices=v_shaped, betas=betas, v_shaped=v_shaped)

    def forward(
        self,
        betas: Optional[Tensor] = None,
        body_pose: Optional[Tensor] = None,
        global_orient: Optional[Tensor] = None,
        transl: Optional[Tensor] = None,
        return_verts=True,
        return_full_pose: bool = False,
        pose2rot: bool = True,
        **kwargs
    ) -> SmplOutput:
        ''' Forward pass for the SMPL model

            Parameters
            ----------
            global_orient: torch.tensor, optional, shape Bx3
                If given, ignore the member variable and use it as the global
                rotation of the body. Useful if someone wishes to predicts this
                with an external model. (default=None)
            betas: torch.tensor, optional, shape BxN_b
                If given, ignore the member variable `betas` and use it
                instead. For example, it can used if shape parameters
                `betas` are predicted from some external model.
                (default=None)
            body_pose: torch.tensor, optional, shape Bx(J*3)
                If given, ignore the member variable `body_pose` and use it
                instead. For example, it can used if someone predicts the
                pose of the body joints are predicted from some external model.
                It should be a tensor that contains joint rotations in
                axis-angle format. (default=None)
            transl: torch.tensor, optional, shape Bx3
                If given, ignore the member variable `transl` and use it
                instead. For example, it can used if the translation
                `transl` is predicted from some external model.
                (default=None)
            return_verts: bool, optional
                Return the vertices. (default=True)
            return_full_pose: bool, optional
                Returns the full axis-angle pose vector (default=False)

            Returns
            -------
        '''
        # If no shape and pose parameters are passed along, then use the
        # ones from the module
        body_pose = body_pose if body_pose is not None else self.body_pose
        bn = body_pose.shape[0]
        global_orient = (global_orient if global_orient is not None else
                         self.global_orient[:bn])
        betas = betas if betas is not None else self.betas[:bn]

        apply_trans = transl is not None or hasattr(self, 'transl')
        if transl is None and hasattr(self, 'transl'):
            transl = self.transl

        full_pose = torch.cat([global_orient, body_pose], dim=1)

        batch_size = max(betas.shape[0], global_orient.shape[0],
                         body_pose.shape[0])

        if betas.shape[0] != batch_size:
            num_repeats = int(batch_size / betas.shape[0])
            betas = betas.expand(num_repeats, -1)

        vertices, joints = lbs(betas, full_pose, self.v_template,
                               self.shapedirs, self.posedirs,
                               self.J_regressor, self.parents,
                               self.lbs_weights, pose2rot=pose2rot)

        joints = self.vertex_joint_selector(vertices, joints)
        # Map the joints to the current dataset
        if self.joint_mapper is not None:
            joints = self.joint_mapper(joints)

        if self.J_regressor_h36m17 is not None:
            # 54 joints = 45 joints + 9 extra joints from different datasets
            joints_h36m17 = vertices2joints(self.J_regressor_h36m17, vertices)
            # use the middle of hip used in the most 2D pose datasets, not the o-th Pelvis of SMPL 24 joint
            joints_h36m17_pelvis = joints_h36m17[:, 14].unsqueeze(1)
            joints_h36m17 = joints_h36m17 - joints_h36m17_pelvis

        if self.J_regressor_extra9 is not None:
            # 54 joints = 45 joints + 9 extra joints from different datasets
            joints = torch.cat([joints, vertices2joints(self.J_regressor_extra9, vertices)], 1)

        if apply_trans:
            joints += transl.unsqueeze(dim=1)
            vertices += transl.unsqueeze(dim=1)
            if self.J_regressor_h36m17 is not None:
                joints_h36m17 += transl.unsqueeze(dim=1)

        output = SmplOutput(vertices=vertices if return_verts else None,
                            global_orient=global_orient,
                            body_pose=body_pose,
                            joints=joints,
                            betas=betas,
                            full_pose=full_pose if return_full_pose else None,
                            joints_h36m17=joints_h36m17 if self.J_regressor_h36m17 is not None else None)

        return output


class SMPLLayer(SMPL):
    def __init__(
        self,
        *args,
        **kwargs
    ) -> None:
        # Just create a SMPL module without any member variables
        super().__init__(
            create_body_pose=False,
            create_betas=False,
            create_global_orient=False,
            create_transl=False,
            *args,
            **kwargs,
        )

    def forward(
        self,
        betas: Optional[Tensor] = None,
        body_pose: Optional[Tensor] = None,
        global_orient: Optional[Tensor] = None,
        transl: Optional[Tensor] = None,
        return_verts=True,
        return_full_pose: bool = False,
        pose2rot: bool = True,
        **kwargs
    ) -> SmplOutput:
        ''' Forward pass for the SMPL model

            Parameters
            ----------
            global_orient: torch.tensor, optional, shape Bx3x3
                Global rotation of the body.  Useful if someone wishes to
                predicts this with an external model. It is expected to be in
                rotation matrix format.  (default=None)
            betas: torch.tensor, optional, shape BxN_b
                Shape parameters. For example, it can used if shape parameters
                `betas` are predicted from some external model.
                (default=None)
            body_pose: torch.tensor, optional, shape BxJx3x3
                Body pose. For example, it can used if someone predicts the
                pose of the body joints are predicted from some external model.
                It should be a tensor that contains joint rotations in
                rotation matrix format. (default=None)
            transl: torch.tensor, optional, shape Bx3
                Translation vector of the body.
                For example, it can used if the translation
                `transl` is predicted from some external model.
                (default=None)
            return_verts: bool, optional
                Return the vertices. (default=True)
            return_full_pose: bool, optional
                Returns the full axis-angle pose vector (default=False)

            Returns
            -------
        '''
        model_vars = [betas, global_orient, body_pose, transl]
        batch_size = 1
        for var in model_vars:
            if var is None:
                continue
            batch_size = max(batch_size, len(var))
        device, dtype = self.shapedirs.device, self.shapedirs.dtype
        if global_orient is None:
            global_orient = torch.eye(3, device=device, dtype=dtype).view(
                1, 1, 3, 3).expand(batch_size, -1, -1, -1).contiguous()
        if body_pose is None:
            body_pose = torch.eye(3, device=device, dtype=dtype).view(
                1, 1, 3, 3).expand(
                    batch_size, self.NUM_BODY_JOINTS, -1, -1).contiguous()
        if betas is None:
            betas = torch.zeros([batch_size, self.num_betas],
                                dtype=dtype, device=device)
        if transl is None:
            transl = torch.zeros([batch_size, 3], dtype=dtype, device=device)

        full_pose = torch.cat(
            [global_orient.reshape(-1, 1, 3, 3),
             body_pose.reshape(-1, self.NUM_BODY_JOINTS, 3, 3)],
            dim=1)

        vertices, joints = lbs(betas, full_pose, self.v_template,
                               self.shapedirs, self.posedirs,
                               self.J_regressor, self.parents,
                               self.lbs_weights,
                               pose2rot=False)

        joints = self.vertex_joint_selector(vertices, joints)
        # Map the joints to the current dataset
        if self.joint_mapper is not None:
            joints = self.joint_mapper(joints)

        if self.J_regressor_h36m17 is not None:
            # 54 joints = 45 joints + 9 extra joints from different datasets
            joints_h36m17 = vertices2joints(self.J_regressor_h36m17, vertices)
            # use the middle of hip used in the most 2D pose datasets, not the o-th Pelvis of SMPL 24 joint
            joints_h36m17_pelvis = joints_h36m17[:, 14].unsqueeze(1)
            joints_h36m17 = joints_h36m17 - joints_h36m17_pelvis

        if self.J_regressor_extra9 is not None:
            # 54 joints = 45 joints + 9 extra joints from different datasets
            joints = torch.cat([joints, vertices2joints(self.J_regressor_extra9, vertices)], 1)

        if transl is not None:
            joints += transl.unsqueeze(dim=1)
            vertices += transl.unsqueeze(dim=1)
            if self.J_regressor_h36m17 is not None:
                joints_h36m17 += transl.unsqueeze(dim=1)

        output = SmplOutput(vertices=vertices if return_verts else None,
                            global_orient=global_orient,
                            body_pose=body_pose,
                            joints=joints,
                            betas=betas,
                            full_pose=full_pose if return_full_pose else None,
                            joints_h36m17=joints_h36m17 if self.J_regressor_h36m17 is not None else None)

        return output


@dataclass
class SmplHOutput(SMPLHOutput):
    joints_h36m17: Optional[Tensor] = None


class SMPLH(SMPL):

    # The hand joints are replaced by MANO
    NUM_BODY_JOINTS = SMPL.NUM_JOINTS - 2
    NUM_HAND_JOINTS = 15
    NUM_JOINTS = NUM_BODY_JOINTS + 2 * NUM_HAND_JOINTS

    def __init__(
        self, model_path,
        kid_template_path: str = '',
        data_struct: Optional[Struct] = None,
        create_betas: bool = True,
        betas: Optional[Tensor] = None,
        num_betas: int = 16,
        create_global_orient: bool = True,
        global_orient: Optional[Tensor] = None,
        create_body_pose: bool = True,
        body_pose: Optional[Tensor] = None,
        create_transl: bool = True,
        transl: Optional[Tensor] = None,
        dtype=torch.float32,
        batch_size: int = 1,
        joint_mapper=None,
        gender: str = 'neutral',
        age: str = 'adult',
        vertex_ids: Dict[str, int] = None,
        v_template: Optional[Union[Tensor, Array]] = None,
        create_left_hand_pose: bool = True,
        left_hand_pose: Optional[Tensor] = None,
        create_right_hand_pose: bool = True,
        right_hand_pose: Optional[Tensor] = None,
        use_pca: bool = False,
        num_pca_comps: int = 6,
        flat_hand_mean: bool = False,
        use_compressed: bool = True,
        ext: str = 'pkl',
        **kwargs
    ) -> None:
        ''' SMPLH model constructor

            Parameters
            ----------
            model_path: str
                The path to the folder or to the file where the model
                parameters are stored
            data_struct: Strct
                A struct object. If given, then the parameters of the model are
                read from the object. Otherwise, the model tries to read the
                parameters from the given `model_path`. (default = None)
            create_left_hand_pose: bool, optional
                Flag for creating a member variable for the pose of the left
                hand. (default = True)
            left_hand_pose: torch.tensor, optional, BxP
                The default value for the left hand pose member variable.
                (default = None)
            create_right_hand_pose: bool, optional
                Flag for creating a member variable for the pose of the right
                hand. (default = True)
            right_hand_pose: torch.tensor, optional, BxP
                The default value for the right hand pose member variable.
                (default = None)
            num_pca_comps: int, optional
                The number of PCA components to use for each hand.
                (default = 6)
            flat_hand_mean: bool, optional
                If False, then the pose of the hand is initialized to False.
            batch_size: int, optional
                The batch size used for creating the member variables
            gender: str, optional
                Which gender to load
            dtype: torch.dtype, optional
                The data type for the created variables
            vertex_ids: dict, optional
                A dictionary containing the indices of the extra vertices that
                will be selected
        '''

        self.num_pca_comps = num_pca_comps
        # If no data structure is passed, then load the data from the given
        # model folder
        if data_struct is None:
            # Load the model
            if osp.isdir(model_path):
                model_fn = 'SMPLH_{}.{ext}'.format(gender.upper(), ext=ext)
                smplh_path = os.path.join(model_path, model_fn)
            else:
                smplh_path = model_path
            assert osp.exists(smplh_path), 'Path {} does not exist!'.format(
                smplh_path)

            if ext == 'pkl':
                with open(smplh_path, 'rb') as smplh_file:
                    model_data = pickle.load(smplh_file, encoding='latin1')
            elif ext == 'npz':
                model_data = np.load(smplh_path, allow_pickle=True)
            else:
                raise ValueError('Unknown extension: {}'.format(ext))
            data_struct = Struct(**model_data)

        if vertex_ids is None:
            vertex_ids = VERTEX_IDS['smplh']

        self.gender = gender
        self.age = age

        super().__init__()
        self.batch_size = batch_size
        shapedirs = data_struct.shapedirs
        if (shapedirs.shape[-1] < self.SHAPE_SPACE_DIM):
            print(f'WARNING: You are using a {self.name()} model, with only'
                  f' {shapedirs.shape[-1]} shape coefficients.')
            num_betas = min(num_betas, 16)
        else:
            num_betas = min(num_betas, self.SHAPE_SPACE_DIM)

        if self.age=='kid':
            v_template_smil = np.load(kid_template_path)
            v_template_smil -= np.mean(v_template_smil, axis=0)
            v_template_diff = np.expand_dims(v_template_smil - data_struct.v_template, axis=2)
            shapedirs = np.concatenate((shapedirs[:, :, :num_betas], v_template_diff), axis=2)
            num_betas = num_betas + 1

        self._num_betas = num_betas
        shapedirs = shapedirs[:, :, :num_betas]
        # The shape components
        self.register_buffer(
            'shapedirs',
            to_tensor(to_np(shapedirs), dtype=dtype))

        if vertex_ids is None:
            # SMPL and SMPL-H share the same topology, so any extra joints can
            # be drawn from the same place
            vertex_ids = VERTEX_IDS['smplh']

        self.dtype = dtype

        self.joint_mapper = joint_mapper

        self.vertex_joint_selector = VertexJointSelector(
            vertex_ids=vertex_ids, **kwargs)

        self.faces = data_struct.f
        self.register_buffer('faces_tensor',
                             to_tensor(to_np(self.faces, dtype=np.int64),
                                       dtype=torch.long))

        if create_betas:
            if betas is None:
                default_betas = torch.zeros(
                    [batch_size, self.num_betas], dtype=dtype)
            else:
                if torch.is_tensor(betas):
                    default_betas = betas.clone().detach()
                else:
                    default_betas = torch.tensor(betas, dtype=dtype)

            self.register_parameter(
                'betas', nn.Parameter(default_betas, requires_grad=True))

        # The tensor that contains the global rotation of the model
        # It is separated from the pose of the joints in case we wish to
        # optimize only over one of them
        if create_global_orient:
            if global_orient is None:
                default_global_orient = torch.zeros(
                    [batch_size, 3], dtype=dtype)
            else:
                if torch.is_tensor(global_orient):
                    default_global_orient = global_orient.clone().detach()
                else:
                    default_global_orient = torch.tensor(
                        global_orient, dtype=dtype)

            global_orient = nn.Parameter(default_global_orient,
                                         requires_grad=True)
            self.register_parameter('global_orient', global_orient)

        if create_body_pose:
            if body_pose is None:
                default_body_pose = torch.zeros(
                    [batch_size, self.NUM_BODY_JOINTS * 3], dtype=dtype)
            else:
                if torch.is_tensor(body_pose):
                    default_body_pose = body_pose.clone().detach()
                else:
                    default_body_pose = torch.tensor(body_pose,
                                                     dtype=dtype)
            self.register_parameter(
                'body_pose',
                nn.Parameter(default_body_pose, requires_grad=True))

        if create_transl:
            if transl is None:
                default_transl = torch.zeros([batch_size, 3],
                                             dtype=dtype,
                                             requires_grad=True)
            else:
                default_transl = torch.tensor(transl, dtype=dtype)
            self.register_parameter(
                'transl', nn.Parameter(default_transl, requires_grad=True))

        if v_template is None:
            v_template = data_struct.v_template
        if not torch.is_tensor(v_template):
            v_template = to_tensor(to_np(v_template), dtype=dtype)
        # The vertices of the template model
        self.register_buffer('v_template', v_template)

        j_regressor = to_tensor(to_np(
            data_struct.J_regressor), dtype=dtype)
        self.register_buffer('J_regressor', j_regressor)

        if J_reg_h36m17_path is not None:
            H36M_TO_J17 = [6, 5, 4, 1, 2, 3, 16, 15, 14, 11, 12, 13, 8, 10, 0, 7, 9]
            J_regressor_h36m17 = np.load(J_reg_h36m17_path)[H36M_TO_J17]
            J_regressor_h36m17 = to_tensor(to_np(J_regressor_h36m17), dtype=dtype)
            self.register_buffer('J_regressor_h36m17', J_regressor_h36m17)
        else:
            self.register_buffer('J_regressor_h36m17', None)

        if J_reg_extra9_path is not None:
            J_regressor_extra9 = np.load(J_reg_extra9_path)
            J_regressor_extra9 = to_tensor(to_np(J_regressor_extra9), dtype=dtype)
            self.register_buffer('J_regressor_extra9', J_regressor_extra9)
        else:
            self.register_buffer('J_regressor_extra9', None)

        # Pose blend shape basis: 6890 x 3 x 207, reshaped to 6890*3 x 207
        num_pose_basis = data_struct.posedirs.shape[-1]
        # 207 x 20670
        posedirs = np.reshape(data_struct.posedirs, [-1, num_pose_basis]).T
        self.register_buffer('posedirs',
                             to_tensor(to_np(posedirs), dtype=dtype))

        # indices of parents for each joints
        parents = to_tensor(to_np(data_struct.kintree_table[0])).long()
        parents[0] = -1
        self.register_buffer('parents', parents)

        lbs_weights = to_tensor(to_np(data_struct.weights), dtype=dtype)
        self.register_buffer('lbs_weights', lbs_weights)

        self.use_pca = use_pca
        self.num_pca_comps = num_pca_comps
        self.flat_hand_mean = flat_hand_mean

        if use_pca:
            left_hand_components = data_struct.hands_componentsl[:num_pca_comps]
            right_hand_components = data_struct.hands_componentsr[:num_pca_comps]

            self.np_left_hand_components = left_hand_components
            self.np_right_hand_components = right_hand_components

            self.register_buffer(
                'left_hand_components',
                torch.tensor(left_hand_components, dtype=dtype))
            self.register_buffer(
                'right_hand_components',
                torch.tensor(right_hand_components, dtype=dtype))

        if self.flat_hand_mean:
            left_hand_mean = np.zeros_like(data_struct.hands_meanl)
        else:
            left_hand_mean = data_struct.hands_meanl

        if self.flat_hand_mean:
            right_hand_mean = np.zeros_like(data_struct.hands_meanr)
        else:
            right_hand_mean = data_struct.hands_meanr

        self.register_buffer('left_hand_mean',
                             to_tensor(left_hand_mean, dtype=self.dtype))
        self.register_buffer('right_hand_mean',
                             to_tensor(right_hand_mean, dtype=self.dtype))

        # Create the buffers for the pose of the left hand
        hand_pose_dim = num_pca_comps if use_pca else 3 * self.NUM_HAND_JOINTS
        if create_left_hand_pose:
            if left_hand_pose is None:
                default_lhand_pose = torch.zeros([batch_size, hand_pose_dim],
                                                 dtype=dtype)
            else:
                default_lhand_pose = torch.tensor(left_hand_pose, dtype=dtype)

            left_hand_pose_param = nn.Parameter(default_lhand_pose,
                                                requires_grad=True)
            self.register_parameter('left_hand_pose',
                                    left_hand_pose_param)

        if create_right_hand_pose:
            if right_hand_pose is None:
                default_rhand_pose = torch.zeros([batch_size, hand_pose_dim],
                                                 dtype=dtype)
            else:
                default_rhand_pose = torch.tensor(right_hand_pose, dtype=dtype)

            right_hand_pose_param = nn.Parameter(default_rhand_pose,
                                                 requires_grad=True)
            self.register_parameter('right_hand_pose',
                                    right_hand_pose_param)

        # Create the buffer for the mean pose.
        pose_mean_tensor = self.create_mean_pose(
            data_struct, flat_hand_mean=flat_hand_mean)
        if not torch.is_tensor(pose_mean_tensor):
            pose_mean_tensor = torch.tensor(pose_mean_tensor, dtype=dtype)
        self.register_buffer('pose_mean', pose_mean_tensor)

    def create_mean_pose(self, data_struct, flat_hand_mean=False):
        # Create the array for the mean pose. If flat_hand is false, then use
        # the mean that is given by the data, rather than the flat open hand
        global_orient_mean = torch.zeros([3], dtype=self.dtype)
        body_pose_mean = torch.zeros([self.NUM_BODY_JOINTS * 3],
                                     dtype=self.dtype)

        pose_mean = torch.cat([global_orient_mean, body_pose_mean,
                               self.left_hand_mean,
                               self.right_hand_mean], dim=0)
        return pose_mean

    def name(self) -> str:
        return 'SMPL+H'

    def extra_repr(self):
        msg = super().extra_repr()
        msg = [msg]
        if self.use_pca:
            msg.append(f'Number of PCA components: {self.num_pca_comps}')
        msg.append(f'Flat hand mean: {self.flat_hand_mean}')
        return '\n'.join(msg)

    def forward(
        self,
        betas: Optional[Tensor] = None,
        global_orient: Optional[Tensor] = None,
        body_pose: Optional[Tensor] = None,
        left_hand_pose: Optional[Tensor] = None,
        right_hand_pose: Optional[Tensor] = None,
        transl: Optional[Tensor] = None,
        return_verts: bool = True,
        return_full_pose: bool = False,
        pose2rot: bool = True,
        **kwargs
    ) -> SmplHOutput:
        '''
        '''

        # If no shape and pose parameters are passed along, then use the
        # ones from the module
        body_pose = body_pose if body_pose is not None else self.body_pose
        bn = body_pose.shape[0]
        global_orient = (global_orient if global_orient is not None else
                         self.global_orient[:bn])
        betas = betas if betas is not None else self.betas[:bn]
        left_hand_pose = (left_hand_pose if left_hand_pose is not None else
                          self.left_hand_pose[:bn])
        right_hand_pose = (right_hand_pose if right_hand_pose is not None else
                           self.right_hand_pose[:bn])

        apply_trans = transl is not None or hasattr(self, 'transl')
        if transl is None:
            if hasattr(self, 'transl'):
                transl = self.transl[:bn]

        if self.use_pca:
            left_hand_pose = torch.einsum(
                'bi,ij->bj', [left_hand_pose, self.left_hand_components])
            right_hand_pose = torch.einsum(
                'bi,ij->bj', [right_hand_pose, self.right_hand_components])

        full_pose = torch.cat([global_orient[:bn], body_pose[:bn],
                               left_hand_pose[:bn],
                               right_hand_pose[:bn]], dim=1)
        full_pose += self.pose_mean

        vertices, joints = lbs(betas, full_pose, self.v_template,
                               self.shapedirs, self.posedirs,
                               self.J_regressor, self.parents,
                               self.lbs_weights, pose2rot=pose2rot)

        # # Add any extra joints that might be needed
        # joints = self.vertex_joint_selector(vertices, joints)
        # if self.joint_mapper is not None:
        #     joints = self.joint_mapper(joints)

        if self.J_regressor_h36m17 is not None:
            # 54 joints = 45 joints + 9 extra joints from different datasets
            joints_h36m17 = vertices2joints(self.J_regressor_h36m17, vertices)
            # use the middle of hip used in the most 2D pose datasets, not the o-th Pelvis of SMPL 24 joint
            joints_h36m17_pelvis = joints_h36m17[:, 14].unsqueeze(1)
            joints_h36m17 = joints_h36m17 - joints_h36m17_pelvis

        if apply_trans:
            joints += transl.unsqueeze(dim=1)
            vertices += transl.unsqueeze(dim=1)
            if self.J_regressor_h36m17 is not None:
                joints_h36m17 += transl.unsqueeze(dim=1)

        output = SmplHOutput(vertices=vertices if return_verts else None,
                             joints=joints,
                             betas=betas,
                             global_orient=global_orient,
                             body_pose=body_pose,
                             left_hand_pose=left_hand_pose,
                             right_hand_pose=right_hand_pose,
                             full_pose=full_pose if return_full_pose else None,
                             joints_h36m17=joints_h36m17 if self.J_regressor_h36m17 is not None else None)

        return output


class SMPLHLayer(SMPLH):

    def __init__(
        self, *args, **kwargs
    ) -> None:
        ''' SMPL+H as a layer model constructor
        '''
        super().__init__(
            create_global_orient=False,
            create_body_pose=False,
            create_left_hand_pose=False,
            create_right_hand_pose=False,
            create_betas=False,
            create_transl=False,
            *args,
            **kwargs)

    def forward(
        self,
        betas: Optional[Tensor] = None,
        global_orient: Optional[Tensor] = None,
        body_pose: Optional[Tensor] = None,
        left_hand_pose: Optional[Tensor] = None,
        right_hand_pose: Optional[Tensor] = None,
        transl: Optional[Tensor] = None,
        return_verts: bool = True,
        return_full_pose: bool = False,
        pose2rot: bool = True,
        **kwargs
    ) -> SmplHOutput:
        ''' Forward pass for the SMPL+H model

            Parameters
            ----------
            global_orient: torch.tensor, optional, shape Bx3x3
                Global rotation of the body. Useful if someone wishes to
                predicts this with an external model. It is expected to be in
                rotation matrix format. (default=None)
            betas: torch.tensor, optional, shape BxN_b
                Shape parameters. For example, it can used if shape parameters
                `betas` are predicted from some external model.
                (default=None)
            body_pose: torch.tensor, optional, shape BxJx3x3
                If given, ignore the member variable `body_pose` and use it
                instead. For example, it can used if someone predicts the
                pose of the body joints are predicted from some external model.
                It should be a tensor that contains joint rotations in
                rotation matrix format. (default=None)
            left_hand_pose: torch.tensor, optional, shape Bx15x3x3
                If given, contains the pose of the left hand.
                It should be a tensor that contains joint rotations in
                rotation matrix format. (default=None)
            right_hand_pose: torch.tensor, optional, shape Bx15x3x3
                If given, contains the pose of the right hand.
                It should be a tensor that contains joint rotations in
                rotation matrix format. (default=None)
            transl: torch.tensor, optional, shape Bx3
                Translation vector of the body.
                For example, it can used if the translation
                `transl` is predicted from some external model.
                (default=None)
            return_verts: bool, optional
                Return the vertices. (default=True)
            return_full_pose: bool, optional
                Returns the full axis-angle pose vector (default=False)

            Returns
            -------
        '''
        model_vars = [betas, global_orient, body_pose, transl, left_hand_pose,
                      right_hand_pose]
        batch_size = 1
        for var in model_vars:
            if var is None:
                continue
            batch_size = max(batch_size, len(var))
        device, dtype = self.shapedirs.device, self.shapedirs.dtype
        if global_orient is None:
            global_orient = torch.eye(3, device=device, dtype=dtype).view(
                1, 1, 3, 3).expand(batch_size, -1, -1, -1).contiguous()
        if body_pose is None:
            body_pose = torch.eye(3, device=device, dtype=dtype).view(
                1, 1, 3, 3).expand(batch_size, 21, -1, -1).contiguous()
        if left_hand_pose is None:
            left_hand_pose = torch.eye(3, device=device, dtype=dtype).view(
                1, 1, 3, 3).expand(batch_size, 15, -1, -1).contiguous()
        if right_hand_pose is None:
            right_hand_pose = torch.eye(3, device=device, dtype=dtype).view(
                1, 1, 3, 3).expand(batch_size, 15, -1, -1).contiguous()
        if betas is None:
            betas = torch.zeros([batch_size, self.num_betas],
                                dtype=dtype, device=device)
        if transl is None:
            transl = torch.zeros([batch_size, 3], dtype=dtype, device=device)

        # Concatenate all pose vectors
        full_pose = torch.cat(
            [global_orient.reshape(-1, 1, 3, 3),
             body_pose.reshape(-1, self.NUM_BODY_JOINTS, 3, 3),
             left_hand_pose.reshape(-1, self.NUM_HAND_JOINTS, 3, 3),
             right_hand_pose.reshape(-1, self.NUM_HAND_JOINTS, 3, 3)],
            dim=1)

        vertices, joints = lbs(betas, full_pose, self.v_template,
                               self.shapedirs, self.posedirs,
                               self.J_regressor, self.parents,
                               self.lbs_weights, pose2rot=False)

        # Add any extra joints that might be needed
        joints = self.vertex_joint_selector(vertices, joints)
        if self.joint_mapper is not None:
            joints = self.joint_mapper(joints)

        if self.J_regressor_h36m17 is not None:
            # 54 joints = 45 joints + 9 extra joints from different datasets
            joints_h36m17 = vertices2joints(self.J_regressor_h36m17, vertices)
            # use the middle of hip used in the most 2D pose datasets, not the o-th Pelvis of SMPL 24 joint
            joints_h36m17_pelvis = joints_h36m17[:, 14].unsqueeze(1)
            joints_h36m17 = joints_h36m17 - joints_h36m17_pelvis

        if transl is not None:
            joints += transl.unsqueeze(dim=1)
            vertices += transl.unsqueeze(dim=1)
            if self.J_regressor_h36m17 is not None:
                joints_h36m17 += transl.unsqueeze(dim=1)

        output = SmplHOutput(vertices=vertices if return_verts else None,
                             joints=joints,
                             betas=betas,
                             global_orient=global_orient,
                             body_pose=body_pose,
                             left_hand_pose=left_hand_pose,
                             right_hand_pose=right_hand_pose,
                             full_pose=full_pose if return_full_pose else None,
                             joints_h36m17=joints_h36m17 if self.J_regressor_h36m17 is not None else None)

        return output


@dataclass
class SmplXOutput(SMPLXOutput):
    joints_h36m17: Optional[Tensor] = None


class SMPLX(SMPLH):
    '''
    SMPL-X (SMPL eXpressive) is a unified body model, with shape parameters
    trained jointly for the face, hands and body.
    SMPL-X uses standard vertex based linear blend skinning with learned
    corrective blend shapes, has N=10475 vertices and K=54 joints,
    which includes joints for the neck, jaw, eyeballs and fingers.
    '''

    NUM_BODY_JOINTS = SMPLH.NUM_BODY_JOINTS
    NUM_HAND_JOINTS = 15
    NUM_FACE_JOINTS = 3
    NUM_JOINTS = NUM_BODY_JOINTS + 2 * NUM_HAND_JOINTS + NUM_FACE_JOINTS
    EXPRESSION_SPACE_DIM = 100
    NECK_IDX = 12

    def __init__(
        self, model_path: str,
        kid_template_path: str = '',
        num_expression_coeffs: int = 10,
        create_expression: bool = True,
        expression: Optional[Tensor] = None,
        create_jaw_pose: bool = True,
        jaw_pose: Optional[Tensor] = None,
        create_leye_pose: bool = True,
        leye_pose: Optional[Tensor] = None,
        create_reye_pose=True,
        reye_pose: Optional[Tensor] = None,
        use_face_contour: bool = False,
        batch_size: int = 1,
        gender: str = 'neutral',
        age: str = 'adult',
        dtype=torch.float32,
        ext: str = 'npz',
        **kwargs
    ) -> None:
        ''' SMPLX model constructor

            Parameters
            ----------
            model_path: str
                The path to the folder or to the file where the model
                parameters are stored
            num_expression_coeffs: int, optional
                Number of expression components to use
                (default = 10).
            create_expression: bool, optional
                Flag for creating a member variable for the expression space
                (default = True).
            expression: torch.tensor, optional, Bx10
                The default value for the expression member variable.
                (default = None)
            create_jaw_pose: bool, optional
                Flag for creating a member variable for the jaw pose.
                (default = False)
            jaw_pose: torch.tensor, optional, Bx3
                The default value for the jaw pose variable.
                (default = None)
            create_leye_pose: bool, optional
                Flag for creating a member variable for the left eye pose.
                (default = False)
            leye_pose: torch.tensor, optional, Bx10
                The default value for the left eye pose variable.
                (default = None)
            create_reye_pose: bool, optional
                Flag for creating a member variable for the right eye pose.
                (default = False)
            reye_pose: torch.tensor, optional, Bx10
                The default value for the right eye pose variable.
                (default = None)
            use_face_contour: bool, optional
                Whether to compute the keypoints that form the facial contour
            batch_size: int, optional
                The batch size used for creating the member variables
            gender: str, optional
                Which gender to load
            dtype: torch.dtype
                The data type for the created variables
        '''

        # Load the model
        if osp.isdir(model_path):
            model_fn = 'SMPLX_{}.{ext}'.format(gender.upper(), ext=ext)
            smplx_path = os.path.join(model_path, model_fn)
        else:
            smplx_path = model_path
        assert osp.exists(smplx_path), 'Path {} does not exist!'.format(
            smplx_path)

        if ext == 'pkl':
            with open(smplx_path, 'rb') as smplx_file:
                model_data = pickle.load(smplx_file, encoding='latin1')
        elif ext == 'npz':
            model_data = np.load(smplx_path, allow_pickle=True)
        else:
            raise ValueError('Unknown extension: {}'.format(ext))

        data_struct = Struct(**model_data)

        super().__init__(
            model_path=model_path,
            kid_template_path=kid_template_path,
            data_struct=data_struct,
            dtype=dtype,
            batch_size=batch_size,
            vertex_ids=VERTEX_IDS['smplx'],
            gender=gender, age=age, ext=ext,
            **kwargs)

        lmk_faces_idx = data_struct.lmk_faces_idx
        self.register_buffer('lmk_faces_idx',
                             torch.tensor(lmk_faces_idx, dtype=torch.long))
        lmk_bary_coords = data_struct.lmk_bary_coords
        self.register_buffer('lmk_bary_coords',
                             torch.tensor(lmk_bary_coords, dtype=dtype))

        self.use_face_contour = use_face_contour
        if self.use_face_contour:
            dynamic_lmk_faces_idx = data_struct.dynamic_lmk_faces_idx
            dynamic_lmk_faces_idx = torch.tensor(
                dynamic_lmk_faces_idx,
                dtype=torch.long)
            self.register_buffer('dynamic_lmk_faces_idx',
                                 dynamic_lmk_faces_idx)

            dynamic_lmk_bary_coords = data_struct.dynamic_lmk_bary_coords
            dynamic_lmk_bary_coords = torch.tensor(
                dynamic_lmk_bary_coords, dtype=dtype)
            self.register_buffer('dynamic_lmk_bary_coords',
                                 dynamic_lmk_bary_coords)

            neck_kin_chain = find_joint_kin_chain(self.NECK_IDX, self.parents)
            self.register_buffer(
                'neck_kin_chain',
                torch.tensor(neck_kin_chain, dtype=torch.long))

        if create_jaw_pose:
            if jaw_pose is None:
                default_jaw_pose = torch.zeros([batch_size, 3], dtype=dtype)
            else:
                default_jaw_pose = torch.tensor(jaw_pose, dtype=dtype)
            jaw_pose_param = nn.Parameter(default_jaw_pose,
                                          requires_grad=True)
            self.register_parameter('jaw_pose', jaw_pose_param)

        if create_leye_pose:
            if leye_pose is None:
                default_leye_pose = torch.zeros([batch_size, 3], dtype=dtype)
            else:
                default_leye_pose = torch.tensor(leye_pose, dtype=dtype)
            leye_pose_param = nn.Parameter(default_leye_pose,
                                           requires_grad=True)
            self.register_parameter('leye_pose', leye_pose_param)

        if create_reye_pose:
            if reye_pose is None:
                default_reye_pose = torch.zeros([batch_size, 3], dtype=dtype)
            else:
                default_reye_pose = torch.tensor(reye_pose, dtype=dtype)
            reye_pose_param = nn.Parameter(default_reye_pose,
                                           requires_grad=True)
            self.register_parameter('reye_pose', reye_pose_param)

        shapedirs = data_struct.shapedirs
        if len(shapedirs.shape) < 3:
            shapedirs = shapedirs[:, :, None]
        if (shapedirs.shape[-1] < self.SHAPE_SPACE_DIM +
                self.EXPRESSION_SPACE_DIM):
            print(f'WARNING: You are using a {self.name()} model, with only'
                  f' {shapedirs.shape[-1]} shape and {20-shapedirs.shape[-1]} expression coefficients.')
            expr_start_idx = shapedirs.shape[-1]
            expr_end_idx = 20
            num_expression_coeffs = min(num_expression_coeffs, 20-shapedirs.shape[-1])
        else:
            expr_start_idx = self.SHAPE_SPACE_DIM
            expr_end_idx = self.SHAPE_SPACE_DIM + num_expression_coeffs
            num_expression_coeffs = min(
                num_expression_coeffs, self.EXPRESSION_SPACE_DIM)

        self._num_expression_coeffs = num_expression_coeffs

        expr_dirs = shapedirs[:, :, expr_start_idx:expr_end_idx]
        self.register_buffer(
            'expr_dirs', to_tensor(to_np(expr_dirs), dtype=dtype))

        if create_expression:
            if expression is None:
                default_expression = torch.zeros(
                    [batch_size, self.num_expression_coeffs], dtype=dtype)
            else:
                default_expression = torch.tensor(expression, dtype=dtype)
            expression_param = nn.Parameter(default_expression,
                                            requires_grad=True)
            self.register_parameter('expression', expression_param)

    def name(self) -> str:
        return 'SMPL-X'

    @property
    def num_expression_coeffs(self):
        return self._num_expression_coeffs

    def create_mean_pose(self, data_struct, flat_hand_mean=False):
        # Create the array for the mean pose. If flat_hand is false, then use
        # the mean that is given by the data, rather than the flat open hand
        global_orient_mean = torch.zeros([3], dtype=self.dtype)
        body_pose_mean = torch.zeros([self.NUM_BODY_JOINTS * 3],
                                     dtype=self.dtype)
        jaw_pose_mean = torch.zeros([3], dtype=self.dtype)
        leye_pose_mean = torch.zeros([3], dtype=self.dtype)
        reye_pose_mean = torch.zeros([3], dtype=self.dtype)

        pose_mean = np.concatenate([global_orient_mean, body_pose_mean,
                                    jaw_pose_mean,
                                    leye_pose_mean, reye_pose_mean,
                                    self.left_hand_mean, self.right_hand_mean],
                                   axis=0)

        return pose_mean

    def extra_repr(self):
        msg = super().extra_repr()
        msg = [
            msg,
            f'Number of Expression Coefficients: {self.num_expression_coeffs}'
        ]
        return '\n'.join(msg)

    def forward(
        self,
        betas: Optional[Tensor] = None,
        global_orient: Optional[Tensor] = None,
        body_pose: Optional[Tensor] = None,
        left_hand_pose: Optional[Tensor] = None,
        right_hand_pose: Optional[Tensor] = None,
        transl: Optional[Tensor] = None,
        expression: Optional[Tensor] = None,
        jaw_pose: Optional[Tensor] = None,
        leye_pose: Optional[Tensor] = None,
        reye_pose: Optional[Tensor] = None,
        return_verts: bool = True,
        return_full_pose: bool = False,
        pose2rot: bool = True,
        return_shaped: bool = True,
        **kwargs
    ) -> SmplXOutput:
        '''
        Forward pass for the SMPLX model

            Parameters
            ----------
            global_orient: torch.tensor, optional, shape Bx3
                If given, ignore the member variable and use it as the global
                rotation of the body. Useful if someone wishes to predicts this
                with an external model. (default=None)
            betas: torch.tensor, optional, shape BxN_b
                If given, ignore the member variable `betas` and use it
                instead. For example, it can used if shape parameters
                `betas` are predicted from some external model.
                (default=None)
            expression: torch.tensor, optional, shape BxN_e
                If given, ignore the member variable `expression` and use it
                instead. For example, it can used if expression parameters
                `expression` are predicted from some external model.
            body_pose: torch.tensor, optional, shape Bx(J*3)
                If given, ignore the member variable `body_pose` and use it
                instead. For example, it can used if someone predicts the
                pose of the body joints are predicted from some external model.
                It should be a tensor that contains joint rotations in
                axis-angle format. (default=None)
            left_hand_pose: torch.tensor, optional, shape BxP
                If given, ignore the member variable `left_hand_pose` and
                use this instead. It should either contain PCA coefficients or
                joint rotations in axis-angle format.
            right_hand_pose: torch.tensor, optional, shape BxP
                If given, ignore the member variable `right_hand_pose` and
                use this instead. It should either contain PCA coefficients or
                joint rotations in axis-angle format.
            jaw_pose: torch.tensor, optional, shape Bx3
                If given, ignore the member variable `jaw_pose` and
                use this instead. It should either joint rotations in
                axis-angle format.
            transl: torch.tensor, optional, shape Bx3
                If given, ignore the member variable `transl` and use it
                instead. For example, it can used if the translation
                `transl` is predicted from some external model.
                (default=None)
            return_verts: bool, optional
                Return the vertices. (default=True)
            return_full_pose: bool, optional
                Returns the full axis-angle pose vector (default=False)

            Returns
            -------
                output: ModelOutput
                A named tuple of type `ModelOutput`
        '''

        # If no shape and pose parameters are passed along, then use the
        # ones from the module
        body_pose = body_pose if body_pose is not None else self.body_pose
        bn = body_pose.shape[0]
        global_orient = (global_orient if global_orient is not None else
                         self.global_orient[:bn])
        betas = betas if betas is not None else self.betas[:bn]
        left_hand_pose = (left_hand_pose if left_hand_pose is not None else
                          self.left_hand_pose)
        right_hand_pose = (right_hand_pose if right_hand_pose is not None else
                           self.right_hand_pose)
        jaw_pose = jaw_pose if jaw_pose is not None else self.jaw_pose[:bn]
        leye_pose = leye_pose if leye_pose is not None else self.leye_pose[:bn]
        reye_pose = reye_pose if reye_pose is not None else self.reye_pose[:bn]
        expression = expression if expression is not None else self.expression[:bn]

        apply_trans = transl is not None or hasattr(self, 'transl')
        if transl is None:
            if hasattr(self, 'transl'):
                transl = self.transl[:self.bn]

        if self.use_pca:
            left_hand_pose = torch.einsum(
                'bi,ij->bj', [left_hand_pose, self.left_hand_components])
            right_hand_pose = torch.einsum(
                'bi,ij->bj', [right_hand_pose, self.right_hand_components])

        full_pose = torch.cat([global_orient.reshape(-1, 1, 3)[:bn],
                               body_pose.reshape(-1, self.NUM_BODY_JOINTS, 3)[:bn],
                               jaw_pose.reshape(-1, 1, 3)[:bn],
                               leye_pose.reshape(-1, 1, 3)[:bn],
                               reye_pose.reshape(-1, 1, 3)[:bn],
                               left_hand_pose.reshape(-1, 15, 3)[:bn],
                               right_hand_pose.reshape(-1, 15, 3)[:bn]],
                              dim=1).reshape(-1, 165)

        # Add the mean pose of the model. Does not affect the body, only the
        # hands when flat_hand_mean == False
        full_pose += self.pose_mean

        batch_size = max(betas.shape[0], global_orient.shape[0],
                         body_pose.shape[0])
        # Concatenate the shape and expression coefficients
        scale = int(batch_size / betas.shape[0])
        if scale > 1:
            betas = betas.expand(scale, -1)
        shape_components = torch.cat([betas, expression], dim=-1)

        shapedirs = torch.cat([self.shapedirs, self.expr_dirs], dim=-1)

        vertices, joints = lbs(shape_components, full_pose, self.v_template,
                               shapedirs, self.posedirs,
                               self.J_regressor, self.parents,
                               self.lbs_weights, pose2rot=pose2rot,
                               )

        lmk_faces_idx = self.lmk_faces_idx.unsqueeze(
            dim=0).expand(batch_size, -1).contiguous()
        lmk_bary_coords = self.lmk_bary_coords.unsqueeze(dim=0).repeat(
            batch_size, 1, 1)
        if self.use_face_contour:
            lmk_idx_and_bcoords = find_dynamic_lmk_idx_and_bcoords(
                vertices, full_pose, self.dynamic_lmk_faces_idx,
                self.dynamic_lmk_bary_coords,
                self.neck_kin_chain,
                pose2rot=True,
            )
            dyn_lmk_faces_idx, dyn_lmk_bary_coords = lmk_idx_and_bcoords

            lmk_faces_idx = torch.cat([lmk_faces_idx,
                                       dyn_lmk_faces_idx], 1)
            lmk_bary_coords = torch.cat(
                [lmk_bary_coords.expand(batch_size, -1, -1),
                 dyn_lmk_bary_coords], 1)

        landmarks = vertices2landmarks(vertices, self.faces_tensor,
                                       lmk_faces_idx,
                                       lmk_bary_coords)

        # Add any extra joints that might be needed
        joints = self.vertex_joint_selector(vertices, joints)
        # Add the landmarks to the joints
        joints = torch.cat([joints, landmarks], dim=1)
        # Map the joints to the current dataset

        if self.joint_mapper is not None:
            joints = self.joint_mapper(joints=joints, vertices=vertices)

        if self.J_regressor_h36m17 is not None:
            # 54 joints = 45 joints + 9 extra joints from different datasets
            joints_h36m17 = vertices2joints(self.J_regressor_h36m17, vertices)
            # use the middle of hip used in the most 2D pose datasets, not the o-th Pelvis of SMPL 24 joint
            joints_h36m17_pelvis = joints_h36m17[:, 14].unsqueeze(1)
            joints_h36m17 = joints_h36m17 - joints_h36m17_pelvis

        if apply_trans:
            joints += transl.unsqueeze(dim=1)
            vertices += transl.unsqueeze(dim=1)
            if self.J_regressor_h36m17 is not None:
                joints_h36m17 += transl.unsqueeze(dim=1)

        v_shaped = None
        if return_shaped:
            v_shaped = self.v_template + blend_shapes(betas, self.shapedirs)
        output = SmplXOutput(vertices=vertices if return_verts else None,
                             joints=joints,
                             betas=betas,
                             expression=expression,
                             global_orient=global_orient,
                             body_pose=body_pose,
                             left_hand_pose=left_hand_pose,
                             right_hand_pose=right_hand_pose,
                             jaw_pose=jaw_pose,
                             v_shaped=v_shaped,
                             full_pose=full_pose if return_full_pose else None,
                             joints_h36m17=joints_h36m17 if self.J_regressor_h36m17 is not None else None)
        return output


class SMPLXLayer(SMPLX):
    def __init__(
        self,
        *args,
        **kwargs
    ) -> None:
        # Just create a SMPLX module without any member variables
        super().__init__(
            create_global_orient=False,
            create_body_pose=False,
            create_left_hand_pose=False,
            create_right_hand_pose=False,
            create_jaw_pose=False,
            create_leye_pose=False,
            create_reye_pose=False,
            create_betas=False,
            create_expression=False,
            create_transl=False,
            *args, **kwargs,
        )

    def forward(
        self,
        betas: Optional[Tensor] = None,
        global_orient: Optional[Tensor] = None,
        body_pose: Optional[Tensor] = None,
        left_hand_pose: Optional[Tensor] = None,
        right_hand_pose: Optional[Tensor] = None,
        transl: Optional[Tensor] = None,
        expression: Optional[Tensor] = None,
        jaw_pose: Optional[Tensor] = None,
        leye_pose: Optional[Tensor] = None,
        reye_pose: Optional[Tensor] = None,
        return_verts: bool = True,
        return_full_pose: bool = False,
        **kwargs
    ) -> SmplXOutput:
        '''
        Forward pass for the SMPLX model

            Parameters
            ----------
            global_orient: torch.tensor, optional, shape Bx3x3
                If given, ignore the member variable and use it as the global
                rotation of the body. Useful if someone wishes to predicts this
                with an external model. It is expected to be in rotation matrix
                format. (default=None)
            betas: torch.tensor, optional, shape BxN_b
                If given, ignore the member variable `betas` and use it
                instead. For example, it can used if shape parameters
                `betas` are predicted from some external model.
                (default=None)
            expression: torch.tensor, optional, shape BxN_e
                Expression coefficients.
                For example, it can used if expression parameters
                `expression` are predicted from some external model.
            body_pose: torch.tensor, optional, shape BxJx3x3
                If given, ignore the member variable `body_pose` and use it
                instead. For example, it can used if someone predicts the
                pose of the body joints are predicted from some external model.
                It should be a tensor that contains joint rotations in
                rotation matrix format. (default=None)
            left_hand_pose: torch.tensor, optional, shape Bx15x3x3
                If given, contains the pose of the left hand.
                It should be a tensor that contains joint rotations in
                rotation matrix format. (default=None)
            right_hand_pose: torch.tensor, optional, shape Bx15x3x3
                If given, contains the pose of the right hand.
                It should be a tensor that contains joint rotations in
                rotation matrix format. (default=None)
            jaw_pose: torch.tensor, optional, shape Bx3x3
                Jaw pose. It should either joint rotations in
                rotation matrix format.
            transl: torch.tensor, optional, shape Bx3
                Translation vector of the body.
                For example, it can used if the translation
                `transl` is predicted from some external model.
                (default=None)
            return_verts: bool, optional
                Return the vertices. (default=True)
            return_full_pose: bool, optional
                Returns the full pose vector (default=False)
            Returns
            -------
                output: ModelOutput
                A data class that contains the posed vertices and joints
        '''
        device, dtype = self.shapedirs.device, self.shapedirs.dtype

        model_vars = [betas, global_orient, body_pose, transl,
                      expression, left_hand_pose, right_hand_pose, jaw_pose]
        batch_size = 1
        for var in model_vars:
            if var is None:
                continue
            batch_size = max(batch_size, len(var))

        if global_orient is None:
            global_orient = torch.eye(3, device=device, dtype=dtype).view(
                1, 1, 3, 3).expand(batch_size, -1, -1, -1).contiguous()
        if body_pose is None:
            body_pose = torch.eye(3, device=device, dtype=dtype).view(
                1, 1, 3, 3).expand(
                    batch_size, self.NUM_BODY_JOINTS, -1, -1).contiguous()
        if left_hand_pose is None:
            left_hand_pose = torch.eye(3, device=device, dtype=dtype).view(
                1, 1, 3, 3).expand(batch_size, 15, -1, -1).contiguous()
        if right_hand_pose is None:
            right_hand_pose = torch.eye(3, device=device, dtype=dtype).view(
                1, 1, 3, 3).expand(batch_size, 15, -1, -1).contiguous()
        if jaw_pose is None:
            jaw_pose = torch.eye(3, device=device, dtype=dtype).view(
                1, 1, 3, 3).expand(batch_size, -1, -1, -1).contiguous()
        if leye_pose is None:
            leye_pose = torch.eye(3, device=device, dtype=dtype).view(
                1, 1, 3, 3).expand(batch_size, -1, -1, -1).contiguous()
        if reye_pose is None:
            reye_pose = torch.eye(3, device=device, dtype=dtype).view(
                1, 1, 3, 3).expand(batch_size, -1, -1, -1).contiguous()
        if expression is None:
            expression = torch.zeros([batch_size, self.num_expression_coeffs],
                                     dtype=dtype, device=device)
        if betas is None:
            betas = torch.zeros([batch_size, self.num_betas],
                                dtype=dtype, device=device)
        if transl is None:
            transl = torch.zeros([batch_size, 3], dtype=dtype, device=device)

        # Concatenate all pose vectors
        full_pose = torch.cat(
            [global_orient.reshape(-1, 1, 3, 3),
             body_pose.reshape(-1, self.NUM_BODY_JOINTS, 3, 3),
             jaw_pose.reshape(-1, 1, 3, 3),
             leye_pose.reshape(-1, 1, 3, 3),
             reye_pose.reshape(-1, 1, 3, 3),
             left_hand_pose.reshape(-1, self.NUM_HAND_JOINTS, 3, 3),
             right_hand_pose.reshape(-1, self.NUM_HAND_JOINTS, 3, 3)],
            dim=1)
        shape_components = torch.cat([betas, expression], dim=-1)

        shapedirs = torch.cat([self.shapedirs, self.expr_dirs], dim=-1)

        vertices, joints = lbs(shape_components, full_pose, self.v_template,
                               shapedirs, self.posedirs,
                               self.J_regressor, self.parents,
                               self.lbs_weights,
                               pose2rot=False,
                               )

        lmk_faces_idx = self.lmk_faces_idx.unsqueeze(
            dim=0).expand(batch_size, -1).contiguous()
        lmk_bary_coords = self.lmk_bary_coords.unsqueeze(dim=0).repeat(
            batch_size, 1, 1)
        if self.use_face_contour:
            lmk_idx_and_bcoords = find_dynamic_lmk_idx_and_bcoords(
                vertices, full_pose,
                self.dynamic_lmk_faces_idx,
                self.dynamic_lmk_bary_coords,
                self.neck_kin_chain,
                pose2rot=False,
            )
            dyn_lmk_faces_idx, dyn_lmk_bary_coords = lmk_idx_and_bcoords

            lmk_faces_idx = torch.cat([lmk_faces_idx, dyn_lmk_faces_idx], 1)
            lmk_bary_coords = torch.cat(
                [lmk_bary_coords.expand(batch_size, -1, -1),
                 dyn_lmk_bary_coords], 1)

        landmarks = vertices2landmarks(vertices, self.faces_tensor,
                                       lmk_faces_idx,
                                       lmk_bary_coords)

        # Add any extra joints that might be needed
        joints = self.vertex_joint_selector(vertices, joints)
        # Add the landmarks to the joints
        joints = torch.cat([joints, landmarks], dim=1)
        # Map the joints to the current dataset

        if self.joint_mapper is not None:
            joints = self.joint_mapper(joints=joints, vertices=vertices)

        if self.J_regressor_h36m17 is not None:
            # 54 joints = 45 joints + 9 extra joints from different datasets
            joints_h36m17 = vertices2joints(self.J_regressor_h36m17, vertices)
            # use the middle of hip used in the most 2D pose datasets, not the o-th Pelvis of SMPL 24 joint
            joints_h36m17_pelvis = joints_h36m17[:, 14].unsqueeze(1)
            joints_h36m17 = joints_h36m17 - joints_h36m17_pelvis

        if transl is not None:
            joints += transl.unsqueeze(dim=1)
            vertices += transl.unsqueeze(dim=1)
            if self.J_regressor_h36m17 is not None:
                joints_h36m17 += transl.unsqueeze(dim=1)

        output = SmplXOutput(vertices=vertices if return_verts else None,
                             joints=joints,
                             betas=betas,
                             expression=expression,
                             global_orient=global_orient,
                             body_pose=body_pose,
                             left_hand_pose=left_hand_pose,
                             right_hand_pose=right_hand_pose,
                             jaw_pose=jaw_pose,
                             transl=transl,
                             full_pose=full_pose if return_full_pose else None,
                             joints_h36m17=joints_h36m17 if self.J_regressor_h36m17 is not None else None)
        return output


@dataclass
class SmplFkOutput(SmplOutput):
    transforms: Optional[Tensor] = None


class SmplFK(SMPL):
    def __init__(self, models_path: str, model_name: str):
        smpl_downloader = SmplModelDownloader(models_path=hydra.utils.to_absolute_path(models_path))
        model_path = smpl_downloader.pull(model_name)
        J_reg_extra9_path = smpl_downloader.pull("J_regressor_extra")
        J_reg_h36m17_path = smpl_downloader.pull("J_regressor_h36m")
        super().__init__(model_path=model_path,
                         J_reg_extra9_path=J_reg_extra9_path,
                         J_reg_h36m17_path=J_reg_h36m17_path)
        

    def forward(
        self,
        betas: Optional[Tensor] = None,
        body_pose: Optional[Tensor] = None,
        global_orient: Optional[Tensor] = None,
        transl: Optional[Tensor] = None,
        return_verts=True,
        return_full_pose: bool = False,
        pose2rot: bool = True,
        **kwargs
    ) -> SmplFkOutput:
        # If no shape and pose parameters are passed along, then use the
        # ones from the module
        global_orient = (global_orient if global_orient is not None else
                         self.global_orient)
        body_pose = body_pose if body_pose is not None else self.body_pose
        betas = betas if betas is not None else self.betas

        apply_trans = transl is not None or hasattr(self, 'transl')
        if transl is None and hasattr(self, 'transl'):
            transl = self.transl

        full_pose = torch.cat([global_orient, body_pose], dim=1)

        vertices, joints, transforms = lbsFK(betas, full_pose, self.v_template, self.shapedirs, self.posedirs,
                                             self.J_regressor, self.parents, self.lbs_weights, pose2rot=pose2rot)

        joints = self.vertex_joint_selector(vertices, joints)
        # Map the joints to the current dataset
        if self.joint_mapper is not None:
            joints = self.joint_mapper(joints)
        
        if self.J_regressor_h36m17 is not None:
            # 54 joints = 45 joints + 9 extra joints from different datasets
            joints_h36m17 = vertices2joints(self.J_regressor_h36m17, vertices)
            # use the middle of hip used in the most 2D pose datasets, not the o-th Pelvis of SMPL 24 joint
            joints_h36m17_pelvis = joints_h36m17[:, 14].unsqueeze(1)
            joints_h36m17 = joints_h36m17 - joints_h36m17_pelvis
        
        if self.J_regressor_extra9 is not None:
            # 54 joints = 45 joints + 9 extra joints from different datasets
            joints = torch.cat([joints, vertices2joints(self.J_regressor_extra9, vertices)], 1)

        if apply_trans:
            joints += transl.unsqueeze(dim=1)
            vertices += transl.unsqueeze(dim=1)
            if self.J_regressor_h36m17 is not None:
                joints_h36m17 += transl.unsqueeze(dim=1)

        output = SmplFkOutput(vertices=vertices if return_verts else None,
                              global_orient=global_orient,
                              body_pose=body_pose,
                              joints=joints,
                              betas=betas,
                              full_pose=full_pose if return_full_pose else None,
                              joints_h36m17=joints_h36m17 if self.J_regressor_h36m17 is not None else None,
                              transforms=transforms)

        return output


@dataclass
class SmplHFkOutput(SmplHOutput):
    transforms: Optional[Tensor] = None


class SmplHFK(SMPLH):
    def __init__(self, models_path: str):
        # smpl_downloader = SmplModelDownloader(models_path=hydra.utils.to_absolute_path(models_path))
        # filepath = smpl_downloader.pull(model_name)
        # super().__init__(filepath)
        super().__init__(models_path, create_transl=False)

    def forward(
        self,
        betas: Optional[Tensor] = None,
        global_orient: Optional[Tensor] = None,
        body_pose: Optional[Tensor] = None,
        left_hand_pose: Optional[Tensor] = None,
        right_hand_pose: Optional[Tensor] = None,
        transl: Optional[Tensor] = None,
        return_verts: bool = True,
        return_full_pose: bool = False,
        pose2rot: bool = True,
        **kwargs
    ) -> SmplHFkOutput:
        # If no shape and pose parameters are passed along, then use the
        # ones from the module
        body_pose = body_pose if body_pose is not None else self.body_pose
        bn = body_pose.shape[0]
        global_orient = (global_orient if global_orient is not None else
                         self.global_orient[:bn])
        betas = betas if betas is not None else self.betas[:bn]
        left_hand_pose = (left_hand_pose if left_hand_pose is not None else
                          self.left_hand_pose[:bn])
        right_hand_pose = (right_hand_pose if right_hand_pose is not None else
                           self.right_hand_pose[:bn])

        apply_trans = transl is not None or hasattr(self, 'transl')
        if transl is None:
            if hasattr(self, 'transl'):
                transl = self.transl[:bn]

        if self.use_pca:
            left_hand_pose = torch.einsum(
                'bi,ij->bj', [left_hand_pose, self.left_hand_components])
            right_hand_pose = torch.einsum(
                'bi,ij->bj', [right_hand_pose, self.right_hand_components])

        full_pose = torch.cat([global_orient[:bn], body_pose[:bn],
                               left_hand_pose[:bn],
                               right_hand_pose[:bn]], dim=1)
        full_pose += self.pose_mean

        vertices, joints, transforms = lbsFK(betas, full_pose, self.v_template,
                                             self.shapedirs, self.posedirs,
                                             self.J_regressor, self.parents,
                                             self.lbs_weights, pose2rot=pose2rot)

        # Add any extra joints that might be needed
        joints = self.vertex_joint_selector(vertices, joints)
        if self.joint_mapper is not None:
            joints = self.joint_mapper(joints)

        if self.J_regressor_h36m17 is not None:
            # 54 joints = 45 joints + 9 extra joints from different datasets
            joints_h36m17 = vertices2joints(self.J_regressor_h36m17, vertices)
            # use the middle of hip used in the most 2D pose datasets, not the o-th Pelvis of SMPL 24 joint
            joints_h36m17_pelvis = joints_h36m17[:, 14].unsqueeze(1)
            joints_h36m17 = joints_h36m17 - joints_h36m17_pelvis

        if apply_trans:
            joints += transl.unsqueeze(dim=1)
            vertices += transl.unsqueeze(dim=1)
            if self.J_regressor_h36m17 is not None:
                joints_h36m17 += transl.unsqueeze(dim=1)

        output = SmplHFkOutput(vertices=vertices if return_verts else None,
                               joints=joints,
                               betas=betas,
                               global_orient=global_orient,
                               body_pose=body_pose,
                               left_hand_pose=left_hand_pose,
                               right_hand_pose=right_hand_pose,
                               full_pose=full_pose if return_full_pose else None,
                               joints_h36m17=joints_h36m17 if self.J_regressor_h36m17 is not None else None,
                               transforms=transforms)

        return output


@dataclass
class SmplXFkOutput(SmplXOutput):
    transforms: Optional[Tensor] = None


class SmplXFK(SMPLX):
    def __init__(self, models_path: str):
        # smpl_downloader = SmplModelDownloader(models_path=hydra.utils.to_absolute_path(models_path))
        # filepath = smpl_downloader.pull(model_name)
        # super().__init__(filepath)
        super().__init__(models_path, create_transl=False)

    def forward(
        self,
        betas: Optional[Tensor] = None,
        global_orient: Optional[Tensor] = None,
        body_pose: Optional[Tensor] = None,
        left_hand_pose: Optional[Tensor] = None,
        right_hand_pose: Optional[Tensor] = None,
        transl: Optional[Tensor] = None,
        expression: Optional[Tensor] = None,
        jaw_pose: Optional[Tensor] = None,
        leye_pose: Optional[Tensor] = None,
        reye_pose: Optional[Tensor] = None,
        return_verts: bool = True,
        return_full_pose: bool = False,
        pose2rot: bool = True,
        return_shaped: bool = True,
        **kwargs
    ) -> SmplXFkOutput:
        '''
        Forward pass for the SMPLX model

            Parameters
            ----------
            global_orient: torch.tensor, optional, shape Bx3
                If given, ignore the member variable and use it as the global
                rotation of the body. Useful if someone wishes to predicts this
                with an external model. (default=None)
            betas: torch.tensor, optional, shape BxN_b
                If given, ignore the member variable `betas` and use it
                instead. For example, it can used if shape parameters
                `betas` are predicted from some external model.
                (default=None)
            expression: torch.tensor, optional, shape BxN_e
                If given, ignore the member variable `expression` and use it
                instead. For example, it can used if expression parameters
                `expression` are predicted from some external model.
            body_pose: torch.tensor, optional, shape Bx(J*3)
                If given, ignore the member variable `body_pose` and use it
                instead. For example, it can used if someone predicts the
                pose of the body joints are predicted from some external model.
                It should be a tensor that contains joint rotations in
                axis-angle format. (default=None)
            left_hand_pose: torch.tensor, optional, shape BxP
                If given, ignore the member variable `left_hand_pose` and
                use this instead. It should either contain PCA coefficients or
                joint rotations in axis-angle format.
            right_hand_pose: torch.tensor, optional, shape BxP
                If given, ignore the member variable `right_hand_pose` and
                use this instead. It should either contain PCA coefficients or
                joint rotations in axis-angle format.
            jaw_pose: torch.tensor, optional, shape Bx3
                If given, ignore the member variable `jaw_pose` and
                use this instead. It should either joint rotations in
                axis-angle format.
            transl: torch.tensor, optional, shape Bx3
                If given, ignore the member variable `transl` and use it
                instead. For example, it can used if the translation
                `transl` is predicted from some external model.
                (default=None)
            return_verts: bool, optional
                Return the vertices. (default=True)
            return_full_pose: bool, optional
                Returns the full axis-angle pose vector (default=False)

            Returns
            -------
                output: ModelOutput
                A named tuple of type `ModelOutput`
        '''

        # If no shape and pose parameters are passed along, then use the
        # ones from the module
        body_pose = body_pose if body_pose is not None else self.body_pose
        bn = body_pose.shape[0]
        global_orient = (global_orient if global_orient is not None else
                         self.global_orient[:bn])
        betas = betas if betas is not None else self.betas[:bn]
        left_hand_pose = (left_hand_pose if left_hand_pose is not None else
                          self.left_hand_pose)
        right_hand_pose = (right_hand_pose if right_hand_pose is not None else
                           self.right_hand_pose)
        jaw_pose = jaw_pose if jaw_pose is not None else self.jaw_pose[:bn]
        leye_pose = leye_pose if leye_pose is not None else self.leye_pose[:bn]
        reye_pose = reye_pose if reye_pose is not None else self.reye_pose[:bn]
        expression = expression if expression is not None else self.expression[:bn]

        apply_trans = transl is not None or hasattr(self, 'transl')
        if transl is None:
            if hasattr(self, 'transl'):
                transl = self.transl[:self.bn]

        if self.use_pca:
            left_hand_pose = torch.einsum(
                'bi,ij->bj', [left_hand_pose, self.left_hand_components])
            right_hand_pose = torch.einsum(
                'bi,ij->bj', [right_hand_pose, self.right_hand_components])

        full_pose = torch.cat([global_orient.reshape(-1, 1, 3)[:bn],
                               body_pose.reshape(-1, self.NUM_BODY_JOINTS, 3)[:bn],
                               jaw_pose.reshape(-1, 1, 3)[:bn],
                               leye_pose.reshape(-1, 1, 3)[:bn],
                               reye_pose.reshape(-1, 1, 3)[:bn],
                               left_hand_pose.reshape(-1, 15, 3)[:bn],
                               right_hand_pose.reshape(-1, 15, 3)[:bn]],
                              dim=1).reshape(-1, 165)

        # Add the mean pose of the model. Does not affect the body, only the
        # hands when flat_hand_mean == False
        full_pose += self.pose_mean

        batch_size = max(betas.shape[0], global_orient.shape[0],
                         body_pose.shape[0])
        # Concatenate the shape and expression coefficients
        scale = int(batch_size / betas.shape[0])
        if scale > 1:
            betas = betas.expand(scale, -1)
        shape_components = torch.cat([betas, expression], dim=-1)

        shapedirs = torch.cat([self.shapedirs, self.expr_dirs], dim=-1)

        vertices, joints, transforms = lbsFK(shape_components, full_pose, self.v_template,
                                             shapedirs, self.posedirs,
                                             self.J_regressor, self.parents,
                                             self.lbs_weights, pose2rot=pose2rot)

        lmk_faces_idx = self.lmk_faces_idx.unsqueeze(
            dim=0).expand(batch_size, -1).contiguous()
        lmk_bary_coords = self.lmk_bary_coords.unsqueeze(dim=0).repeat(
            batch_size, 1, 1)
        if self.use_face_contour:
            lmk_idx_and_bcoords = find_dynamic_lmk_idx_and_bcoords(
                vertices, full_pose, self.dynamic_lmk_faces_idx,
                self.dynamic_lmk_bary_coords,
                self.neck_kin_chain,
                pose2rot=True,
            )
            dyn_lmk_faces_idx, dyn_lmk_bary_coords = lmk_idx_and_bcoords

            lmk_faces_idx = torch.cat([lmk_faces_idx,
                                       dyn_lmk_faces_idx], 1)
            lmk_bary_coords = torch.cat(
                [lmk_bary_coords.expand(batch_size, -1, -1),
                 dyn_lmk_bary_coords], 1)

        landmarks = vertices2landmarks(vertices, self.faces_tensor,
                                       lmk_faces_idx,
                                       lmk_bary_coords)

        # Add any extra joints that might be needed
        joints = self.vertex_joint_selector(vertices, joints)

        # Add the face landmarks to the joints
        joints = torch.cat([joints, landmarks], dim=1)

        # Map the joints to the current dataset
        if self.joint_mapper is not None:
            joints = self.joint_mapper(joints=joints, vertices=vertices)

        if self.J_regressor_h36m17 is not None:
            # 54 joints = 45 joints + 9 extra joints from different datasets
            joints_h36m17 = vertices2joints(self.J_regressor_h36m17, vertices)
            # use the middle of hip used in the most 2D pose datasets, not the o-th Pelvis of SMPL 24 joint
            joints_h36m17_pelvis = joints_h36m17[:, 14].unsqueeze(1)
            joints_h36m17 = joints_h36m17 - joints_h36m17_pelvis

        if apply_trans:
            joints += transl.unsqueeze(dim=1)
            vertices += transl.unsqueeze(dim=1)
            if self.J_regressor_h36m17 is not None:
                joints_h36m17 += transl.unsqueeze(dim=1)

        v_shaped = None
        if return_shaped:
            v_shaped = self.v_template + blend_shapes(betas, self.shapedirs)

        output = SmplXFkOutput(vertices=vertices if return_verts else None,
                               joints=joints,
                               betas=betas,
                               expression=expression,
                               global_orient=global_orient,
                               body_pose=body_pose,
                               left_hand_pose=left_hand_pose,
                               right_hand_pose=right_hand_pose,
                               jaw_pose=jaw_pose,
                               v_shaped=v_shaped,
                               full_pose=full_pose if return_full_pose else None,
                               joints_h36m17=joints_h36m17 if self.J_regressor_h36m17 is not None else None,
                               transforms=transforms)

        return output
