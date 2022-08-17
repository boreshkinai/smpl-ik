import torch
import numpy as np

from protores.geometry.vector import normalize_vector, cross_product


def get_4x4_rotation_matrix_from_3x3_rotation_matrix(m):
    batch_size = m.shape[0]

    row4 = torch.autograd.Variable(torch.zeros(batch_size, 1, 3).type_as(m))
    m43 = torch.cat((m, row4), 1)  # batch*4,3
    col4 = torch.autograd.Variable(torch.zeros(batch_size, 4, 1).type_as(m))
    col4[:, 3, 0] = col4[:, 3, 0] + 1
    out = torch.cat((m43, col4), 2)  # batch*4*4

    return out

def compute_ortho6d_from_rotation_matrix(mat):
    mat = mat.reshape([-1, 3, 3])
    r6d = torch.cat([mat[..., 0], mat[..., 1]], dim=-1)
    return r6d

# poses batch*6
# poses
def compute_rotation_matrix_from_ortho6d(ortho6d, eps: float = 1e-8):
    x_raw = ortho6d[:, 0:3]  # batch*3
    y_raw = ortho6d[:, 3:6]  # batch*3

    x = normalize_vector(x_raw, eps=eps)  # batch*3
    z = cross_product(x, y_raw)  # batch*3
    z = normalize_vector(z, eps=eps)  # batch*3
    y = cross_product(z, x)  # batch*3

    x = x.view(-1, 3, 1)
    y = y.view(-1, 3, 1)
    z = z.view(-1, 3, 1)
    matrix = torch.cat((x, y, z), 2)  # batch*3*3
    return matrix


# euler batch*4
# output cuda batch*3*3 matrices in the rotation order of XZ'Y'' (intrinsic) or YZX (extrinsic)
def compute_rotation_matrix_from_euler(euler):
    batch = euler.shape[0]

    c1 = torch.cos(euler[:, 0]).view(batch, 1)  # batch*1
    s1 = torch.sin(euler[:, 0]).view(batch, 1)  # batch*1
    c2 = torch.cos(euler[:, 2]).view(batch, 1)  # batch*1
    s2 = torch.sin(euler[:, 2]).view(batch, 1)  # batch*1
    c3 = torch.cos(euler[:, 1]).view(batch, 1)  # batch*1
    s3 = torch.sin(euler[:, 1]).view(batch, 1)  # batch*1

    row1 = torch.cat((c2 * c3, -s2, c2 * s3), 1).view(-1, 1, 3)  # batch*1*3
    row2 = torch.cat((c1 * s2 * c3 + s1 * s3, c1 * c2, c1 * s2 * s3 - s1 * c3), 1).view(-1, 1, 3)  # batch*1*3
    row3 = torch.cat((s1 * s2 * c3 - c1 * s3, s1 * c2, s1 * s2 * s3 + c1 * c3), 1).view(-1, 1, 3)  # batch*1*3

    matrix = torch.cat((row1, row2, row3), 1)  # batch*3*3

    return matrix


# matrices batch*3*3
# both matrix are orthogonal rotation matrices
# out theta between 0 to 180 degree batch
def compute_geodesic_distance_from_two_matrices(m1, m2):
    batch = m1.shape[0]
    m = torch.bmm(m1, m2.transpose(1, 2))  # batch*3*3

    cos = (m[:, 0, 0] + m[:, 1, 1] + m[:, 2, 2] - 1) / 2
    cos = torch.min(cos, torch.autograd.Variable(torch.ones(batch).type_as(m1)))
    cos = torch.max(cos, torch.autograd.Variable(torch.ones(batch).type_as(m1)) * -1)

    theta = torch.acos(cos)

    # theta = torch.min(theta, 2*np.pi - theta)

    return theta


# q of size (N, 4)
# quaternion order is w, x, y, z
# output of size (N, 3, 3)
def compute_rotation_matrix_from_unit_quaternion(q):
    batch = q.shape[0]

    qw = q[..., 0].contiguous().view(batch, 1)
    qx = q[..., 1].contiguous().view(batch, 1)
    qy = q[..., 2].contiguous().view(batch, 1)
    qz = q[..., 3].contiguous().view(batch, 1)

    # Unit quaternion rotation matrices computation
    xx = qx * qx
    yy = qy * qy
    zz = qz * qz
    xy = qx * qy
    xz = qx * qz
    yz = qy * qz
    xw = qx * qw
    yw = qy * qw
    zw = qz * qw

    row0 = torch.cat((1 - 2 * yy - 2 * zz, 2 * xy - 2 * zw, 2 * xz + 2 * yw), 1)  # batch*3
    row1 = torch.cat((2 * xy + 2 * zw, 1 - 2 * xx - 2 * zz, 2 * yz - 2 * xw), 1)  # batch*3
    row2 = torch.cat((2 * xz - 2 * yw, 2 * yz + 2 * xw, 1 - 2 * xx - 2 * yy), 1)  # batch*3

    matrix = torch.cat((row0.view(batch, 1, 3), row1.view(batch, 1, 3), row2.view(batch, 1, 3)), 1)  # batch*3*3

    return matrix


# q of size (N, 4)
# quaternion order is w, x, y, z
# output of size (N, 3, 3)
def compute_rotation_matrix_from_quaternion(q, eps: float = 1e-8):
    return compute_rotation_matrix_from_unit_quaternion(normalize_vector(q, eps=eps).contiguous())


# axis: size (batch, 3)
# output quat order is w, x, y, z
def get_random_rotation_around_axis(axis, return_quaternion=False, eps: float = 1e-8):
    batch = axis.shape[0]
    axis = normalize_vector(axis, eps=eps)  # batch*3
    theta = torch.FloatTensor(axis.shape[0]).uniform_(-np.pi, np.pi).type_as(axis)  # [0, pi] #[-180, 180]
    sin = torch.sin(theta)
    qw = torch.cos(theta)
    qx = axis[:, 0] * sin
    qy = axis[:, 1] * sin
    qz = axis[:, 2] * sin

    quaternion = torch.cat((qw.view(batch, 1), qx.view(batch, 1), qy.view(batch, 1), qz.view(batch, 1)), 1)
    matrix = compute_rotation_matrix_from_unit_quaternion(quaternion)

    if (return_quaternion == True):
        return matrix, quaternion
    else:
        return matrix

# axisAngle batch*4 angle, x,y,z
# output quat order is w, x, y, z
def get_random_rotation_matrices_around_random_axis(batch, return_quaternion=False):
    axis = torch.autograd.Variable(torch.randn(batch.shape[0], 3).type_as(batch))
    return get_random_rotation_around_axis(axis, return_quaternion=return_quaternion)


# matrices batch*3*3
# both matrix are orthogonal rotation matrices
# out theta between 0 to 180 degree batch
def compute_geodesic_distance_from_two_matrices(m1, m2):
    batch = m1.shape[0]
    m = torch.bmm(m1, m2.transpose(1, 2))  # batch*3*3

    cos = (m[:, 0, 0] + m[:, 1, 1] + m[:, 2, 2] - 1) / 2
    cos = torch.min(cos, torch.autograd.Variable(torch.ones(batch).type_as(m1)))
    cos = torch.max(cos, torch.autograd.Variable(torch.ones(batch).type_as(m1)) * -1)

    theta = torch.acos(cos)

    # theta = torch.min(theta, 2*np.pi - theta)

    return theta


def geodesic_loss(gt_r_matrix, out_r_matrix):
    theta = compute_geodesic_distance_from_two_matrices(gt_r_matrix, out_r_matrix)
    error = theta.mean()
    return error


def geodesic_loss_matrix3x3_matrix3x3(gt_r_matrix, out_r_matrix):
    return geodesic_loss(gt_r_matrix, out_r_matrix)
