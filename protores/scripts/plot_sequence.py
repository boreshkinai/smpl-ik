import argparse

import torch
import numpy as np
from plotly import graph_objects as go

from protores.data.dataset.typed_table import TypedColumnSequenceDataset
from protores.geometry.rotations import compute_rotation_matrix_from_quaternion
from protores.geometry.skeleton import Skeleton
from protores.data.datasets import SplitFileDatabaseLoader

from protores.geometry.quaternions import multiply_quaternions, apply_quaternions, compute_quaternions_from_rotation_matrices
from protores.geometry.rotations import compute_rotation_matrix_from_euler


# applies multiple steps of root motion to a full sequence
def apply_root_motion(joint_positions, joint_rotations, velocity, angular_velocity, dt):
    offset, rotations = accumulate_root_motion(velocity, angular_velocity, dt)

    new_joint_positions = apply_quaternions(rotations, joint_positions) + offset
    new_joint_rotations = multiply_quaternions(rotations, joint_rotations)

    return new_joint_positions, new_joint_rotations


def accumulate_root_motion(velocity, angular_velocity, dt):
    euler_rotations = torch.cumsum(dt * angular_velocity, dim=0)
    # rotations = quaternion_from_euler_numpy(euler_rotations.cpu().numpy(), order='zxy')
    # rotations = torch.from_numpy(rotations).float().cuda()
    rotations = compute_quaternions_from_rotation_matrices(compute_rotation_matrix_from_euler(euler_rotations))

    frame_velocity = apply_quaternions(rotations, dt * velocity)
    offset = torch.cumsum(frame_velocity, dim=0)

    return offset, rotations


class AnimationPlot:
    def __init__(self, skeleton, timestep, horizontal_extent, vertical_extent):
        self.skeleton = skeleton
        self.timestep = timestep
        self.horizontal_extent = horizontal_extent
        self.vertical_extent = vertical_extent
        self.frames = None

        self.root_name = None
        for name in self.skeleton.all_joints:
            parent_name = self.skeleton.bone_parent[name]
            if parent_name is None or parent_name is "":
                assert self.root_name is None, "Skeleton can only have one root bone"
                self.root_name = name

    def add_sequence(self, joint_positions, feet_contacts=None, feet_positions=None, root_position=None, root_rotation=None, color='rgb(0,0,0)'):
        frames_data = self._create_plot_sequence(joint_positions, feet_contacts, feet_positions, root_position,
                                                 root_rotation, color=color)
        self.frames = frames_data if self.frames is None else self._merge_plot_frames((self.frames, frames_data))

    def add_ground_sequence(self, height=0.0):
        data = self._create_ground_plot(height)
        for i in range(len(self.frames)):
            self.frames[i] += data

    def add_lookat_sequence(self, head_positions, lookat_directions, vector_length=0.5):
        frames_data = self._create_lookat_plot(head_positions, lookat_directions, vector_length)
        self.frames = frames_data if self.frames is None else self._merge_plot_frames((self.frames, frames_data))

    def plot(self):
        fig = self.create_plot()
        fig.show()

    def create_plot(self):
        fig_dict, sliders_dict = self._create_figure_and_sliders()

        # make data
        times = []
        for i in range(len(self.frames)):
            time = i * self.timestep
            times.append(time)

            frame = {
                "data": self.frames[i],
                "name": str(time)
            }

            if i == 0:
                fig_dict["data"] = frame["data"]
            fig_dict["frames"].append(frame)

            slider_step = {"args": [
                [time],
                {"frame": {"duration": self.timestep * 1000, "redraw": True},
                 "mode": "immediate",
                 "transition": {"duration": 0}}
            ],
                "label": i,
                "method": "animate"}
            sliders_dict["steps"].append(slider_step)

        fig_dict["layout"]["sliders"] = [sliders_dict]

        fig = go.Figure(fig_dict)
        return fig

    def _create_figure_and_sliders(self):
        # make figure
        fig_dict = {
            "data": [],
            "layout": {},
            "frames": []
        }

        # fill in most of layout
        fig_dict["layout"]["font"] = {"size": 18}
        fig_dict["layout"]["scene"] = {"xaxis": {"showticklabels": False, "range": [-self.horizontal_extent, self.horizontal_extent]},
                                       "yaxis": {"showticklabels": False, "range": [-self.horizontal_extent, self.horizontal_extent]},
                                       "zaxis": {"showticklabels": False, "range": [-self.vertical_extent, self.vertical_extent]},
                                       "aspectratio": {"x": 1, "y": 1, "z": self.vertical_extent / self.horizontal_extent}}
        fig_dict["layout"]["updatemenus"] = [
            {
                "buttons": [
                    {
                        "args": [None, {"frame": {"duration": self.timestep * 1000, "redraw": True},
                                        "fromcurrent": True, "transition": {"duration": 0}}],
                        "label": "Play",
                        "method": "animate"
                    },
                    {
                        "args": [[None], {"frame": {"duration": 0, "redraw": True},
                                          "mode": "immediate",
                                          "transition": {"duration": 0}}],
                        "label": "Pause",
                        "method": "animate"
                    }
                ],
                "direction": "left",
                "pad": {"r": 10, "t": 87},
                "showactive": False,
                "type": "buttons",
                "x": 0.1,
                "xanchor": "right",
                "y": 0,
                "yanchor": "top"
            }
        ]

        sliders_dict = {
            "active": 0,
            "yanchor": "top",
            "xanchor": "left",
            "currentvalue": {
                "font": {"size": 20},
                "prefix": "Frame:",
                "visible": True,
                "xanchor": "right"
            },
            "transition": {"duration": 0},
            "pad": {"b": 10, "t": 50},
            "len": 0.9,
            "x": 0.1,
            "y": 0,
            "steps": []
        }

        return fig_dict, sliders_dict

    def _create_plot_sequence(self, joint_positions, feet_contacts, feet_positions, root_position, root_rotation, color):
        # sequence properties
        frames_count = joint_positions.shape[0]

        if root_position is not None and root_rotation is not None:
            root_motion_frames = self._plot_root_motion(root_position, root_rotation, size=1.0, color=color)

        # make data
        frames = []
        for i in range(frames_count):
            frame = []

            # joint positions
            frame += self._plot_skeleton(joint_positions=joint_positions[i, :], color=color)

            # feet contacts
            if feet_contacts is not None and feet_positions is not None:
                frame += self._plot_feet_contacts(feet_positions[i, :], feet_contacts[i, :], size=10.0, color=color)

            # root motion
            if root_position is not None and root_rotation is not None:
                frame += root_motion_frames

            frames.append(frame)

        return frames

    @staticmethod
    def _merge_plot_frames(frame_sequence_list):
        assert len(frame_sequence_list) >= 1, "No frame sequences"

        if len(frame_sequence_list) == 1:
            return frame_sequence_list[0]

        frames_count = len(frame_sequence_list[0])
        for seq in frame_sequence_list:
            assert len(seq) == frames_count, "All frame sequences must have identical length"

        frames = []
        for i in range(frames_count):
            frame_data = []
            for seq in frame_sequence_list:
                frame_data += seq[i]
            frames.append(frame_data)

        return frames

    def _create_lookat_plot(self, head_positions, lookat_directions, length, color='rgb(255,0,0)'):
        frames_count = head_positions.shape[0]

        sx = head_positions.view(-1, 3)[:, 0].cpu().numpy()
        sy = head_positions.view(-1, 3)[:, 2].cpu().numpy()
        sz = head_positions.view(-1, 3)[:, 1].cpu().numpy()
        tx = sx + length * lookat_directions.view(-1, 3)[:, 0].cpu().numpy()
        ty = sy + length * lookat_directions.view(-1, 3)[:, 2].cpu().numpy()
        tz = sz + length * lookat_directions.view(-1, 3)[:, 1].cpu().numpy()

        # make data
        frames = []
        for i in range(frames_count):
            frame = go.Scatter3d(x=[sx[i], tx[i]], y=[sy[i], ty[i]], z=[sz[i], tz[i]], mode="lines", line={"color": color})
            frames.append([frame])

        return frames

    def _create_ground_plot(self, height=0.0):
        ground_steps = 2
        ground_x = np.linspace(-self.horizontal_extent, self.horizontal_extent, ground_steps)
        ground_y = ground_x
        ground_z = height * np.ones((ground_steps, ground_steps))

        return [go.Surface(x=ground_x, y=ground_y, z=ground_z, showscale=False, opacity=0.3)]

    @staticmethod
    def _plot_root_motion(root_position, root_rotation, size, color):
        px = root_position.view(-1, 3)[:, 0].cpu().numpy()
        py = root_position.view(-1, 3)[:, 2].cpu().numpy()
        pz = root_position.view(-1, 3)[:, 1].cpu().numpy()

        return [go.Scatter3d(x=px, y=py, z=pz, mode='markers', marker=dict(size=size, color=color, opacity=1.0))]

    @staticmethod
    def _plot_feet_contacts(feet_positions, feet_contacts, size, color):
        px = feet_positions.view(-1, 3)[:, 0].cpu().numpy()
        py = feet_positions.view(-1, 3)[:, 2].cpu().numpy()
        pz = feet_positions.view(-1, 3)[:, 1].cpu().numpy()
        ps = size * feet_contacts.cpu().numpy()

        return [go.Scatter3d(x=px, y=py, z=pz, mode='markers', marker=dict(size=ps, color=color, opacity=1.0))]

    def _plot_skeleton(self, joint_positions, color):
        px = joint_positions.view(-1, 3)[:, 0].cpu().numpy()
        py = joint_positions.view(-1, 3)[:, 2].cpu().numpy()
        pz = joint_positions.view(-1, 3)[:, 1].cpu().numpy()
        return self._generate_skeleton_lines(px=px, py=py, pz=pz, color=color)

    def _generate_skeleton_lines(self, px, py, pz, color='rgb(0,0,0)'):
        plots = []
        self._generate_skeleton_lines_rec(px, py, pz, self.root_name, plots, [], [], [], color)
        return plots

    def _generate_skeleton_lines_rec(self, px, py, pz, parent_name, plots, all_x, all_y, all_z, color):
        parent_idx = self.skeleton.bone_indexes[parent_name]
        parent_x = px[parent_idx]
        parent_y = py[parent_idx]
        parent_z = pz[parent_idx]

        children = self.skeleton.bone_children.get(parent_name, [])

        if len(children) == 0 or len(children) > 1:
            if len(all_x) > 0:
                data_dict = go.Scatter3d(x=all_x, y=all_y, z=all_z, mode="lines", line={"color": color})
                plots.append(data_dict)
                all_x = []
                all_y = []
                all_z = []

        for child_name in children:
            child_idx = self.skeleton.bone_indexes[child_name]
            child_x = px[child_idx]
            child_y = py[child_idx]
            child_z = pz[child_idx]

            if len(all_x) == 0:
                self._generate_skeleton_lines_rec(px, py, pz, child_name, plots,
                                             [parent_x, child_x], [parent_y, child_y], [parent_z, child_z], color)
            else:
                all_x.append(child_x)
                all_y.append(child_y)
                all_z.append(child_z)
                self._generate_skeleton_lines_rec(px, py, pz, child_name, plots, all_x, all_y, all_z, color)


def plot_dataset_sequence(dataset, idx: int = 0, frame_rate: int = 60, plot_ground: bool = True,
                          plot_joint_positions: bool = True, plot_joint_rotations: bool = True,
                          plot_joint_rotations_after_root_motion: bool = True, plot_contacts: bool = True,
                          plot_lookat: bool = True):

    sequence_data = dataset.__getitem__(idx)

    # dataset properties
    timestep = 1.0 / frame_rate  # TODO: use data timestamp instead
    frames_count = sequence_data.shape[0]
    skeleton_data = dataset.config["skeleton"]
    skeleton = Skeleton(skeleton_data)

    # list of all joint names
    all_joints = skeleton.all_joints

    # root joint
    root_name = "Hips"
    root_pos_idx = dataset.get_feature_indices(["BonePositions"], [root_name])
    root_positions = sequence_data[:, root_pos_idx]
    root_rot_idx = dataset.get_feature_indices(["BoneRotations"], [root_name])
    root_rotations = sequence_data[:, root_rot_idx][:, [3, 0, 1, 2]]  # Change quat from x, y, z, w to w, x, y, z

    # joint positions in skeleton order
    joint_pos_idx = dataset.get_feature_indices(["BonePositions"], all_joints)
    joint_pos_idx_sorted = np.argsort(np.array(joint_pos_idx).reshape(-1, 3), axis=0)[:, 0]
    joint_positions = sequence_data[:, joint_pos_idx].view(-1, skeleton.nb_joints, 3)[:, joint_pos_idx_sorted, :].view(
        -1, skeleton.nb_joints * 3)

    # feet contact
    if plot_contacts:
        feet_names = ["ToeLeft", "ToeLeftEnd", "ToeRight", "ToeRightEnd"]
        feed_idx = []
        for foot_name in feet_names:
            idx = skeleton.bone_indexes[foot_name]
            feed_idx += [idx * 3, idx * 3 + 1, idx * 3 + 2]
        feet_positions = joint_positions[:, feed_idx]
        feet_contact_idx = dataset.get_feature_indices(["FootContact"], feet_names)
        feet_contacts = sequence_data[:, feet_contact_idx]
    else:
        feet_contacts = None
        feet_positions = None

    # joint rotations in skeleton order
    joint_rot_idx = dataset.get_feature_indices(["BoneRotations"], all_joints)
    joint_rot_idx_sorted = np.argsort(np.array(joint_rot_idx).reshape(-1, 4), axis=0)[:, 0]
    joint_rotations = sequence_data[:, joint_rot_idx].view(-1, skeleton.nb_joints, 4)[:, joint_rot_idx_sorted, :][:, :,
                      [3, 0, 1, 2]].view(
        -1, skeleton.nb_joints * 4)  # Change quat from x, y, z, w to w, x, y, z

    # root motion
    if plot_joint_rotations_after_root_motion:
        root_velocity_idx = dataset.get_feature_indices(["RootMotion"], ["Velocity"])
        root_velocity = sequence_data[:, root_velocity_idx]
        root_angular_velocity_idx = dataset.get_feature_indices(["RootMotion"], ["AngularVelocity"])
        root_angular_velocity = sequence_data[:, root_angular_velocity_idx]
        root_positions_with_motion, root_rotations_with_motion = apply_root_motion(root_positions, root_rotations,
                                                                                   root_velocity, root_angular_velocity,
                                                                                   timestep)

    # forward kinematics
    rot_shaped = joint_rotations.view(-1, 4)
    joint_rotations_matrices = compute_rotation_matrix_from_quaternion(rot_shaped).view(-1, skeleton.nb_joints, 3, 3)
    joint_fk_position, joint_fk_rotation = skeleton.forward(joint_rotations_matrices, root_positions)
    joint_fk_position = joint_fk_position.view(-1, skeleton.nb_joints * 3)

    if plot_joint_rotations_after_root_motion:
        skeleton_root_idx = skeleton.bone_indexes[root_name]
        skeleton_root_rot_idx = range(skeleton_root_idx * 4, (skeleton_root_idx + 1) * 4)
        joint_rotations[:, skeleton_root_rot_idx] = root_rotations_with_motion

        rot_shaped = joint_rotations.view(-1, 4)
        joint_rotations_matrices = compute_rotation_matrix_from_quaternion(rot_shaped).view(-1, skeleton.nb_joints, 3, 3)
        joint_fk_position_with_root_motion, _ = skeleton.forward(joint_rotations_matrices, root_positions_with_motion)
        joint_fk_position_with_root_motion = joint_fk_position_with_root_motion.view(-1, skeleton.nb_joints * 3)

        if plot_contacts:
            feet_positions_root_motion = joint_fk_position_with_root_motion[:, feed_idx]
        else:
            feet_positions_root_motion = None

    # look-at
    if plot_lookat:
        lookat_name = "Head"
        skeleton_lookat_idx = skeleton.bone_indexes[lookat_name]
        lookat_position = joint_fk_position.view(-1, skeleton.nb_joints, 3)[:, skeleton_lookat_idx, :]
        lookat_rotation_mat = joint_fk_rotation[:, skeleton_lookat_idx, :, :]
        lookat_direction = torch.zeros((lookat_rotation_mat.shape[0], 3))
        lookat_direction[:, 2] = 1.0
        lookat_direction = torch.matmul(lookat_rotation_mat, lookat_direction.unsqueeze(2)).squeeze(1)

    # display space
    pos_range = np.max(np.abs(joint_positions.view(-1, 3).cpu().numpy()), axis=0)
    horizontal_extent = 4 * max(pos_range[0], pos_range[2])
    vertical_extent = pos_range[1]

    plot = AnimationPlot(skeleton, timestep, horizontal_extent, vertical_extent)

    # joint positions
    if plot_joint_positions:
        plot.add_sequence(joint_positions, feet_contacts if plot_contacts else None, feet_positions, None, None, color='rgb(0,0,0)')

    # joint rotations (using FK)
    if plot_joint_rotations:
        plot.add_sequence(joint_fk_position, None, None, None, None, color='rgb(0,0,255)')

    # floor
    if plot_ground:
        plot.add_ground_sequence(height=0.0)

    # root motion (using FK)
    if plot_joint_rotations_after_root_motion:
        plot.add_sequence(joint_fk_position_with_root_motion, feet_contacts if plot_contacts else None,
                          feet_positions_root_motion, None, None, color='rgb(255,0,0)')

    if plot_lookat:
        plot.add_lookat_sequence(lookat_position, lookat_direction)

    plot.plot()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=True, description="Plot sequence")
    parser.add_argument('--dataset_path', type=str, default="./datasets", help='Path to dataset storage')
    parser.add_argument('--dataset_name', type=str, default="deeppose_master_v1_fps60", help='Name of the dataset')
    parser.add_argument('--subset', type=str, default="Training", help='Subset of the dataset')
    parser.add_argument('--rate', type=float, default=60.0, help='Frame rate')
    parser.add_argument('--clip', type=int, default=0, help='Clip index')
    parser.add_argument('--plot_ground', dest='ground', action='store_true', help="Plot the ground plane")
    parser.add_argument('--no-plot_ground', dest='ground', action='store_false', help="Don't plot the gorund plane")
    parser.set_defaults(ground=True)
    parser.add_argument('--positions', dest='positions', action='store_true', help="Plot raw positions")
    parser.add_argument('--no-positions', dest='positions', action='store_false', help="Don't raw positions")
    parser.set_defaults(positions=True)
    parser.add_argument('--rotations', dest='rotations', action='store_true', help="Plot positions obtained by forward kinematics")
    parser.add_argument('--no-rotations', dest='rotations', action='store_false', help="PDon't plot positions obtained by forward kinematics")
    parser.set_defaults(rotations=True)
    parser.add_argument('--root_motion', dest='root_motion', action='store_true', help="Plot positions obtained after root motion and forward kinematics")
    parser.add_argument('--no-root_motion', dest='root_motion', action='store_false', help="Don't plot positions obtained after root motion and forward kinematics")
    parser.set_defaults(root_motion=True)
    parser.add_argument('--contacts', dest='contacts', action='store_true', help="Plot feet contacts")
    parser.add_argument('--no-contacts', dest='contacts', action='store_false', help="Don't plot feet contacts")
    parser.set_defaults(contacts=True)
    parser.add_argument('--lookat', dest='lookat', action='store_true', help="Plot head look-at direction")
    parser.add_argument('--no-lookat', dest='lookat', action='store_false', help="Don't plot head look-at direction")
    parser.set_defaults(lookat=True)
    args = parser.parse_args()

    dataset_path = "./datasets"
    dataset_name = "deeppose_master_v3_fps60"
    split = SplitFileDatabaseLoader(args.dataset_path).pull(args.dataset_name)
    dataset = TypedColumnSequenceDataset(split, subset=args.subset)
    plot_dataset_sequence(dataset, idx=args.clip, frame_rate=args.rate, plot_ground=args.ground,
                          plot_joint_positions=args.positions, plot_joint_rotations=args.rotations,
                          plot_joint_rotations_after_root_motion=args.root_motion, plot_contacts=args.contacts,
                          plot_lookat=args.lookat)
