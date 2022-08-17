import os
import json
import pandas as pd
import torch
import numpy as np
from omegaconf import OmegaConf

from protores.models.optional_lookat_model import TYPE_VALUES
from protores.utils.model_factory import ModelFactory
from protores.data.base_dataset import BaseDataset
from protores.geometry.skeleton import Skeleton
from protores.data.dataset.typed_table import FlatTypedColumnDataset
from protores.geometry.rotations import compute_rotation_matrix_from_quaternion,\
    compute_ortho6d_from_rotation_matrix, compute_rotation_matrix_from_ortho6d
from protores.geometry.quaternions import compute_quaternions_from_rotation_matrices
from glob import glob
from protores.geometry.rotations import compute_geodesic_distance_from_two_matrices

        
class BatchedRandomEffectorDataPacked:
    def __init__(self, dataframe, device):
        """
        Building a single huge batch of packed data
        """
        df = dataframe
        self.length = len(df)
        self.frame_indices = list(df.values[:, 0])
        self.device = device
        
        n_effectors = [int(c.split("_")[0].split("effector")[-1]) for c in df.columns if 'effector' in c]
        n_effectors = max(n_effectors) + 1
        self.n_effectors = n_effectors

        batch = {}
        batchsize = len(dataframe)

        effector_type = df[[f"effector{i}_type" for i in range(n_effectors)]].values
        batch["effector_type"] = torch.tensor(effector_type, dtype=torch.int64).to(device)
        
        effector_id = df[[f"effector{i}_id" for i in range(n_effectors)]].values
        batch["effector_id"] = torch.tensor(effector_id, dtype=torch.int64).to(device)
        
        effectors_columns = []
        effectors_data_types = ['X', 'Y', 'Z', 'A', 'B', 'C', 'T']
        for i in range(n_effectors):
            effectors_columns.extend([f"effector{i}_data_{data_type}" for data_type in effectors_data_types])
        effectors = df[effectors_columns].values.reshape(batchsize, n_effectors, len(effectors_data_types))
        batch["effectors"] = torch.tensor(effectors, dtype=torch.float).to(device)
        
        effector_weight = df[[f"effector{i}_weight" for i in range(n_effectors)]].values
        batch["effector_weight"] = torch.tensor(effector_weight, dtype=torch.float).to(device)

        self.batch = batch


def fixed_points_effector_batch(csv_path, effectors, skeleton, device, with_bone_rotation=False, bone="Hips"):
    df = pd.read_csv(csv_path)#.reset_index(drop=True, inplace=True)
    batchsize = len(df)
    njoints = len(effectors)
    ids = torch.tensor([skeleton.bone_indexes[e] for e in effectors], dtype=torch.int64).unsqueeze(0).repeat(batchsize, 1)
    weights = torch.ones((batchsize, njoints), dtype=torch.float)
    tolerances = torch.zeros((batchsize, njoints), dtype=torch.float)

    if with_bone_rotation:
        position_data = torch.FloatTensor(df.values[:, :-4]).pin_memory().view(-1, njoints, 3)
        bone_rotations = torch.FloatTensor(df.values[:, -4:]).pin_memory().view(-1, 1, 4)[:, :, [3, 0, 1, 2]]  # Quat order
        bone_rotations = compute_rotation_matrix_from_quaternion(bone_rotations.view(-1, 4))
        bone_rotations = compute_ortho6d_from_rotation_matrix(bone_rotations).view(-1, 1, 6)

        batch = {"position_data": position_data,
                 "position_weight": weights,
                 "position_tolerance": tolerances,
                 "position_id": ids,
                 "rotation_data": bone_rotations,
                 "rotation_weight": torch.ones((batchsize, 1)).type_as(position_data),
                 "rotation_tolerance": torch.zeros((batchsize, 1)).type_as(position_data),
                 "rotation_id": torch.zeros((batchsize, 1), dtype=torch.int64).to(device) + skeleton.bone_indexes[bone],  # Hips ID = 0
                 "lookat_data": torch.zeros((batchsize, 0, 6)).type_as(position_data),
                 "lookat_weight": torch.zeros((batchsize, 0)).type_as(position_data),
                 "lookat_tolerance": torch.zeros((batchsize, 0)).type_as(position_data),
                 "lookat_id": torch.zeros((batchsize, 0), dtype=torch.int64).to(device)}

    else:
        position_data = torch.FloatTensor(df.values).pin_memory().view(-1, njoints, 3)
        batch = {"position_data": position_data,
                      "position_weight": weights,
                      "position_tolerance": tolerances,
                      "position_id": ids,
                      "rotation_data": torch.zeros((batchsize, 0, 6)).type_as(position_data),
                      "rotation_weight": torch.zeros((batchsize, 0)).type_as(position_data),
                      "rotation_tolerance": torch.zeros((batchsize, 0)).type_as(position_data),
                      "rotation_id": torch.zeros((batchsize, 0), dtype=torch.int64).to(device),
                      "lookat_data": torch.zeros((batchsize, 0, 6)).type_as(position_data),
                      "lookat_weight": torch.zeros((batchsize, 0)).type_as(position_data),
                      "lookat_tolerance": torch.zeros((batchsize, 0)).type_as(position_data),
                      "lookat_id": torch.zeros((batchsize, 0), dtype=torch.int64).to(device)}

    return batch


def fixed_points_to_packed_format(fixedpoints_batch,  n_effectors, outfile):
    ids = fixedpoints_batch['position_id'].detach().cpu().numpy()
    types = np.zeros_like(ids) + TYPE_VALUES['position']
    pos_data = fixedpoints_batch['position_data'].detach().cpu().numpy()
    pos_data = np.concatenate([pos_data, np.zeros((pos_data.shape[0], pos_data.shape[1], 4), dtype=np.float32)], axis=-1)  # zero-padding

    bone_rotations = False
    if fixedpoints_batch['rotation_data'].shape[1] > 0:
        bone_rotations = True
        rot_data = fixedpoints_batch['rotation_data'].detach().cpu().numpy()
        rot_data = np.concatenate([rot_data, np.zeros((rot_data.shape[0], rot_data.shape[1], 1), dtype=np.float32)], axis=-1) # zero-padding
        rot_ids = fixedpoints_batch['rotation_id'].detach().cpu().numpy()
        rot_types = np.zeros_like(ids) + TYPE_VALUES['rotation']

    output_df = pd.DataFrame()
    loc = 0
    data_dims_id = ['X', 'Y', 'Z', 'A', 'B', 'C', 'T']

    # Populate df
    for eff_i in range(n_effectors):
        output_df.insert(loc, "effector{}_type".format(eff_i), types[:, eff_i])
        loc += 1
        output_df.insert(loc, "effector{}_id".format(eff_i), ids[:, eff_i])
        loc += 1
        output_df.insert(loc, "effector{}_weight".format(eff_i), np.ones_like(types[:, eff_i], dtype=np.float32))
        loc += 1
        for dim in range(pos_data.shape[-1]):
            output_df.insert(loc, "effector{0}_data_{1}".format(eff_i, data_dims_id[dim]), pos_data[:, eff_i, dim])
            loc += 1

    if bone_rotations:
        output_df.insert(loc, "effector{}_type".format(n_effectors), rot_types[:, 0])
        loc += 1
        output_df.insert(loc, "effector{}_id".format(n_effectors), rot_ids[:, 0])
        loc += 1
        output_df.insert(loc, "effector{}_weight".format(n_effectors), np.ones_like(types[:, 0], dtype=np.float32))
        loc += 1
        for dim in range(rot_data.shape[-1]):
            output_df.insert(loc, "effector{0}_data_{1}".format(n_effectors, data_dims_id[dim]), rot_data[:, 0, dim])
            loc += 1

    output_df['Frame'] = list(range(pos_data.shape[0]))
    output_df.set_index('Frame', inplace=True)

    # Save df
    output_df.to_csv(outfile, index=True)


def remove_df_whitespaces(df):
    replace_dict = {}
    for k in df.keys():
        replace_dict[k] = k.strip(' ')
    df.rename(columns=replace_dict, inplace=True)


def build_dataset(dataset_settings_path, csv_path, effectors=None):
    with open(dataset_settings_path, "r") as f:
        config = json.load(f)

    skeleton = Skeleton(config['skeleton'])
    df = pd.read_csv(csv_path)
    remove_df_whitespaces(df)
    if 'Frame' in df.columns:
        frame_indices = df['Frame'].values
        del df['Frame']  # Frame info messed up in indices for the dataset.
    else:
        frame_indices = list(range(df.shape[0]))
    dataset = FlatTypedColumnDataset(df, config)
    dataset = BaseDataset(source=dataset, skeleton=skeleton)
    return dataset, skeleton, config, frame_indices


def setup_model(model_folder, checkpoint_name, skeleton, device="cuda:0"):
    # Load the model checkpoint and hyper-params and initialize everything
    
    if checkpoint_name == "best":
        # This is if we want to load the best checkpoint
        models = glob(os.path.join(model_folder, 'checkpoints/*.ckpt'))
        models = [os.path.basename(m) for m in models]
        models = {int(m.split("=")[-1].split(".")[0]):m for m in models if m.startswith("epoch=")}
    
        k = max(models.keys())
        checkpoint_name = models[k]
        
        print(f"Will load the best checckpoint {checkpoint_name}")
    
    checkpoint_path = os.path.join(model_folder, 'checkpoints', checkpoint_name)
    hparams_path = os.path.join(model_folder, "hparams.yaml")

    # Restore model config
    cfg = OmegaConf.load(hparams_path)
    cfg = OmegaConf.to_container(cfg, resolve=True)
    cfg = OmegaConf.create(cfg)
    OmegaConf.set_struct(cfg, True)
    print(OmegaConf.to_yaml(cfg))

    # Create model, load state_dict
    model = ModelFactory.instantiate(cfg, skeleton=skeleton)
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict['state_dict'], strict=False)
    model.eval()
    print()

    return model


def save_predictions(skeleton, predictions, out_path, frame_indices=None):
    predicted_joint_rotations = predictions["joint_rotations"]
    
    if predictions["joint_positions"] is None:
        predicted_joint_positions = predictions["root_joint_position"].unsqueeze(1).cpu().detach().numpy()
        predicted_joint_positions = np.tile(predicted_joint_positions, (1, skeleton.nb_joints, 1))
    else:
        predicted_joint_positions = predictions["joint_positions"].cpu().detach().numpy()
    
    predicted_joint_rotations = compute_rotation_matrix_from_ortho6d(predicted_joint_rotations.view(-1, 6))
    predicted_joint_rotations = compute_quaternions_from_rotation_matrices(predicted_joint_rotations)
    predicted_joint_rotations = predicted_joint_rotations.view(-1, skeleton.nb_joints, 4)
    predicted_joint_rotations = predicted_joint_rotations.cpu().detach().numpy()

    all_joints = [skeleton.index_bones[i] for i in range(skeleton.nb_joints)]
    predicted_df = pd.DataFrame()

    # populate df:
    for joint in all_joints:
        predicted_df["BonePositions_{}_X".format(joint)] = predicted_joint_positions[:, skeleton.bone_indexes[joint], 0]
        predicted_df["BonePositions_{}_Y".format(joint)] = predicted_joint_positions[:, skeleton.bone_indexes[joint], 1]
        predicted_df["BonePositions_{}_Z".format(joint)] = predicted_joint_positions[:, skeleton.bone_indexes[joint], 2]

    for joint in all_joints:
        predicted_df["BoneRotations_{}_X".format(joint)] = predicted_joint_rotations[:, skeleton.bone_indexes[joint], 1]
        predicted_df["BoneRotations_{}_Y".format(joint)] = predicted_joint_rotations[:, skeleton.bone_indexes[joint], 2]
        predicted_df["BoneRotations_{}_Z".format(joint)] = predicted_joint_rotations[:, skeleton.bone_indexes[joint], 3]
        predicted_df["BoneRotations_{}_W".format(joint)] = predicted_joint_rotations[:, skeleton.bone_indexes[joint], 0]

    if frame_indices is None:
        predicted_df['Frame'] = list(range(len(predicted_df)))
    else:
        predicted_df['Frame'] = frame_indices

    predicted_df.set_index('Frame', inplace=True)
    predicted_df.to_csv(out_path, index=True)

    return out_path


def benchmark_from_csv(model, dataset_config_path, target_file, predictions_file):
    target_dataset, skeleton, _, _ = build_dataset(dataset_config_path, target_file)
    predictions_dataset, _, _, all_predicted_frames = build_dataset(dataset_config_path, predictions_file)

    target_batch = target_dataset[all_predicted_frames]
    predictions_batch = predictions_dataset[:]

    _, target_data = model.get_data_from_batch(target_batch, fixed_effector_setup=True)

    # Test metrics expects predictions in ortho6D format, let's not upset them.
    _, predictions = model.get_data_from_batch(predictions_batch, fixed_effector_setup=True)
    predicted_joint_rotations = predictions["joint_rotations"]
    predicted_joint_rotations = compute_rotation_matrix_from_quaternion(predicted_joint_rotations.view(-1, 4))
    predicted_joint_rotations = compute_ortho6d_from_rotation_matrix(predicted_joint_rotations)
    predicted_joint_rotations = predicted_joint_rotations.view(-1, skeleton.nb_joints, 6)
    predictions["joint_rotations"] = predicted_joint_rotations

    model.update_test_metrics(predictions, target_data)

    metrics = {"fk": model.test_fk_metric.compute(),
               "position": model.test_position_metric.compute(),
               "rotation": model.test_rotation_metric.compute()[0]}

    for k, v in metrics.items():
        print("\t {0} : {1}".format(k, v))

    margins = get_confidence_interval_margins(predictions, target_data, model.skeleton)

    # Reset for now.
    model.test_fk_metric.reset()
    model.test_position_metric.reset()
    model.test_rotation_metric.reset()

    return metrics, margins


def get_confidence_interval_margins(predicted, target, skeleton):
    # Used to get per-sample error samples. Can be useful to get std, ou confidence intervals.
    # This was wuickly hacked to mimick base_posing.update_test_metrics
    target_joint_positions = target["joint_positions"]
    target_root_joint_position = target["root_joint_position"]
    target_joint_rotations = target["joint_rotations"]
    predicted_root_joint_position = predicted["root_joint_position"]
    predicted_joint_rotations = predicted["joint_rotations"]
    # compute rotation matrices
    target_joint_rotations_mat = compute_rotation_matrix_from_quaternion(target_joint_rotations.view(-1, 4)).view(-1, skeleton.nb_joints, 3, 3)
    predicted_joint_rotations_mat = compute_rotation_matrix_from_ortho6d(predicted_joint_rotations.view(-1, 6)).view(-1, skeleton.nb_joints, 3, 3)

    # forward kinematics
    predicted_joint_positions, _ = skeleton.forward(predicted_joint_rotations_mat, predicted_root_joint_position)

    # Get actual error samples
    fk_samples = torch.mean(torch.pow(predicted_joint_positions.view(-1, 3) - target_joint_positions.view(-1, 3), 2), dim=-1).detach().cpu().numpy()
    pos_samples = torch.mean((predicted_root_joint_position.view(-1, 3) - target_root_joint_position.view(-1, 3)) ** 2, dim=-1).detach().cpu().numpy()
    rot_samples = compute_geodesic_distance_from_two_matrices(predicted_joint_rotations_mat.view(-1, 3, 3), target_joint_rotations_mat.view(-1, 3, 3)).detach().cpu().numpy()

    # For debug purposes. These should match the metric outputs.
    fk_mean = np.mean(fk_samples)
    pos_mean = np.mean(pos_samples)
    rot_mean = np.mean(rot_samples)

    fk_std = np.std(fk_samples)
    pos_std = np.std(pos_samples)
    rot_std = np.std(rot_samples)

    # For 95% confidence, t=1.960
    fk_confidence_margin = 1.96 * fk_std / np.sqrt(fk_samples.shape[0])
    pos_confidence_margin = 1.96 * pos_std / np.sqrt(pos_samples.shape[0])
    rot_confidence_margin = 1.96 * rot_std / np.sqrt(rot_samples.shape[0])

    margins = {
        "fk": fk_confidence_margin,
        "pos": pos_confidence_margin,
        "rot": rot_confidence_margin
    }

    return margins
